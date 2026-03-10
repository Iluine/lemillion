use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// SpectralGraph — Spectral Graph Laplacian for co-occurrence community detection.
///
/// Builds a weighted co-occurrence graph of the N numbers, computes the
/// normalized Laplacian's smallest non-trivial eigenvectors via power iteration,
/// and predicts by proximity to the last draw's spectral centroid.
///
/// VERY orthogonal to TE: TE measures temporal causality between pairs,
/// spectral analysis measures static community structure in the co-occurrence graph.
pub struct SpectralGraphModel {
    n_components: usize,
    window: usize,
    smoothing: f64,
    min_draws: usize,
}

impl Default for SpectralGraphModel {
    fn default() -> Self {
        Self {
            n_components: 3,
            window: 500,
            smoothing: 0.25,
            min_draws: 100,
        }
    }
}

/// Compute spectral embedding of the normalized graph Laplacian.
/// Returns k eigenvectors of (I - L_sym) = D^{-1/2} A D^{-1/2}
/// corresponding to the largest eigenvalues (= smallest Laplacian eigenvalues).
fn spectral_embedding(adjacency: &[f64], n: usize, k: usize) -> Vec<Vec<f64>> {
    // Degree vector
    let mut degree = vec![0.0f64; n];
    for i in 0..n {
        degree[i] = (0..n).map(|j| adjacency[i * n + j]).sum::<f64>();
    }
    let d_inv_sqrt: Vec<f64> = degree.iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // M = (I + D^{-1/2} A D^{-1/2}) / 2
    // Shifts eigenvalues from [-1, 1] to [0, 1], ensuring power iteration convergence.
    let mut a_norm = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a_norm[i * n + j] = 0.5 * d_inv_sqrt[i] * adjacency[i * n + j] * d_inv_sqrt[j];
        }
        a_norm[i * n + i] += 0.5;
    }

    // Trivial eigenvector of A_norm (eigenvalue=1) is D^{1/2}*ones (normalized)
    let mut trivial = degree.iter().map(|&d| d.sqrt()).collect::<Vec<f64>>();
    let tn = trivial.iter().map(|x| x * x).sum::<f64>().sqrt();
    if tn > 1e-15 {
        for x in &mut trivial { *x /= tn; }
    }

    // Power iteration for top-k non-trivial eigenvectors
    let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(k);

    for comp in 0..k {
        // Initialize with deterministic pseudo-random values
        let mut v: Vec<f64> = (0..n)
            .map(|i| ((i + comp * 7 + 1) as f64).sin())
            .collect();

        for _iter in 0..200 {
            // Matrix-vector multiply: A_norm * v
            let mut new_v = vec![0.0f64; n];
            for i in 0..n {
                let row = i * n;
                for j in 0..n {
                    new_v[i] += a_norm[row + j] * v[j];
                }
            }

            // Orthogonalize against trivial eigenvector D^{1/2}*ones
            let dot_trivial: f64 = new_v.iter().zip(trivial.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n { new_v[i] -= dot_trivial * trivial[i]; }

            // Orthogonalize against previous eigenvectors (Gram-Schmidt)
            for prev in &eigenvecs {
                let dot: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for i in 0..n { new_v[i] -= dot * prev[i]; }
            }

            // Normalize
            let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in &mut new_v { *x /= norm; }
            }
            v = new_v;
        }
        eigenvecs.push(v);
    }
    eigenvecs
}

impl ForecastModel for SpectralGraphModel {
    fn name(&self) -> &str {
        "SpectralGraph"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let window = self.window.min(draws.len());
        let recent = &draws[..window];

        // 1. Build adjacency matrix
        let mut adj = vec![0.0f64; size * size];
        for d in recent {
            let nums: Vec<usize> = pool.numbers_from(d).iter()
                .map(|&x| (x - 1) as usize).collect();
            for a in 0..nums.len() {
                for b in (a + 1)..nums.len() {
                    adj[nums[a] * size + nums[b]] += 1.0;
                    adj[nums[b] * size + nums[a]] += 1.0;
                }
            }
        }

        // 2. Spectral embedding
        let embedding = spectral_embedding(&adj, size, self.n_components);

        // 3. Last draw centroid in spectral space
        let last_nums: Vec<usize> = pool.numbers_from(&draws[0])
            .iter().map(|&x| (x - 1) as usize).collect();

        let mut centroid = vec![0.0f64; self.n_components];
        for &idx in &last_nums {
            for c in 0..self.n_components {
                centroid[c] += embedding[c][idx];
            }
        }
        let n_last = last_nums.len() as f64;
        for c in &mut centroid { *c /= n_last; }

        // 4. Frequency baseline
        let mut freq = vec![0.0f64; size];
        for d in recent {
            for &num in pool.numbers_from(d) {
                freq[(num - 1) as usize] += 1.0;
            }
        }
        let freq_total: f64 = freq.iter().sum();
        if freq_total > 0.0 {
            for f in &mut freq { *f /= freq_total; }
        }

        // 5. Score = frequency × (1 + proximity to centroid)
        let mut probs = vec![0.0f64; size];
        for i in 0..size {
            let dist_sq: f64 = (0..self.n_components)
                .map(|c| (embedding[c][i] - centroid[c]).powi(2))
                .sum();
            let proximity = 1.0 / (1.0 + dist_sq * 10.0);
            probs[i] = freq[i] * (1.0 + proximity);
        }

        // 6. Smooth & normalize
        let uniform_val = 1.0 / size as f64;
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = if pool == Pool::Balls { PROB_FLOOR_BALLS } else { PROB_FLOOR_STARS };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_components".into(), self.n_components as f64),
            ("window".into(), self.window as f64),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_spectral_graph_balls_sums_to_one() {
        let model = SpectralGraphModel::default();
        let draws = make_test_draws(150);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_spectral_graph_stars_sums_to_one() {
        let model = SpectralGraphModel::default();
        let draws = make_test_draws(150);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_spectral_graph_few_draws_uniform() {
        let model = SpectralGraphModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spectral_graph_deterministic() {
        let model = SpectralGraphModel::default();
        let draws = make_test_draws(150);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_spectral_graph_no_negative() {
        let model = SpectralGraphModel::default();
        let draws = make_test_draws(150);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_spectral_embedding_basic() {
        // Simple 4-node graph: two pairs connected
        let n = 4;
        let mut adj = vec![0.0f64; n * n];
        // Pair (0,1) and pair (2,3) strongly connected
        adj[0 * n + 1] = 10.0; adj[1 * n + 0] = 10.0;
        adj[2 * n + 3] = 10.0; adj[3 * n + 2] = 10.0;
        // Weak inter-group connections
        adj[0 * n + 2] = 1.0; adj[2 * n + 0] = 1.0;
        adj[1 * n + 3] = 1.0; adj[3 * n + 1] = 1.0;

        let embedding = spectral_embedding(&adj, n, 1);
        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 4);

        // Fiedler vector should separate the two groups:
        // nodes 0,1 should have similar values, nodes 2,3 similar values,
        // and the two groups should have opposite signs
        let v = &embedding[0];
        let same_group_01 = (v[0] - v[1]).abs();
        let same_group_23 = (v[2] - v[3]).abs();
        let diff_groups = (v[0] - v[2]).abs();
        assert!(same_group_01 < diff_groups, "Same-group distance ({}) should be < inter-group ({})", same_group_01, diff_groups);
        assert!(same_group_23 < diff_groups, "Same-group distance ({}) should be < inter-group ({})", same_group_23, diff_groups);
    }
}
