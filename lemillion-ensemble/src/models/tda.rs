use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// TDA (Topological Data Analysis) — détection de structures géométriques.
///
/// Utilise l'homologie persistante H0 (composantes connexes) via Union-Find
/// sur des nuages de points de tirages encodés en 5D. Extrait des features
/// topologiques (persistence entropy, max persistence, Betti-0) et corrèle
/// avec l'apparition de chaque numéro dans les tirages suivants.
pub struct TdaModel {
    window_size: usize,
    n_epsilon: usize,
    max_epsilon: f64,
    correlation_window: usize,
    smoothing: f64,
}

impl TdaModel {
    pub fn new(
        window_size: usize,
        n_epsilon: usize,
        max_epsilon: f64,
        correlation_window: usize,
        smoothing: f64,
    ) -> Self {
        Self {
            window_size,
            n_epsilon,
            max_epsilon,
            correlation_window,
            smoothing,
        }
    }
}

impl Default for TdaModel {
    fn default() -> Self {
        Self {
            window_size: 30,
            n_epsilon: 20,
            max_epsilon: 2.0,
            correlation_window: 50,
            smoothing: 0.6,
        }
    }
}

// --- Union-Find ---

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    n_components: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            n_components: n,
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
        self.n_components -= 1;
        true
    }
}

/// Encode un tirage en vecteur 5D (même encoding que NVAR).
fn encode_draw(numbers: &[u8], pool_size: usize) -> [f64; 5] {
    let n = numbers.len() as f64;
    let max_val = pool_size as f64;

    let sum: f64 = numbers.iter().map(|&x| x as f64).sum();
    let sum_norm = sum / (n * max_val);

    let min_v = numbers.iter().copied().min().unwrap_or(1) as f64;
    let max_v = numbers.iter().copied().max().unwrap_or(1) as f64;
    let spread_norm = (max_v - min_v) / max_val;

    let odd_ratio = numbers.iter().filter(|&&x| x % 2 == 1).count() as f64 / n;

    let centroid = sum / n / max_val;

    let mean = sum / n;
    let variance = numbers.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let var_norm = variance / (max_val * max_val);

    [sum_norm, spread_norm, odd_ratio, centroid, var_norm]
}

/// Distance euclidienne entre deux points 5D.
fn dist5d(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Features topologiques extraites de l'homologie persistante H0.
struct TopoFeatures {
    persistence_entropy: f64,
    max_persistence: f64,
    n_significant: f64,
    betti0_at_median: f64,
}

/// Calcule les features topologiques H0 d'un nuage de points via Vietoris-Rips filtration.
fn compute_topo_features(points: &[[f64; 5]]) -> TopoFeatures {
    let n = points.len();
    if n < 2 {
        return TopoFeatures {
            persistence_entropy: 0.0,
            max_persistence: 0.0,
            n_significant: 0.0,
            betti0_at_median: 1.0,
        };
    }

    // Calculer toutes les distances pairwise et trier
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((dist5d(&points[i], &points[j]), i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find pour tracker les fusions (= diagramme de persistance H0)
    let mut uf = UnionFind::new(n);
    let _birth_times = vec![0.0f64; n]; // Toutes les composantes naissent à ε=0
    let mut persistences: Vec<f64> = Vec::new();
    let mut betti0_at_median = n;
    let median_edge_idx = edges.len() / 2;

    for (idx, &(dist, i, j)) in edges.iter().enumerate() {
        if uf.union(i, j) {
            // Une composante meurt à distance `dist`
            // La persistence = death - birth = dist - 0 = dist
            persistences.push(dist);
        }
        if idx == median_edge_idx {
            betti0_at_median = uf.n_components;
        }
    }

    if persistences.is_empty() {
        return TopoFeatures {
            persistence_entropy: 0.0,
            max_persistence: 0.0,
            n_significant: 0.0,
            betti0_at_median: betti0_at_median as f64,
        };
    }

    // Max persistence
    let max_p = persistences.iter().cloned().fold(0.0f64, f64::max);

    // Persistence entropy : H = -Σ p_i log(p_i) où p_i = persistence_i / total
    let total_p: f64 = persistences.iter().sum();
    let entropy = if total_p > 0.0 {
        persistences
            .iter()
            .map(|&p| {
                let ratio = p / total_p;
                if ratio > 1e-15 {
                    -ratio * ratio.ln()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    // N significant : persistence > median
    let mut sorted_p = persistences.clone();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_p = sorted_p[sorted_p.len() / 2];
    let n_sig = persistences.iter().filter(|&&p| p > median_p).count();

    // Normaliser par n pour que les features soient comparables entre fenêtres
    TopoFeatures {
        persistence_entropy: entropy,
        max_persistence: max_p,
        n_significant: n_sig as f64 / n as f64,
        betti0_at_median: betti0_at_median as f64 / n as f64,
    }
}

impl ForecastModel for TdaModel {
    fn name(&self) -> &str {
        "TDA"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        let min_required = self.window_size + self.correlation_window + 1;
        if draws.len() < min_required {
            return uniform;
        }

        // Ordre chronologique
        let chronological: Vec<&Draw> = draws.iter().rev().collect();
        let n = chronological.len();

        // Encoder tous les tirages
        let encoded: Vec<[f64; 5]> = chronological
            .iter()
            .map(|d| encode_draw(pool.numbers_from(d), size))
            .collect();

        // Pour chaque fenêtre glissante, calculer les features topologiques
        let n_windows = n.saturating_sub(self.window_size);
        if n_windows < 2 {
            return uniform;
        }

        let mut topo_features: Vec<[f64; 4]> = Vec::with_capacity(n_windows);
        for t in 0..n_windows {
            let window = &encoded[t..t + self.window_size];
            let feat = compute_topo_features(window);
            topo_features.push([
                feat.persistence_entropy,
                feat.max_persistence,
                feat.n_significant,
                feat.betti0_at_median,
            ]);
        }

        // Pour chaque numéro k : corréler les features topologiques avec l'apparition
        // de k dans le tirage SUIVANT la fenêtre
        let corr_window = self.correlation_window.min(n_windows.saturating_sub(1));
        if corr_window < 2 {
            return uniform;
        }

        let mut scores = vec![0.0f64; size];

        for num in 1..=size as u8 {
            let num_idx = (num - 1) as usize;

            // Série d'apparition de ce numéro dans les tirages post-fenêtre
            // et features topologiques correspondantes
            let start = n_windows.saturating_sub(corr_window);
            let end = n_windows;

            let mut appearances: Vec<f64> = Vec::with_capacity(end - start);
            let mut feat_means = [0.0f64; 4];

            for t in start..end {
                let target_idx = t + self.window_size;
                if target_idx < n {
                    let appeared = if pool
                        .numbers_from(chronological[target_idx])
                        .contains(&num)
                    {
                        1.0
                    } else {
                        0.0
                    };
                    appearances.push(appeared);
                    for (k, m) in feat_means.iter_mut().enumerate() {
                        *m += topo_features[t][k];
                    }
                }
            }

            let n_samples = appearances.len();
            if n_samples < 2 {
                continue;
            }

            // Moyenne des features
            for m in &mut feat_means {
                *m /= n_samples as f64;
            }

            // Corrélation entre chaque feature et l'apparition
            let mean_app = appearances.iter().sum::<f64>() / n_samples as f64;
            let var_app: f64 = appearances
                .iter()
                .map(|&a| (a - mean_app).powi(2))
                .sum::<f64>();

            let mut total_corr = 0.0f64;
            for feat_idx in 0..4 {
                let feat_mean = feat_means[feat_idx];
                let mut cov = 0.0f64;
                let mut var_f = 0.0f64;

                for (i, t) in (start..end).enumerate() {
                    if i < appearances.len() {
                        let f_val = topo_features[t][feat_idx] - feat_mean;
                        let a_val = appearances[i] - mean_app;
                        cov += f_val * a_val;
                        var_f += f_val * f_val;
                    }
                }

                if var_f > 1e-15 && var_app > 1e-15 {
                    let corr = cov / (var_f.sqrt() * var_app.sqrt());
                    total_corr += corr.abs();
                }
            }

            // Score basé sur la corrélation absolue moyenne + fréquence de base
            scores[num_idx] = mean_app + 0.5 * total_corr / 4.0;
        }

        // Smooth + normalize
        let uniform_val = 1.0 / size as f64;
        for p in &mut scores {
            *p = (1.0 - self.smoothing) * (*p).max(1e-10) + self.smoothing * uniform_val;
        }
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for p in &mut scores {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("window_size".into(), self.window_size as f64),
            ("n_epsilon".into(), self.n_epsilon as f64),
            ("max_epsilon".into(), self.max_epsilon),
            ("correlation_window".into(), self.correlation_window as f64),
            ("smoothing".into(), self.smoothing),
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
    fn test_tda_balls_sums_to_one() {
        let model = TdaModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_tda_stars_sums_to_one() {
        let model = TdaModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_tda_no_negative() {
        let model = TdaModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_tda_empty_draws() {
        let model = TdaModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tda_few_draws() {
        let model = TdaModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tda_deterministic() {
        let model = TdaModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "TDA should be deterministic");
        }
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.n_components, 5);
        assert!(uf.union(0, 1));
        assert_eq!(uf.n_components, 4);
        assert!(!uf.union(0, 1)); // Already same component
        assert_eq!(uf.n_components, 4);
        assert!(uf.union(2, 3));
        assert_eq!(uf.n_components, 3);
        assert!(uf.union(0, 2));
        assert_eq!(uf.n_components, 2);
    }

    #[test]
    fn test_topo_features_single_point() {
        let points = vec![[0.5, 0.3, 0.5, 0.25, 0.1]];
        let feat = compute_topo_features(&points);
        assert_eq!(feat.max_persistence, 0.0);
    }

    #[test]
    fn test_topo_features_two_points() {
        let points = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let feat = compute_topo_features(&points);
        assert!(feat.max_persistence > 0.0);
    }

    #[test]
    fn test_encode_draw_range() {
        let numbers = [1u8, 25, 50];
        let enc = encode_draw(&numbers, 50);
        for &v in &enc {
            assert!(v >= 0.0 && v <= 1.0, "Encoded value out of range: {v}");
        }
    }
}
