use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS};

/// TensorModel — CP Tensor Decomposition on 5th-order co-occurrence tensor.
///
/// Builds a sparse co-occurrence tensor from observed ball quintuplets and
/// decomposes it via simplified ALS (Alternating Least Squares) with R components.
///
/// Each component r is an "archetype" draw pattern represented by factor vectors
/// a_r ∈ R^50 for each of the 5 positions, with weight λ_r.
///
/// The CP decomposition approximates:
///   T(i1,i2,i3,i4,i5) ≈ Σ_r λ_r · a1_r(i1) · a2_r(i2) · a3_r(i3) · a4_r(i4) · a5_r(i5)
///
/// Prediction:
/// 1. Project recent draws onto the R components to get activation weights
/// 2. Marginal probability for number k = Σ_r activation_r · Σ_pos a_pos_r(k)
///
/// Stars: returns uniform (balls-only model).
pub struct TensorModel {
    smoothing: f64,
    n_components: usize,
    min_draws: usize,
    als_iterations: usize,
}

impl Default for TensorModel {
    fn default() -> Self {
        Self {
            smoothing: 0.30,
            n_components: 10,
            min_draws: 50,
            als_iterations: 20,
        }
    }
}

/// A single CP component: weight λ and 5 factor vectors (one per ball position).
struct CpComponent {
    lambda: f64,
    /// factors[pos][ball_idx] — factor vector for each of the 5 positions
    factors: [[f64; 50]; 5],
}

/// Sparse co-occurrence entry: (ball indices as sorted tuple, count).
/// Since the full 50^5 tensor is too large, we only store observed quintuplets.
struct SparseTensor {
    /// Map from (b1, b2, b3, b4, b5) sorted indices to count
    entries: HashMap<[u8; 5], f64>,
    /// Marginal counts per position: marginals[pos][ball_idx]
    marginals: [[f64; 50]; 5],
    total: f64,
}

impl SparseTensor {
    fn from_draws(draws: &[Draw]) -> Self {
        let mut entries: HashMap<[u8; 5], f64> = HashMap::new();
        let mut marginals = [[0.0f64; 50]; 5];
        let total = draws.len() as f64;

        for d in draws {
            // balls are already sorted ascending
            *entries.entry(d.balls).or_insert(0.0) += 1.0;
            for (pos, &b) in d.balls.iter().enumerate() {
                marginals[pos][(b - 1) as usize] += 1.0;
            }
        }

        Self { entries, marginals, total }
    }
}

/// Initialize CP components with deterministic pseudo-random values seeded
/// from the marginal distributions.
fn init_components(tensor: &SparseTensor, n_components: usize) -> Vec<CpComponent> {
    let mut components = Vec::with_capacity(n_components);

    for r in 0..n_components {
        let mut factors = [[0.0f64; 50]; 5];
        for pos in 0..5 {
            for k in 0..50 {
                // Mix marginal frequency with deterministic pseudo-random perturbation
                let marginal_val = if tensor.total > 0.0 {
                    tensor.marginals[pos][k] / tensor.total
                } else {
                    1.0 / 50.0
                };
                // Deterministic seed: sin(r*50*5 + pos*50 + k + 1)
                let perturb = ((r * 250 + pos * 50 + k + 1) as f64).sin().abs() * 0.1;
                factors[pos][k] = marginal_val + perturb;
            }
            // Normalize factor vector to unit L1 norm
            let sum: f64 = factors[pos].iter().sum();
            if sum > 0.0 {
                for k in 0..50 {
                    factors[pos][k] /= sum;
                }
            }
        }
        // Initial lambda proportional to component index (larger components first)
        let lambda = 1.0 / (r + 1) as f64;
        components.push(CpComponent { lambda, factors });
    }

    components
}

/// One ALS update step: fix all factors except `update_pos`, solve for it.
///
/// For each ball index k at position `update_pos`:
///   a[update_pos][k] = Σ_{entries} count · Π_{pos≠update_pos} a[pos][entry.balls[pos]]
///                      where entry.balls[update_pos] == k
fn als_update(
    components: &mut [CpComponent],
    tensor: &SparseTensor,
    update_pos: usize,
) {
    for comp in components.iter_mut() {
        let mut numerator = [0.0f64; 50];
        let mut denominator = 0.0f64;

        for (balls, &count) in &tensor.entries {
            // Product of factors at all OTHER positions for this entry
            let mut prod = comp.lambda;
            for pos in 0..5 {
                if pos != update_pos {
                    let ball_idx = (balls[pos] - 1) as usize;
                    prod *= comp.factors[pos][ball_idx];
                }
            }
            let target_idx = (balls[update_pos] - 1) as usize;
            numerator[target_idx] += count * prod;
            denominator += count * prod * comp.factors[update_pos][target_idx];
        }

        // Update factor vector
        let norm_sq: f64 = numerator.iter().map(|x| x * x).sum::<f64>();
        if norm_sq > 1e-30 {
            let norm = norm_sq.sqrt();
            for k in 0..50 {
                comp.factors[update_pos][k] = numerator[k].max(0.0) / norm;
            }
            // Absorb norm into lambda
            if denominator > 1e-30 {
                comp.lambda = denominator;
            }
        }
    }
}

/// Compute marginal probabilities from CP decomposition.
/// P(ball k) ∝ Σ_r λ_r · Σ_{pos} a_r[pos][k]
#[cfg(test)]
fn marginal_from_cp(components: &[CpComponent]) -> Vec<f64> {
    let mut probs = vec![0.0f64; 50];

    for comp in components {
        for k in 0..50 {
            let mut sum_across_positions = 0.0;
            for pos in 0..5 {
                sum_across_positions += comp.factors[pos][k];
            }
            probs[k] += comp.lambda * sum_across_positions;
        }
    }

    // Normalize
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        for p in &mut probs {
            *p = 1.0 / 50.0;
        }
    }

    probs
}

/// Project recent draws onto components to get activation weights.
/// activation_r = Σ_{recent draws} Π_{pos} a_r[pos][ball[pos]]
fn project_recent(
    components: &[CpComponent],
    recent_draws: &[Draw],
) -> Vec<f64> {
    let n_comp = components.len();
    let mut activations = vec![0.0f64; n_comp];

    for d in recent_draws {
        for (r, comp) in components.iter().enumerate() {
            let mut prod = 1.0;
            for (pos, &b) in d.balls.iter().enumerate() {
                prod *= comp.factors[pos][(b - 1) as usize];
            }
            activations[r] += prod;
        }
    }

    // Normalize activations
    let sum: f64 = activations.iter().sum();
    if sum > 0.0 {
        for a in &mut activations {
            *a /= sum;
        }
    } else {
        for a in &mut activations {
            *a = 1.0 / n_comp as f64;
        }
    }

    activations
}

impl ForecastModel for TensorModel {
    fn name(&self) -> &str {
        "TensorCP"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Stars: uniform (balls-only model)
        if pool == Pool::Stars {
            return uniform;
        }

        if draws.len() < self.min_draws {
            return uniform;
        }

        // 1. Build sparse co-occurrence tensor from all draws
        let tensor = SparseTensor::from_draws(draws);
        if tensor.total < 1.0 {
            return uniform;
        }

        // 2. Initialize CP components
        let mut components = init_components(&tensor, self.n_components);

        // 3. ALS iterations: cycle through positions
        for _iter in 0..self.als_iterations {
            for pos in 0..5 {
                als_update(&mut components, &tensor, pos);
            }
        }

        // 4. Project recent draws (last 10) to weight components
        let n_recent = 10.min(draws.len());
        let recent = &draws[..n_recent];
        let activations = project_recent(&components, recent);

        // 5. Weighted marginal: P(k) ∝ Σ_r activation_r · Σ_pos a_r[pos][k]
        let mut probs = vec![0.0f64; 50];
        for (r, comp) in components.iter().enumerate() {
            let weight = activations[r] * comp.lambda;
            for k in 0..50 {
                let mut sum_pos = 0.0;
                for pos in 0..5 {
                    sum_pos += comp.factors[pos][k];
                }
                probs[k] += weight * sum_pos;
            }
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smooth towards uniform
        let uniform_val = 1.0 / size as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = PROB_FLOOR_BALLS;
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("n_components".into(), self.n_components as f64),
            ("min_draws".into(), self.min_draws as f64),
            ("als_iterations".into(), self.als_iterations as f64),
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
    fn test_tensor_valid_ball_distribution() {
        let draws = make_test_draws(100);
        let model = TensorModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_tensor_stars_uniform() {
        let draws = make_test_draws(100);
        let model = TensorModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!(
                (p - expected).abs() < 1e-10,
                "Stars should be uniform for TensorCP, got {}",
                p
            );
        }
    }

    #[test]
    fn test_tensor_few_draws_uniform() {
        let draws = make_test_draws(10);
        let model = TensorModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!(
                (p - expected).abs() < 1e-10,
                "Few draws should return uniform"
            );
        }
    }

    #[test]
    fn test_tensor_no_negative() {
        let draws = make_test_draws(100);
        let model = TensorModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_tensor_deterministic() {
        let draws = make_test_draws(100);
        let model = TensorModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-12, "TensorCP should be deterministic");
        }
    }

    #[test]
    fn test_tensor_empty_draws() {
        let model = TensorModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tensor_large_draws() {
        let draws = make_test_draws(300);
        let model = TensorModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_sparse_tensor_counts() {
        let draws = make_test_draws(20);
        let tensor = SparseTensor::from_draws(&draws);
        assert_eq!(tensor.total, 20.0);
        // Each draw contributes 1 entry; with 10 patterns cycling, we should see
        // counts summing to 20
        let entry_sum: f64 = tensor.entries.values().sum();
        assert!((entry_sum - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_marginal_from_cp_normalized() {
        let draws = make_test_draws(100);
        let tensor = SparseTensor::from_draws(&draws);
        let components = init_components(&tensor, 5);
        let probs = marginal_from_cp(&components);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Marginal should sum to 1.0, got {}",
            sum
        );
    }
}
