use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// GnnCooccurrenceModel — Simplified 1-layer message-passing GNN on co-occurrence graph.
///
/// Builds a co-occurrence graph where nodes are numbers (50 balls or 12 stars)
/// and edge weights are EWMA co-occurrence frequencies.
///
/// Node features (5D): [frequency_ewma, gap, mod8_class, decade, trend]
///
/// Message passing (1 layer):
///   h_i = tanh(W1·x_i + W2·Σ_{j∈N(i)} e_ij·x_j)
///
/// Readout via ridge regression (not SGD):
///   p_i = softmax(W3·h_i)
///
/// W1, W2 are deterministic (based on co-occurrence structure).
/// W3 is learned by ridge regression on historical draws.
///
/// This captures graph-structural information: numbers that frequently co-occur
/// with recently-drawn numbers get boosted through message passing.
pub struct GnnCooccurrenceModel {
    smoothing: f64,
    ewma_alpha: f64,
    ridge_lambda: f64,
    min_draws: usize,
}

impl Default for GnnCooccurrenceModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            ewma_alpha: 0.10,
            ridge_lambda: 0.01,
            min_draws: 50,
        }
    }
}

const N_FEATURES: usize = 5;
const N_HIDDEN: usize = 8; // Hidden dimension after message passing

/// Node features for each number in the pool.
struct NodeFeatures {
    /// features[node_idx] = [freq_ewma, gap, mod8_class, decade, trend]
    features: Vec<[f64; N_FEATURES]>,
}

impl NodeFeatures {
    fn compute(draws: &[Draw], pool: Pool, ewma_alpha: f64) -> Self {
        let size = pool.size();
        let pick = pool.pick_count();
        let mut features = vec![[0.0f64; N_FEATURES]; size];

        if draws.is_empty() {
            return Self { features };
        }

        // 1. EWMA frequency (chronological: iterate from oldest to newest)
        let mut ewma = vec![pick as f64 / size as f64; size]; // init at expected frequency
        for d in draws.iter().rev() {
            for i in 0..size {
                let num = (i + 1) as u8;
                let present = if pool.numbers_from(d).contains(&num) {
                    1.0
                } else {
                    0.0
                };
                ewma[i] = ewma_alpha * present + (1.0 - ewma_alpha) * ewma[i];
            }
        }
        for i in 0..size {
            features[i][0] = ewma[i];
        }

        // 2. Gap (draws since last appearance)
        let mut gap = vec![0u32; size];
        let mut found = vec![false; size];
        for (t, d) in draws.iter().enumerate() {
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < size && !found[idx] {
                    gap[idx] = t as u32;
                    found[idx] = true;
                }
            }
        }
        // Max gap for normalization
        let max_gap = (*gap.iter().max().unwrap_or(&1)).max(1) as f64;
        for i in 0..size {
            features[i][1] = gap[i] as f64 / max_gap;
        }

        // 3. Mod-8 class (normalized to [0, 1])
        for i in 0..size {
            let num = (i + 1) as u8;
            features[i][2] = (num % 8) as f64 / 7.0;
        }

        // 4. Decade (normalized to [0, 1])
        for i in 0..size {
            let num = (i + 1) as u8;
            features[i][3] = ((num - 1) / 10) as f64 / ((size as f64 - 1.0) / 10.0).max(1.0);
        }

        // 5. Trend: frequency in last 20 draws vs overall frequency
        let window = 20.min(draws.len());
        let mut recent_freq = vec![0.0f64; size];
        for d in &draws[..window] {
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < size {
                    recent_freq[idx] += 1.0;
                }
            }
        }
        let mut overall_freq = vec![0.0f64; size];
        for d in draws {
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < size {
                    overall_freq[idx] += 1.0;
                }
            }
        }
        for i in 0..size {
            let recent_rate = recent_freq[i] / window as f64;
            let overall_rate = overall_freq[i] / draws.len() as f64;
            // Trend: positive = trending up, negative = trending down
            // Normalize to roughly [-1, 1] via ratio
            let raw_trend = if overall_rate > 1e-10 {
                (recent_rate - overall_rate) / overall_rate
            } else {
                0.0
            };
            features[i][4] = raw_trend.clamp(-2.0, 2.0) / 2.0; // to [-1, 1]
        }

        Self { features }
    }
}

/// Co-occurrence graph with EWMA edge weights.
struct CooccurrenceGraph {
    /// adjacency[i * size + j] = EWMA co-occurrence weight (row-normalized)
    adjacency: Vec<f64>,
    size: usize,
}

impl CooccurrenceGraph {
    fn build(draws: &[Draw], pool: Pool, ewma_alpha: f64) -> Self {
        let size = pool.size();
        let mut adjacency = vec![0.0f64; size * size];

        // Process draws from oldest to newest (chronological EWMA)
        for d in draws.iter().rev() {
            let nums: Vec<usize> = pool
                .numbers_from(d)
                .iter()
                .map(|&x| (x - 1) as usize)
                .filter(|&x| x < size)
                .collect();

            // Decay all existing edges
            for w in adjacency.iter_mut() {
                *w *= 1.0 - ewma_alpha;
            }

            // Excite edges for co-occurring pairs
            for a in 0..nums.len() {
                for b in (a + 1)..nums.len() {
                    adjacency[nums[a] * size + nums[b]] += ewma_alpha;
                    adjacency[nums[b] * size + nums[a]] += ewma_alpha;
                }
            }
        }

        // Normalize rows for message passing (row-stochastic)
        for i in 0..size {
            let row_sum: f64 = (0..size).map(|j| adjacency[i * size + j]).sum();
            if row_sum > 1e-15 {
                for j in 0..size {
                    adjacency[i * size + j] /= row_sum;
                }
            }
        }

        Self { adjacency, size }
    }
}

/// Message passing: h_i = tanh(W1·x_i + W2·Σ_j e_ij·x_j)
///
/// W1, W2 are deterministic projection matrices derived from a fixed seed.
/// This is a "random feature" approach -- the GNN layer acts as a nonlinear
/// feature extractor, and only the readout layer is trained.
fn message_passing(
    features: &NodeFeatures,
    graph: &CooccurrenceGraph,
) -> Vec<[f64; N_HIDDEN]> {
    let size = graph.size;
    let mut hidden = vec![[0.0f64; N_HIDDEN]; size];

    // Deterministic W1 and W2 matrices (N_HIDDEN x N_FEATURES)
    // Using sin-based pseudo-random initialization for reproducibility
    let mut w1 = [[0.0f64; N_FEATURES]; N_HIDDEN];
    let mut w2 = [[0.0f64; N_FEATURES]; N_HIDDEN];
    for h in 0..N_HIDDEN {
        for f in 0..N_FEATURES {
            w1[h][f] = ((h * N_FEATURES + f + 1) as f64 * 0.7).sin() * 0.5;
            w2[h][f] = ((h * N_FEATURES + f + 1) as f64 * 1.3 + 2.0).sin() * 0.5;
        }
    }

    for i in 0..size {
        // Self-transform: W1 * x_i
        let mut self_proj = [0.0f64; N_HIDDEN];
        for h in 0..N_HIDDEN {
            for f in 0..N_FEATURES {
                self_proj[h] += w1[h][f] * features.features[i][f];
            }
        }

        // Neighbor aggregation: sum_j e_ij * x_j, then W2 * aggregate
        let mut agg = [0.0f64; N_FEATURES];
        for j in 0..size {
            let edge_weight = graph.adjacency[i * size + j];
            if edge_weight > 1e-15 {
                for f in 0..N_FEATURES {
                    agg[f] += edge_weight * features.features[j][f];
                }
            }
        }
        let mut neighbor_proj = [0.0f64; N_HIDDEN];
        for h in 0..N_HIDDEN {
            for f in 0..N_FEATURES {
                neighbor_proj[h] += w2[h][f] * agg[f];
            }
        }

        // Combine and apply tanh activation
        for h in 0..N_HIDDEN {
            hidden[i][h] = (self_proj[h] + neighbor_proj[h]).tanh();
        }
    }

    hidden
}

/// Ridge regression readout: learn W3 from historical data.
///
/// For each historical draw t, the target is whether each number was drawn.
/// Features are the hidden representations h_i from message passing on
/// draws[t+1..] (no future leakage).
///
/// We solve: w = (H^T H + lambda*I)^{-1} H^T y
///
/// Since N_HIDDEN is small (8), we can solve in primal form with Cholesky.
fn ridge_readout(
    draws: &[Draw],
    pool: Pool,
    ewma_alpha: f64,
    ridge_lambda: f64,
) -> [f64; N_HIDDEN] {
    let size = pool.size();
    let n_train = draws.len().saturating_sub(1);

    if n_train < 30 {
        // Not enough data: return uniform weights
        let mut w = [0.0f64; N_HIDDEN];
        w[0] = 1.0; // bias towards first feature (frequency)
        return w;
    }

    // Subsample training points for efficiency
    let stride = (n_train / 40).max(1);

    // Accumulate H^T H (N_HIDDEN x N_HIDDEN) and H^T y (N_HIDDEN) incrementally
    let mut hth = [[0.0f64; N_HIDDEN]; N_HIDDEN];
    let mut hty = [0.0f64; N_HIDDEN];
    let mut n_samples = 0usize;

    for t in (0..n_train).step_by(stride) {
        let history = &draws[t + 1..];
        if history.len() < 30 {
            continue;
        }

        let node_feats = NodeFeatures::compute(history, pool, ewma_alpha);
        let graph = CooccurrenceGraph::build(history, pool, ewma_alpha);
        let hidden = message_passing(&node_feats, &graph);

        // Target: which numbers were drawn at time t
        let drawn: Vec<usize> = pool
            .numbers_from(&draws[t])
            .iter()
            .map(|&x| (x - 1) as usize)
            .filter(|&x| x < size)
            .collect();

        for i in 0..size {
            let target = if drawn.contains(&i) { 1.0 } else { 0.0 };
            for a in 0..N_HIDDEN {
                hty[a] += hidden[i][a] * target;
                for b in 0..N_HIDDEN {
                    hth[a][b] += hidden[i][a] * hidden[i][b];
                }
            }
            n_samples += 1;
        }
    }

    if n_samples < N_HIDDEN {
        let mut w = [0.0f64; N_HIDDEN];
        w[0] = 1.0;
        return w;
    }

    // Add regularization
    for a in 0..N_HIDDEN {
        hth[a][a] += ridge_lambda * n_samples as f64;
    }

    // Solve via Cholesky
    solve_cholesky_small(&hth, &hty)
}

/// Simple Cholesky solve for small (N_HIDDEN x N_HIDDEN) system.
/// Solves A*x = b where A is symmetric positive definite.
fn solve_cholesky_small(
    a: &[[f64; N_HIDDEN]; N_HIDDEN],
    b: &[f64; N_HIDDEN],
) -> [f64; N_HIDDEN] {
    // Cholesky decomposition: A = L * L^T
    let mut l = [[0.0f64; N_HIDDEN]; N_HIDDEN];

    for i in 0..N_HIDDEN {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                l[i][j] = if diag > 0.0 { diag.sqrt() } else { 1e-10 };
            } else {
                l[i][j] = if l[j][j].abs() > 1e-15 {
                    (a[i][j] - sum) / l[j][j]
                } else {
                    0.0
                };
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = [0.0f64; N_HIDDEN];
    for i in 0..N_HIDDEN {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * y[j];
        }
        y[i] = if l[i][i].abs() > 1e-15 {
            (b[i] - sum) / l[i][i]
        } else {
            0.0
        };
    }

    // Backward substitution: L^T * x = y
    let mut x = [0.0f64; N_HIDDEN];
    for i in (0..N_HIDDEN).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..N_HIDDEN {
            sum += l[j][i] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-15 {
            (y[i] - sum) / l[i][i]
        } else {
            0.0
        };
    }

    x
}

impl ForecastModel for GnnCooccurrenceModel {
    fn name(&self) -> &str {
        "GnnCooccurrence"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // 1. Build current node features and graph
        let node_features = NodeFeatures::compute(draws, pool, self.ewma_alpha);
        let graph = CooccurrenceGraph::build(draws, pool, self.ewma_alpha);

        // 2. Message passing
        let hidden = message_passing(&node_features, &graph);

        // 3. Ridge regression readout (trained on historical data)
        let readout_weights = ridge_readout(draws, pool, self.ewma_alpha, self.ridge_lambda);

        // 4. Compute scores: score_i = w . h_i
        let mut scores = vec![0.0f64; size];
        for i in 0..size {
            let mut score = 0.0;
            for h in 0..N_HIDDEN {
                score += readout_weights[h] * hidden[i][h];
            }
            scores[i] = score;
        }

        // 5. Softmax
        let max_score = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // 6. Smooth towards uniform
        let uniform_val = 1.0 / size as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = if pool == Pool::Balls {
            PROB_FLOOR_BALLS
        } else {
            PROB_FLOOR_STARS
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("ridge_lambda".into(), self.ridge_lambda),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn calibration_stride(&self) -> usize {
        2 // ridge readout is expensive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_gnn_valid_ball_distribution() {
        let draws = make_test_draws(100);
        let model = GnnCooccurrenceModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_gnn_valid_star_distribution() {
        let draws = make_test_draws(100);
        let model = GnnCooccurrenceModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_gnn_few_draws_uniform() {
        let draws = make_test_draws(10);
        let model = GnnCooccurrenceModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!(
                (p - expected).abs() < 1e-10,
                "Few draws should return uniform, got {}",
                p
            );
        }
    }

    #[test]
    fn test_gnn_no_negative() {
        let draws = make_test_draws(100);
        let model = GnnCooccurrenceModel::default();
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            for &p in &dist {
                assert!(p >= 0.0, "Negative probability: {} for {:?}", p, pool);
            }
        }
    }

    #[test]
    fn test_gnn_deterministic() {
        let draws = make_test_draws(100);
        let model = GnnCooccurrenceModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "GnnCooccurrence should be deterministic"
            );
        }
    }

    #[test]
    fn test_gnn_empty_draws() {
        let model = GnnCooccurrenceModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_node_features_dimensions() {
        let draws = make_test_draws(100);
        let features = NodeFeatures::compute(&draws, Pool::Balls, 0.10);
        assert_eq!(features.features.len(), 50);
        for f in &features.features {
            assert_eq!(f.len(), N_FEATURES);
        }
    }

    #[test]
    fn test_cooccurrence_graph_nonnegative() {
        let draws = make_test_draws(100);
        let graph = CooccurrenceGraph::build(&draws, Pool::Stars, 0.10);
        let size = 12;
        for i in 0..size {
            for j in 0..size {
                let w = graph.adjacency[i * size + j];
                assert!(w.is_finite(), "Edge weight should be finite");
                assert!(w >= 0.0, "Edge weight should be non-negative");
            }
        }
    }

    #[test]
    fn test_message_passing_bounded() {
        let draws = make_test_draws(100);
        let features = NodeFeatures::compute(&draws, Pool::Balls, 0.10);
        let graph = CooccurrenceGraph::build(&draws, Pool::Balls, 0.10);
        let hidden = message_passing(&features, &graph);
        assert_eq!(hidden.len(), 50);
        for h in &hidden {
            for &val in h {
                assert!(val.is_finite(), "Hidden value should be finite");
                assert!(
                    val.abs() <= 1.0,
                    "tanh output should be in [-1, 1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_cholesky_small_identity() {
        // Solve I * x = b --> x = b
        let mut a = [[0.0f64; N_HIDDEN]; N_HIDDEN];
        for i in 0..N_HIDDEN {
            a[i][i] = 1.0;
        }
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = solve_cholesky_small(&a, &b);
        for i in 0..N_HIDDEN {
            assert!(
                (x[i] - b[i]).abs() < 1e-10,
                "Cholesky solve with identity failed at {}: {} vs {}",
                i,
                x[i],
                b[i]
            );
        }
    }
}
