use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Hawkes Process — Self-exciting point process for co-occurrences.
///
/// For each pair (i,j), tracks temporal co-occurrence patterns and models
/// clustering via a self-exciting intensity:
///   lambda_ij(t) = mu_ij + alpha * sum_{t_k < t} exp(-beta * (t - t_k))
///
/// where:
/// - mu_ij = baseline rate of pair co-occurrence
/// - alpha = excitation strength (how much a recent occurrence boosts future intensity)
/// - beta = decay rate (how quickly excitation fades)
///
/// Only tracks top n_top_pairs by z-score to keep computation tractable.
/// The last 3 draws serve as trigger context for computing current intensities.
/// Marginalizes pair intensities to individual number scores.
pub struct HawkesModel {
    decay_rate: f64,
    excitation: f64,
    n_top_pairs: usize,
    smoothing: f64,
    min_draws: usize,
}

impl Default for HawkesModel {
    fn default() -> Self {
        Self {
            decay_rate: 0.1,
            excitation: 0.5,
            n_top_pairs: 100,
            smoothing: 0.12,
            min_draws: 50,
        }
    }
}

/// A significant pair with its baseline rate and co-occurrence times.
struct TrackedPair {
    i: usize,
    j: usize,
    baseline: f64,
    /// Times (in draws-from-end, 0 = most recent) when both i and j appeared.
    occurrence_times: Vec<usize>,
}

impl HawkesModel {
    /// Find the top pairs by z-score excess and track their occurrence times.
    fn find_top_pairs(&self, draws: &[Draw], pool: Pool) -> Vec<TrackedPair> {
        let n = pool.size();
        let k = pool.pick_count();
        let n_draws = draws.len();

        // Count pair co-occurrences (flat matrix for cache locality)
        let mut pair_counts = vec![0u32; n * n];
        for d in draws {
            let nums: Vec<usize> = pool
                .numbers_from(d)
                .iter()
                .map(|&x| (x - 1) as usize)
                .collect();
            for a in 0..nums.len() {
                for b in (a + 1)..nums.len() {
                    pair_counts[nums[a] * n + nums[b]] += 1;
                    pair_counts[nums[b] * n + nums[a]] += 1;
                }
            }
        }

        // Expected pair count
        let p_pair = (k as f64 * (k as f64 - 1.0)) / (n as f64 * (n as f64 - 1.0));
        let expected = n_draws as f64 * p_pair;
        let std_dev = (n_draws as f64 * p_pair * (1.0 - p_pair)).sqrt().max(1.0);

        // Collect all pairs with their z-scores
        let mut pair_scores: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let obs = pair_counts[i * n + j] as f64;
                let z = (obs - expected) / std_dev;
                pair_scores.push((i, j, z.abs()));
            }
        }

        // Sort by z-score descending, take top N
        pair_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        pair_scores.truncate(self.n_top_pairs);

        // For each top pair, collect occurrence times
        pair_scores
            .into_iter()
            .map(|(i, j, _z)| {
                let mut occurrence_times = Vec::new();
                for (t, d) in draws.iter().enumerate() {
                    let nums = pool.numbers_from(d);
                    let has_i = nums.iter().any(|&x| (x - 1) as usize == i);
                    let has_j = nums.iter().any(|&x| (x - 1) as usize == j);
                    if has_i && has_j {
                        occurrence_times.push(t);
                    }
                }
                let baseline = if n_draws > 0 {
                    pair_counts[i * n + j] as f64 / n_draws as f64
                } else {
                    p_pair
                };
                TrackedPair {
                    i,
                    j,
                    baseline,
                    occurrence_times,
                }
            })
            .collect()
    }

    /// Compute the Hawkes intensity for a pair at the current time (t=0, just after draws[0]).
    /// The intensity is:
    ///   lambda = mu + alpha * sum_{t_k in context} exp(-beta * t_k)
    fn compute_intensity(&self, pair: &TrackedPair, n_context: usize) -> f64 {
        let mu = pair.baseline;
        let alpha = self.excitation * mu; // Scale excitation by baseline

        // Sum excitation from recent co-occurrences
        let mut excitation_sum = 0.0;
        for &t in &pair.occurrence_times {
            if t < n_context {
                // t=0 is the most recent draw; the trigger decays with distance
                excitation_sum += (-self.decay_rate * (t as f64 + 1.0)).exp();
            }
        }

        mu + alpha * excitation_sum
    }
}

impl ForecastModel for HawkesModel {
    fn name(&self) -> &str {
        "Hawkes"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let uniform = vec![1.0 / n as f64; n];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Find top pairs and their occurrence histories
        let tracked_pairs = self.find_top_pairs(draws, pool);

        // Compute intensity for each tracked pair using last 3 draws as context
        let n_context = 3;

        // Accumulate intensity contributions per number
        let mut number_scores = vec![0.0f64; n];
        let mut number_pair_count = vec![0u32; n];

        for pair in &tracked_pairs {
            let intensity = self.compute_intensity(pair, n_context);

            // Distribute intensity to both members of the pair
            number_scores[pair.i] += intensity;
            number_scores[pair.j] += intensity;
            number_pair_count[pair.i] += 1;
            number_pair_count[pair.j] += 1;
        }

        // Normalize per-number scores by pair count (average intensity)
        for i in 0..n {
            if number_pair_count[i] > 0 {
                number_scores[i] /= number_pair_count[i] as f64;
            }
        }

        // Add baseline frequency contribution for numbers not in any tracked pair
        let k = pool.pick_count();
        let base_rate = k as f64 / n as f64;
        for i in 0..n {
            if number_pair_count[i] == 0 {
                number_scores[i] = base_rate;
            }
        }

        // Also blend in raw frequency from recent draws for stability
        let recent_window = 20.min(draws.len());
        let mut recent_freq = vec![0.0f64; n];
        for d in &draws[..recent_window] {
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < n {
                    recent_freq[idx] += 1.0;
                }
            }
        }
        let freq_sum: f64 = recent_freq.iter().sum();
        if freq_sum > 0.0 {
            for f in &mut recent_freq {
                *f /= freq_sum;
            }
        }

        // Blend: 60% Hawkes intensity + 40% recent frequency
        for i in 0..n {
            number_scores[i] = 0.6 * number_scores[i] + 0.4 * recent_freq[i];
        }

        // Normalize to distribution
        let total: f64 = number_scores.iter().sum();
        if total <= 0.0 {
            return uniform;
        }
        let mut probs: Vec<f64> = number_scores.iter().map(|&s| s / total).collect();

        // Smooth towards uniform
        let uniform_val = 1.0 / n as f64;
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
            ("decay_rate".into(), self.decay_rate),
            ("excitation".into(), self.excitation),
            ("n_top_pairs".into(), self.n_top_pairs as f64),
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
    fn test_hawkes_valid_distribution() {
        let draws = make_test_draws(100);
        let model = HawkesModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_hawkes_stars() {
        let draws = make_test_draws(100);
        let model = HawkesModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_hawkes_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = HawkesModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hawkes_no_negative() {
        let draws = make_test_draws(100);
        let model = HawkesModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_hawkes_deterministic() {
        let draws = make_test_draws(100);
        let model = HawkesModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Hawkes should be deterministic");
        }
    }

    #[test]
    fn test_hawkes_empty_draws() {
        let model = HawkesModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hawkes_intensity_decay() {
        // A pair with a recent occurrence should have higher intensity
        // than one without
        let pair_recent = TrackedPair {
            i: 0,
            j: 1,
            baseline: 0.1,
            occurrence_times: vec![0], // just occurred
        };
        let pair_old = TrackedPair {
            i: 0,
            j: 1,
            baseline: 0.1,
            occurrence_times: vec![10], // occurred 10 draws ago
        };
        let model = HawkesModel::default();
        let int_recent = model.compute_intensity(&pair_recent, 3);
        let int_old = model.compute_intensity(&pair_old, 3);
        // Recent occurrence should produce higher intensity
        assert!(
            int_recent > int_old,
            "Recent: {}, Old: {}",
            int_recent,
            int_old
        );
    }

    #[test]
    fn test_hawkes_large_draws() {
        let draws = make_test_draws(200);
        let model = HawkesModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
