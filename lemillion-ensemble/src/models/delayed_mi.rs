use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// DelayedMI — exploits lag-reappearance structure at specific offsets.
///
/// The research module (informational/delayed MI) finds mutual information
/// structure at specific lags in the draw sequence. Rather than computing
/// expensive MI per prediction call, this model uses a simplified but
/// effective proxy: lag-reappearance rates.
///
/// For each number k, we measure how often k reappears exactly L draws
/// after a previous appearance, for L in {1, 2, 3, 5, 7}. These lags
/// correspond to the delays where MI was found to be significant.
///
/// At prediction time, we check which of those lag offsets match the
/// current position (i.e., k appeared exactly L draws ago) and boost k
/// by its empirical reappearance rate at that lag. Lags are weighted
/// inversely by distance (1/L) to favor short-range structure.
///
/// The lag signal is blended with EWMA frequency for stability.
pub struct DelayedMiModel {
    smoothing: f64,
    min_draws: usize,
    alpha: f64,
    lags: Vec<usize>,
}

impl Default for DelayedMiModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            min_draws: 25,
            alpha: 0.06,
            lags: vec![1, 2, 3, 5, 7],
        }
    }
}

impl DelayedMiModel {
    /// Compute the lag-reappearance rate for number `num_idx` at lag `lag`.
    ///
    /// For all positions t where the number appeared, count how often it
    /// also appeared at position t + lag. Returns the fraction of such
    /// co-occurrences relative to total appearances.
    ///
    /// `presence` is a chronological binary series (0 = earliest draw).
    fn lag_reappearance_rate(presence: &[u8], lag: usize) -> f64 {
        if presence.len() <= lag {
            return 0.0;
        }

        let mut trigger_count = 0u32;
        let mut reappear_count = 0u32;

        for t in 0..presence.len() - lag {
            if presence[t] == 1 {
                trigger_count += 1;
                if presence[t + lag] == 1 {
                    reappear_count += 1;
                }
            }
        }

        if trigger_count == 0 {
            return 0.0;
        }
        reappear_count as f64 / trigger_count as f64
    }

    /// Compute EWMA frequency for a number given draws (draws[0] = most recent).
    fn ewma_frequency(&self, draws: &[Draw], num: u8, pool: Pool) -> f64 {
        let base = pool.pick_count() as f64 / pool.size() as f64;
        let mut freq = base;
        // Iterate oldest to newest for correct EWMA accumulation
        for d in draws.iter().rev() {
            let present = pool.numbers_from(d).contains(&num);
            let val = if present { 1.0 } else { 0.0 };
            freq = self.alpha * val + (1.0 - self.alpha) * freq;
        }
        freq
    }
}

impl ForecastModel for DelayedMiModel {
    fn name(&self) -> &str {
        "DelayedMI"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Build per-number presence series in chronological order.
        // draws[0] = most recent, so reverse for chronological.
        let n_draws = draws.len();
        let mut presence = vec![vec![0u8; n_draws]; size];
        for (rev_idx, d) in draws.iter().enumerate() {
            let chrono_idx = n_draws - 1 - rev_idx;
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < size {
                    presence[idx][chrono_idx] = 1;
                }
            }
        }

        // Compute lag weights: 1/L, normalized
        let lag_weights: Vec<f64> = self.lags.iter().map(|&l| 1.0 / l as f64).collect();
        let lag_weight_sum: f64 = lag_weights.iter().sum();

        let mut probs = vec![0.0f64; size];

        for k in 0..size {
            let num = (k + 1) as u8;

            // Phase 1: Lag-reappearance signal
            let mut lag_score = 0.0;

            for (lag_idx, &lag) in self.lags.iter().enumerate() {
                // Compute the empirical reappearance rate at this lag
                let rate = Self::lag_reappearance_rate(&presence[k], lag);

                // Check if number k appeared exactly `lag` draws ago.
                // In the chronological series, the current position is n_draws,
                // so "lag draws ago" means chrono index n_draws - lag.
                if n_draws >= lag {
                    let check_idx = n_draws - lag;
                    if check_idx < n_draws && presence[k][check_idx] == 1 {
                        // Number appeared at this lag offset -- boost by its rate
                        lag_score += rate * lag_weights[lag_idx];
                    }
                }
            }

            // Normalize by total lag weight
            if lag_weight_sum > 0.0 {
                lag_score /= lag_weight_sum;
            }

            // Phase 2: EWMA frequency
            let ewma = self.ewma_frequency(draws, num, pool);

            // Blend: 40% lag signal + 60% EWMA
            probs[k] = 0.4 * lag_score + 0.6 * ewma;
        }

        // Normalize to distribution
        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return uniform;
        }
        for p in &mut probs {
            *p /= total;
        }

        // Smooth towards uniform
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
            ("min_draws".into(), self.min_draws as f64),
            ("alpha".into(), self.alpha),
            ("n_lags".into(), self.lags.len() as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn calibration_stride(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_delayed_mi_valid_distribution_balls() {
        let draws = make_test_draws(100);
        let model = DelayedMiModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_delayed_mi_valid_distribution_stars() {
        let draws = make_test_draws(100);
        let model = DelayedMiModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_delayed_mi_few_draws_returns_uniform() {
        let draws = make_test_draws(10);
        let model = DelayedMiModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_delayed_mi_empty_draws_returns_uniform() {
        let model = DelayedMiModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_delayed_mi_no_negative() {
        let draws = make_test_draws(100);
        let model = DelayedMiModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_delayed_mi_deterministic() {
        let draws = make_test_draws(100);
        let model = DelayedMiModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "DelayedMI should be deterministic");
        }
    }

    #[test]
    fn test_delayed_mi_sampling_strategy() {
        let model = DelayedMiModel::default();
        assert_eq!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        );
    }

    #[test]
    fn test_delayed_mi_calibration_stride() {
        let model = DelayedMiModel::default();
        assert_eq!(model.calibration_stride(), 2);
    }

    #[test]
    fn test_delayed_mi_large_draws() {
        let draws = make_test_draws(200);
        let model = DelayedMiModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_lag_reappearance_rate_basic() {
        // Number appears at positions 0, 3, 6 (lag 3 reappearance)
        let presence = vec![1, 0, 0, 1, 0, 0, 1, 0, 0, 0];
        let rate = DelayedMiModel::lag_reappearance_rate(&presence, 3);
        // At t=0: present, t+3=3 present -> hit
        // At t=3: present, t+3=6 present -> hit
        // At t=6: present, t+3=9 present? no -> miss
        // But t=6 is at index 6, and 6+3=9 < 10, so it counts as trigger
        // 3 triggers (t=0,3,6 but t=6 checks t=9 which is 0), 2 hits
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lag_reappearance_rate_never() {
        let presence = vec![0, 0, 0, 0, 0];
        let rate = DelayedMiModel::lag_reappearance_rate(&presence, 1);
        assert!((rate).abs() < 1e-10);
    }

    #[test]
    fn test_lag_reappearance_rate_always() {
        let presence = vec![1, 1, 1, 1, 1];
        let rate = DelayedMiModel::lag_reappearance_rate(&presence, 1);
        assert!((rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lag_reappearance_rate_too_short() {
        let presence = vec![1, 0];
        let rate = DelayedMiModel::lag_reappearance_rate(&presence, 5);
        assert!((rate).abs() < 1e-10);
    }
}
