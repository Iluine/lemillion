use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Wavelet — multi-scale frequency analysis via Discrete Wavelet Transform (Haar).
///
/// For each number in the pool, builds a binary occurrence time series over the
/// last `window` draws, then decomposes it via 3-level Haar DWT. The most recent
/// coefficients at each scale capture different temporal dynamics:
///
/// - Level 2 approximation (8-16 draw scale): background trend
/// - Level 1 detail (4-8 draw scale): medium-term oscillations
/// - Level 0 detail (2-4 draw scale): short-term cycles
///
/// These are combined with weights [0.5, 0.3, 0.2] (long > medium > short),
/// passed through softmax, smoothed toward uniform, and floor-normalized.
pub struct WaveletModel {
    n_levels: usize,
    window: usize,
    smoothing: f64,
    min_draws: usize,
}

impl Default for WaveletModel {
    fn default() -> Self {
        Self {
            n_levels: 3,
            window: 64,
            smoothing: 0.30,
            min_draws: 30,
        }
    }
}

/// Single-level Haar DWT: splits a signal into approximation and detail coefficients.
///
/// approx[i] = (signal[2i] + signal[2i+1]) / sqrt(2)
/// detail[i] = (signal[2i] - signal[2i+1]) / sqrt(2)
fn haar_dwt(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len() / 2;
    let mut approx = vec![0.0; n];
    let mut detail = vec![0.0; n];
    let sqrt2 = std::f64::consts::SQRT_2;
    for i in 0..n {
        approx[i] = (signal[2 * i] + signal[2 * i + 1]) / sqrt2;
        detail[i] = (signal[2 * i] - signal[2 * i + 1]) / sqrt2;
    }
    (approx, detail)
}

/// Multi-level Haar DWT: recursively decomposes the approximation coefficients.
///
/// Returns (final_approx, detail_coefficients) where detail_coefficients[0] is
/// the finest scale (level 0) and detail_coefficients[n_levels-1] is the coarsest.
fn haar_dwt_multilevel(signal: &[f64], n_levels: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut approx = signal.to_vec();
    let mut details = Vec::with_capacity(n_levels);

    for _ in 0..n_levels {
        if approx.len() < 2 {
            break;
        }
        let (a, d) = haar_dwt(&approx);
        details.push(d);
        approx = a;
    }

    (approx, details)
}

impl ForecastModel for WaveletModel {
    fn name(&self) -> &str {
        "Wavelet"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Take the last `window` draws (or fewer, rounded down to even for DWT)
        let available = self.window.min(draws.len());
        // Round down to the largest power-of-2 <= available that supports n_levels of DWT.
        // Each level halves the length, so we need at least 2^n_levels samples.
        let min_len = 1usize << self.n_levels; // 2^n_levels = 8 for n_levels=3
        if available < min_len {
            return uniform;
        }
        // Round down to largest power-of-2 <= available
        let window_len = 1usize << (usize::BITS - 1 - available.leading_zeros() as u32);
        let window_len = window_len.max(min_len);

        // draws[0] = most recent; take draws[0..window_len] and reverse for chronological order
        let recent = &draws[..window_len];

        // Weights for combining levels: [level2_approx, level1_detail, level0_detail]
        let level_weights = [0.5, 0.3, 0.2];

        let mut scores = vec![0.0f64; size];

        for k in 0..size {
            let number = (k + 1) as u8;

            // Build binary time series: 1 if number was drawn, 0 otherwise
            // Chronological order (oldest first)
            let signal: Vec<f64> = recent
                .iter()
                .rev()
                .map(|draw| {
                    let numbers = pool.numbers_from(draw);
                    if numbers.contains(&number) { 1.0 } else { 0.0 }
                })
                .collect();

            // Apply multi-level Haar DWT
            let (approx, details) = haar_dwt_multilevel(&signal, self.n_levels);

            // Extract the most recent coefficient from each level
            // Level 2 (coarsest): last element of final approximation → background trend
            let approx_val = approx.last().copied().unwrap_or(0.0);

            // Level 1 (medium): last element of details[1] (if exists)
            let detail1_val = if details.len() >= 2 {
                details[1].last().copied().unwrap_or(0.0)
            } else {
                0.0
            };

            // Level 0 (finest): last element of details[0]
            let detail0_val = if !details.is_empty() {
                details[0].last().copied().unwrap_or(0.0)
            } else {
                0.0
            };

            // Combine: weight * coefficient for each scale
            // level_weights[0] = level2_approx, [1] = level1_detail, [2] = level0_detail
            scores[k] = level_weights[0] * approx_val
                + level_weights[1] * detail1_val
                + level_weights[2] * detail0_val;
        }

        // Softmax to convert scores to probabilities
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = probs.iter().sum();
        if sum_exp > 0.0 {
            for p in &mut probs {
                *p /= sum_exp;
            }
        } else {
            return uniform;
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
            ("n_levels".into(), self.n_levels as f64),
            ("window".into(), self.window as f64),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Consecutive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_wavelet_valid_distribution_balls() {
        let draws = make_test_draws(100);
        let model = WaveletModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_wavelet_valid_distribution_stars() {
        let draws = make_test_draws(100);
        let model = WaveletModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_wavelet_deterministic() {
        let draws = make_test_draws(100);
        let model = WaveletModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Wavelet should be deterministic");
        }
    }

    #[test]
    fn test_wavelet_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = WaveletModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_wavelet_empty_draws() {
        let model = WaveletModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_wavelet_no_negative() {
        let draws = make_test_draws(100);
        let model = WaveletModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_wavelet_sampling_strategy() {
        let model = WaveletModel::default();
        assert_eq!(model.sampling_strategy(), SamplingStrategy::Consecutive);
    }

    #[test]
    fn test_wavelet_name() {
        let model = WaveletModel::default();
        assert_eq!(model.name(), "Wavelet");
    }

    #[test]
    fn test_wavelet_params() {
        let model = WaveletModel::default();
        let params = model.params();
        assert_eq!(params["n_levels"], 3.0);
        assert_eq!(params["window"], 64.0);
        assert_eq!(params["smoothing"], 0.30);
        assert_eq!(params["min_draws"], 30.0);
    }

    #[test]
    fn test_haar_dwt_basic() {
        let signal = vec![1.0, 3.0, 5.0, 7.0];
        let (approx, detail) = haar_dwt(&signal);
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!((approx[0] - 4.0 / sqrt2).abs() < 1e-10);
        assert!((approx[1] - 12.0 / sqrt2).abs() < 1e-10);
        assert!((detail[0] - (-2.0 / sqrt2)).abs() < 1e-10);
        assert!((detail[1] - (-2.0 / sqrt2)).abs() < 1e-10);
    }

    #[test]
    fn test_haar_dwt_multilevel() {
        let signal = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let (approx, details) = haar_dwt_multilevel(&signal, 3);
        assert_eq!(approx.len(), 1);
        assert_eq!(details.len(), 3);
        assert_eq!(details[0].len(), 4); // level 0: finest
        assert_eq!(details[1].len(), 2); // level 1: medium
        assert_eq!(details[2].len(), 1); // level 2: coarsest
    }

    #[test]
    fn test_wavelet_large_draws() {
        let draws = make_test_draws(200);
        let model = WaveletModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
