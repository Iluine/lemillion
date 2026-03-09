use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Renewal Process — Inter-arrival time analysis via Weibull hazard.
///
/// For each number i, collects all gaps (inter-arrival times) from recent draws
/// and fits a Weibull(shape, scale) distribution by maximum likelihood.
///
/// The Weibull shape parameter reveals the number's temporal behavior:
/// - shape < 1: anti-persistence (tends to return quickly after appearing)
/// - shape = 1: memoryless (geometric/iid process)
/// - shape > 1: persistence (long absences, clustered appearances)
///
/// Prediction uses the Weibull hazard function (conditional failure rate):
///   h(t) = (shape/scale) × (t/scale)^{shape-1}
///
/// which gives P(gap ends at time t | gap >= t). Numbers with high hazard
/// at their current gap are predicted more likely to appear next.
pub struct RenewalModel {
    window: usize,
    min_gaps: usize,
    smoothing: f64,
    min_draws: usize,
}

impl Default for RenewalModel {
    fn default() -> Self {
        Self {
            window: 200,
            min_gaps: 8,
            smoothing: 0.25,
            min_draws: 30,
        }
    }
}

/// Weibull distribution parameters.
struct WeibullParams {
    shape: f64,
    scale: f64,
}

impl WeibullParams {
    /// Weibull hazard function: h(t) = (shape/scale) × (t/scale)^{shape-1}
    /// Returns the instantaneous failure rate at time t.
    fn hazard(&self, t: f64) -> f64 {
        if t <= 0.0 {
            // At t=0 with shape<1, hazard is infinite; clamp to a large value
            if self.shape < 1.0 {
                return 10.0;
            }
            return self.shape / self.scale;
        }
        (self.shape / self.scale) * (t / self.scale).powf(self.shape - 1.0)
    }
}

/// Fit a Weibull distribution to gap data using maximum likelihood estimation.
///
/// MLE for Weibull:
/// - scale = (sum(g_i^shape) / m)^{1/shape}
/// - shape is found by Newton-Raphson on the profile log-likelihood equation:
///   f(k) = 1/k + (sum(ln g_i))/m - sum(g_i^k × ln g_i) / sum(g_i^k) = 0
///
/// Returns None if the data is insufficient or degenerate.
fn fit_weibull(gaps: &[f64]) -> Option<WeibullParams> {
    let m = gaps.len();
    if m < 2 {
        return None;
    }

    // All gaps must be positive
    if gaps.iter().any(|&g| g <= 0.0) {
        return None;
    }

    let ln_gaps: Vec<f64> = gaps.iter().map(|g| g.ln()).collect();
    let mean_ln: f64 = ln_gaps.iter().sum::<f64>() / m as f64;

    // Newton-Raphson for shape parameter
    let mut k = 1.0_f64; // initial guess
    let max_iter = 20;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // Compute sums needed for the profile equation
        let mut sum_gk = 0.0_f64;
        let mut sum_gk_ln = 0.0_f64;
        let mut sum_gk_ln2 = 0.0_f64;

        for i in 0..m {
            let gk = gaps[i].powf(k);
            let lng = ln_gaps[i];
            sum_gk += gk;
            sum_gk_ln += gk * lng;
            sum_gk_ln2 += gk * lng * lng;
        }

        if sum_gk < 1e-300 {
            break;
        }

        // f(k) = 1/k + mean_ln - sum_gk_ln / sum_gk
        let f = 1.0 / k + mean_ln - sum_gk_ln / sum_gk;

        // f'(k) = -1/k^2 - (sum_gk_ln2 × sum_gk - sum_gk_ln^2) / sum_gk^2
        let f_prime =
            -1.0 / (k * k) - (sum_gk_ln2 * sum_gk - sum_gk_ln * sum_gk_ln) / (sum_gk * sum_gk);

        if f_prime.abs() < 1e-300 {
            break;
        }

        let delta = f / f_prime;
        k -= delta;

        // Clamp for numerical stability
        k = k.clamp(0.3, 3.0);

        if delta.abs() < tol {
            break;
        }
    }

    // Compute scale from converged shape
    let mut sum_gk = 0.0_f64;
    for &g in gaps {
        sum_gk += g.powf(k);
    }
    let scale = (sum_gk / m as f64).powf(1.0 / k);

    if scale <= 0.0 || !scale.is_finite() || !k.is_finite() {
        return None;
    }

    Some(WeibullParams {
        shape: k,
        scale,
    })
}

impl RenewalModel {
    /// Collect gaps (inter-arrival times) for a specific number from the draw history.
    /// Returns a vector of gap lengths (in draws). A gap of 1 means the number appeared
    /// in consecutive draws.
    fn collect_gaps(&self, draws: &[Draw], pool: Pool, number: u8) -> Vec<f64> {
        let window = self.window.min(draws.len());
        let mut gaps = Vec::new();
        let mut last_seen: Option<usize> = None;

        for (t, draw) in draws[..window].iter().enumerate() {
            let nums = pool.numbers_from(draw);
            if nums.contains(&number) {
                if let Some(prev_t) = last_seen {
                    let gap = (t - prev_t) as f64;
                    if gap > 0.0 {
                        gaps.push(gap);
                    }
                }
                last_seen = Some(t);
            }
        }

        gaps
    }

    /// Compute the current gap for a number: how many draws since it last appeared.
    /// Returns None if the number appeared in the most recent draw (gap = 0).
    fn current_gap(&self, draws: &[Draw], pool: Pool, number: u8) -> f64 {
        for (t, draw) in draws.iter().enumerate() {
            let nums = pool.numbers_from(draw);
            if nums.contains(&number) {
                return t as f64;
            }
        }
        // Never seen: return a large gap
        draws.len() as f64
    }
}

impl ForecastModel for RenewalModel {
    fn name(&self) -> &str {
        "Renewal"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let uniform = vec![1.0 / n as f64; n];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let mut scores = vec![0.0_f64; n];
        let base_rate = pool.pick_count() as f64 / n as f64;

        for i in 0..n {
            let number = (i + 1) as u8;
            let gaps = self.collect_gaps(draws, pool, number);
            let current_gap = self.current_gap(draws, pool, number);

            if gaps.len() >= self.min_gaps {
                // Fit Weibull and use hazard
                if let Some(params) = fit_weibull(&gaps) {
                    // Hazard at current gap gives conditional probability of appearing
                    let h = params.hazard(current_gap);
                    // Clamp hazard to avoid extreme values
                    scores[i] = h.clamp(1e-6, 10.0);
                } else {
                    // Fit failed: use base rate
                    scores[i] = base_rate;
                }
            } else {
                // Not enough gaps: use base rate
                scores[i] = base_rate;
            }
        }

        // Normalize to distribution
        let total: f64 = scores.iter().sum();
        if total <= 0.0 {
            return uniform;
        }
        let mut probs: Vec<f64> = scores.iter().map(|&s| s / total).collect();

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
            ("window".into(), self.window as f64),
            ("min_gaps".into(), self.min_gaps as f64),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_renewal_valid_distribution_balls() {
        let draws = make_test_draws(200);
        let model = RenewalModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_renewal_valid_distribution_stars() {
        let draws = make_test_draws(200);
        let model = RenewalModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_renewal_deterministic() {
        let draws = make_test_draws(200);
        let model = RenewalModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Renewal should be deterministic");
        }
    }

    #[test]
    fn test_renewal_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = RenewalModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_renewal_empty_draws() {
        let model = RenewalModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_renewal_no_negative() {
        let draws = make_test_draws(200);
        let model = RenewalModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_renewal_params() {
        let model = RenewalModel::default();
        let params = model.params();
        assert_eq!(params["window"], 200.0);
        assert_eq!(params["min_gaps"], 8.0);
        assert_eq!(params["smoothing"], 0.25);
        assert_eq!(params["min_draws"], 30.0);
    }

    #[test]
    fn test_renewal_sampling_strategy() {
        let model = RenewalModel::default();
        assert_eq!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 4 }
        );
    }

    #[test]
    fn test_weibull_fit_known() {
        // Constant gaps should give shape ~= very high (approaching delta)
        // but clamped to 3.0
        let gaps = vec![5.0; 20];
        let params = fit_weibull(&gaps).unwrap();
        assert!(params.shape >= 2.5, "Constant gaps should have high shape: {}", params.shape);
        assert!((params.scale - 5.0).abs() < 1.0, "Scale should be near 5.0: {}", params.scale);
    }

    #[test]
    fn test_weibull_fit_exponential() {
        // Gaps that roughly follow exponential (shape ~1) distribution
        let gaps = vec![1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 2.0, 6.0, 1.0, 3.0];
        let params = fit_weibull(&gaps).unwrap();
        // Shape should be roughly near 1.0 for exponential-like data
        assert!(params.shape > 0.3 && params.shape < 3.0, "Shape: {}", params.shape);
        assert!(params.scale > 0.0, "Scale should be positive: {}", params.scale);
    }

    #[test]
    fn test_weibull_fit_insufficient() {
        let gaps = vec![5.0]; // only 1 gap
        assert!(fit_weibull(&gaps).is_none());
    }

    #[test]
    fn test_weibull_hazard_increasing() {
        // shape > 1 means increasing hazard (the longer you wait, the more likely)
        let params = WeibullParams { shape: 2.0, scale: 5.0 };
        let h1 = params.hazard(2.0);
        let h2 = params.hazard(4.0);
        assert!(h2 > h1, "Increasing hazard: h(2)={}, h(4)={}", h1, h2);
    }

    #[test]
    fn test_weibull_hazard_decreasing() {
        // shape < 1 means decreasing hazard (anti-persistence)
        let params = WeibullParams { shape: 0.5, scale: 5.0 };
        let h1 = params.hazard(2.0);
        let h2 = params.hazard(4.0);
        assert!(h1 > h2, "Decreasing hazard: h(2)={}, h(4)={}", h1, h2);
    }

    #[test]
    fn test_renewal_large_draws() {
        let draws = make_test_draws(500);
        let model = RenewalModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
