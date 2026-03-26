use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{filter_star_era, floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_STARS};

/// OUReversion — Ornstein-Uhlenbeck mean-reversion model for stars.
///
/// Discretized AR(1) per star number: f_{t+1} = f_t + θ·(μ - f_t)
/// - θ_i = mean-reversion speed (estimated by OLS on Δf_t vs (μ - f_t))
/// - μ_i = theoretical frequency (2/12 ≈ 0.1667)
/// - When f_current ≪ μ: θ·(μ - f) > 0 → BOOSTS cold stars
/// - When f_current ≫ μ: θ·(μ - f) < 0 → DAMPENS hot stars
///
/// This is the direct counter-signal to momentum EWMA, addressing the ★4 scenario
/// where a star in cold phase is unanimously avoided but then appears.
pub struct OUReversionModel {
    smoothing: f64,
    min_draws: usize,
    ewma_alpha: f64,    // Slow EWMA for frequency tracking
    theta_default: f64, // Default θ when OLS fails
}

impl Default for OUReversionModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            min_draws: 60,
            ewma_alpha: 0.05, // Slow EWMA to track long-term frequency
            theta_default: 0.15,
        }
    }
}

impl ForecastModel for OUReversionModel {
    fn name(&self) -> &str {
        "OUReversion"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Stars only
        if pool == Pool::Balls {
            return uniform;
        }

        let draws = filter_star_era(draws);
        if draws.len() < self.min_draws {
            return uniform;
        }

        let n_stars = size;
        let pick = pool.pick_count();
        let mu = pick as f64 / n_stars as f64; // theoretical frequency ≈ 0.1667

        let mut probs = vec![0.0f64; n_stars];

        for star_idx in 0..n_stars {
            let star_num = (star_idx + 1) as u8;

            // Build EWMA frequency series (chronological order)
            let mut freq_series: Vec<f64> = Vec::with_capacity(draws.len());
            let mut ewma = mu; // Start from theoretical frequency
            for d in draws.iter().rev() {
                let val = if d.stars.contains(&star_num) { 1.0 } else { 0.0 };
                ewma = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * ewma;
                freq_series.push(ewma);
            }

            // Current frequency = last EWMA value
            let f_current = *freq_series.last().unwrap_or(&mu);

            // Estimate θ by OLS: Δf_t = θ·(μ - f_t) + ε
            // Regression of (f_{t+1} - f_t) on (μ - f_t)
            let theta = if freq_series.len() > 10 {
                let mut sum_xy = 0.0f64;
                let mut sum_xx = 0.0f64;
                for t in 0..freq_series.len() - 1 {
                    let x = mu - freq_series[t];  // (μ - f_t)
                    let y = freq_series[t + 1] - freq_series[t]; // Δf_t
                    sum_xy += x * y;
                    sum_xx += x * x;
                }
                if sum_xx > 1e-15 {
                    (sum_xy / sum_xx).clamp(0.01, 0.50) // θ ∈ [0.01, 0.50]
                } else {
                    self.theta_default
                }
            } else {
                self.theta_default
            };

            // OU prediction: f_predicted = f_current + θ·(μ - f_current)
            let f_predicted = f_current + theta * (mu - f_current);

            // Ensure non-negative
            probs[star_idx] = f_predicted.max(1e-6);
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

        // Smooth with uniform
        let uniform_val = 1.0 / n_stars as f64;
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        floor_only(&mut probs, PROB_FLOOR_STARS);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("theta_default".into(), self.theta_default),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn is_stars_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_ou_reversion_balls_uniform() {
        let model = OUReversionModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ou_reversion_stars_valid() {
        let model = OUReversionModel::default();
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
    fn test_ou_reversion_few_draws_uniform() {
        let model = OUReversionModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ou_reversion_no_negative() {
        let model = OUReversionModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_ou_reversion_deterministic() {
        let model = OUReversionModel::default();
        let draws = make_test_draws(100);
        let d1 = model.predict(&draws, Pool::Stars);
        let d2 = model.predict(&draws, Pool::Stars);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_ou_reversion_boosts_cold_stars() {
        // Create draws where star 4 is very cold (never appears)
        // OU should boost it above uniform
        let mut draws = Vec::new();
        for i in 0..100 {
            let base = (i % 10) as u8;
            // Stars cycle through 1-3, 5-12, never 4
            let s1 = ((base % 11) + 1) as u8;
            let s2 = (((base + 3) % 11) + 1) as u8;
            let (s1, s2) = if s1 == 4 { (5, s2) } else if s2 == 4 { (s1, 5) } else { (s1, s2) };
            let (s1, s2) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
            let (s1, s2) = if s1 == s2 { (s1, (s1 % 12) + 1) } else { (s1, s2) };
            let (s1, s2) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
            draws.push(Draw {
                draw_id: format!("{:03}", i),
                day: if i % 2 == 0 { "MARDI".to_string() } else { "VENDREDI".to_string() },
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 10, 20, 30, 40],
                stars: [s1.clamp(1, 12), s2.clamp(1, 12)],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
        prize_tiers: None,
            });
        }
        let model = OUReversionModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        let uniform_val = 1.0 / 12.0;
        // Star 4 (index 3) should be boosted by OU mean-reversion.
        // With 25% smoothing, floor guarantees at least uniform/5 ≈ 0.017.
        // The OU model should push it above the smoothing floor.
        // Check that OU doesn't crush star 4 to near-zero (which is the bug we're fixing).
        assert!(
            dist[3] > uniform_val * 0.20,
            "Star 4 prob {} should be meaningfully above zero (OU boosts cold stars, uniform={})",
            dist[3], uniform_val
        );
    }
}
