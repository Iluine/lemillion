use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{filter_star_era, floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_STARS};

/// StarVolatility — GARCH(1,1) volatility model for stars.
///
/// Models the volatility of each star's frequency using GARCH(1,1):
///   h_t = ω + α·e²_{t-1} + β·h_{t-1}
///
/// When volatility is HIGH → predict towards uniform (uncertainty).
/// When volatility is LOW → trust the current frequency.
///
/// This is an anti-overconfidence filter: when frequencies oscillate strongly
/// (normal for 12 numbers, 2 drawn), the model recommends caution.
pub struct StarVolatilityModel {
    smoothing: f64,
    min_draws: usize,
    ewma_alpha: f64,
    garch_omega: f64, // baseline volatility
    garch_alpha: f64, // reaction to shocks
    garch_beta: f64,  // persistence of volatility
}

impl Default for StarVolatilityModel {
    fn default() -> Self {
        Self {
            smoothing: 0.30,
            min_draws: 60,
            ewma_alpha: 0.08,
            garch_omega: 0.001,
            garch_alpha: 0.15,
            garch_beta: 0.80,
        }
    }
}

impl ForecastModel for StarVolatilityModel {
    fn name(&self) -> &str {
        "StarVolatility"
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
        let base_freq = pick as f64 / n_stars as f64;

        let mut probs = vec![0.0f64; n_stars];
        let mut volatilities = vec![0.0f64; n_stars];

        for star_idx in 0..n_stars {
            let star_num = (star_idx + 1) as u8;

            // Build EWMA frequency and GARCH volatility (chronological)
            let mut ewma = base_freq;
            let mut h = self.garch_omega / (1.0 - self.garch_alpha - self.garch_beta).max(0.01);
            // Initial unconditional variance

            for d in draws.iter().rev() {
                let val = if d.stars.contains(&star_num) { 1.0 } else { 0.0 };

                // Residual
                let e = val - ewma;
                let e_sq = e * e;

                // GARCH(1,1) update
                h = self.garch_omega + self.garch_alpha * e_sq + self.garch_beta * h;
                h = h.clamp(1e-6, 1.0); // prevent explosion

                // EWMA frequency update
                ewma = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * ewma;
            }

            probs[star_idx] = ewma;
            volatilities[star_idx] = h;
        }

        // Compute aggregated volatility metric
        let mean_vol: f64 = volatilities.iter().sum::<f64>() / n_stars as f64;

        // Blend frequency with uniform based on volatility
        // High volatility → more uniform (less trust in frequency)
        // Low volatility → more frequency (trust the signal)
        // vol_weight ∈ [0.2, 0.8]: how much to trust frequency vs uniform
        let vol_scale = (mean_vol * 20.0).clamp(0.0, 1.0); // 0=low vol, 1=high vol
        let freq_trust = 0.8 - 0.6 * vol_scale; // [0.2, 0.8]

        // Per-star inverse-volatility weighting
        let inv_vols: Vec<f64> = volatilities.iter()
            .map(|&v| 1.0 / v.max(1e-6))
            .collect();
        let inv_vol_sum: f64 = inv_vols.iter().sum();

        for i in 0..n_stars {
            // Weight each star by inverse volatility
            let vol_weight = if inv_vol_sum > 0.0 {
                inv_vols[i] / inv_vol_sum * n_stars as f64 // normalized to avg=1
            } else {
                1.0
            };

            // Blend: freq_trust * (freq * vol_weight) + (1-freq_trust) * uniform
            let uniform_val = 1.0 / n_stars as f64;
            probs[i] = freq_trust * probs[i] * vol_weight + (1.0 - freq_trust) * uniform_val;
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
            ("garch_omega".into(), self.garch_omega),
            ("garch_alpha".into(), self.garch_alpha),
            ("garch_beta".into(), self.garch_beta),
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
    fn test_star_volatility_balls_uniform() {
        let model = StarVolatilityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_volatility_stars_valid() {
        let model = StarVolatilityModel::default();
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
    fn test_star_volatility_few_draws_uniform() {
        let model = StarVolatilityModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_volatility_no_negative() {
        let model = StarVolatilityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_star_volatility_deterministic() {
        let model = StarVolatilityModel::default();
        let draws = make_test_draws(100);
        let d1 = model.predict(&draws, Pool::Stars);
        let d2 = model.predict(&draws, Pool::Stars);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
