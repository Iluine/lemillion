use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// SpreadModel — exploite le clustering des boules (spread moyen 33.6 vs 40.8 attendu).
///
/// Prédit un "centre de masse" des boules via EWMA et booste les numéros
/// proches de ce centre via un noyau gaussien. Blend avec fréquence EWMA.
///
/// Pour les étoiles: fréquence EWMA simple avec smoothing élevé.
pub struct SpreadModel {
    smoothing: f64,
    min_draws: usize,
    center_alpha: f64,
    spread_alpha: f64,
    sigma_factor: f64,
}

impl Default for SpreadModel {
    fn default() -> Self {
        Self {
            smoothing: 0.35,
            min_draws: 30,
            center_alpha: 0.08,
            spread_alpha: 0.08,
            sigma_factor: 3.5,
        }
    }
}

impl ForecastModel for SpreadModel {
    fn name(&self) -> &str {
        "Spread"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        match pool {
            Pool::Balls => self.predict_balls(draws, size),
            Pool::Stars => self.predict_stars(draws, size),
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("center_alpha".into(), self.center_alpha),
            ("spread_alpha".into(), self.spread_alpha),
            ("sigma_factor".into(), self.sigma_factor),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

impl SpreadModel {
    fn predict_balls(&self, draws: &[Draw], size: usize) -> Vec<f64> {
        let uniform = vec![1.0 / size as f64; size];

        // EWMA du centre de masse et du spread (itérer en ordre chronologique)
        let mut center_ewma = 25.5f64; // centre théorique (1+50)/2
        let mut spread_ewma = 40.8f64; // spread théorique attendu

        // Fréquence EWMA par numéro
        let mut freq_ewma = vec![1.0 / size as f64; size];

        for draw in draws.iter().rev() {
            let balls = &draw.balls;
            let mut sorted = *balls;
            sorted.sort();

            let center = sorted.iter().map(|&b| b as f64).sum::<f64>() / 5.0;
            let spread = (sorted[4] - sorted[0]) as f64;

            center_ewma = self.center_alpha * center + (1.0 - self.center_alpha) * center_ewma;
            spread_ewma = self.spread_alpha * spread + (1.0 - self.spread_alpha) * spread_ewma;

            // Mise à jour fréquence EWMA
            for (idx, freq) in freq_ewma.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if balls.contains(&num) { 1.0 } else { 0.0 };
                *freq = self.center_alpha * present + (1.0 - self.center_alpha) * *freq;
            }
        }

        // Noyau gaussien centré sur center_ewma
        let sigma = (spread_ewma / self.sigma_factor).max(1.0);
        let mut kernel_scores = vec![0.0f64; size];
        for k in 0..size {
            let num = (k + 1) as f64;
            let diff = num - center_ewma;
            kernel_scores[k] = (-diff * diff / (2.0 * sigma * sigma)).exp();
        }

        // Normaliser le noyau
        let kernel_sum: f64 = kernel_scores.iter().sum();
        if kernel_sum > 0.0 {
            for s in &mut kernel_scores {
                *s /= kernel_sum;
            }
        }

        // Blend : 60% noyau spatial + 40% fréquence EWMA
        let mut probs = vec![0.0f64; size];
        for k in 0..size {
            probs[k] = 0.6 * kernel_scores[k] + 0.4 * freq_ewma[k];
        }

        // Normaliser
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smoothing avec uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    fn predict_stars(&self, draws: &[Draw], size: usize) -> Vec<f64> {
        let uniform = vec![1.0 / size as f64; size];
        let star_smoothing = 0.45;

        // Fréquence EWMA pour les étoiles
        let mut freq_ewma = vec![1.0 / size as f64; size];

        for draw in draws.iter().rev() {
            for (idx, freq) in freq_ewma.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if draw.stars.contains(&num) { 1.0 } else { 0.0 };
                *freq = self.center_alpha * present + (1.0 - self.center_alpha) * *freq;
            }
        }

        // Normaliser
        let sum: f64 = freq_ewma.iter().sum();
        if sum > 0.0 {
            for p in &mut freq_ewma {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smoothing élevé avec uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut freq_ewma {
            *p = (1.0 - star_smoothing) * *p + star_smoothing * uniform_val;
        }

        floor_only(&mut freq_ewma, PROB_FLOOR_STARS);
        freq_ewma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_spread_balls_valid() {
        let model = SpreadModel::default();
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
    fn test_spread_stars_valid() {
        let model = SpreadModel::default();
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
    fn test_spread_few_draws() {
        let model = SpreadModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Too few draws → uniform");
        }
    }

    #[test]
    fn test_spread_no_negative() {
        let model = SpreadModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_spread_deterministic() {
        let model = SpreadModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_spread_sparse_strategy() {
        let model = SpreadModel::default();
        assert!(matches!(model.sampling_strategy(), SamplingStrategy::Sparse { span_multiplier: 3 }));
    }

    #[test]
    fn test_spread_balls_not_uniform() {
        // With enough draws, the spread model should produce non-uniform output
        let model = SpreadModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        let all_uniform = dist.iter().all(|&p| (p - expected).abs() < 1e-6);
        // With the test data, the model should have some signal
        assert!(!all_uniform || draws.len() < model.min_draws, "Should have some signal");
    }
}
