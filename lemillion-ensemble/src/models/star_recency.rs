use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// StarRecencyModel — EWMA multi-échelle sur fréquence par étoile.
///
/// 3 alphas (rapide, moyen, lent) fusionnés à poids égaux.
/// Retourne uniforme pour Pool::Balls.
pub struct StarRecencyModel {
    alphas: [f64; 3],
    smoothing: f64,
    min_draws: usize,
}

impl Default for StarRecencyModel {
    fn default() -> Self {
        Self {
            alphas: [0.20, 0.10, 0.05],
            smoothing: 0.30,
            min_draws: 20,
        }
    }
}

impl ForecastModel for StarRecencyModel {
    fn name(&self) -> &str {
        "StarRecency"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if pool == Pool::Balls {
            return uniform;
        }

        if draws.len() < self.min_draws {
            return uniform;
        }

        let mut combined = vec![0.0f64; size];

        for &alpha in &self.alphas {
            let mut ewma = vec![1.0 / size as f64; size];

            // Itérer du plus ancien au plus récent
            for draw in draws.iter().rev() {
                let mut indicator = vec![0.0f64; size];
                for &s in &draw.stars {
                    let idx = (s - 1) as usize;
                    if idx < size {
                        indicator[idx] = 1.0;
                    }
                }
                for i in 0..size {
                    ewma[i] = (1.0 - alpha) * ewma[i] + alpha * indicator[i];
                }
            }

            for i in 0..size {
                combined[i] += ewma[i];
            }
        }

        // Normalize combined (3 alphas, merge 1/3 chaque)
        let sum: f64 = combined.iter().sum();
        if sum > 0.0 {
            for c in &mut combined {
                *c /= sum;
            }
        }

        // Smooth with uniform
        let uniform_val = 1.0 / size as f64;
        for c in &mut combined {
            *c = (1.0 - self.smoothing) * *c + self.smoothing * uniform_val;
        }

        // Renormalize
        let sum: f64 = combined.iter().sum();
        if sum > 0.0 {
            for c in &mut combined {
                *c /= sum;
            }
        }

        combined
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("alpha_fast".into(), self.alphas[0]),
            ("alpha_mid".into(), self.alphas[1]),
            ("alpha_slow".into(), self.alphas[2]),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
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
    fn test_star_recency_sums_to_one() {
        let model = StarRecencyModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_star_recency_balls_uniform() {
        let model = StarRecencyModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_recency_few_draws_uniform() {
        let model = StarRecencyModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_recency_no_negative() {
        let model = StarRecencyModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Stars);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_star_recency_deterministic() {
        let model = StarRecencyModel::default();
        let draws = make_test_draws(60);
        let d1 = model.predict(&draws, Pool::Stars);
        let d2 = model.predict(&draws, Pool::Stars);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
