pub mod calibration;
pub mod consensus;
pub mod meta;

use lemillion_db::models::{Draw, Pool};
use crate::models::ForecastModel;

pub struct EnsembleCombiner {
    pub models: Vec<Box<dyn ForecastModel>>,
    pub ball_weights: Vec<f64>,
    pub star_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    pub distribution: Vec<f64>,
    pub model_distributions: Vec<(String, Vec<f64>)>,
    pub spread: Vec<f64>,
}

impl EnsembleCombiner {
    pub fn new(models: Vec<Box<dyn ForecastModel>>) -> Self {
        let n = models.len();
        let uniform_weight = 1.0 / n as f64;
        Self {
            models,
            ball_weights: vec![uniform_weight; n],
            star_weights: vec![uniform_weight; n],
        }
    }

    pub fn with_weights(models: Vec<Box<dyn ForecastModel>>, ball_weights: Vec<f64>, star_weights: Vec<f64>) -> Self {
        Self { models, ball_weights, star_weights }
    }

    pub fn predict(&self, draws: &[Draw], pool: Pool) -> EnsemblePrediction {
        let weights = match pool {
            Pool::Balls => &self.ball_weights,
            Pool::Stars => &self.star_weights,
        };

        let mut model_distributions = Vec::new();
        let size = pool.size();
        let mut combined = vec![0.0f64; size];

        for (i, model) in self.models.iter().enumerate() {
            let dist = model.predict(draws, pool);
            let w = weights[i];
            for j in 0..size {
                combined[j] += w * dist[j];
            }
            model_distributions.push((model.name().to_string(), dist));
        }

        // Normaliser au cas où les poids ne somment pas exactement à 1
        let total: f64 = combined.iter().sum();
        if total > 0.0 {
            for p in &mut combined {
                *p /= total;
            }
        }

        let spread = compute_spread(&model_distributions, size);

        EnsemblePrediction {
            distribution: combined,
            model_distributions,
            spread,
        }
    }
}

pub fn compute_spread(model_dists: &[(String, Vec<f64>)], size: usize) -> Vec<f64> {
    let n = model_dists.len() as f64;
    (0..size)
        .map(|j| {
            let mean = model_dists.iter().map(|(_, d)| d[j]).sum::<f64>() / n;
            let variance = model_dists.iter().map(|(_, d)| (d[j] - mean).powi(2)).sum::<f64>() / n;
            variance.sqrt()
        })
        .collect()
}

impl EnsembleCombiner {
    /// Prédit avec agreement boost : les numéros où les modèles convergent
    /// reçoivent un boost proportionnel à leur agreement (1 - spread/max_spread).
    ///
    /// `strength` contrôle l'intensité du boost (0 = pas de boost, 1 = doublement max).
    pub fn predict_with_agreement_boost(
        &self,
        draws: &[Draw],
        pool: Pool,
        strength: f64,
    ) -> EnsemblePrediction {
        let base = self.predict(draws, pool);

        let max_spread = base.spread.iter().cloned().fold(0.0f64, f64::max);
        if max_spread < 1e-15 {
            return base; // pas de spread → pas de boost
        }

        let mut boosted = base.distribution.clone();
        for (i, p) in boosted.iter_mut().enumerate() {
            let agreement = 1.0 - base.spread[i] / max_spread;
            *p *= 1.0 + strength * agreement;
        }

        // Renormaliser
        let sum: f64 = boosted.iter().sum();
        if sum > 0.0 {
            for p in &mut boosted {
                *p /= sum;
            }
        }

        EnsemblePrediction {
            distribution: boosted,
            model_distributions: base.model_distributions,
            spread: base.spread,
        }
    }
}

/// Online Hedge : recalcule les poids en rejouant les N derniers tirages.
///
/// Mise à jour multiplicative : w[m] *= exp(-eta * loss_m)
/// où loss = -log(P(tirage observé | modèle m)).
pub fn compute_hedge_weights(
    models: &[Box<dyn ForecastModel>],
    draws: &[Draw],
    base_ball_weights: &[f64],
    base_star_weights: &[f64],
    n_recent: usize,
    eta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = n_recent.min(draws.len().saturating_sub(1));

    if n == 0 {
        return (base_ball_weights.to_vec(), base_star_weights.to_vec());
    }

    let mut ball_weights = base_ball_weights.to_vec();
    let mut star_weights = base_star_weights.to_vec();

    // Rejouer les n derniers tirages (draws[0..n])
    for t in (0..n).rev() {
        let test_draw = &draws[t];
        let training_draws = &draws[t + 1..];

        if training_draws.len() < 10 {
            continue;
        }

        for (m, model) in models.iter().enumerate() {
            // Loss boules
            let ball_dist = model.predict(training_draws, Pool::Balls);
            let ball_loss: f64 = test_draw.balls.iter()
                .map(|&b| -ball_dist[(b - 1) as usize].max(1e-15).ln())
                .sum();
            ball_weights[m] *= (-eta * ball_loss).exp();

            // Loss étoiles
            let star_dist = model.predict(training_draws, Pool::Stars);
            let star_loss: f64 = test_draw.stars.iter()
                .map(|&s| -star_dist[(s - 1) as usize].max(1e-15).ln())
                .sum();
            star_weights[m] *= (-eta * star_loss).exp();
        }

        // Normaliser après chaque tirage
        let bs: f64 = ball_weights.iter().sum();
        if bs > 0.0 {
            for w in &mut ball_weights {
                *w /= bs;
            }
        }
        let ss: f64 = star_weights.iter().sum();
        if ss > 0.0 {
            for w in &mut star_weights {
                *w /= ss;
            }
        }
    }

    (ball_weights, star_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;
    use crate::models::validate_distribution;

    #[test]
    fn test_ensemble_prediction_sums_to_one() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&pred.distribution, Pool::Balls));
    }

    #[test]
    fn test_ensemble_spread_length() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict(&draws, Pool::Balls);
        assert_eq!(pred.spread.len(), 50);
    }

    #[test]
    fn test_agreement_boost_sums_to_one() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict_with_agreement_boost(&draws, Pool::Balls, 0.5);
        assert!(validate_distribution(&pred.distribution, Pool::Balls));
    }

    #[test]
    fn test_agreement_boost_zero_strength_equals_base() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let base = combiner.predict(&draws, Pool::Balls);
        let boosted = combiner.predict_with_agreement_boost(&draws, Pool::Balls, 0.0);
        for (a, b) in base.distribution.iter().zip(boosted.distribution.iter()) {
            assert!((a - b).abs() < 1e-10, "Zero-strength boost should match base");
        }
    }

    #[test]
    fn test_hedge_weights_sum_to_one() {
        let models = crate::models::all_models();
        let n = models.len();
        let draws = make_test_draws(30);
        let uniform = vec![1.0 / n as f64; n];
        let (bw, sw) = compute_hedge_weights(&models, &draws, &uniform, &uniform, 5, 0.1);
        let bs: f64 = bw.iter().sum();
        let ss: f64 = sw.iter().sum();
        assert!((bs - 1.0).abs() < 1e-9, "Ball weights sum = {}", bs);
        assert!((ss - 1.0).abs() < 1e-9, "Star weights sum = {}", ss);
    }
}
