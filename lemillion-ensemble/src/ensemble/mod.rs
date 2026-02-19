pub mod calibration;
pub mod consensus;

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

fn compute_spread(model_dists: &[(String, Vec<f64>)], size: usize) -> Vec<f64> {
    let n = model_dists.len() as f64;
    (0..size)
        .map(|j| {
            let mean = model_dists.iter().map(|(_, d)| d[j]).sum::<f64>() / n;
            let variance = model_dists.iter().map(|(_, d)| (d[j] - mean).powi(2)).sum::<f64>() / n;
            variance.sqrt()
        })
        .collect()
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
}
