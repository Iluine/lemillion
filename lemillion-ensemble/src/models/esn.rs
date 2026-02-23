use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};
use lemillion_esn::config::EsnConfig;
use lemillion_esn::training::{predict_next, train_and_evaluate};

use super::ForecastModel;

pub struct EsnModel {
    config: EsnConfig,
}

impl EsnModel {
    pub fn new(config: EsnConfig) -> Self {
        Self { config }
    }

    fn uniform(pool: Pool) -> Vec<f64> {
        vec![1.0 / pool.size() as f64; pool.size()]
    }
}

impl ForecastModel for EsnModel {
    fn name(&self) -> &str {
        "ESN"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        if draws.len() < 10 {
            return Self::uniform(pool);
        }

        let mut config = self.config.clone();

        // Adjust washout for small windows: need washout < train_size - 1
        // DataSplit gives train_end = (n * 0.80) as usize
        let train_size = (draws.len() as f64 * 0.80) as usize;
        if train_size < 3 {
            return Self::uniform(pool);
        }
        config.washout = config.washout.min(train_size - 2);

        let (mut esn, _result) = match train_and_evaluate(draws, &config) {
            Ok(v) => v,
            Err(_) => return Self::uniform(pool),
        };

        let (ball_probs, star_probs) = predict_next(&mut esn, draws);

        match pool {
            Pool::Balls => ball_probs,
            Pool::Stars => star_probs,
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("reservoir_size".to_string(), self.config.reservoir_size as f64),
            ("spectral_radius".to_string(), self.config.spectral_radius),
            ("sparsity".to_string(), self.config.sparsity),
            ("leaking_rate".to_string(), self.config.leaking_rate),
            ("ridge_lambda".to_string(), self.config.ridge_lambda),
            ("input_scaling".to_string(), self.config.input_scaling),
            ("washout".to_string(), self.config.washout as f64),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};
    use lemillion_esn::config::Encoding;

    fn test_config() -> EsnConfig {
        EsnConfig {
            reservoir_size: 20,
            spectral_radius: 0.9,
            sparsity: 0.8,
            leaking_rate: 0.3,
            ridge_lambda: 1e-2,
            input_scaling: 0.1,
            encoding: Encoding::OneHot,
            washout: 5,
            noise_amplitude: 0.0,
            seed: 42,
        }
    }

    #[test]
    fn test_esn_balls_sums_to_one() {
        let model = EsnModel::new(test_config());
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
    fn test_esn_stars_sums_to_one() {
        let model = EsnModel::new(test_config());
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
    fn test_esn_small_draws_uniform() {
        let model = EsnModel::new(test_config());
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_esn_washout_adjustment() {
        // With washout=100 and only 30 draws, the model should adjust and not crash
        let config = EsnConfig {
            washout: 100,
            reservoir_size: 20,
            spectral_radius: 0.9,
            sparsity: 0.8,
            leaking_rate: 0.3,
            ridge_lambda: 1e-2,
            input_scaling: 0.1,
            encoding: Encoding::OneHot,
            noise_amplitude: 0.0,
            seed: 42,
        };
        let model = EsnModel::new(config);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum = {sum}");
    }
}
