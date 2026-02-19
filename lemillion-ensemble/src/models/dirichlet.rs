use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::ForecastModel;

pub struct DirichletModel {
    alpha: f64,
}

impl DirichletModel {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ForecastModel for DirichletModel {
    fn name(&self) -> &str {
        "Dirichlet"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let mut counts = vec![0u32; size];

        for draw in draws {
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size {
                    counts[idx] += 1;
                }
            }
        }

        let total: u32 = counts.iter().sum();
        let denominator = size as f64 * self.alpha + total as f64;

        counts
            .iter()
            .map(|&count| (self.alpha + count as f64) / denominator)
            .collect()
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([("alpha".to_string(), self.alpha)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;
    use crate::models::validate_distribution;

    #[test]
    fn test_dirichlet_balls_sums_to_one() {
        let model = DirichletModel::new(1.0);
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_dirichlet_stars_sums_to_one() {
        let model = DirichletModel::new(1.0);
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars));
    }

    #[test]
    fn test_dirichlet_uniform_empty() {
        let model = DirichletModel::new(1.0);
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }
}
