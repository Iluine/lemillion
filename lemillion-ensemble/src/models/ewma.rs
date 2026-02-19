use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::ForecastModel;

pub struct EwmaModel {
    alpha: f64,
}

impl EwmaModel {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ForecastModel for EwmaModel {
    fn name(&self) -> &str {
        "EWMA"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let mut scores = vec![0.0f64; size];

        let floor = self.alpha.powi(draws.len() as i32 + 1);

        for (t, draw) in draws.iter().enumerate() {
            let weight = self.alpha.powi(t as i32);
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size {
                    scores[idx] += weight;
                }
            }
        }

        for score in &mut scores {
            if *score < floor {
                *score = floor;
            }
        }

        let total: f64 = scores.iter().sum();

        if total > 0.0 {
            scores.iter().map(|&s| s / total).collect()
        } else {
            vec![1.0 / size as f64; size]
        }
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
    fn test_ewma_balls_sums_to_one() {
        let model = EwmaModel::new(0.9);
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_ewma_stars_sums_to_one() {
        let model = EwmaModel::new(0.9);
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars));
    }

    #[test]
    fn test_ewma_uniform_empty() {
        let model = EwmaModel::new(0.9);
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ewma_recent_higher() {
        let draws = vec![
            Draw {
                draw_id: "001".into(), day: "MARDI".into(), date: "2024-01-02".into(),
                balls: [1, 2, 3, 4, 5], stars: [1, 2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
            Draw {
                draw_id: "000".into(), day: "VENDREDI".into(), date: "2024-01-01".into(),
                balls: [6, 7, 8, 9, 10], stars: [3, 4],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
        ];
        let dist = EwmaModel::new(0.9).predict(&draws, Pool::Balls);
        assert!(dist[0] > dist[5], "Recent ball 1 should have higher prob than older ball 6");
    }
}
