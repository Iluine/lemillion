use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::ForecastModel;

pub struct RetardModel {
    gamma: f64,
}

impl RetardModel {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl ForecastModel for RetardModel {
    fn name(&self) -> &str {
        "Retard"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        if draws.is_empty() {
            return vec![1.0 / size as f64; size];
        }

        // Calculer le gap actuel pour chaque numéro
        let mut gaps = vec![draws.len(); size];
        for (t, draw) in draws.iter().enumerate() {
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size && gaps[idx] == draws.len() {
                    gaps[idx] = t;
                }
            }
        }

        // Calculer le gap moyen pour chaque numéro
        let mut mean_gaps = vec![0.0f64; size];
        for i in 0..size {
            let mut gap_list = Vec::new();
            let mut last_seen: Option<usize> = None;

            for (t, draw) in draws.iter().enumerate() {
                let number = (i + 1) as u8;
                if pool.numbers_from(draw).contains(&number) {
                    if let Some(prev) = last_seen {
                        gap_list.push((t - prev) as f64);
                    }
                    last_seen = Some(t);
                }
            }

            mean_gaps[i] = if gap_list.is_empty() {
                // Estimation par le ratio théorique
                size as f64 / pool.pick_count() as f64
            } else {
                gap_list.iter().sum::<f64>() / gap_list.len() as f64
            };
        }

        // score(i) = (gap_actuel / gap_moyen)^gamma
        let mut scores: Vec<f64> = (0..size)
            .map(|i| {
                let ratio = (gaps[i] as f64 + 1.0) / mean_gaps[i].max(1.0);
                ratio.powf(self.gamma)
            })
            .collect();

        // Normaliser
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in &mut scores {
                *s /= total;
            }
        } else {
            scores = vec![1.0 / size as f64; size];
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([("gamma".to_string(), self.gamma)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_retard_balls_sums_to_one() {
        let model = RetardModel::new(1.5);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_retard_stars_sums_to_one() {
        let model = RetardModel::new(1.5);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_retard_no_negative() {
        let model = RetardModel::new(1.5);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_retard_empty_draws() {
        let model = RetardModel::new(1.5);
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }
}
