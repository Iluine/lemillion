use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::ForecastModel;

pub struct HotStreakModel {
    k: usize,
}

impl HotStreakModel {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl ForecastModel for HotStreakModel {
    fn name(&self) -> &str {
        "HotStreak"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        if draws.is_empty() {
            return vec![1.0 / size as f64; size];
        }

        let effective_k = self.k.min(draws.len());
        let mut scores = vec![0.0f64; size];

        // Plancher pour éviter les poids nuls
        let floor = 1.0 / (size as f64 * 10.0);

        // Poids linéaires décroissants : w_0 = K, w_1 = K-1, ..., w_{K-1} = 1
        for (t, draw) in draws.iter().take(effective_k).enumerate() {
            let weight = (effective_k - t) as f64;
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size {
                    scores[idx] += weight;
                }
            }
        }

        // Appliquer le plancher
        for s in &mut scores {
            if *s < floor {
                *s = floor;
            }
        }

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
        HashMap::from([("k".to_string(), self.k as f64)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_hot_streak_balls_sums_to_one() {
        let model = HotStreakModel::new(5);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_hot_streak_stars_sums_to_one() {
        let model = HotStreakModel::new(5);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_hot_streak_no_negative() {
        let model = HotStreakModel::new(5);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_hot_streak_empty_draws() {
        let model = HotStreakModel::new(5);
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hot_streak_recent_favored() {
        let draws = vec![
            Draw {
                draw_id: "002".into(), day: "MARDI".into(), date: "2024-01-03".into(),
                balls: [1, 2, 3, 4, 5], stars: [1, 2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
            Draw {
                draw_id: "001".into(), day: "VENDREDI".into(), date: "2024-01-02".into(),
                balls: [1, 2, 3, 4, 5], stars: [1, 2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
            Draw {
                draw_id: "000".into(), day: "MARDI".into(), date: "2024-01-01".into(),
                balls: [6, 7, 8, 9, 10], stars: [3, 4],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
        ];
        let dist = HotStreakModel::new(3).predict(&draws, Pool::Balls);
        // Boule 1 (présente dans 2 tirages récents) devrait avoir une prob plus haute que boule 6 (1 tirage ancien)
        assert!(dist[0] > dist[5]);
    }
}
