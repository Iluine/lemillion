use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::{ForecastModel, SamplingStrategy};

pub struct DirichletModel {
    alpha: f64,
    window: Option<usize>,
}

impl DirichletModel {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, window: None }
    }

    pub fn with_window(alpha: f64, window: Option<usize>) -> Self {
        Self { alpha, window }
    }
}

impl ForecastModel for DirichletModel {
    fn name(&self) -> &str {
        "Dirichlet"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let effective_draws = match self.window {
            Some(w) => &draws[..w.min(draws.len())],
            None => draws,
        };

        let mut counts = vec![0u32; size];

        for draw in effective_draws {
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
        let mut params = HashMap::from([("alpha".to_string(), self.alpha)]);
        if let Some(w) = self.window {
            params.insert("window".to_string(), w as f64);
        }
        params
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
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

    #[test]
    fn test_dirichlet_windowed_more_peaked() {
        // 10 premiers tirages : toujours boule 1 (très concentré)
        // 90 tirages suivants : numéros variés répartis uniformément
        let mut draws = Vec::new();
        for i in 0..10 {
            draws.push(Draw {
                draw_id: format!("{i}"),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", i + 1),
                balls: [1, 2, 3, 4, 5],
                stars: [1, 2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            });
        }
        for i in 10..100 {
            let base = ((i * 7) % 46) as u8;
            draws.push(Draw {
                draw_id: format!("{i}"),
                day: "VENDREDI".to_string(),
                date: format!("2024-02-{:02}", (i % 28) + 1),
                balls: [base + 1, base + 2, base + 3, base + 4, base + 5],
                stars: [((i % 11) + 1) as u8, ((i % 11) + 2) as u8],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            });
        }

        // Windowed(10) avec alpha=0.1 : très concentré sur 1-5
        // Full avec alpha=1.0 : dilué par les 90 tirages variés
        let full = DirichletModel::new(1.0);
        let windowed = DirichletModel::with_window(0.1, Some(10));

        let dist_full = full.predict(&draws, Pool::Balls);
        let dist_windowed = windowed.predict(&draws, Pool::Balls);

        // Vérifier que le max de la distribution fenêtrée est plus élevé
        let max_full = dist_full.iter().cloned().fold(0.0f64, f64::max);
        let max_windowed = dist_windowed.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_windowed > max_full,
            "windowed max ({:.4}) devrait > full max ({:.4})", max_windowed, max_full);
    }

    #[test]
    fn test_dirichlet_windowed_sums_to_one() {
        let draws = make_test_draws(50);
        let model = DirichletModel::with_window(0.1, Some(10));
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
