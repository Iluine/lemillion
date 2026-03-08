use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::{ForecastModel, SamplingStrategy, floor_and_normalize, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Modèle contextuel basé sur la taille du jackpot.
///
/// Quand le jackpot monte (gros lots), plus de joueurs occasionnels participent.
/// Ceux-ci jouent davantage des numéros "birthday" (1-31) et des numéros "chanceux" (7, 13, etc).
/// Ce modèle surpondère les numéros hauts (31-50) quand on détecte un gros jackpot
/// (pas de gagnant récent), car les joueurs réguliers sont sous-représentés dans les petits jackpots.
///
/// Ne change pas P(win) mais concentre sur des numéros moins joués → meilleur EV et diversification.
pub struct JackpotContextModel {
    smoothing: f64,
    min_draws: usize,
}

impl Default for JackpotContextModel {
    fn default() -> Self {
        Self {
            smoothing: 0.40,
            min_draws: 30,
        }
    }
}

impl ForecastModel for JackpotContextModel {
    fn name(&self) -> &str {
        "JackpotCtx"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let uniform = 1.0 / n as f64;

        if draws.len() < self.min_draws {
            return vec![uniform; n];
        }

        // Estimer la "phase" du jackpot : combien de tirages sans gagnant
        // On regarde le winner_count des tirages récents
        let no_winner_streak = draws.iter()
            .take(20)
            .take_while(|d| d.winner_count == 0)
            .count();

        // Normaliser le streak en [0, 1] — 20 tirages sans gagnant = jackpot très élevé
        let jackpot_phase = (no_winner_streak as f64 / 15.0).min(1.0);

        match pool {
            Pool::Balls => {
                let mut probs = vec![0.0f64; 50];

                // Base : fréquence historique récente (100 tirages)
                let window = draws.len().min(100);
                for draw in &draws[..window] {
                    for &b in &draw.balls {
                        probs[(b - 1) as usize] += 1.0;
                    }
                }

                // Normaliser
                let total: f64 = probs.iter().sum();
                if total > 0.0 {
                    for p in probs.iter_mut() {
                        *p /= total;
                    }
                }

                // Appliquer le biais jackpot : surpondérer les hauts numéros quand jackpot monte
                // Ratio : numeros > 31 reçoivent un bonus proportionnel au jackpot_phase
                let high_bonus = 1.0 + 0.3 * jackpot_phase; // jusqu'à +30% pour hauts numéros
                let low_malus = 1.0 - 0.15 * jackpot_phase; // -15% pour bas numéros

                for (i, p) in probs.iter_mut().enumerate() {
                    let num = (i + 1) as u8;
                    if num > 31 {
                        *p *= high_bonus;
                    } else if num <= 12 {
                        // Numéros très "birthday" (mois)
                        *p *= low_malus * 0.95;
                    } else {
                        *p *= low_malus;
                    }
                }

                // Smoothing avec uniforme
                for p in probs.iter_mut() {
                    *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
                }

                floor_and_normalize(&mut probs, PROB_FLOOR_BALLS);
                probs
            }
            Pool::Stars => {
                let mut probs = vec![0.0f64; 12];

                // Base fréquence
                let window = draws.len().min(100);
                for draw in &draws[..window] {
                    for &s in &draw.stars {
                        probs[(s - 1) as usize] += 1.0;
                    }
                }

                let total: f64 = probs.iter().sum();
                if total > 0.0 {
                    for p in probs.iter_mut() {
                        *p /= total;
                    }
                }

                // Étoiles hautes (>6) légèrement favorisées en phase jackpot élevé
                let high_bonus = 1.0 + 0.15 * jackpot_phase;
                for (i, p) in probs.iter_mut().enumerate() {
                    if i + 1 > 6 {
                        *p *= high_bonus;
                    }
                }

                for p in probs.iter_mut() {
                    *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
                }

                floor_and_normalize(&mut probs, PROB_FLOOR_STARS);
                probs
            }
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws".to_string(), self.min_draws as f64);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_jackpot_context_returns_valid_distribution() {
        let model = JackpotContextModel::default();
        let draws = make_test_draws(50);

        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            assert_eq!(dist.len(), pool.size());
            let sum: f64 = dist.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
            assert!(dist.iter().all(|&p| p >= 0.0));
        }
    }

    #[test]
    fn test_jackpot_context_few_draws_uniform() {
        let model = JackpotContextModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9);
        }
    }
}
