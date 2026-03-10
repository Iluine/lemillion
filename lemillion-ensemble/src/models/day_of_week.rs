use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// DayOfWeekModel — exploite les différences MARDI/VENDREDI.
///
/// EuroMillions alterne MARDI et VENDREDI. Si les machines, opérateurs ou
/// conditions physiques diffèrent par jour, la distribution pourrait varier.
///
/// Algorithme :
/// 1. Déterminer le jour du prochain tirage (inverse du dernier)
/// 2. Filtrer l'historique par jour correspondant
/// 3. Fréquence EWMA sur les tirages filtrés
/// 4. Blend avec fréquences globales via smoothing
pub struct DayOfWeekModel {
    smoothing: f64,
    ewma_alpha: f64,
    min_draws: usize,
}

impl Default for DayOfWeekModel {
    fn default() -> Self {
        Self {
            smoothing: 0.35,
            ewma_alpha: 0.10,
            min_draws: 50,
        }
    }
}

impl ForecastModel for DayOfWeekModel {
    fn name(&self) -> &str {
        "DayOfWeek"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Déterminer le jour du prochain tirage (inverse du dernier)
        let last_is_friday = draws[0].day.contains("VENDREDI");
        let next_is_friday = !last_is_friday;

        // Filtrer tirages par jour correspondant (chronologique inversé → draws[0] = récent)
        let day_filter = if next_is_friday { "VENDREDI" } else { "MARDI" };
        let same_day_draws: Vec<&Draw> = draws.iter()
            .filter(|d| d.day.contains(day_filter))
            .collect();

        if same_day_draws.len() < 20 {
            return uniform;
        }

        // EWMA sur les tirages du même jour (itérer en chronologique = inversé)
        let mut freq_ewma = vec![1.0 / size as f64; size];

        for draw in same_day_draws.iter().rev() {
            let nums = pool.numbers_from(draw);
            for idx in 0..size {
                let num = (idx + 1) as u8;
                let present = if nums.contains(&num) { 1.0 } else { 0.0 };
                freq_ewma[idx] = self.ewma_alpha * present + (1.0 - self.ewma_alpha) * freq_ewma[idx];
            }
        }

        // Normaliser
        let sum: f64 = freq_ewma.iter().sum();
        if sum <= 0.0 {
            return uniform;
        }
        for p in &mut freq_ewma {
            *p /= sum;
        }

        // Smoothing avec uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut freq_ewma {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = match pool {
            Pool::Balls => PROB_FLOOR_BALLS,
            Pool::Stars => PROB_FLOOR_STARS,
        };
        floor_only(&mut freq_ewma, floor);
        freq_ewma
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_day_of_week_balls_valid() {
        let model = DayOfWeekModel::default();
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
    fn test_day_of_week_stars_valid() {
        let model = DayOfWeekModel::default();
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
    fn test_day_of_week_few_draws_uniform() {
        let model = DayOfWeekModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_day_of_week_deterministic() {
        let model = DayOfWeekModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_day_of_week_sparse_strategy() {
        let model = DayOfWeekModel::default();
        assert!(matches!(model.sampling_strategy(), SamplingStrategy::Sparse { span_multiplier: 3 }));
    }
}
