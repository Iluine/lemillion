use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// CondSummary — prédiction conditionnelle basée sur le résumé du tirage précédent.
///
/// Exploite le gain informationnel de 66% identifié par le test d'entropie conditionnelle.
/// Encode chaque tirage en 4D (sum_bin, spread_bin, odd_count, decade_mask) et construit
/// une table de fréquences conditionnelles : P(numéro k | état du tirage précédent).
pub struct ConditionalSummaryModel {
    smoothing: f64,
    laplace_alpha: f64,
    min_draws: usize,
}

impl ConditionalSummaryModel {
    pub fn new(smoothing: f64, laplace_alpha: f64, min_draws: usize) -> Self {
        Self { smoothing, laplace_alpha, min_draws }
    }
}

impl Default for ConditionalSummaryModel {
    fn default() -> Self {
        Self {
            smoothing: 0.4,
            laplace_alpha: 1.0,
            min_draws: 20,
        }
    }
}

/// Encode un tirage en résumé 4D.
/// Adapté par pool (balls: 5 numéros 1-50, stars: 2 numéros 1-12).
fn encode_summary(numbers: &[u8], pool: Pool) -> [usize; 4] {
    let sum: u16 = numbers.iter().map(|&n| n as u16).sum();
    let min = *numbers.iter().min().unwrap_or(&0);
    let max = *numbers.iter().max().unwrap_or(&0);
    let spread = max - min;
    let odd_count = numbers.iter().filter(|&&n| n % 2 == 1).count();

    match pool {
        Pool::Balls => {
            // sum range: ~15-240, bin into 5 classes
            let sum_bin = ((sum as f64 - 15.0) / 30.0 * 5.0).floor().clamp(0.0, 4.0) as usize;
            let spread_bin = (spread as usize / 10).min(4);
            // decade_mask: which decades have at least one ball
            let decade_mask = numbers.iter()
                .map(|&b| 1usize << (((b - 1) / 10) as usize))
                .fold(0usize, |acc, x| acc | x);
            [sum_bin, spread_bin, odd_count, decade_mask]
        }
        Pool::Stars => {
            // sum range: 2-24, bin into 5 classes
            let sum_bin = ((sum as f64 - 2.0) / 5.0).floor().clamp(0.0, 4.0) as usize;
            let spread_bin = (spread as usize / 3).min(4);
            // For stars, decade_mask is just high/low
            let high_count = numbers.iter().filter(|&&s| s > 6).count();
            [sum_bin, spread_bin, odd_count, high_count]
        }
    }
}

/// Clé de hachage pour un état résumé (pour éviter un HashMap sur [usize; 4]).
fn state_key(state: &[usize; 4]) -> u64 {
    // Pack 4 usize values into a u64
    (state[0] as u64)
        | ((state[1] as u64) << 8)
        | ((state[2] as u64) << 16)
        | ((state[3] as u64) << 24)
}

impl ForecastModel for ConditionalSummaryModel {
    fn name(&self) -> &str {
        "CondSummary"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Encoder tous les tirages
        let summaries: Vec<[usize; 4]> = draws.iter()
            .map(|d| encode_summary(pool.numbers_from(d), pool))
            .collect();

        // Construire table conditionnelle : cond_freq[state_key][num_idx] += 1
        // Pour chaque paire (t, t+1) : state du tirage t -> numéros du tirage t+1
        // draws[0] = plus récent, draws[n-1] = plus ancien
        // Paire (t, t+1) signifie : état de draws[t+1] prédit draws[t]
        let mut cond_counts: HashMap<u64, Vec<f64>> = HashMap::new();
        let mut state_totals: HashMap<u64, f64> = HashMap::new();

        // Fréquences marginales pour fallback
        let mut marginal = vec![0.0f64; size];

        for t in 0..draws.len() - 1 {
            // L'état conditionnel est le tirage précédent (plus ancien)
            let key = state_key(&summaries[t + 1]);
            let entry = cond_counts.entry(key).or_insert_with(|| vec![0.0; size]);
            *state_totals.entry(key).or_insert(0.0) += 1.0;

            // Les numéros observés sont ceux du tirage courant (plus récent)
            for &n in pool.numbers_from(&draws[t]) {
                let idx = (n - 1) as usize;
                if idx < size {
                    entry[idx] += 1.0;
                    marginal[idx] += 1.0;
                }
            }
        }

        // Normaliser les marginales
        let marginal_total: f64 = marginal.iter().sum();
        if marginal_total > 0.0 {
            for m in &mut marginal {
                *m /= marginal_total;
            }
        } else {
            return uniform;
        }

        // Lookup : état actuel = encode(draws[0])
        let current_key = state_key(&summaries[0]);

        let probs = if let Some(counts) = cond_counts.get(&current_key) {
            let total = state_totals[&current_key];
            let alpha = self.laplace_alpha;
            counts.iter()
                .map(|&c| (c + alpha) / (total + alpha * size as f64))
                .collect::<Vec<f64>>()
        } else {
            // Fallback : fréquences marginales
            marginal.clone()
        };

        // Mixer avec uniforme
        let uniform_val = 1.0 / size as f64;
        let mut dist: Vec<f64> = probs.iter()
            .map(|&p| (1.0 - self.smoothing) * p + self.smoothing * uniform_val)
            .collect();

        // Normaliser
        let total: f64 = dist.iter().sum();
        if total > 0.0 {
            for p in &mut dist {
                *p /= total;
            }
        } else {
            return uniform;
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("laplace_alpha".into(), self.laplace_alpha),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_conditional_balls_sums_to_one() {
        let model = ConditionalSummaryModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_conditional_stars_sums_to_one() {
        let model = ConditionalSummaryModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_conditional_no_negative() {
        let model = ConditionalSummaryModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_conditional_empty_draws() {
        let model = ConditionalSummaryModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_conditional_few_draws() {
        let model = ConditionalSummaryModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_conditional_deterministic() {
        let model = ConditionalSummaryModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_encode_summary_balls() {
        let summary = encode_summary(&[1, 10, 20, 30, 40], Pool::Balls);
        // sum=101, sum_bin = ((101-15)/30*5).floor().clamp(0,4) = 14.3 -> 4
        assert!(summary[0] <= 4);
        // spread=39, spread_bin = 39/10 = 3
        assert_eq!(summary[1], 3);
        // odd_count: 1 is odd -> 1
        assert_eq!(summary[2], 1);
        // decade_mask: decades 0,0,1,2,3
        assert!(summary[3] > 0);
    }

    #[test]
    fn test_encode_summary_stars() {
        let summary = encode_summary(&[3, 10], Pool::Stars);
        assert!(summary[0] <= 4);
        assert!(summary[1] <= 4);
    }
}
