use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::ForecastModel;

pub struct MarkovModel;

impl MarkovModel {
    pub fn new() -> Self {
        Self
    }
}

impl ForecastModel for MarkovModel {
    fn name(&self) -> &str {
        "Markov"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        if draws.len() < 2 {
            return vec![1.0 / size as f64; size];
        }

        // Définir les plages
        let (n_ranges, range_size) = match pool {
            Pool::Balls => (5usize, 10usize),  // 5 plages de 10 : [1-10], [11-20], ..., [41-50]
            Pool::Stars => (3usize, 4usize),   // 3 plages de 4 : [1-4], [5-8], [9-12]
        };

        // Matrice de transition: n_ranges x n_ranges
        // transition[i][j] = P(tirer dans plage j | le tirage précédent contenait la plage i)
        let mut transition = vec![vec![0.0f64; n_ranges]; n_ranges];
        let mut from_counts = vec![0.0f64; n_ranges];

        // Calculer les comptages de transitions
        for t in 0..draws.len() - 1 {
            let current = &draws[t];
            let next = &draws[t + 1]; // plus ancien

            let current_ranges = numbers_to_ranges(pool.numbers_from(current), range_size);
            let next_ranges = numbers_to_ranges(pool.numbers_from(next), range_size);

            for &from_range in &current_ranges {
                if from_range < n_ranges {
                    from_counts[from_range] += 1.0;
                    for &to_range in &next_ranges {
                        if to_range < n_ranges {
                            transition[from_range][to_range] += 1.0;
                        }
                    }
                }
            }
        }

        // Normaliser les transitions
        for i in 0..n_ranges {
            if from_counts[i] > 0.0 {
                for j in 0..n_ranges {
                    transition[i][j] /= from_counts[i];
                }
            } else {
                // Uniforme si pas de données
                for j in 0..n_ranges {
                    transition[i][j] = 1.0 / n_ranges as f64;
                }
            }
        }

        // Obtenir les plages du tirage le plus récent
        let latest_ranges = numbers_to_ranges(pool.numbers_from(&draws[0]), range_size);

        // Prédire les probabilités par plage (moyenne sur les plages actives)
        let mut range_probs = vec![0.0f64; n_ranges];
        if !latest_ranges.is_empty() {
            for &from_range in &latest_ranges {
                if from_range < n_ranges {
                    for j in 0..n_ranges {
                        range_probs[j] += transition[from_range][j];
                    }
                }
            }
            let total: f64 = range_probs.iter().sum();
            if total > 0.0 {
                for p in &mut range_probs {
                    *p /= total;
                }
            }
        } else {
            for p in &mut range_probs {
                *p = 1.0 / n_ranges as f64;
            }
        }

        // Calculer les fréquences individuelles pour redistribuer intra-plage
        let mut freq = vec![0u32; size];
        let window = draws.len().min(50);
        for draw in &draws[..window] {
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size {
                    freq[idx] += 1;
                }
            }
        }

        // Construire la distribution finale
        let mut scores = vec![0.0f64; size];
        for i in 0..size {
            let range_idx = i / range_size;
            let range_idx = range_idx.min(n_ranges - 1);

            // Poids intra-plage basé sur la fréquence (+ lissage)
            let intra_weight = freq[i] as f64 + 1.0;
            scores[i] = range_probs[range_idx] * intra_weight;
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
        HashMap::new()
    }
}

fn numbers_to_ranges(numbers: &[u8], range_size: usize) -> Vec<usize> {
    let mut ranges: Vec<usize> = numbers
        .iter()
        .map(|&n| ((n - 1) as usize) / range_size)
        .collect();
    ranges.sort();
    ranges.dedup();
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_markov_balls_sums_to_one() {
        let model = MarkovModel::new();
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_markov_stars_sums_to_one() {
        let model = MarkovModel::new();
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_markov_no_negative() {
        let model = MarkovModel::new();
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_markov_empty_draws() {
        let model = MarkovModel::new();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }
}
