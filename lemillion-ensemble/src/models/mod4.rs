use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// Mod4Trans — matrice de transition sur les classes de résidus mod 4.
///
/// Exploite la corrélation inter-tirages des résidus mod 4 (cosine 0.68 vs 0.38 attendu),
/// liée aux 4 pales de la machine Stresa. Le tirage t prédit la distribution mod 4 du
/// tirage t+1 via une matrice de transition avec lissage de Laplace.
pub struct Mod4TransitionModel {
    smoothing: f64,
    min_draws: usize,
}

impl Mod4TransitionModel {
    pub fn new(smoothing: f64, min_draws: usize) -> Self {
        Self { smoothing, min_draws }
    }
}

impl Default for Mod4TransitionModel {
    fn default() -> Self {
        Self {
            smoothing: 0.5,
            min_draws: 10,
        }
    }
}

/// Encode un tirage en vecteur de comptages mod 4.
fn mod4_counts(numbers: &[u8]) -> [f64; 4] {
    let mut counts = [0.0f64; 4];
    for &n in numbers {
        let r = ((n - 1) % 4) as usize;
        counts[r] += 1.0;
    }
    counts
}

impl ForecastModel for Mod4TransitionModel {
    fn name(&self) -> &str {
        "Mod4Trans"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let numbers_list: Vec<Vec<u8>> = draws.iter()
            .map(|d| pool.numbers_from(d).to_vec())
            .collect();

        // Matrice de transition T[i][j] : probabilité de passer du résidu i au résidu j
        // Avec lissage de Laplace (+1 partout)
        let mut transition = [[1.0f64; 4]; 4];

        // Parcourir les paires consécutives (draws[0] = plus récent)
        for t in 0..draws.len() - 1 {
            let current = mod4_counts(&numbers_list[t]);
            let next = mod4_counts(&numbers_list[t + 1]);

            for (i, &c_i) in current.iter().enumerate() {
                if c_i > 0.0 {
                    for (j, &n_j) in next.iter().enumerate() {
                        transition[i][j] += c_i * n_j;
                    }
                }
            }
        }

        // Normaliser les lignes
        for row in &mut transition {
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
        }

        // Prédiction : utiliser le dernier tirage (draws[0])
        let current = mod4_counts(&numbers_list[0]);
        let total_current: f64 = current.iter().sum();

        let mut p_next = [0.0f64; 4];
        if total_current > 0.0 {
            for (i, &c) in current.iter().enumerate() {
                let w = c / total_current;
                for (j, p) in p_next.iter_mut().enumerate() {
                    *p += w * transition[i][j];
                }
            }
        } else {
            p_next = [0.25; 4];
        }

        // Fréquences historiques par numéro (pour redistribution intra-classe)
        let mut freq = vec![1.0f64; size]; // Laplace +1
        for nums in &numbers_list {
            for &n in nums {
                let idx = (n - 1) as usize;
                if idx < size {
                    freq[idx] += 1.0;
                }
            }
        }

        // Fréquences normalisées au sein de chaque classe mod 4
        let mut class_freq_sum = [0.0f64; 4];
        for (k, &f) in freq.iter().enumerate() {
            class_freq_sum[k % 4] += f;
        }

        // Redistribuer selon les fréquences historiques au sein de chaque classe
        let mut prob = vec![0.0f64; size];
        for (k, p) in prob.iter_mut().enumerate() {
            let r = k % 4;
            if class_freq_sum[r] > 0.0 {
                *p = p_next[r] * freq[k] / class_freq_sum[r];
            }
        }

        // Mixer avec uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut prob {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Normaliser
        let total: f64 = prob.iter().sum();
        if total > 0.0 {
            for p in &mut prob {
                *p /= total;
            }
        } else {
            return uniform;
        }

        prob
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
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
    fn test_mod4_balls_sums_to_one() {
        let model = Mod4TransitionModel::default();
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
    fn test_mod4_stars_sums_to_one() {
        let model = Mod4TransitionModel::default();
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
    fn test_mod4_no_negative() {
        let model = Mod4TransitionModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_mod4_empty_draws() {
        let model = Mod4TransitionModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mod4_few_draws() {
        let model = Mod4TransitionModel::default();
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mod4_deterministic() {
        let model = Mod4TransitionModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_mod4_counts() {
        let counts = mod4_counts(&[1, 2, 3, 4, 5]);
        // 1 -> (1-1)%4=0, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 0
        assert_eq!(counts, [2.0, 1.0, 1.0, 1.0]);
    }
}
