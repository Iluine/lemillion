use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// TripletBoost — scoring par triplets co-occurrents à z-score élevé.
///
/// Exploite les 81 triplets excédentaires (|z|>3 vs ~15 attendus) identifiés par
/// l'analyse de co-occurrence. Pour chaque candidat k, on calcule combien de
/// triplets à z-score élevé il forme avec les paires du dernier tirage.
pub struct TripletBoostModel {
    z_threshold: f64,
    smoothing: f64,
    min_draws: usize,
}

impl TripletBoostModel {
    pub fn new(z_threshold: f64, smoothing: f64, min_draws: usize) -> Self {
        Self { z_threshold, smoothing, min_draws }
    }
}

impl Default for TripletBoostModel {
    fn default() -> Self {
        Self {
            z_threshold: 2.0,
            smoothing: 0.5,
            min_draws: 30,
        }
    }
}

impl ForecastModel for TripletBoostModel {
    fn name(&self) -> &str {
        "TripletBoost"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Étoiles : pas de triplets possibles (seulement 2 étoiles par tirage)
        if pool == Pool::Stars {
            return uniform;
        }

        let pick = pool.pick_count();
        let n = draws.len() as f64;

        // Probabilité qu'un triplet spécifique apparaisse dans un tirage
        // p = C(size-3, pick-3) / C(size, pick)
        // p = C(size-3, pick-3) / C(size, pick) = pick*(pick-1)*(pick-2) / (size*(size-1)*(size-2))
        let p_triplet = if size >= 3 && pick >= 3 {
            (pick as f64 * (pick as f64 - 1.0) * (pick as f64 - 2.0))
                / (size as f64 * (size as f64 - 1.0) * (size as f64 - 2.0))
        } else {
            return uniform;
        };

        let expected = n * p_triplet;
        let std_dev = (n * p_triplet * (1.0 - p_triplet)).sqrt().max(1e-10);

        // Compter les co-occurrences de tous triplets
        let mut triplet_counts: HashMap<(u8, u8, u8), u32> = HashMap::new();
        for draw in draws {
            let nums = pool.numbers_from(draw);
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    for k in (j + 1)..nums.len() {
                        let mut triple = [nums[i], nums[j], nums[k]];
                        triple.sort();
                        *triplet_counts.entry((triple[0], triple[1], triple[2])).or_insert(0) += 1;
                    }
                }
            }
        }

        // Paires du dernier tirage
        let last_nums = pool.numbers_from(&draws[0]);

        // Pour chaque candidat k, scorer les triplets {a, b, k} formés avec les paires du dernier tirage
        let mut scores = vec![0.0f64; size];
        for (k_idx, score_val) in scores.iter_mut().enumerate() {
            let k = (k_idx + 1) as u8;

            let mut score = 0.0f64;
            for i in 0..last_nums.len() {
                for j in (i + 1)..last_nums.len() {
                    let a = last_nums[i];
                    let b = last_nums[j];
                    if k == a || k == b {
                        continue;
                    }
                    let mut triple = [a, b, k];
                    triple.sort();
                    let count = triplet_counts.get(&(triple[0], triple[1], triple[2]))
                        .copied()
                        .unwrap_or(0) as f64;
                    let z = (count - expected) / std_dev;
                    if z > self.z_threshold {
                        score += z;
                    }
                }
            }
            *score_val = score;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Mixer avec uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Normaliser
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            for p in &mut probs {
                *p /= total;
            }
        } else {
            return uniform;
        }

        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("z_threshold".into(), self.z_threshold),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_triplet_balls_sums_to_one() {
        let model = TripletBoostModel::default();
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
    fn test_triplet_stars_uniform() {
        let model = TripletBoostModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
        // Stars should be uniform
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Stars should be uniform");
        }
    }

    #[test]
    fn test_triplet_no_negative() {
        let model = TripletBoostModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_triplet_empty_draws() {
        let model = TripletBoostModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_triplet_few_draws() {
        let model = TripletBoostModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_triplet_deterministic() {
        let model = TripletBoostModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
