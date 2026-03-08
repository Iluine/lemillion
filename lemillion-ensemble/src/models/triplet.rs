use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS};

/// TripletBoost — scoring par triplets co-occurrents à z-score élevé.
///
/// Exploite les 83 triplets excédentaires (|z|>3 vs ~15 attendus) identifiés par
/// l'analyse de co-occurrence. Pour chaque candidat k, on calcule combien de
/// triplets à z-score élevé il forme avec les paires des 5 derniers tirages.
/// Comptage temporellement pondéré + top-K triplets par |z|.
pub struct TripletBoostModel {
    z_threshold: f64,
    smoothing: f64,
    min_draws: usize,
    n_context_draws: usize,
    temporal_alpha: f64,
    max_triplets: usize,
}

impl Default for TripletBoostModel {
    fn default() -> Self {
        Self {
            z_threshold: 2.0,
            smoothing: 0.30,
            min_draws: 50,
            n_context_draws: 5,
            temporal_alpha: 0.03,
            max_triplets: 300,
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
        if size < 3 || pick < 3 {
            return uniform;
        }

        // Probabilité théorique d'un triplet
        let p_triplet = (pick as f64 * (pick as f64 - 1.0) * (pick as f64 - 2.0))
            / (size as f64 * (size as f64 - 1.0) * (size as f64 - 2.0));

        // Comptage temporellement pondéré des triplets
        let mut triplet_weighted: HashMap<(u8, u8, u8), f64> = HashMap::new();
        let mut total_weight = 0.0f64;
        for (t, draw) in draws.iter().enumerate() {
            let w = (-self.temporal_alpha * t as f64).exp();
            total_weight += w;
            let nums = pool.numbers_from(draw);
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    for k in (j + 1)..nums.len() {
                        let mut triple = [nums[i], nums[j], nums[k]];
                        triple.sort();
                        *triplet_weighted.entry((triple[0], triple[1], triple[2])).or_insert(0.0) += w;
                    }
                }
            }
        }

        let expected = total_weight * p_triplet;
        let std_dev = (total_weight * p_triplet * (1.0 - p_triplet)).sqrt().max(1e-10);

        // Calculer z-scores et garder top-K par |z|
        let mut triplet_z: Vec<((u8, u8, u8), f64)> = triplet_weighted.iter()
            .map(|(&trip, &count)| (trip, (count - expected) / std_dev))
            .filter(|&(_, z)| z.abs() > self.z_threshold)
            .collect();
        triplet_z.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        triplet_z.truncate(self.max_triplets);

        let top_triplets: HashMap<(u8, u8, u8), f64> = triplet_z.into_iter().collect();

        if top_triplets.is_empty() {
            return uniform;
        }

        // Collecter les paires des N derniers tirages avec poids par distance
        let n_ctx = self.n_context_draws.min(draws.len());
        let context_weights = [1.0, 0.8, 0.6, 0.4, 0.2];
        let mut context_pairs: Vec<((u8, u8), f64)> = Vec::new();
        for d in 0..n_ctx {
            let pair_weight = if d < context_weights.len() { context_weights[d] } else { 0.1 };
            let nums = pool.numbers_from(&draws[d]);
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    context_pairs.push(((nums[i], nums[j]), pair_weight));
                }
            }
        }

        // Pour chaque candidat k, scorer via triplets formés avec les paires de contexte
        let mut scores = vec![0.0f64; size];
        for (k_idx, score_val) in scores.iter_mut().enumerate() {
            let k = (k_idx + 1) as u8;

            let mut score = 0.0f64;
            for &((a, b), pair_weight) in &context_pairs {
                if k == a || k == b {
                    continue;
                }
                let mut triple = [a, b, k];
                triple.sort();
                if let Some(&z) = top_triplets.get(&(triple[0], triple[1], triple[2])) {
                    score += pair_weight * z;
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

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("z_threshold".into(), self.z_threshold),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("n_context_draws".into(), self.n_context_draws as f64),
            ("temporal_alpha".into(), self.temporal_alpha),
            ("max_triplets".into(), self.max_triplets as f64),
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
    fn test_triplet_balls_sums_to_one() {
        let model = TripletBoostModel::default();
        let draws = make_test_draws(80);
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
        let draws = make_test_draws(80);
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
        let draws = make_test_draws(80);
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
        let draws = make_test_draws(80);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_triplet_sparse_strategy() {
        let model = TripletBoostModel::default();
        assert!(matches!(model.sampling_strategy(), SamplingStrategy::Sparse { span_multiplier: 3 }));
    }
}
