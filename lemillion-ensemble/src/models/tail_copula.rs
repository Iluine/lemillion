/// TailCopula v21: Copule de Clayton pour dépendance de queue.
///
/// L'ancien modèle Copula (retiré) utilisait une copule Gaussienne — insensible
/// aux événements extrêmes. Pour le jackpot, les co-occurrences RARES sont le signal.
///
/// Algorithme:
/// 1. Pour chaque paire (i,j) de boules, compter co-occurrences dans les N derniers tirages
/// 2. Estimer θ de Clayton via méthode des moments (Kendall's tau)
/// 3. Clayton amplifie la dépendance de queue basse
/// 4. Score marginal = Σ_j copula_score(i,j) × P(j tiré)

use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS};

pub struct TailCopulaModel {
    smoothing: f64,
    min_draws: usize,
    window: usize,
}

impl Default for TailCopulaModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            min_draws: 50,
            window: 300,
        }
    }
}

impl ForecastModel for TailCopulaModel {
    fn name(&self) -> &str { "TailCopula" }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("smoothing".into(), self.smoothing);
        m.insert("window".into(), self.window as f64);
        m
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;
        let pick = pool.pick_count();

        if draws.len() < self.min_draws || pool == Pool::Stars {
            // Stars have too few pairs (C(12,2)=66) for reliable copula estimation
            return vec![uniform; size];
        }

        let n = draws.len().min(self.window);

        // 1. Compute empirical marginal frequencies and co-occurrence counts
        let mut freq = vec![0.0_f64; size];
        let mut cooccurrence = vec![vec![0.0_f64; size]; size];

        for draw in &draws[..n] {
            let nums = pool.numbers_from(draw);
            for &num in nums {
                freq[(num - 1) as usize] += 1.0;
            }
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    let a = (nums[i] - 1) as usize;
                    let b = (nums[j] - 1) as usize;
                    cooccurrence[a][b] += 1.0;
                    cooccurrence[b][a] += 1.0;
                }
            }
        }

        let n_f = n as f64;
        for f in freq.iter_mut() { *f /= n_f; }
        for row in cooccurrence.iter_mut() {
            for c in row.iter_mut() { *c /= n_f; }
        }

        // Expected co-occurrence under independence
        let expected_cooc = (pick as f64 * (pick as f64 - 1.0)) / (size as f64 * (size as f64 - 1.0));

        // 2. Estimate Clayton θ per pair via excess co-occurrence
        // Clayton copula: C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
        // Kendall's tau = θ/(θ+2) → θ = 2τ/(1-τ)
        // We estimate τ from concordance/discordance of co-occurrence vs expected

        // Global Kendall tau estimation from co-occurrence excess
        let mut concordant = 0.0_f64;
        let mut discordant = 0.0_f64;

        for i in 0..size {
            for j in (i + 1)..size {
                let excess = cooccurrence[i][j] - expected_cooc;
                if excess > 0.0 {
                    concordant += excess;
                } else {
                    discordant += -excess;
                }
            }
        }

        let tau = if (concordant + discordant) > 0.0 {
            ((concordant - discordant) / (concordant + discordant)).clamp(-0.5, 0.9)
        } else {
            0.0
        };

        // Clayton θ from tau (θ must be > 0 for lower tail dependence)
        let theta = if tau > 0.01 {
            (2.0 * tau / (1.0 - tau)).clamp(0.01, 10.0)
        } else {
            0.01 // near-independence
        };

        // 3. Score each number using Clayton survival copula
        // For each number i, compute score based on its co-occurrence with all others
        // weighted by Clayton tail dependence
        let mut scores = vec![0.0_f64; size];

        for i in 0..size {
            let u_i = freq[i].max(1e-10);
            let mut copula_sum = 0.0_f64;

            for j in 0..size {
                if i == j { continue; }
                let u_j = freq[j].max(1e-10);

                // Clayton copula value
                let u_i_neg_theta = u_i.powf(-theta);
                let u_j_neg_theta = u_j.powf(-theta);
                let copula_val = (u_i_neg_theta + u_j_neg_theta - 1.0).max(1e-15).powf(-1.0 / theta);

                // Excess from observed co-occurrence
                let observed = cooccurrence[i][j];
                let independent = freq[i] * freq[j] * (pick as f64 * (pick as f64 - 1.0));
                let excess_ratio = if independent > 1e-15 {
                    (observed / independent).max(0.1).min(5.0)
                } else {
                    1.0
                };

                // Clayton-weighted score: copula amplifies tail co-occurrence
                copula_sum += copula_val * excess_ratio;
            }

            scores[i] = u_i * (1.0 + copula_sum / (size as f64 - 1.0));
        }

        // 4. Normalize and apply smoothing
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in scores.iter_mut() { *s /= total; }
        } else {
            return vec![uniform; size];
        }

        for p in scores.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        floor_only(&mut scores, PROB_FLOOR_BALLS);
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_tail_copula_balls_sums_to_one() {
        let model = TailCopulaModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum = {}", sum);
    }

    #[test]
    fn test_tail_copula_stars_uniform() {
        let model = TailCopulaModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let uniform = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Stars should be uniform");
        }
    }

    #[test]
    fn test_tail_copula_short_history() {
        let model = TailCopulaModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Short history should be uniform");
        }
    }

    #[test]
    fn test_tail_copula_name() {
        let model = TailCopulaModel::default();
        assert_eq!(model.name(), "TailCopula");
    }

    #[test]
    fn test_tail_copula_no_negative_probs() {
        let model = TailCopulaModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }
}
