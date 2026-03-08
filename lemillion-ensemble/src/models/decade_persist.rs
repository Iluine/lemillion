use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// DecadePersist — matrice de transition entre profils de décades.
///
/// Les 5 décades (1-10, 11-20, ..., 41-50) correspondent aux 5 rangées du rack Stresa.
/// C'est une vraie caractéristique physique: les boules sont rangées par décade.
///
/// Modélise la persistence inter-tirages au niveau des rangées via:
/// 1. Profil de décade = (n₀, n₁, n₂, n₃, n₄) avec Σnᵢ = 5
/// 2. Matrice de transition profil→profil (avec Laplace smoothing)
/// 3. Redistribution intra-rangée par fréquence historique
///
/// Orthogonal aux modèles modulaires (mod-8/mod-4).
pub struct DecadePersistModel {
    smoothing: f64,
    laplace_alpha: f64,
    min_draws: usize,
}

impl Default for DecadePersistModel {
    fn default() -> Self {
        Self {
            smoothing: 0.30,
            laplace_alpha: 1.0,
            min_draws: 30,
        }
    }
}

/// Extrait le profil de décade d'un tirage de boules.
/// Décade 0 = 1-10, décade 1 = 11-20, ..., décade 4 = 41-50.
fn decade_profile(balls: &[u8]) -> [u8; 5] {
    let mut profile = [0u8; 5];
    for &b in balls {
        let dec = ((b - 1) / 10) as usize;
        if dec < 5 {
            profile[dec] += 1;
        }
    }
    profile
}

/// Énumère tous les profils de décade: compositions de 5 dans 5 parts = C(9,4) = 126.
fn enumerate_decade_profiles() -> Vec<[u8; 5]> {
    let mut profiles = Vec::new();
    for n0 in 0..=5u8 {
        for n1 in 0..=(5 - n0) {
            for n2 in 0..=(5 - n0 - n1) {
                for n3 in 0..=(5 - n0 - n1 - n2) {
                    let n4 = 5 - n0 - n1 - n2 - n3;
                    profiles.push([n0, n1, n2, n3, n4]);
                }
            }
        }
    }
    profiles
}

fn profile_index(profile: &[u8; 5], profiles: &[[u8; 5]]) -> Option<usize> {
    profiles.iter().position(|p| p == profile)
}

impl ForecastModel for DecadePersistModel {
    fn name(&self) -> &str {
        "DecadePersist"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Only works for balls — stars don't have decades
        if pool == Pool::Stars || draws.len() < self.min_draws {
            return uniform;
        }

        let profiles = enumerate_decade_profiles();
        let n_profiles = profiles.len();

        // Extract decade profile for each draw
        let draw_profiles: Vec<[u8; 5]> = draws
            .iter()
            .map(|d| decade_profile(&d.balls))
            .collect();

        // Build transition matrix T[from][to] + Laplace
        let mut transition = vec![vec![self.laplace_alpha; n_profiles]; n_profiles];

        for t in 0..draws.len() - 1 {
            let from = &draw_profiles[t + 1]; // older
            let to = &draw_profiles[t];       // newer
            if let (Some(from_idx), Some(to_idx)) = (
                profile_index(from, &profiles),
                profile_index(to, &profiles),
            ) {
                transition[from_idx][to_idx] += 1.0;
            }
        }

        // Normalize rows
        for row in &mut transition {
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
        }

        // Predict from current profile
        let current = &draw_profiles[0];
        let current_idx = match profile_index(current, &profiles) {
            Some(idx) => idx,
            None => return uniform,
        };
        let p_profile = &transition[current_idx];

        // Historical frequency per ball (for intra-decade redistribution)
        let mut freq = vec![1.0f64; size]; // Laplace +1
        for d in draws {
            for &b in &d.balls {
                let idx = (b - 1) as usize;
                if idx < size {
                    freq[idx] += 1.0;
                }
            }
        }

        // Frequency per decade
        let mut decade_freq_sum = [0.0f64; 5];
        for (k, &f) in freq.iter().enumerate() {
            decade_freq_sum[k / 10] += f;
        }

        // Marginalize: P(ball_k) = Σ_j P(profile_j) × n_dec(profile_j) × freq[k] / decade_sum[dec]
        let mut prob = vec![0.0f64; size];
        for (j, profile) in profiles.iter().enumerate() {
            let p_j = p_profile[j];
            if p_j < 1e-15 { continue; }
            for (k, p_k) in prob.iter_mut().enumerate() {
                let dec = k / 10;
                let n_dec = profile[dec] as f64;
                if n_dec > 0.0 && decade_freq_sum[dec] > 0.0 {
                    *p_k += p_j * n_dec * freq[k] / decade_freq_sum[dec];
                }
            }
        }

        // Normalize
        let total: f64 = prob.iter().sum();
        if total > 0.0 {
            for p in &mut prob {
                *p /= total;
            }
        } else {
            return uniform;
        }

        // Smooth with uniform
        let uniform_val = 1.0 / size as f64;
        for p in &mut prob {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Renormalize
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
            ("laplace_alpha".into(), self.laplace_alpha),
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
    fn test_decade_profile() {
        let profile = decade_profile(&[1, 12, 23, 34, 45]);
        // 1→dec0, 12→dec1, 23→dec2, 34→dec3, 45→dec4
        assert_eq!(profile, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_decade_profile_same_decade() {
        let profile = decade_profile(&[1, 2, 3, 4, 5]);
        assert_eq!(profile, [5, 0, 0, 0, 0]);
    }

    #[test]
    fn test_enumerate_decade_profiles() {
        let profiles = enumerate_decade_profiles();
        // Compositions of 5 in 5 parts = C(9,4) = 126
        assert_eq!(profiles.len(), 126);
        for p in &profiles {
            assert_eq!(p.iter().sum::<u8>(), 5);
        }
    }

    #[test]
    fn test_decade_persist_balls() {
        let model = DecadePersistModel::default();
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
    fn test_decade_persist_stars_returns_uniform() {
        let model = DecadePersistModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        let uniform = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decade_persist_deterministic() {
        let model = DecadePersistModel::default();
        let draws = make_test_draws(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_decade_persist_few_draws() {
        let model = DecadePersistModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }
}
