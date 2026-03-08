use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};
use super::mod4::modulus;

/// ModProfile — matrice de transition sur les profils modulaires.
///
/// Au lieu de transitions blade→blade (M×M), modélise les PROFILS modulaires complets.
/// Un profil = (n₀,n₁,...,n_{M-1}) avec Σnᵢ = pick_count.
/// - Boules mod-8: compositions de 5 dans 8 parts = C(12,7) = 792 profils
/// - Étoiles mod-4: compositions de 2 dans 4 parts = C(5,3) = 10 profils
///
/// Matrice de transition profil→profil avec lissage Laplace.
pub struct Mod4ProfileModel {
    smoothing: f64,
    laplace_alpha: f64,
    min_draws: usize,
}

impl Mod4ProfileModel {
    pub fn new(smoothing: f64, laplace_alpha: f64, min_draws: usize) -> Self {
        Self { smoothing, laplace_alpha, min_draws }
    }
}

impl Default for Mod4ProfileModel {
    fn default() -> Self {
        Self {
            smoothing: 0.4,
            laplace_alpha: 1.0,
            min_draws: 20,
        }
    }
}

/// Énumère tous les profils (n₀,n₁,...,n_{M-1}) avec Σnᵢ = total.
/// Ce sont les compositions faibles de `total` en `m` parts.
pub fn enumerate_profiles_m(total: u8, m: usize) -> Vec<Vec<u8>> {
    let mut profiles = Vec::new();
    let mut current = vec![0u8; m];
    enumerate_recursive(&mut profiles, &mut current, 0, total, m);
    profiles
}

fn enumerate_recursive(
    profiles: &mut Vec<Vec<u8>>,
    current: &mut Vec<u8>,
    depth: usize,
    remaining: u8,
    m: usize,
) {
    if depth == m - 1 {
        current[depth] = remaining;
        profiles.push(current.clone());
        return;
    }
    for v in 0..=remaining {
        current[depth] = v;
        enumerate_recursive(profiles, current, depth + 1, remaining - v, m);
    }
}

/// Legacy 4-part enumeration (for backward compatibility with tests).
pub fn enumerate_profiles(total: u8) -> Vec<[u8; 4]> {
    let mut profiles = Vec::new();
    for n0 in 0..=total {
        for n1 in 0..=(total - n0) {
            for n2 in 0..=(total - n0 - n1) {
                let n3 = total - n0 - n1 - n2;
                profiles.push([n0, n1, n2, n3]);
            }
        }
    }
    profiles
}

/// Extrait le profil modulaire d'un ensemble de numéros.
/// Chaque numéro k → classe (k-1) % m.
pub fn extract_profile_m(numbers: &[u8], m: usize) -> Vec<u8> {
    let mut profile = vec![0u8; m];
    for &n in numbers {
        let r = ((n - 1) as usize) % m;
        profile[r] += 1;
    }
    profile
}

/// Legacy 4-class extraction.
pub fn extract_profile(numbers: &[u8]) -> [u8; 4] {
    let mut profile = [0u8; 4];
    for &n in numbers {
        let r = ((n - 1) % 4) as usize;
        profile[r] += 1;
    }
    profile
}

/// Index d'un profil dans la liste énumérée.
fn profile_index_m(profile: &[u8], profiles: &[Vec<u8>]) -> Option<usize> {
    profiles.iter().position(|p| p.as_slice() == profile)
}

impl ForecastModel for Mod4ProfileModel {
    fn name(&self) -> &str {
        "ModProfile"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let m = modulus(pool);
        let pick_count = pool.pick_count() as u8;
        let profiles = enumerate_profiles_m(pick_count, m);
        let n_profiles = profiles.len();

        // Extraire le profil de chaque tirage
        let draw_profiles: Vec<Vec<u8>> = draws
            .iter()
            .map(|d| extract_profile_m(pool.numbers_from(d), m))
            .collect();

        // Construire matrice de transition T[from][to] + Laplace
        let mut transition = vec![vec![self.laplace_alpha; n_profiles]; n_profiles];

        // draws[0] = plus récent, draws[n-1] = plus ancien
        // Paire (t, t+1) : draws[t+1] (plus ancien) -> draws[t] (plus récent)
        for t in 0..draws.len() - 1 {
            let from_profile = &draw_profiles[t + 1];
            let to_profile = &draw_profiles[t];
            if let (Some(from_idx), Some(to_idx)) = (
                profile_index_m(from_profile, &profiles),
                profile_index_m(to_profile, &profiles),
            ) {
                transition[from_idx][to_idx] += 1.0;
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

        // Prédire : profil actuel = draws[0]
        let current_profile = &draw_profiles[0];
        let current_idx = match profile_index_m(current_profile, &profiles) {
            Some(idx) => idx,
            None => return uniform,
        };

        // P(profil_suivant) = transition[current_idx]
        let p_profile = &transition[current_idx];

        // Fréquences historiques par numéro (pour redistribution intra-classe)
        let mut freq = vec![1.0f64; size]; // Laplace +1
        for d in draws {
            for &n in pool.numbers_from(d) {
                let idx = (n - 1) as usize;
                if idx < size {
                    freq[idx] += 1.0;
                }
            }
        }

        // Fréquences par classe modulaire
        let mut class_freq_sum = vec![0.0f64; m];
        for (k, &f) in freq.iter().enumerate() {
            class_freq_sum[k % m] += f;
        }

        // Convertir en marginales :
        // P(boule_k) = Σ_j P(profil_j) × n_r(profil_j) × freq[k] / class_sum[r]
        // où r = (k-1) % m, n_r = count de la classe r dans le profil j
        let mut prob = vec![0.0f64; size];
        for (j, profile) in profiles.iter().enumerate() {
            let p_j = p_profile[j];
            if p_j < 1e-15 {
                continue;
            }
            for (k, p_k) in prob.iter_mut().enumerate() {
                let r = k % m;
                let n_r = profile[r] as f64;
                if n_r > 0.0 && class_freq_sum[r] > 0.0 {
                    *p_k += p_j * n_r * freq[k] / class_freq_sum[r];
                }
            }
        }

        // Normaliser (la somme devrait être ~pick_count, pas 1.0)
        let total: f64 = prob.iter().sum();
        if total > 0.0 {
            for p in &mut prob {
                *p /= total;
            }
        } else {
            return uniform;
        }

        // Mixer avec uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut prob {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Renormaliser
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
    fn test_enumerate_profiles_balls_mod8() {
        // Compositions of 5 in 8 parts = C(12,7) = 792
        let profiles = enumerate_profiles_m(5, 8);
        assert_eq!(profiles.len(), 792, "C(12,7) = 792 profils pour 5 boules mod-8");
        for p in &profiles {
            assert_eq!(p.iter().map(|&x| x as u16).sum::<u16>(), 5);
            assert_eq!(p.len(), 8);
        }
    }

    #[test]
    fn test_enumerate_profiles_stars_mod4() {
        let profiles = enumerate_profiles_m(2, 4);
        assert_eq!(profiles.len(), 10, "C(5,3) = 10 profils pour 2 étoiles mod-4");
        for p in &profiles {
            assert_eq!(p.iter().map(|&x| x as u16).sum::<u16>(), 2);
        }
    }

    #[test]
    fn test_enumerate_profiles_legacy() {
        let profiles = enumerate_profiles(5);
        assert_eq!(profiles.len(), 56, "C(8,3) = 56 profils legacy");
        for p in &profiles {
            assert_eq!(p.iter().sum::<u8>(), 5);
        }
    }

    #[test]
    fn test_extract_profile_mod8() {
        // 1→(1-1)%8=0, 2→1, 3→2, 4→3, 5→4
        let profile = extract_profile_m(&[1, 2, 3, 4, 5], 8);
        assert_eq!(profile, vec![1, 1, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_extract_profile_mod4() {
        // 1→0, 2→1, 3→2, 4→3, 5→0
        let profile = extract_profile_m(&[1, 2, 3, 4, 5], 4);
        assert_eq!(profile, vec![2, 1, 1, 1]);
    }

    #[test]
    fn test_extract_profile_legacy() {
        let profile = extract_profile(&[1, 2, 3, 4, 5]);
        assert_eq!(profile, [2, 1, 1, 1]);
    }

    #[test]
    fn test_profile_balls_sums_to_one() {
        let model = Mod4ProfileModel::default();
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
    fn test_profile_stars_sums_to_one() {
        let model = Mod4ProfileModel::default();
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
    fn test_profile_no_negative() {
        let model = Mod4ProfileModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_profile_empty_draws() {
        let model = Mod4ProfileModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_profile_few_draws() {
        let model = Mod4ProfileModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_profile_deterministic() {
        let model = Mod4ProfileModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
