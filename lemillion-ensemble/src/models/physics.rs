use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// Physics — simulation de biais mécaniques hypothétiques.
///
/// Modélise les biais potentiels de la machine de tirage via :
/// 1. Fréquences pondérées exponentiellement (EWMA lent, α=0.05)
/// 2. Biais log-ratio avec shrinkage bayésien
/// 3. Drift temporel (comparaison récent vs ancien)
/// 4. Lissage spatial gaussien (hypothèse : boules voisines ~ propriétés similaires)
/// 5. Détection de changement de régime (CUSUM via chi²)
pub struct PhysicsModel {
    alpha: f64,
    prior_strength: f64,
    drift_scale: f64,
    drift_window: usize,
    spatial_sigma: f64,
    changepoint_threshold: f64,
    smoothing: f64,
}

impl PhysicsModel {
    pub fn new(
        alpha: f64,
        prior_strength: f64,
        drift_scale: f64,
        drift_window: usize,
        spatial_sigma: f64,
        changepoint_threshold: f64,
        smoothing: f64,
    ) -> Self {
        Self {
            alpha,
            prior_strength,
            drift_scale,
            drift_window,
            spatial_sigma,
            changepoint_threshold,
            smoothing,
        }
    }
}

impl Default for PhysicsModel {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            prior_strength: 50.0,
            drift_scale: 0.3,
            drift_window: 20,
            spatial_sigma: 3.0,
            changepoint_threshold: 5.0,
            smoothing: 0.4,
        }
    }
}

/// Calcule les fréquences EWMA (exponentially weighted moving average).
/// draws[0] = plus récent, on parcourt du plus ancien au plus récent.
fn ewma_frequencies(draws: &[Draw], pool: Pool, alpha: f64) -> Vec<f64> {
    let size = pool.size();
    let mut freq = vec![1.0 / size as f64; size];

    // Parcourir du plus ancien au plus récent
    for draw in draws.iter().rev() {
        let numbers = pool.numbers_from(draw);
        let mut current = vec![0.0; size];
        for &n in numbers {
            let idx = (n - 1) as usize;
            if idx < size {
                current[idx] = 1.0;
            }
        }
        for i in 0..size {
            freq[i] = (1.0 - alpha) * freq[i] + alpha * current[i];
        }
    }

    // Normaliser
    let sum: f64 = freq.iter().sum();
    if sum > 0.0 {
        for f in &mut freq {
            *f /= sum;
        }
    }
    freq
}

/// Calcule le biais log-ratio avec shrinkage bayésien.
/// bias_k = log(freq_k / uniform) * n_eff / (n_eff + prior)
fn log_ratio_bias(freq: &[f64], prior_strength: f64, n_draws: usize) -> Vec<f64> {
    let size = freq.len();
    let uniform = 1.0 / size as f64;
    let n_eff = n_draws as f64;

    freq.iter()
        .map(|&f| {
            let ratio = (f.max(1e-15) / uniform).ln();
            ratio * n_eff / (n_eff + prior_strength)
        })
        .collect()
}

/// Drift temporel : différence entre fréquences récentes et anciennes.
fn temporal_drift(draws: &[Draw], pool: Pool, drift_window: usize) -> Vec<f64> {
    let size = pool.size();
    let window = drift_window.min(draws.len() / 2).max(1);

    let recent = &draws[..window];
    let old_start = draws.len().saturating_sub(window);
    let old = &draws[old_start..];

    let count_freq = |subset: &[Draw]| -> Vec<f64> {
        let mut counts = vec![0.0f64; size];
        for draw in subset {
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size {
                    counts[idx] += 1.0;
                }
            }
        }
        let total: f64 = counts.iter().sum();
        if total > 0.0 {
            for c in &mut counts {
                *c /= total;
            }
        }
        counts
    };

    let freq_recent = count_freq(recent);
    let freq_old = count_freq(old);

    freq_recent
        .iter()
        .zip(freq_old.iter())
        .map(|(&r, &o)| r - o)
        .collect()
}

/// Lissage spatial gaussien des biais.
/// Hypothèse : les boules voisines (numéros proches) ont des propriétés physiques similaires.
fn spatial_smooth(bias: &[f64], sigma: f64) -> Vec<f64> {
    let size = bias.len();
    let mut smoothed = vec![0.0; size];

    for i in 0..size {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for j in 0..size {
            let dist = (i as f64 - j as f64).abs();
            let w = (-dist * dist / (2.0 * sigma * sigma)).exp();
            weighted_sum += w * bias[j];
            weight_sum += w;
        }
        if weight_sum > 0.0 {
            smoothed[i] = weighted_sum / weight_sum;
        }
    }
    smoothed
}

/// Détection de changement de régime via chi² cumulatif (CUSUM simplifié).
/// Retourne le poids à donner aux données récentes vs historiques (0.5 à 1.0).
fn changepoint_weight(draws: &[Draw], pool: Pool, threshold: f64) -> f64 {
    let size = pool.size();
    if draws.len() < 20 {
        return 0.5;
    }

    let half = draws.len() / 2;
    let recent = &draws[..half];
    let old = &draws[half..];

    // Comptages récents
    let mut counts_recent = vec![0.0f64; size];
    for draw in recent {
        for &n in pool.numbers_from(draw) {
            let idx = (n - 1) as usize;
            if idx < size {
                counts_recent[idx] += 1.0;
            }
        }
    }

    // Comptages anciens
    let mut counts_old = vec![0.0f64; size];
    for draw in old {
        for &n in pool.numbers_from(draw) {
            let idx = (n - 1) as usize;
            if idx < size {
                counts_old[idx] += 1.0;
            }
        }
    }

    // Chi² entre récent et ancien (normalisé par le nombre de tirages)
    let n_recent = recent.len() as f64;
    let n_old = old.len() as f64;
    let mut chi2 = 0.0;
    for i in 0..size {
        let freq_r = counts_recent[i] / n_recent;
        let freq_o = counts_old[i] / n_old;
        let exp = (freq_r + freq_o) / 2.0;
        if exp > 0.0 {
            chi2 += (freq_r - freq_o).powi(2) / exp;
        }
    }
    // Normaliser par le nombre de degrés de liberté
    chi2 /= (size - 1) as f64;

    // Si chi² dépasse le seuil, pondérer davantage les données récentes
    if chi2 > threshold {
        0.8 // Régime changeant : favoriser les données récentes
    } else {
        0.5 // Pas de changement détecté : équilibrer
    }
}

impl ForecastModel for PhysicsModel {
    fn name(&self) -> &str {
        "Physics"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 10 {
            return uniform;
        }

        // 1. Fréquences EWMA
        let freq = ewma_frequencies(draws, pool, self.alpha);

        // 2. Biais log-ratio avec shrinkage bayésien
        let bias = log_ratio_bias(&freq, self.prior_strength, draws.len());

        // 3. Drift temporel
        let drift = temporal_drift(draws, pool, self.drift_window);

        // 4. Combiner biais + drift
        let combined: Vec<f64> = bias
            .iter()
            .zip(drift.iter())
            .map(|(&b, &d)| b + self.drift_scale * d)
            .collect();

        // 5. Lissage spatial gaussien
        let smoothed = spatial_smooth(&combined, self.spatial_sigma);

        // 6. Détection de changement de régime
        let cp_weight = changepoint_weight(draws, pool, self.changepoint_threshold);

        // Fréquences récentes (fenêtre drift_window) pour le mélange post-changepoint
        let recent_window = self.drift_window.min(draws.len());
        let mut freq_recent = vec![0.0f64; size];
        for draw in &draws[..recent_window] {
            for &n in pool.numbers_from(draw) {
                let idx = (n - 1) as usize;
                if idx < size {
                    freq_recent[idx] += 1.0;
                }
            }
        }
        let total_recent: f64 = freq_recent.iter().sum();
        if total_recent > 0.0 {
            for f in &mut freq_recent {
                *f /= total_recent;
            }
        }

        // 7. Softmax des biais lissés
        let max_b = smoothed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_bias: Vec<f64> = smoothed.iter().map(|&b| (b - max_b).exp()).collect();
        let sum_exp: f64 = exp_bias.iter().sum();
        if sum_exp > 0.0 {
            for e in &mut exp_bias {
                *e /= sum_exp;
            }
        }

        // 8. Mélange pondéré par changepoint : si régime changeant, plus de poids aux récents
        let mut dist: Vec<f64> = exp_bias
            .iter()
            .zip(freq_recent.iter())
            .map(|(&bias_p, &recent_p)| {
                (1.0 - cp_weight) * bias_p + cp_weight * recent_p
            })
            .collect();

        // 9. Smooth + normalize
        let uniform_val = 1.0 / size as f64;
        for p in &mut dist {
            *p = (1.0 - self.smoothing) * (*p).max(1e-10) + self.smoothing * uniform_val;
        }
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for p in &mut dist {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("alpha".into(), self.alpha),
            ("prior_strength".into(), self.prior_strength),
            ("drift_scale".into(), self.drift_scale),
            ("drift_window".into(), self.drift_window as f64),
            ("spatial_sigma".into(), self.spatial_sigma),
            ("changepoint_threshold".into(), self.changepoint_threshold),
            ("smoothing".into(), self.smoothing),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_physics_balls_sums_to_one() {
        let model = PhysicsModel::default();
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
    fn test_physics_stars_sums_to_one() {
        let model = PhysicsModel::default();
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
    fn test_physics_no_negative() {
        let model = PhysicsModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_physics_empty_draws() {
        let model = PhysicsModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_physics_few_draws() {
        let model = PhysicsModel::default();
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_physics_deterministic() {
        let model = PhysicsModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Physics should be deterministic");
        }
    }

    #[test]
    fn test_ewma_frequencies_normalized() {
        let draws = make_test_draws(30);
        let freq = ewma_frequencies(&draws, Pool::Balls, 0.05);
        let sum: f64 = freq.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum = {sum}");
    }

    #[test]
    fn test_spatial_smooth_preserves_shape() {
        let bias = vec![0.0, 0.1, 0.5, 0.1, 0.0];
        let smoothed = spatial_smooth(&bias, 1.0);
        assert_eq!(smoothed.len(), bias.len());
        // Peak should still be near center
        let max_idx = smoothed
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_changepoint_weight_range() {
        let draws = make_test_draws(50);
        let w = changepoint_weight(&draws, Pool::Balls, 5.0);
        assert!(w >= 0.0 && w <= 1.0, "Weight = {w}");
    }

    #[test]
    fn test_physics_large_draws() {
        let model = PhysicsModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
