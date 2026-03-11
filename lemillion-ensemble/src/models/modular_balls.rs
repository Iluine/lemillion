use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// ModularBalls — teste les 3 symétries (mod-3, mod-8, mod-24) pour les boules.
///
/// La machine Stresa a 3 barres centrales et 8 barres extérieures,
/// ce qui crée potentiellement des symétries mod-3, mod-8, ou mod-24 (LCM).
///
/// Ce modèle construit des matrices de transition pour chaque modulus et
/// sélectionne celle qui donne la meilleure prédiction (plus éloignée de l'uniforme).
/// Pour mod-24 (seulement ~2 numéros/classe), shrinkage vers mod-8.
///
/// Retourne uniforme pour les étoiles (utiliser ModTrans mod-4 à la place).
pub struct ModularBallsModel {
    smoothing: f64,
    min_draws: usize,
}

impl Default for ModularBallsModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            min_draws: 20,
        }
    }
}

fn mod_counts(numbers: &[u8], m: usize) -> Vec<f64> {
    let mut counts = vec![0.0f64; m];
    for &n in numbers {
        let r = ((n - 1) as usize) % m;
        counts[r] += 1.0;
    }
    counts
}

/// Build transition matrix and predict for a given modulus.
fn predict_with_modulus(draws: &[Draw], m: usize, smoothing: f64) -> Vec<f64> {
    let size = 50;
    let uniform_val = 1.0 / size as f64;

    let numbers_list: Vec<Vec<u8>> = draws.iter()
        .map(|d| d.balls.to_vec())
        .collect();

    // Transition matrix T[i][j] with Laplace
    let mut transition = vec![vec![1.0f64; m]; m];

    for t in 0..draws.len() - 1 {
        let current = mod_counts(&numbers_list[t], m);
        let next = mod_counts(&numbers_list[t + 1], m);

        for (i, &c_i) in current.iter().enumerate() {
            if c_i > 0.0 {
                for (j, &n_j) in next.iter().enumerate() {
                    transition[i][j] += c_i * n_j;
                }
            }
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

    // Predict from last draw
    let current = mod_counts(&numbers_list[0], m);
    let total_current: f64 = current.iter().sum();

    let mut p_next = vec![0.0f64; m];
    if total_current > 0.0 {
        for (i, &c) in current.iter().enumerate() {
            let w = c / total_current;
            for (j, p) in p_next.iter_mut().enumerate() {
                *p += w * transition[i][j];
            }
        }
    } else {
        for p in &mut p_next {
            *p = 1.0 / m as f64;
        }
    }

    // Historical frequency per ball
    let mut freq = vec![1.0f64; size];
    for nums in &numbers_list {
        for &n in nums {
            let idx = (n - 1) as usize;
            if idx < size {
                freq[idx] += 1.0;
            }
        }
    }

    // Frequency per class
    let mut class_freq_sum = vec![0.0f64; m];
    for (k, &f) in freq.iter().enumerate() {
        class_freq_sum[k % m] += f;
    }

    // Redistribute
    let mut prob = vec![0.0f64; size];
    for (k, p) in prob.iter_mut().enumerate() {
        let r = k % m;
        if class_freq_sum[r] > 0.0 {
            *p = p_next[r] * freq[k] / class_freq_sum[r];
        }
    }

    // Normalize
    let total: f64 = prob.iter().sum();
    if total > 0.0 {
        for p in &mut prob {
            *p /= total;
        }
    }

    // For mod-24: shrinkage towards mod-8 prediction (only ~2 nums per class)
    if m == 24 {
        let mod8_pred = predict_with_modulus_raw(draws, 8);
        let shrink = 0.5; // 50% shrinkage towards mod-8
        for (k, p) in prob.iter_mut().enumerate() {
            *p = (1.0 - shrink) * *p + shrink * mod8_pred[k];
        }
    }

    // Smooth
    for p in &mut prob {
        *p = (1.0 - smoothing) * *p + smoothing * uniform_val;
    }

    // Renormalize
    let total: f64 = prob.iter().sum();
    if total > 0.0 {
        for p in &mut prob {
            *p /= total;
        }
    }

    prob
}

/// Raw prediction without smoothing (for shrinkage targets).
fn predict_with_modulus_raw(draws: &[Draw], m: usize) -> Vec<f64> {
    let size = 50;
    let numbers_list: Vec<Vec<u8>> = draws.iter()
        .map(|d| d.balls.to_vec())
        .collect();

    let mut transition = vec![vec![1.0f64; m]; m];
    for t in 0..draws.len() - 1 {
        let current = mod_counts(&numbers_list[t], m);
        let next = mod_counts(&numbers_list[t + 1], m);
        for (i, &c_i) in current.iter().enumerate() {
            if c_i > 0.0 {
                for (j, &n_j) in next.iter().enumerate() {
                    transition[i][j] += c_i * n_j;
                }
            }
        }
    }
    for row in &mut transition {
        let total: f64 = row.iter().sum();
        if total > 0.0 { for v in row.iter_mut() { *v /= total; } }
    }

    let current = mod_counts(&numbers_list[0], m);
    let total_current: f64 = current.iter().sum();
    let mut p_next = vec![0.0f64; m];
    if total_current > 0.0 {
        for (i, &c) in current.iter().enumerate() {
            let w = c / total_current;
            for (j, p) in p_next.iter_mut().enumerate() {
                *p += w * transition[i][j];
            }
        }
    } else {
        for p in &mut p_next { *p = 1.0 / m as f64; }
    }

    let mut freq = vec![1.0f64; size];
    for nums in &numbers_list {
        for &n in nums { let idx = (n - 1) as usize; if idx < size { freq[idx] += 1.0; } }
    }
    let mut class_freq_sum = vec![0.0f64; m];
    for (k, &f) in freq.iter().enumerate() { class_freq_sum[k % m] += f; }

    let mut prob = vec![0.0f64; size];
    for (k, p) in prob.iter_mut().enumerate() {
        let r = k % m;
        if class_freq_sum[r] > 0.0 { *p = p_next[r] * freq[k] / class_freq_sum[r]; }
    }
    let total: f64 = prob.iter().sum();
    if total > 0.0 { for p in &mut prob { *p /= total; } }
    prob
}

/// Measure how much a distribution deviates from uniform (KL divergence).
fn kl_from_uniform(probs: &[f64]) -> f64 {
    let n = probs.len() as f64;
    let uniform = 1.0 / n;
    probs.iter()
        .map(|&p| {
            let p = p.max(1e-15);
            p * (p / uniform).ln()
        })
        .sum()
}

impl ForecastModel for ModularBallsModel {
    fn name(&self) -> &str {
        "ModularBalls"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Only for balls
        if pool == Pool::Stars || draws.len() < self.min_draws {
            return uniform;
        }

        // v18: Test mod-3, mod-8, mod-24 and combine via KL-weighted blend
        // instead of selecting the single best modulus.
        // The Stresa physics is an INTERACTION between 3 central bars AND 8 annular bars.
        let moduli = [3, 8, 24];
        let predictions: Vec<(f64, Vec<f64>)> = moduli.iter()
            .map(|&m| {
                let pred = predict_with_modulus(draws, m, self.smoothing);
                let kl = kl_from_uniform(&pred);
                (kl, pred)
            })
            .collect();

        let eps = 1e-15;
        let kl_total: f64 = predictions.iter().map(|(kl, _)| kl).sum::<f64>() + eps;

        let mut blended = vec![0.0f64; size];
        for (kl, pred) in &predictions {
            let w = kl / kl_total;
            for j in 0..size {
                blended[j] += w * pred[j];
            }
        }

        // Normalize
        let sum: f64 = blended.iter().sum();
        if sum > 0.0 {
            for p in &mut blended {
                *p /= sum;
            }
            blended
        } else {
            uniform
        }
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
    fn test_modular_balls_valid_distribution() {
        let model = ModularBallsModel::default();
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
    fn test_modular_balls_stars_returns_uniform() {
        let model = ModularBallsModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_modular_balls_few_draws() {
        let model = ModularBallsModel::default();
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_modular_balls_deterministic() {
        let model = ModularBallsModel::default();
        let draws = make_test_draws(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_kl_divergence_uniform_is_zero() {
        let uniform = vec![0.02; 50]; // 1/50 = 0.02
        let kl = kl_from_uniform(&uniform);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_concentrated_is_positive() {
        let mut probs = vec![0.01; 50];
        probs[0] = 0.5;
        let total: f64 = probs.iter().sum();
        let probs: Vec<f64> = probs.iter().map(|&p| p / total).collect();
        let kl = kl_from_uniform(&probs);
        assert!(kl > 0.0);
    }
}
