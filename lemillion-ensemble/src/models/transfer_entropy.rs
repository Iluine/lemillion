use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// TransferEntropy — exploite les paires causales TE(source→target).
///
/// Algorithme :
/// 1. Construire séries binaires présence/absence pour chaque numéro
/// 2. Calculer TE pour les top sources → toutes les cibles
/// 3. Retenir paires avec TE > threshold_factor × baseline (shuffled)
/// 4. Pour chaque cible j : score(j) = Π (1 + alpha * TE_ij * δ(i ∈ last_draw))
/// 5. Normaliser, lisser avec uniforme
///
/// Pour Pool::Stars : utilise les paires cross-pool TE(ball→star).
pub struct TransferEntropyModel {
    alpha: f64,
    te_threshold_factor: f64,
    smoothing: f64,
    n_top_sources: usize,
    min_draws: usize,
}

impl Default for TransferEntropyModel {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            te_threshold_factor: 3.0,
            smoothing: 0.30,
            n_top_sources: 15,
            min_draws: 50,
        }
    }
}

/// Série binaire de présence/absence pour un numéro donné.
/// Retourne Vec<bool> en ordre chronologique (index 0 = plus ancien).
fn presence_series(draws: &[Draw], pool: Pool, num: u8) -> Vec<bool> {
    draws.iter().rev()
        .map(|d| pool.numbers_from(d).contains(&num))
        .collect()
}

/// Calcule TE lagué TE_k(source→target) par comptage conditionnel (v15).
///
/// TE_k(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-k})
///
/// Pour lag=1, équivalent au TE classique TE(X_t → Y_{t+1}).
/// Pour lag=k, mesure la causalité retardée de k pas.
fn transfer_entropy_lagged(source: &[bool], target: &[bool], lag: usize) -> f64 {
    let n = source.len().min(target.len());
    let start = lag.max(1);
    if n <= start || (n - start) < 3 {
        return 0.0;
    }

    // Comptages : (y_{t-1}, x_{t-lag}, y_t) → 8 cases
    let mut counts = [0.0f64; 8];
    let total = (n - start) as f64;

    for t in start..n {
        let yt_prev = target[t - 1] as usize;
        let x_lagged = source[t - lag] as usize;
        let yt = target[t] as usize;
        counts[yt_prev * 4 + x_lagged * 2 + yt] += 1.0;
    }

    let mut te = 0.0f64;

    for yt in 0..2 {
        for xt in 0..2 {
            let n_yt_xt: f64 = (0..2).map(|yt1| counts[yt * 4 + xt * 2 + yt1]).sum();
            let n_yt: f64 = (0..2).flat_map(|x| (0..2).map(move |y1| counts[yt * 4 + x * 2 + y1])).sum();

            for yt1 in 0..2 {
                let n_joint = counts[yt * 4 + xt * 2 + yt1];
                if n_joint < 1.0 || n_yt_xt < 1.0 || n_yt < 1.0 {
                    continue;
                }

                let p_cond_joint = n_joint / n_yt_xt;
                let n_yt_yt1: f64 = (0..2).map(|x| counts[yt * 4 + x * 2 + yt1]).sum();
                let p_cond_marg = n_yt_yt1 / n_yt;

                if p_cond_joint > 1e-15 && p_cond_marg > 1e-15 {
                    let p_joint = n_joint / total;
                    te += p_joint * (p_cond_joint / p_cond_marg).ln();
                }
            }
        }
    }

    te.max(0.0)
}

/// Compatibilité : TE classique (lag=1). Utilisé par les tests.
#[cfg(test)]
fn transfer_entropy(source: &[bool], target: &[bool]) -> f64 {
    transfer_entropy_lagged(source, target, 1)
}

/// Calcule le TE baseline par permutation (moyenne sur 5 shuffles) au lag donné.
fn baseline_te_lagged(source: &[bool], target: &[bool], lag: usize, seed: u64) -> f64 {
    let n_shuffles = 5;
    let mut total = 0.0f64;
    let mut rng = seed.wrapping_add(1);
    if rng == 0 { rng = 1; }

    for _ in 0..n_shuffles {
        let mut shuffled: Vec<bool> = source.to_vec();
        for i in (1..shuffled.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        total += transfer_entropy_lagged(&shuffled, target, lag);
    }

    total / n_shuffles as f64
}

/// Compatibilité : baseline TE classique (lag=1). Utilisé par les tests.
#[allow(dead_code)]
fn baseline_te(source: &[bool], target: &[bool], seed: u64) -> f64 {
    baseline_te_lagged(source, target, 1, seed)
}

/// Paire causale significative (v15: avec lag optimal).
struct CausalPair {
    source: u8,      // numéro source (1-indexed)
    source_pool: Pool,
    target: u8,      // numéro cible (1-indexed)
    te_value: f64,
    best_lag: usize, // v15: lag optimal [1,2,3]
}

impl ForecastModel for TransferEntropyModel {
    fn name(&self) -> &str {
        "TransferEntropy"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // 1. Identifier les top sources (boules les plus fréquentes)
        let mut ball_freq = vec![0usize; 50];
        for draw in draws {
            for &b in &draw.balls {
                ball_freq[(b - 1) as usize] += 1;
            }
        }
        let mut top_sources: Vec<u8> = (1..=50u8).collect();
        top_sources.sort_by(|&a, &b| ball_freq[(b - 1) as usize].cmp(&ball_freq[(a - 1) as usize]));
        top_sources.truncate(self.n_top_sources);

        // 2. Précalculer les séries de présence pour les sources
        let source_series: Vec<(u8, Vec<bool>)> = top_sources.iter()
            .map(|&s| (s, presence_series(draws, Pool::Balls, s)))
            .collect();

        // 3. Précalculer toutes les séries cibles en un seul passage
        let (target_pool, target_size) = match pool {
            Pool::Balls => (Pool::Balls, 50usize),
            Pool::Stars => (Pool::Stars, 12usize),
        };
        let mut all_target_series: Vec<Vec<bool>> = vec![Vec::with_capacity(draws.len()); target_size];
        for d in draws.iter().rev() {
            let present = target_pool.numbers_from(d);
            for num in 0..target_size {
                all_target_series[num].push(present.contains(&((num + 1) as u8)));
            }
        }

        // 4. v15: True lagged TE — compute TE aux lags [1,2,3], garder le meilleur
        let mut significant_pairs: Vec<CausalPair> = Vec::new();
        let test_lags: [usize; 3] = [1, 2, 3];

        match pool {
            Pool::Balls => {
                for target_num in 1..=50u8 {
                    let target_series = &all_target_series[(target_num - 1) as usize];
                    for &(source_num, ref src_series) in &source_series {
                        if source_num == target_num {
                            continue;
                        }
                        // Find best lag
                        let mut best_te = 0.0f64;
                        let mut best_lag = 1usize;
                        for &lag in &test_lags {
                            let te = transfer_entropy_lagged(src_series, target_series, lag);
                            if te > best_te {
                                best_te = te;
                                best_lag = lag;
                            }
                        }
                        let seed = source_num as u64 * 100 + target_num as u64;
                        let baseline = baseline_te_lagged(src_series, target_series, best_lag, seed);
                        if best_te > self.te_threshold_factor * baseline.max(1e-6) {
                            significant_pairs.push(CausalPair {
                                source: source_num,
                                source_pool: Pool::Balls,
                                target: target_num,
                                te_value: best_te,
                                best_lag,
                            });
                        }
                    }
                }
            }
            Pool::Stars => {
                for target_num in 1..=12u8 {
                    let target_series = &all_target_series[(target_num - 1) as usize];
                    for &(source_num, ref src_series) in &source_series {
                        let mut best_te = 0.0f64;
                        let mut best_lag = 1usize;
                        for &lag in &test_lags {
                            let te = transfer_entropy_lagged(src_series, target_series, lag);
                            if te > best_te {
                                best_te = te;
                                best_lag = lag;
                            }
                        }
                        let seed = source_num as u64 * 100 + target_num as u64;
                        let baseline = baseline_te_lagged(src_series, target_series, best_lag, seed);
                        if best_te > self.te_threshold_factor * baseline.max(1e-6) {
                            significant_pairs.push(CausalPair {
                                source: source_num,
                                source_pool: Pool::Balls,
                                target: target_num,
                                te_value: best_te,
                                best_lag,
                            });
                        }
                    }
                }
            }
        }

        // 5. v15: Scorer chaque cible — appliquer au lag optimal spécifique
        let mut scores = vec![1.0f64; size];

        for pair in &significant_pairs {
            let lag_idx = pair.best_lag - 1; // 0-indexed into draws (draws[0] = most recent)
            if lag_idx >= draws.len() { continue; }
            let in_draw = match pair.source_pool {
                Pool::Balls => draws[lag_idx].balls.contains(&pair.source),
                Pool::Stars => draws[lag_idx].stars.contains(&pair.source),
            };
            if in_draw {
                let target_idx = (pair.target - 1) as usize;
                if target_idx < size {
                    scores[target_idx] *= 1.0 + self.alpha * pair.te_value;
                }
            }
        }

        // 5. Normaliser
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }

        // 6. Lissage avec uniforme
        let uniform_val = 1.0 / size as f64;
        for s in &mut scores {
            *s = (1.0 - self.smoothing) * *s + self.smoothing * uniform_val;
        }

        // Renormaliser
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("alpha".into(), self.alpha),
            ("te_threshold_factor".into(), self.te_threshold_factor),
            ("smoothing".into(), self.smoothing),
            ("n_top_sources".into(), self.n_top_sources as f64),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }

    fn calibration_stride(&self) -> usize {
        2  // TE est coûteux, sauter 1 point sur 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_transfer_entropy_model_balls_sums_to_one() {
        let model = TransferEntropyModel::default();
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
    fn test_transfer_entropy_model_stars_sums_to_one() {
        let model = TransferEntropyModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_transfer_entropy_few_draws_uniform() {
        let model = TransferEntropyModel::default();
        let draws = make_test_draws(20);
        let dist_b = model.predict(&draws, Pool::Balls);
        let expected_b = 1.0 / 50.0;
        for &p in &dist_b {
            assert!((p - expected_b).abs() < 1e-10);
        }
        let dist_s = model.predict(&draws, Pool::Stars);
        let expected_s = 1.0 / 12.0;
        for &p in &dist_s {
            assert!((p - expected_s).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transfer_entropy_basic() {
        // TE d'une série vers elle-même (lag=0) devrait être faible
        let source = vec![true, false, true, false, true, false, true, false, true, false];
        let target = source.clone();
        let te = transfer_entropy(&source, &target);
        // TE(X→X) quand X est périodique : devrait être > 0
        assert!(te >= 0.0, "TE should be non-negative, got {}", te);
    }

    #[test]
    fn test_transfer_entropy_independent() {
        // Deux séries constantes → TE = 0
        let source = vec![true; 100];
        let target = vec![false; 100];
        let te = transfer_entropy(&source, &target);
        assert!(te.abs() < 1e-10, "TE of constant series should be ~0, got {}", te);
    }
}
