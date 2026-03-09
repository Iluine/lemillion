use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// RényiTE — Transfer Entropy de Rényi (α=0.7) pour amplifier les événements rares.
///
/// Le TE de Shannon (TransferEntropy) mesure la causalité « en moyenne ».
/// Le TE de Rényi avec α < 1 donne plus de poids aux événements rares
/// (co-occurrences peu fréquentes mais fortement prédictives).
///
/// Pour le jackpot (5+2), ce sont exactement les patterns rares qui comptent.
///
/// Même architecture que TransferEntropy : top-15 sources, 5 shuffles,
/// multi-lag scoring (decay [1.0, 0.5, 0.25]).
pub struct RenyiTEModel {
    alpha_boost: f64,
    renyi_alpha: f64,
    te_threshold_factor: f64,
    smoothing: f64,
    n_top_sources: usize,
    min_draws: usize,
}

impl Default for RenyiTEModel {
    fn default() -> Self {
        Self {
            alpha_boost: 2.0,
            renyi_alpha: 0.7,
            te_threshold_factor: 2.5, // more sensitive than Shannon (3.0)
            smoothing: 0.30,
            n_top_sources: 15,
            min_draws: 50,
        }
    }
}

/// Binary presence series for a number (chronological order, index 0 = oldest).
fn presence_series(draws: &[Draw], pool: Pool, num: u8) -> Vec<bool> {
    draws.iter().rev()
        .map(|d| pool.numbers_from(d).contains(&num))
        .collect()
}

/// Rényi Transfer Entropy (α < 1).
///
/// TE_α(X→Y) = (1/(α-1)) × ln[ Σ p(y_{t+1}, y_t, x_t) × (p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t))^(α-1) ]
///
/// For α=0.7, terms where p(cond_joint) >> p(cond_marg) (rare events with strong causality) are amplified.
fn renyi_transfer_entropy(source: &[bool], target: &[bool], alpha: f64) -> f64 {
    let n = source.len().min(target.len());
    if n < 3 { return 0.0; }

    // Counts: (y_t, x_t, y_{t+1}) → 8 cells
    let mut counts = [0.0f64; 8];
    let total = (n - 1) as f64;

    for t in 0..n - 1 {
        let yt = target[t] as usize;
        let xt = source[t] as usize;
        let yt1 = target[t + 1] as usize;
        counts[yt * 4 + xt * 2 + yt1] += 1.0;
    }

    let mut sum = 0.0f64;
    for yt in 0..2 {
        for xt in 0..2 {
            let n_yt_xt: f64 = (0..2).map(|yt1| counts[yt * 4 + xt * 2 + yt1]).sum();
            let n_yt: f64 = (0..2).flat_map(|x| (0..2).map(move |y1| counts[yt * 4 + x * 2 + y1])).sum();

            for yt1 in 0..2 {
                let n_joint = counts[yt * 4 + xt * 2 + yt1];
                if n_joint < 1.0 || n_yt_xt < 1.0 || n_yt < 1.0 { continue; }

                let p_cond_joint = n_joint / n_yt_xt;
                let n_yt_yt1: f64 = (0..2).map(|x| counts[yt * 4 + x * 2 + yt1]).sum();
                let p_cond_marg = n_yt_yt1 / n_yt;

                if p_cond_joint > 1e-15 && p_cond_marg > 1e-15 {
                    let p_joint = n_joint / total;
                    let ratio = p_cond_joint / p_cond_marg;
                    sum += p_joint * ratio.powf(alpha - 1.0);
                }
            }
        }
    }

    if sum <= 0.0 { return 0.0; }
    let te = (1.0 / (alpha - 1.0)) * sum.ln();
    te.max(0.0)
}

/// Baseline Rényi TE via permutation (5 shuffles).
fn baseline_renyi_te(source: &[bool], target: &[bool], alpha: f64, seed: u64) -> f64 {
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
        total += renyi_transfer_entropy(&shuffled, target, alpha);
    }

    total / n_shuffles as f64
}

struct CausalPair {
    source: u8,
    source_pool: Pool,
    target: u8,
    te_value: f64,
}

impl ForecastModel for RenyiTEModel {
    fn name(&self) -> &str {
        "RényiTE"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // 1. Top sources (most frequent balls)
        let mut ball_freq = vec![0usize; 50];
        for draw in draws {
            for &b in &draw.balls {
                ball_freq[(b - 1) as usize] += 1;
            }
        }
        let mut top_sources: Vec<u8> = (1..=50u8).collect();
        top_sources.sort_by(|&a, &b| ball_freq[(b - 1) as usize].cmp(&ball_freq[(a - 1) as usize]));
        top_sources.truncate(self.n_top_sources);

        // 2. Source presence series
        let source_series: Vec<(u8, Vec<bool>)> = top_sources.iter()
            .map(|&s| (s, presence_series(draws, Pool::Balls, s)))
            .collect();

        // 3. Target presence series
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

        // 4. Compute Rényi TE for each source→target pair
        let mut significant_pairs: Vec<CausalPair> = Vec::new();

        match pool {
            Pool::Balls => {
                for target_num in 1..=50u8 {
                    let target_series = &all_target_series[(target_num - 1) as usize];
                    for &(source_num, ref src_series) in &source_series {
                        if source_num == target_num { continue; }
                        let te = renyi_transfer_entropy(src_series, target_series, self.renyi_alpha);
                        let baseline = baseline_renyi_te(src_series, target_series, self.renyi_alpha,
                            source_num as u64 * 100 + target_num as u64);
                        if te > self.te_threshold_factor * baseline.max(1e-6) {
                            significant_pairs.push(CausalPair {
                                source: source_num,
                                source_pool: Pool::Balls,
                                target: target_num,
                                te_value: te,
                            });
                        }
                    }
                }
            }
            Pool::Stars => {
                for target_num in 1..=12u8 {
                    let target_series = &all_target_series[(target_num - 1) as usize];
                    for &(source_num, ref src_series) in &source_series {
                        let te = renyi_transfer_entropy(src_series, target_series, self.renyi_alpha);
                        let baseline = baseline_renyi_te(src_series, target_series, self.renyi_alpha,
                            source_num as u64 * 100 + target_num as u64);
                        if te > self.te_threshold_factor * baseline.max(1e-6) {
                            significant_pairs.push(CausalPair {
                                source: source_num,
                                source_pool: Pool::Balls,
                                target: target_num,
                                te_value: te,
                            });
                        }
                    }
                }
            }
        }

        // 5. Multi-lag scoring (lags 1-3)
        let mut scores = vec![1.0f64; size];
        let decay_weights = [1.0, 0.5, 0.25];

        for pair in &significant_pairs {
            for (lag_idx, &weight) in decay_weights.iter().enumerate() {
                if lag_idx >= draws.len() { break; }
                let in_draw = match pair.source_pool {
                    Pool::Balls => draws[lag_idx].balls.contains(&pair.source),
                    Pool::Stars => draws[lag_idx].stars.contains(&pair.source),
                };
                if in_draw {
                    let target_idx = (pair.target - 1) as usize;
                    if target_idx < size {
                        scores[target_idx] *= 1.0 + self.alpha_boost * pair.te_value * weight;
                    }
                }
            }
        }

        // 6. Normalize
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores { *s /= sum; }
        }

        // 7. Smooth with uniform
        let uniform_val = 1.0 / size as f64;
        for s in &mut scores {
            *s = (1.0 - self.smoothing) * *s + self.smoothing * uniform_val;
        }

        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores { *s /= sum; }
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("alpha_boost".into(), self.alpha_boost),
            ("renyi_alpha".into(), self.renyi_alpha),
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
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_renyi_te_balls_sums_to_one() {
        let model = RenyiTEModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_renyi_te_stars_sums_to_one() {
        let model = RenyiTEModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_renyi_te_few_draws_uniform() {
        let model = RenyiTEModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_renyi_te_deterministic() {
        let model = RenyiTEModel::default();
        let draws = make_test_draws(80);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_renyi_te_basic() {
        let source = vec![true, false, true, false, true, false, true, false, true, false];
        let target = source.clone();
        let te = renyi_transfer_entropy(&source, &target, 0.7);
        assert!(te >= 0.0, "Rényi TE should be non-negative, got {}", te);
    }

    #[test]
    fn test_renyi_te_independent() {
        let source = vec![true; 100];
        let target = vec![false; 100];
        let te = renyi_transfer_entropy(&source, &target, 0.7);
        assert!(te.abs() < 1e-10, "Rényi TE of constant series should be ~0, got {}", te);
    }
}
