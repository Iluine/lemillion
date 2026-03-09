use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// CrossTE — Cross-pool Transfer Entropy (star→ball et star→star).
///
/// Le TE actuel (TransferEntropy) fait ball→ball et ball→star.
/// CrossTE exploite le signal inverse : les étoiles influencent-elles
/// les boules du tirage suivant (même machine, même processus physique) ?
///
/// Pour Pool::Balls : TE(star_i → ball_j) pour les 12 étoiles × 50 boules = 600 paires
/// Pour Pool::Stars : TE(star_i → star_j) pour les 12 × 12 = 144 paires
///
/// Mêmes hyperparamètres que TransferEntropy.
pub struct CrossTEModel {
    alpha: f64,
    te_threshold_factor: f64,
    smoothing: f64,
    min_draws: usize,
}

impl Default for CrossTEModel {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            te_threshold_factor: 3.0,
            smoothing: 0.30,
            min_draws: 50,
        }
    }
}

/// Binary presence series (chronological: index 0 = oldest).
fn presence_series(draws: &[Draw], pool: Pool, num: u8) -> Vec<bool> {
    draws.iter().rev()
        .map(|d| pool.numbers_from(d).contains(&num))
        .collect()
}

/// Shannon TE(source→target).
fn transfer_entropy(source: &[bool], target: &[bool]) -> f64 {
    let n = source.len().min(target.len());
    if n < 3 { return 0.0; }

    let mut counts = [0.0f64; 8];
    let total = (n - 1) as f64;

    for t in 0..n - 1 {
        let yt = target[t] as usize;
        let xt = source[t] as usize;
        let yt1 = target[t + 1] as usize;
        counts[yt * 4 + xt * 2 + yt1] += 1.0;
    }

    let mut te = 0.0f64;
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
                    te += p_joint * (p_cond_joint / p_cond_marg).ln();
                }
            }
        }
    }
    te.max(0.0)
}

/// Baseline TE via permutation (5 shuffles).
fn baseline_te(source: &[bool], target: &[bool], seed: u64) -> f64 {
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
        total += transfer_entropy(&shuffled, target);
    }
    total / n_shuffles as f64
}

struct CausalPair {
    source: u8,
    target: u8,
    te_value: f64,
}

impl ForecastModel for CrossTEModel {
    fn name(&self) -> &str {
        "CrossTE"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Star source series (all 12 stars)
        let star_series: Vec<(u8, Vec<bool>)> = (1..=12u8)
            .map(|s| (s, presence_series(draws, Pool::Stars, s)))
            .collect();

        // Target series
        let target_size = size;
        let mut all_target_series: Vec<Vec<bool>> = vec![Vec::with_capacity(draws.len()); target_size];
        for d in draws.iter().rev() {
            let present = pool.numbers_from(d);
            for num in 0..target_size {
                all_target_series[num].push(present.contains(&((num + 1) as u8)));
            }
        }

        // Compute TE for each star→target pair
        let mut significant_pairs: Vec<CausalPair> = Vec::new();

        match pool {
            Pool::Balls => {
                // Star → Ball (600 pairs)
                for target_num in 1..=50u8 {
                    let target_series = &all_target_series[(target_num - 1) as usize];
                    for &(source_star, ref src_series) in &star_series {
                        let te = transfer_entropy(src_series, target_series);
                        let baseline = baseline_te(src_series, target_series,
                            source_star as u64 * 1000 + target_num as u64);
                        if te > self.te_threshold_factor * baseline.max(1e-6) {
                            significant_pairs.push(CausalPair {
                                source: source_star,
                                target: target_num,
                                te_value: te,
                            });
                        }
                    }
                }
            }
            Pool::Stars => {
                // Star → Star (144 pairs)
                for target_num in 1..=12u8 {
                    let target_series = &all_target_series[(target_num - 1) as usize];
                    for &(source_star, ref src_series) in &star_series {
                        if source_star == target_num { continue; }
                        let te = transfer_entropy(src_series, target_series);
                        let baseline = baseline_te(src_series, target_series,
                            source_star as u64 * 1000 + target_num as u64);
                        if te > self.te_threshold_factor * baseline.max(1e-6) {
                            significant_pairs.push(CausalPair {
                                source: source_star,
                                target: target_num,
                                te_value: te,
                            });
                        }
                    }
                }
            }
        }

        // Multi-lag scoring (lags 1-3)
        let mut scores = vec![1.0f64; size];
        let decay_weights = [1.0, 0.5, 0.25];

        for pair in &significant_pairs {
            for (lag_idx, &weight) in decay_weights.iter().enumerate() {
                if lag_idx >= draws.len() { break; }
                let in_draw = draws[lag_idx].stars.contains(&pair.source);
                if in_draw {
                    let target_idx = (pair.target - 1) as usize;
                    if target_idx < size {
                        scores[target_idx] *= 1.0 + self.alpha * pair.te_value * weight;
                    }
                }
            }
        }

        // Normalize
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores { *s /= sum; }
        }

        // Smooth with uniform
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
            ("alpha".into(), self.alpha),
            ("te_threshold_factor".into(), self.te_threshold_factor),
            ("smoothing".into(), self.smoothing),
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
    fn test_cross_te_balls_sums_to_one() {
        let model = CrossTEModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_cross_te_stars_sums_to_one() {
        let model = CrossTEModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_cross_te_few_draws_uniform() {
        let model = CrossTEModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cross_te_deterministic() {
        let model = CrossTEModel::default();
        let draws = make_test_draws(80);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
