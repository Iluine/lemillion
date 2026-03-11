use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// StarPairModel — prédit les paires d'étoiles (66 combinaisons) et marginalise.
///
/// Exploite le signal de dépendance boules/étoiles (p=0.013) détecté par la recherche,
/// plus les transitions de paires et les fréquences par paire.
///
/// 3 signaux fusionnés par Hedge :
/// 1. Fréquence par paire (certaines paires sur/sous-représentées)
/// 2. Transition paire→paire (groupée par catégorie de somme)
/// 3. Ball-conditionné : P(paire | features boules du dernier tirage)
///
/// Le résultat est marginalisé : P(star=k) = Σ P(paire contenant k)
pub struct StarPairModel {
    smoothing: f64,
    learning_rate: f64,
    min_draws: usize,
}

impl Default for StarPairModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            learning_rate: 0.15,
            min_draws: 50,
        }
    }
}

/// Encode a star pair (s1 < s2) into an index in [0, 66)
pub fn pair_index(s1: u8, s2: u8) -> usize {
    let (a, b) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
    // Index = sum of (12-1) + (12-2) + ... for rows before a, plus offset
    let a0 = (a - 1) as usize;
    let b0 = (b - 1) as usize;
    // Triangular index: sum_{i=0}^{a0-1} (12-1-i) + (b0 - a0 - 1)
    let mut idx = 0;
    for i in 0..a0 {
        idx += 11 - i;
    }
    idx + (b0 - a0 - 1)
}

pub const N_PAIRS: usize = 66;

fn pair_from_draw(draw: &Draw) -> (u8, u8) {
    let (s1, s2) = (draw.stars[0], draw.stars[1]);
    if s1 < s2 { (s1, s2) } else { (s2, s1) }
}

/// Marginalize pair probabilities to individual star probabilities.
fn marginalize(pair_probs: &[f64; N_PAIRS]) -> Vec<f64> {
    let mut star_probs = vec![0.0f64; 12];
    for s1 in 1..=12u8 {
        for s2 in (s1 + 1)..=12u8 {
            let idx = pair_index(s1, s2);
            let p = pair_probs[idx];
            star_probs[(s1 - 1) as usize] += p;
            star_probs[(s2 - 1) as usize] += p;
        }
    }
    let sum: f64 = star_probs.iter().sum();
    if sum > 0.0 {
        for p in &mut star_probs {
            *p /= sum;
        }
    }
    star_probs
}

/// Expert 1: Pair frequency with Laplace smoothing
fn pair_frequency(draws: &[Draw]) -> [f64; N_PAIRS] {
    let laplace = 0.5;
    let mut counts = [laplace; N_PAIRS];
    let total_base = laplace * N_PAIRS as f64;

    for draw in draws {
        let (s1, s2) = pair_from_draw(draw);
        counts[pair_index(s1, s2)] += 1.0;
    }

    let total = total_base + draws.len() as f64;
    let mut probs = [0.0; N_PAIRS];
    for i in 0..N_PAIRS {
        probs[i] = counts[i] / total;
    }
    probs
}

/// Expert 2: Pair transition (grouped by sum category for sparsity reduction)
/// Sum categories: low (3-8), mid-low (9-12), mid (13-16), mid-high (17-20), high (21-23)
fn pair_transition(draws: &[Draw]) -> [f64; N_PAIRS] {
    let n_cats = 5;
    let laplace = 0.5;
    // transition[from_cat][to_pair]
    let mut counts = vec![[laplace; N_PAIRS]; n_cats];
    let mut totals = vec![laplace * N_PAIRS as f64; n_cats];

    let sum_cat = |s1: u8, s2: u8| -> usize {
        let sum = s1 as usize + s2 as usize;
        match sum {
            3..=8 => 0,
            9..=12 => 1,
            13..=16 => 2,
            17..=20 => 3,
            _ => 4,
        }
    };

    for i in 0..draws.len().saturating_sub(1) {
        let (fs1, fs2) = pair_from_draw(&draws[i + 1]);
        let cat = sum_cat(fs1, fs2);
        let (ts1, ts2) = pair_from_draw(&draws[i]);
        let tidx = pair_index(ts1, ts2);
        counts[cat][tidx] += 1.0;
        totals[cat] += 1.0;
    }

    // Predict based on last draw's pair category
    if draws.is_empty() {
        return [1.0 / N_PAIRS as f64; N_PAIRS];
    }
    let (ls1, ls2) = pair_from_draw(&draws[0]);
    let last_cat = sum_cat(ls1, ls2);

    let mut probs = [0.0; N_PAIRS];
    let total = totals[last_cat];
    if total > 0.0 {
        for i in 0..N_PAIRS {
            probs[i] = counts[last_cat][i] / total;
        }
    }
    probs
}

/// Expert 4: Extraction-order conditioned pair probability (v18)
/// Uses star_order from previous draw: P(pair | mod-4 class of 1st extracted star)
fn extraction_order_pair(draws: &[Draw]) -> [f64; N_PAIRS] {
    let n_mod4 = 4;
    let laplace = 0.3;

    // Condition on mod-4 class of 1st extracted star from previous draw
    let mut counts = vec![[laplace; N_PAIRS]; n_mod4];
    let mut totals = vec![laplace * N_PAIRS as f64; n_mod4];

    for i in 0..draws.len().saturating_sub(1) {
        let prev = &draws[i + 1]; // previous draw (older)
        let target = &draws[i];   // target draw (newer)

        if let Some(ref order) = prev.star_order {
            if !order.is_empty() && order[0] >= 1 {
                let cls = ((order[0] - 1) % 4) as usize;
                let (s1, s2) = pair_from_draw(target);
                let pidx = pair_index(s1, s2);
                counts[cls][pidx] += 1.0;
                totals[cls] += 1.0;
            }
        }
    }

    // Use last draw's extraction order to condition
    if draws.is_empty() {
        return [1.0 / N_PAIRS as f64; N_PAIRS];
    }

    let last = &draws[0];
    let cls = if let Some(ref order) = last.star_order {
        if !order.is_empty() && order[0] >= 1 {
            ((order[0] - 1) % 4) as usize
        } else {
            return pair_frequency(draws); // fallback to frequency
        }
    } else {
        return pair_frequency(draws); // fallback if no extraction order
    };

    let mut probs = [0.0; N_PAIRS];
    let total = totals[cls];
    if total > 0.0 {
        for i in 0..N_PAIRS {
            probs[i] = counts[cls][i] / total;
        }
    }
    probs
}

/// Expert 3: Ball-conditioned pair probability
/// Condition on (ball_sum_bin, ball_spread_bin) — 5×3 = 15 contexts
fn ball_conditioned_pair(draws: &[Draw]) -> [f64; N_PAIRS] {
    let n_sum_bins = 5;
    let n_spread_bins = 3;
    let n_ctx = n_sum_bins * n_spread_bins;
    let laplace = 0.3;

    let mut counts = vec![[laplace; N_PAIRS]; n_ctx];
    let mut totals = vec![laplace * N_PAIRS as f64; n_ctx];

    let context = |draw: &Draw| -> usize {
        let ball_sum: u32 = draw.balls.iter().map(|&b| b as u32).sum();
        let spread = draw.balls.iter().max().unwrap() - draw.balls.iter().min().unwrap();
        // sum bins: [15,90), [90,115), [115,140), [140,165), [165,250]
        let sum_bin = match ball_sum {
            0..=89 => 0,
            90..=114 => 1,
            115..=139 => 2,
            140..=164 => 3,
            _ => 4,
        };
        // spread bins: [0,20), [20,35), [35,49]
        let spread_bin = match spread {
            0..=19 => 0,
            20..=34 => 1,
            _ => 2,
        };
        sum_bin * n_spread_bins + spread_bin
    };

    // draws[i+1] is the "context" draw (balls observed), draws[i] is the "target" (stars to predict)
    for i in 0..draws.len().saturating_sub(1) {
        let ctx = context(&draws[i + 1]);
        let (s1, s2) = pair_from_draw(&draws[i]);
        let pidx = pair_index(s1, s2);
        counts[ctx][pidx] += 1.0;
        totals[ctx] += 1.0;
    }

    if draws.is_empty() {
        return [1.0 / N_PAIRS as f64; N_PAIRS];
    }
    let ctx = context(&draws[0]);

    let mut probs = [0.0; N_PAIRS];
    let total = totals[ctx];
    if total > 0.0 {
        for i in 0..N_PAIRS {
            probs[i] = counts[ctx][i] / total;
        }
    }
    probs
}

/// Expert 5: Pair autocorrelation lag-1/2 via EWMA (v19 C1)
/// Tracks persistence of specific pairs at short lags.
/// With 66 pairs and only 12 numbers, pair persistence is a stronger signal
/// than individual persistence.
fn pair_autocorrelation(draws: &[Draw]) -> [f64; N_PAIRS] {
    let uniform = [1.0 / N_PAIRS as f64; N_PAIRS];

    if draws.len() < 20 {
        return uniform;
    }

    let alpha_lag1 = 0.15; // Fast EWMA for lag-1
    let alpha_lag2 = 0.08; // Slower EWMA for lag-2

    let mut ewma_lag1 = [1.0 / N_PAIRS as f64; N_PAIRS];
    let mut ewma_lag2 = [1.0 / N_PAIRS as f64; N_PAIRS];

    // Process chronologically (oldest first)
    let len = draws.len();
    for i in (0..len).rev() {
        let (s1, s2) = pair_from_draw(&draws[i]);
        let pidx = pair_index(s1, s2);

        // Lag-1: update EWMA with current draw
        for j in 0..N_PAIRS {
            let val = if j == pidx { 1.0 } else { 0.0 };
            ewma_lag1[j] = alpha_lag1 * val + (1.0 - alpha_lag1) * ewma_lag1[j];
        }

        // Lag-2: only update every other step for lag-2 signal
        if i + 2 < len {
            let (ps1, ps2) = pair_from_draw(&draws[i + 2]);
            let ppidx = pair_index(ps1, ps2);
            // If same pair appeared 2 draws ago, this is a lag-2 persistence signal
            if ppidx == pidx {
                for j in 0..N_PAIRS {
                    let val = if j == pidx { 1.0 } else { 0.0 };
                    ewma_lag2[j] = alpha_lag2 * val + (1.0 - alpha_lag2) * ewma_lag2[j];
                }
            }
        }
    }

    // Blend lag-1 (70%) and lag-2 (30%)
    let mut probs = [0.0f64; N_PAIRS];
    for j in 0..N_PAIRS {
        probs[j] = 0.7 * ewma_lag1[j] + 0.3 * ewma_lag2[j];
    }

    // Normalize
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

impl StarPairModel {
    /// Compute combined pair distribution using Hedge-weighted experts.
    /// Returns None if not enough draws.
    fn compute_pair_distribution(&self, draws: &[Draw]) -> Option<[f64; N_PAIRS]> {
        if draws.len() < self.min_draws {
            return None;
        }

        // v19: 5 experts (added pair autocorrelation C1)
        let experts: Vec<[f64; N_PAIRS]> = vec![
            pair_frequency(draws),
            pair_transition(draws),
            ball_conditioned_pair(draws),
            extraction_order_pair(draws),
            pair_autocorrelation(draws),
        ];
        let n_experts = experts.len();

        let eta = self.learning_rate;
        let mut weights = vec![1.0 / n_experts as f64; n_experts];

        let n_recent = draws.len().min(50);
        for t in (1..n_recent).rev() {
            let train_data = &draws[t..];
            if train_data.len() < 20 {
                continue;
            }

            let preds = [
                pair_frequency(train_data),
                pair_transition(train_data),
                ball_conditioned_pair(train_data),
                extraction_order_pair(train_data),
                pair_autocorrelation(train_data),
            ];

            let target = &draws[t - 1];
            let (ts1, ts2) = pair_from_draw(target);
            let target_idx = pair_index(ts1, ts2);

            for (e, pred) in preds.iter().enumerate() {
                let p = pred[target_idx].max(1e-15);
                let loss = (-p.ln()).clamp(0.0, 10.0);

                // F5: Entropy-weighted Hedge loss — experts making more concentrated
                // (lower entropy) predictions are penalized/rewarded more heavily
                let max_entropy = (N_PAIRS as f64).ln();
                let expert_entropy = -pred.iter()
                    .filter(|&&p| p > 1e-30)
                    .map(|&p| p * p.ln())
                    .sum::<f64>();
                let entropy_weight = 1.0 + (max_entropy - expert_entropy) / max_entropy;

                weights[e] *= (-eta * loss * entropy_weight).exp();
            }

            let wsum: f64 = weights.iter().sum();
            if wsum > 0.0 {
                for w in &mut weights {
                    *w /= wsum;
                }
            }
        }

        let mut combined = [0.0f64; N_PAIRS];
        for (e, expert) in experts.iter().enumerate() {
            for i in 0..N_PAIRS {
                combined[i] += weights[e] * expert[i];
            }
        }

        let pair_sum: f64 = combined.iter().sum();
        if pair_sum > 0.0 {
            for p in &mut combined {
                *p /= pair_sum;
            }
        }

        Some(combined)
    }

    /// Returns the full pair probability distribution (66 pairs), smoothed with uniform.
    /// Returns None if not enough draws.
    pub fn predict_pair_distribution(&self, draws: &[Draw]) -> Option<[f64; N_PAIRS]> {
        let mut combined = self.compute_pair_distribution(draws)?;

        let uniform_pair = 1.0 / N_PAIRS as f64;
        for p in &mut combined {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_pair;
        }

        let sum: f64 = combined.iter().sum();
        if sum > 0.0 {
            for p in &mut combined {
                *p /= sum;
            }
        }

        Some(combined)
    }
}

impl ForecastModel for StarPairModel {
    fn name(&self) -> &str {
        "StarPair"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if pool == Pool::Balls {
            return uniform;
        }

        let combined = match self.compute_pair_distribution(draws) {
            Some(c) => c,
            None => return uniform,
        };

        // Marginalize to individual star probabilities
        let mut dist = marginalize(&combined);

        // Smooth with uniform
        let uniform_val = 1.0 / size as f64;
        for d in &mut dist {
            *d = (1.0 - self.smoothing) * *d + self.smoothing * uniform_val;
        }

        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for d in &mut dist {
                *d /= sum;
            }
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("learning_rate".into(), self.learning_rate),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn is_stars_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_pair_index_range() {
        let mut indices = std::collections::HashSet::new();
        for s1 in 1..=12u8 {
            for s2 in (s1 + 1)..=12u8 {
                let idx = pair_index(s1, s2);
                assert!(idx < N_PAIRS, "pair_index({},{})={} >= {}", s1, s2, idx, N_PAIRS);
                indices.insert(idx);
            }
        }
        assert_eq!(indices.len(), N_PAIRS);
    }

    #[test]
    fn test_marginalize_uniform() {
        let probs = [1.0 / N_PAIRS as f64; N_PAIRS];
        let marginal = marginalize(&probs);
        assert_eq!(marginal.len(), 12);
        let expected = 1.0 / 12.0;
        for &p in &marginal {
            assert!((p - expected).abs() < 1e-9, "p={} expected={}", p, expected);
        }
    }

    #[test]
    fn test_star_pair_balls_uniform() {
        let model = StarPairModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_pair_stars_valid() {
        let model = StarPairModel::default();
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
    fn test_star_pair_few_draws_uniform() {
        let model = StarPairModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_pair_no_negative() {
        let model = StarPairModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Stars);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_star_pair_deterministic() {
        let model = StarPairModel::default();
        let draws = make_test_draws(80);
        let d1 = model.predict(&draws, Pool::Stars);
        let d2 = model.predict(&draws, Pool::Stars);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_pair_frequency_normalized() {
        let draws = make_test_draws(80);
        let probs = pair_frequency(&draws);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "pair_frequency sum = {}", sum);
    }

    #[test]
    fn test_pair_transition_normalized() {
        let draws = make_test_draws(80);
        let probs = pair_transition(&draws);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "pair_transition sum = {}", sum);
    }

    #[test]
    fn test_ball_conditioned_normalized() {
        let draws = make_test_draws(80);
        let probs = ball_conditioned_pair(&draws);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "ball_conditioned sum = {}", sum);
    }
}
