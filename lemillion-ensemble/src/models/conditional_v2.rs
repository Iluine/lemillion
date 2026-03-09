use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// CondSummaryV2 — Naive Bayes factorisé pour l'entropie conditionnelle.
///
/// Corrige le problème de V1 (espace d'états trop grand ~4800 pour 634 tirages)
/// en factorisant en 3 tables indépendantes avec ~100+ observations chacune :
/// - P(num | sum_bin)    : 5 bins, ~127 obs/bin
/// - P(num | spread_bin) : 5 bins, ~127 obs/bin
/// - P(num | odd_count)  : 6 bins (balls) / 3 bins (stars), ~100+ obs/bin
///
/// Combinaison multiplicative : score(k) = P(k|sum) * P(k|spread) * P(k|odd) / P(k)^2
pub struct CondSummaryV2Model {
    smoothing: f64,
    laplace: f64,
    min_draws: usize,
}

impl CondSummaryV2Model {
    pub fn new(smoothing: f64, laplace: f64, min_draws: usize) -> Self {
        Self { smoothing, laplace, min_draws }
    }
}

impl Default for CondSummaryV2Model {
    fn default() -> Self {
        Self {
            smoothing: 0.35,
            laplace: 0.5,
            min_draws: 30,
        }
    }
}

/// Calcule le bin de somme (0..4) pour un ensemble de numéros.
fn sum_bin(numbers: &[u8], pool: Pool) -> usize {
    let sum: u16 = numbers.iter().map(|&n| n as u16).sum();
    match pool {
        Pool::Balls => {
            // Somme théorique : min=15 (1+2+3+4+5), max=240 (46+47+48+49+50)
            // On divise en 5 bins égaux
            let normed = (sum as f64 - 15.0) / (240.0 - 15.0);
            (normed * 5.0).floor().clamp(0.0, 4.0) as usize
        }
        Pool::Stars => {
            // Somme théorique : min=3 (1+2), max=23 (11+12)
            let normed = (sum as f64 - 3.0) / (23.0 - 3.0);
            (normed * 5.0).floor().clamp(0.0, 4.0) as usize
        }
    }
}

/// Calcule le bin de spread (max - min) (0..4).
fn spread_bin(numbers: &[u8], pool: Pool) -> usize {
    let min = *numbers.iter().min().unwrap_or(&0);
    let max = *numbers.iter().max().unwrap_or(&0);
    let spread = (max - min) as usize;
    match pool {
        Pool::Balls => {
            // Spread max = 49, divise en 5 bins de ~10
            (spread / 10).min(4)
        }
        Pool::Stars => {
            // Spread max = 11, divise en 5 bins de ~3
            (spread / 3).min(4)
        }
    }
}

/// Calcule le nombre d'impairs.
fn odd_count(numbers: &[u8]) -> usize {
    numbers.iter().filter(|&&n| n % 2 == 1).count()
}

/// Construit une table conditionnelle P(num | bin) avec lissage de Laplace.
/// `bin_fn` extrait le bin du tirage conditionnel.
/// Retourne la distribution pour `current_bin`.
fn conditional_table(
    draws: &[Draw],
    pool: Pool,
    size: usize,
    laplace: f64,
    n_bins: usize,
    bin_fn: impl Fn(&[u8], Pool) -> usize,
) -> Vec<Vec<f64>> {
    // counts[bin][num_idx]
    let mut counts = vec![vec![0.0f64; size]; n_bins];
    let mut bin_totals = vec![0.0f64; n_bins];

    // draws[0] = plus récent. Paire (t, t+1) : état de draws[t+1] → numéros de draws[t]
    for t in 0..draws.len() - 1 {
        let cond_numbers = pool.numbers_from(&draws[t + 1]);
        let bin = bin_fn(cond_numbers, pool);
        let bin = bin.min(n_bins - 1);
        bin_totals[bin] += 1.0;

        for &n in pool.numbers_from(&draws[t]) {
            let idx = (n - 1) as usize;
            if idx < size {
                counts[bin][idx] += 1.0;
            }
        }
    }

    // Normaliser chaque bin avec lissage de Laplace
    let mut tables = vec![vec![1.0 / size as f64; size]; n_bins];
    for bin in 0..n_bins {
        if bin_totals[bin] > 0.0 {
            for k in 0..size {
                tables[bin][k] = (counts[bin][k] + laplace)
                    / (bin_totals[bin] * pool.pick_count() as f64 + laplace * size as f64);
            }
        }
    }

    tables
}

impl ForecastModel for CondSummaryV2Model {
    fn name(&self) -> &str {
        "CondSummaryV2"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let n_sum_bins = 5;
        let n_spread_bins = 5;
        let n_odd_bins = match pool {
            Pool::Balls => 6,  // 0..=5 impairs parmi 5
            Pool::Stars => 3,  // 0..=2 impairs parmi 2
        };

        // Construire les 3 tables conditionnelles
        let sum_table = conditional_table(draws, pool, size, self.laplace, n_sum_bins, sum_bin);
        let spread_table = conditional_table(draws, pool, size, self.laplace, n_spread_bins, spread_bin);
        let odd_table = conditional_table(draws, pool, size, self.laplace, n_odd_bins, |nums, _| {
            odd_count(nums).min(n_odd_bins - 1)
        });

        // Fréquences marginales P(k)
        let mut marginal = vec![0.0f64; size];
        for t in 0..draws.len() - 1 {
            for &n in pool.numbers_from(&draws[t]) {
                let idx = (n - 1) as usize;
                if idx < size {
                    marginal[idx] += 1.0;
                }
            }
        }
        let marginal_total: f64 = marginal.iter().sum();
        if marginal_total > 0.0 {
            for m in &mut marginal {
                *m /= marginal_total;
            }
        } else {
            return uniform;
        }

        // État actuel = draws[0] (le plus récent)
        let current_numbers = pool.numbers_from(&draws[0]);
        let cur_sum = sum_bin(current_numbers, pool).min(n_sum_bins - 1);
        let cur_spread = spread_bin(current_numbers, pool).min(n_spread_bins - 1);
        let cur_odd = odd_count(current_numbers).min(n_odd_bins - 1);

        // Combinaison Naive Bayes : score(k) = P(k|sum) * P(k|spread) * P(k|odd) / P(k)^2
        let mut scores = Vec::with_capacity(size);
        for k in 0..size {
            let p_k = marginal[k].max(1e-15);
            let score = sum_table[cur_sum][k]
                * spread_table[cur_spread][k]
                * odd_table[cur_odd][k]
                / (p_k * p_k);
            scores.push(score.max(1e-15));
        }

        // Normaliser
        let total: f64 = scores.iter().sum();
        if total <= 0.0 {
            return uniform;
        }
        for s in &mut scores {
            *s /= total;
        }

        // Lisser avec uniforme
        let uniform_val = 1.0 / size as f64;
        for s in &mut scores {
            *s = (1.0 - self.smoothing) * *s + self.smoothing * uniform_val;
        }

        // Re-normaliser
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in &mut scores {
                *s /= total;
            }
        } else {
            return uniform;
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("laplace".into(), self.laplace),
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
    fn test_cond_v2_balls_sums_to_one() {
        let model = CondSummaryV2Model::default();
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
    fn test_cond_v2_stars_sums_to_one() {
        let model = CondSummaryV2Model::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_cond_v2_no_negative() {
        let model = CondSummaryV2Model::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_cond_v2_empty_draws() {
        let model = CondSummaryV2Model::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cond_v2_few_draws() {
        let model = CondSummaryV2Model::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cond_v2_deterministic() {
        let model = CondSummaryV2Model::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "CondSummaryV2 should be deterministic");
        }
    }

    #[test]
    fn test_sum_bin_balls() {
        // min sum = 15, max = 240
        assert_eq!(sum_bin(&[1, 2, 3, 4, 5], Pool::Balls), 0);
        assert_eq!(sum_bin(&[46, 47, 48, 49, 50], Pool::Balls), 4);
    }

    #[test]
    fn test_spread_bin_balls() {
        assert_eq!(spread_bin(&[1, 2, 3, 4, 5], Pool::Balls), 0);   // spread=4
        assert_eq!(spread_bin(&[1, 10, 20, 30, 50], Pool::Balls), 4); // spread=49
    }

    #[test]
    fn test_odd_count() {
        assert_eq!(odd_count(&[1, 3, 5, 7, 9]), 5);
        assert_eq!(odd_count(&[2, 4, 6, 8, 10]), 0);
        assert_eq!(odd_count(&[1, 2, 3, 4, 5]), 3);
    }
}
