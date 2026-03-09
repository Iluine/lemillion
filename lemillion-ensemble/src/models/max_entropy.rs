use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};
use super::mod4::modulus;

/// Maximum Entropy model: starts from uniform distribution and adds
/// only statistically significant constraints as exponential tilts.
///
/// Constraints considered:
/// 1. Mod-4 frequency bias (from Stresa machine 4-blade structure)
/// 2. Gap compression (observed gaps are shorter than geometric)
/// 3. Pair co-occurrence excess (top pairs have higher-than-expected frequency)
///
/// Each constraint is validated by a simple z-test vs expected;
/// only constraints with |z| > 2.0 are included.
pub struct MaxEntropyModel {
    smoothing: f64,
    min_draws: usize,
    z_threshold: f64,
}

impl Default for MaxEntropyModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            min_draws: 50,
            z_threshold: 1.5,
        }
    }
}

impl ForecastModel for MaxEntropyModel {
    fn name(&self) -> &str {
        "MaxEntropy"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();

        if draws.len() < self.min_draws {
            return vec![1.0 / n as f64; n];
        }

        // Start from uniform log-weights (will be exponentiated at the end)
        let mut log_weights = vec![0.0f64; n];

        // Constraint 1: Mod-4 bias
        self.apply_mod4_constraint(draws, pool, &mut log_weights);

        // Constraint 2: Gap-based hazard
        self.apply_gap_constraint(draws, pool, &mut log_weights);

        // Constraint 3: Pair co-occurrence (only for balls)
        if pool == Pool::Balls && draws.len() >= 100 {
            self.apply_pair_constraint(draws, pool, &mut log_weights);
        }

        // Constraint 4: Autocorrelation lag-1 (v13)
        self.apply_autocorrelation_constraint(draws, pool, &mut log_weights);

        // Convert log-weights to probabilities
        let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();

        // Smooth towards uniform
        let uniform = 1.0 / n as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        let floor = if pool == Pool::Balls { PROB_FLOOR_BALLS } else { PROB_FLOOR_STARS };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("smoothing".into(), self.smoothing);
        m.insert("z_threshold".into(), self.z_threshold);
        m.insert("min_draws".into(), self.min_draws as f64);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

impl MaxEntropyModel {
    /// Mod-4 constraint: if the observed mod-4 distribution differs significantly
    /// from uniform, tilt towards over-represented residues.
    fn apply_mod4_constraint(&self, draws: &[Draw], pool: Pool, log_weights: &mut [f64]) {
        let n = pool.size();
        let m = modulus(pool); // 8 for balls, 4 for stars
        let mut mod_counts = vec![0u32; m];
        let mut total = 0u32;

        for d in draws {
            for &num in pool.numbers_from(d) {
                mod_counts[((num - 1) as usize) % m] += 1;
                total += 1;
            }
        }

        if total == 0 {
            return;
        }

        let expected = total as f64 / m as f64;
        let mut any_significant = false;
        let mut mod_tilts = vec![0.0f64; m];

        for r in 0..m {
            let obs = mod_counts[r] as f64;
            let z = (obs - expected) / expected.sqrt();
            if z.abs() > self.z_threshold {
                mod_tilts[r] = z * 0.08; // tilt per z-unit
                any_significant = true;
            }
        }

        if any_significant {
            for i in 0..n {
                let residue = i % m;
                log_weights[i] += mod_tilts[residue];
            }
        }
    }

    /// Gap constraint: numbers with gaps shorter than geometric expectation
    /// get a boost based on the hazard function.
    fn apply_gap_constraint(&self, draws: &[Draw], pool: Pool, log_weights: &mut [f64]) {
        let n = pool.size();
        let k = pool.pick_count();
        let p_appear = k as f64 / n as f64; // geometric parameter
        let expected_gap = 1.0 / p_appear;

        // Compute current gap for each number
        let mut last_seen = vec![draws.len(); n];
        for (t, d) in draws.iter().enumerate() {
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if last_seen[idx] == draws.len() {
                    last_seen[idx] = t;
                }
            }
        }

        // Compute mean gap per number
        let mut gap_sums = vec![0.0f64; n];
        let mut gap_counts = vec![0u32; n];
        for i in 0..n {
            let mut prev_t: Option<usize> = None;
            for (t, d) in draws.iter().enumerate() {
                let present = pool.numbers_from(d).iter().any(|&num| (num - 1) as usize == i);
                if present {
                    if let Some(pt) = prev_t {
                        let gap = (t - pt) as f64;
                        gap_sums[i] += gap;
                        gap_counts[i] += 1;
                    }
                    prev_t = Some(t);
                }
            }
        }

        // z-test: is mean gap significantly different from expected?
        let mut any_significant = false;
        let mut gap_tilts = vec![0.0f64; n];

        for i in 0..n {
            if gap_counts[i] >= 10 {
                let mean_gap = gap_sums[i] / gap_counts[i] as f64;
                // Exact geometric std: sqrt((1-p)/p²)
                let std_gap = ((1.0 - p_appear) / (p_appear * p_appear)).sqrt();
                let z = (expected_gap - mean_gap) / (std_gap / (gap_counts[i] as f64).sqrt());
                if z.abs() > self.z_threshold {
                    // Positive z = gap shorter than expected = number appears more often
                    gap_tilts[i] = z * 0.025;
                    any_significant = true;
                }
            }

            // Empirical hazard: use the shape of the gap distribution, not current gap
            // This avoids gambler's fallacy (boosting "overdue" numbers)
            if gap_counts[i] >= 5 {
                let current_gap = last_seen[i];
                // Count historical gaps >= current_gap and == current_gap
                let mut n_survived = 0u32;
                let mut n_event = 0u32;
                let mut prev_t: Option<usize> = None;
                for (t2, d2) in draws.iter().enumerate() {
                    let present = pool.numbers_from(d2).iter().any(|&num| (num - 1) as usize == i);
                    if present {
                        if let Some(pt) = prev_t {
                            let gap = t2 - pt;
                            if gap >= current_gap { n_survived += 1; }
                            if gap == current_gap { n_event += 1; }
                        }
                        prev_t = Some(t2);
                    }
                }
                if n_survived >= 3 {
                    let emp_hazard = n_event as f64 / n_survived as f64;
                    let geo_hazard = p_appear;
                    let hazard_z = (emp_hazard - geo_hazard) / (geo_hazard * (1.0 - geo_hazard) / n_survived as f64).sqrt().max(1e-10);
                    if hazard_z.abs() > self.z_threshold {
                        gap_tilts[i] += hazard_z * 0.05;
                        any_significant = true;
                    }
                }
            }
        }

        if any_significant {
            for i in 0..n {
                log_weights[i] += gap_tilts[i];
            }
        }
    }

    /// Autocorrelation lag-1 constraint (v13): if a number's persistence rate
    /// (probability of appearing at t+1 given it appeared at t) differs significantly
    /// from its marginal rate, tilt accordingly for numbers present in the last draw.
    fn apply_autocorrelation_constraint(&self, draws: &[Draw], pool: Pool, log_weights: &mut [f64]) {
        if draws.len() < 20 { return; }
        let size = pool.size();
        let last_draw = pool.numbers_from(&draws[0]);

        for num in 1..=size as u8 {
            let idx = (num - 1) as usize;
            let was_present = last_draw.contains(&num);
            if !was_present { continue; }

            // Count transitions 1→1 (persistence) in history
            let mut persist_count = 0usize;
            let mut present_count = 0usize;
            for t in 0..draws.len() - 1 {
                let present = pool.numbers_from(&draws[t]).contains(&num);
                if present {
                    present_count += 1;
                    let next_present = pool.numbers_from(&draws[t + 1]).contains(&num);
                    if next_present { persist_count += 1; }
                }
            }
            if present_count < 10 { continue; }

            let persist_rate = persist_count as f64 / present_count as f64;
            let marginal_rate = present_count as f64 / (draws.len() - 1) as f64;
            let se = (marginal_rate * (1.0 - marginal_rate) / present_count as f64).sqrt();
            if se < 1e-10 { continue; }
            let z = (persist_rate - marginal_rate) / se;

            if z.abs() > self.z_threshold {
                log_weights[idx] += 0.08 * z.signum() * (z.abs() - self.z_threshold).min(3.0);
            }
        }
    }

    /// Pair co-occurrence constraint: if certain pairs appear together more often
    /// than expected, both members get a conditional boost when one appeared recently.
    fn apply_pair_constraint(&self, draws: &[Draw], pool: Pool, log_weights: &mut [f64]) {
        let n = pool.size();
        let k = pool.pick_count();
        let total_draws = draws.len() as f64;
        let p_pair = (k as f64 * (k as f64 - 1.0)) / (n as f64 * (n as f64 - 1.0));

        // Count pair co-occurrences
        let mut pair_counts = vec![vec![0u32; n]; n];
        for d in draws {
            let nums: Vec<usize> = pool.numbers_from(d).iter().map(|&x| (x - 1) as usize).collect();
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    pair_counts[nums[i]][nums[j]] += 1;
                    pair_counts[nums[j]][nums[i]] += 1;
                }
            }
        }

        // Find significant pairs involving numbers from the last draw
        let last_nums: Vec<usize> = if !draws.is_empty() {
            pool.numbers_from(&draws[0]).iter().map(|&x| (x - 1) as usize).collect()
        } else {
            return;
        };

        let expected_count = total_draws * p_pair;
        let std_count = (total_draws * p_pair * (1.0 - p_pair)).sqrt();

        if std_count < 1e-10 {
            return;
        }

        for &last_num in &last_nums {
            for partner in 0..n {
                if partner == last_num {
                    continue;
                }
                let obs = pair_counts[last_num][partner] as f64;
                let z = (obs - expected_count) / std_count;
                if z > self.z_threshold {
                    log_weights[partner] += z * 0.015;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_max_entropy_valid_distribution() {
        let draws = make_test_draws(100);
        let model = MaxEntropyModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_max_entropy_stars() {
        let draws = make_test_draws(100);
        let model = MaxEntropyModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars));
    }

    #[test]
    fn test_max_entropy_few_draws_returns_uniform() {
        let draws = make_test_draws(10);
        let model = MaxEntropyModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9);
        }
    }
}
