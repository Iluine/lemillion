use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// EVT — Extreme Value Theory model for tail dependence.
///
/// Models co-exceedance: when the frequency of a number is extreme
/// (well above average), which other numbers also have extreme frequency?
///
/// The tail dependence coefficient χ(i,j) = P(Z_j > u | Z_i > u)
/// captures co-exceedances that are fundamentally different from:
/// - TE (temporal causality)
/// - Spectral (static community structure)
///
/// Uses rolling sub-windows to compute z-score frequency series,
/// then measures pairwise tail dependence.
pub struct EvtModel {
    window: usize,
    sub_window: usize,
    threshold_pctl: f64,
    alpha: f64,
    smoothing: f64,
    min_draws: usize,
}

impl Default for EvtModel {
    fn default() -> Self {
        Self {
            window: 500,
            sub_window: 30,
            threshold_pctl: 0.80,
            alpha: 1.5,
            smoothing: 0.30,
            min_draws: 100,
        }
    }
}

impl ForecastModel for EvtModel {
    fn name(&self) -> &str {
        "EVT"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let window = self.window.min(draws.len());
        let recent = &draws[..window];

        // 1. Compute rolling frequency z-scores per number
        // Each sub-window gives one observation per number
        let n_windows = if window >= self.sub_window {
            window - self.sub_window + 1
        } else {
            return uniform;
        };

        if n_windows < 5 {
            return uniform;
        }

        // Compute frequency in each sub-window (chronological: recent[0] = newest)
        let mut freq_series: Vec<Vec<f64>> = vec![Vec::with_capacity(n_windows); size];
        for w in 0..n_windows {
            let sub = &recent[w..w + self.sub_window];
            let mut counts = vec![0.0f64; size];
            for d in sub {
                for &num in pool.numbers_from(d) {
                    counts[(num - 1) as usize] += 1.0;
                }
            }
            for i in 0..size {
                freq_series[i].push(counts[i] / self.sub_window as f64);
            }
        }

        // 2. Normalize to z-scores
        let mut z_series: Vec<Vec<f64>> = Vec::with_capacity(size);
        for i in 0..size {
            let series = &freq_series[i];
            let mean = series.iter().sum::<f64>() / series.len() as f64;
            let var = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;
            let std = var.sqrt().max(1e-10);
            z_series.push(series.iter().map(|&x| (x - mean) / std).collect());
        }

        // 3. Determine exceedance threshold per number (80th percentile)
        let mut thresholds = vec![0.0f64; size];
        for i in 0..size {
            let mut sorted: Vec<f64> = z_series[i].clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((n_windows as f64 * self.threshold_pctl) as usize).min(n_windows - 1);
            thresholds[i] = sorted[idx];
        }

        // 4. Identify exceedances
        let mut exceedances: Vec<Vec<bool>> = Vec::with_capacity(size);
        for i in 0..size {
            exceedances.push(z_series[i].iter().enumerate()
                .map(|(t, &z)| z > thresholds[i] || (t == 0 && z >= thresholds[i]))
                .collect());
        }

        // 5. Compute tail dependence coefficient χ(i,j)
        // χ(i,j) = |{t : Z_i > u AND Z_j > u}| / |{t : Z_i > u}|
        // Expected under independence ≈ 1 - threshold_pctl
        let chi_expected = 1.0 - self.threshold_pctl; // 0.20

        // Only compute for numbers in the last draw
        let last_nums: Vec<usize> = pool.numbers_from(&draws[0])
            .iter().map(|&x| (x - 1) as usize).collect();

        // Get current z-scores (window 0 = most recent)
        let current_z: Vec<f64> = (0..size).map(|i| z_series[i][0]).collect();

        // 6. Score each candidate based on tail dependence with last-draw numbers
        let mut scores = vec![0.0f64; size];
        for j in 0..size {
            let mut tail_boost = 0.0f64;
            let mut n_active = 0;

            for &i in &last_nums {
                if i == j { continue; }
                // Only consider if source had a positive z-score recently
                if current_z[i] <= 0.0 { continue; }

                // Count co-exceedances
                let n_i_exceed: usize = exceedances[i].iter().filter(|&&x| x).count();
                if n_i_exceed < 2 { continue; }

                let n_co_exceed: usize = exceedances[i].iter().zip(exceedances[j].iter())
                    .filter(|&(&a, &b)| a && b).count();

                let chi = n_co_exceed as f64 / n_i_exceed as f64;

                // Significant if chi > alpha × expected
                if chi > self.alpha * chi_expected {
                    tail_boost += (chi - chi_expected) * current_z[i].max(0.0);
                    n_active += 1;
                }
            }

            // Normalize by number of active sources
            if n_active > 0 {
                tail_boost /= n_active as f64;
            }

            // Base frequency
            let mut freq_count = 0.0f64;
            for d in recent {
                if pool.numbers_from(d).contains(&((j + 1) as u8)) {
                    freq_count += 1.0;
                }
            }
            let base_freq = freq_count / window as f64;

            scores[j] = base_freq * (1.0 + tail_boost);
        }

        // 7. Smooth & normalize
        let uniform_val = 1.0 / size as f64;
        for s in &mut scores {
            *s = (1.0 - self.smoothing) * *s + self.smoothing * uniform_val;
        }

        let floor = if pool == Pool::Balls { PROB_FLOOR_BALLS } else { PROB_FLOOR_STARS };
        floor_only(&mut scores, floor);
        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("window".into(), self.window as f64),
            ("sub_window".into(), self.sub_window as f64),
            ("threshold_pctl".into(), self.threshold_pctl),
            ("alpha".into(), self.alpha),
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
    fn test_evt_balls_sums_to_one() {
        let model = EvtModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_evt_stars_sums_to_one() {
        let model = EvtModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len()
        );
    }

    #[test]
    fn test_evt_few_draws_uniform() {
        let model = EvtModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_evt_deterministic() {
        let model = EvtModel::default();
        let draws = make_test_draws(200);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_evt_no_negative() {
        let model = EvtModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_evt_empty_draws() {
        let model = EvtModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }
}
