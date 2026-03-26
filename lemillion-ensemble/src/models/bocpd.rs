use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// BOCPD — Bayesian Online Changepoint Detection.
///
/// Adams & MacKay (2007) algorithm. Maintains a run-length distribution
/// P(r_t | data_1:t) where r_t is the number of draws since the last
/// changepoint.
///
/// Hazard function: H(tau) = 1/lambda where lambda is the expected
/// regime length. When a changepoint is detected (short run lengths
/// dominate), the model focuses on recent data.
///
/// The underlying predictive model per run-length segment is a simple
/// frequency counter with Dirichlet-Multinomial conjugate prior:
///   P(num | data in segment) = (count + alpha) / (total + alpha * n)
///
/// Final prediction averages over run lengths weighted by their posterior.
pub struct BocpdModel {
    expected_run_length: f64,
    smoothing: f64,
    min_draws: usize,
}

impl Default for BocpdModel {
    fn default() -> Self {
        Self {
            expected_run_length: 200.0,
            smoothing: 0.22,
            min_draws: 20,
        }
    }
}

impl BocpdModel {
    /// Run BOCPD and return the predictive distribution.
    ///
    /// The algorithm processes draws chronologically (oldest first),
    /// maintaining run-length probabilities and sufficient statistics
    /// (frequency counts) for each run length.
    fn run_bocpd(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let k = pool.pick_count();
        let t_len = draws.len();

        // Hazard: constant 1/lambda
        let hazard = 1.0 / self.expected_run_length;

        // Dirichlet prior concentration (per number), min 0.5 for regularization
        let alpha = (k as f64 / n as f64).max(0.5);

        // Process chronologically (reverse of draws order since draws[0] = most recent)
        // We maintain:
        // - run_length_probs: P(r_t = r | data_1:t) for r = 0..t
        // - sufficient_stats: for each run length, frequency counts of each number
        //
        // To keep memory bounded, we cap run length at min(t, 500).
        let max_run = 500.min(t_len);

        // run_length_probs[r] = P(r_t = r | data_1:t)
        let mut run_length_probs = vec![0.0f64; max_run + 1];
        run_length_probs[0] = 1.0; // Start with run length 0

        // Sufficient statistics: counts[r][num] = number of times `num` appeared
        // in the segment for run length r. We only need counts for current run lengths.
        // To save memory, we use a rolling buffer.
        let mut counts: Vec<Vec<f64>> = vec![vec![0.0f64; n]; max_run + 1];
        let mut totals: Vec<f64> = vec![0.0f64; max_run + 1];

        // Pre-allocate double buffers to avoid per-iteration allocation
        let mut counts_b: Vec<Vec<f64>> = vec![vec![0.0f64; n]; max_run + 1];
        let mut totals_b = vec![0.0f64; max_run + 1];
        let mut new_probs = vec![0.0f64; max_run + 1];
        let mut pred_probs = vec![0.0f64; max_run + 1];

        // Track active run-length range for pruning
        let mut active_min: usize = 0;
        let mut active_max: usize = 0; // inclusive

        // Process each draw chronologically
        for t in 0..t_len {
            let draw = &draws[t_len - 1 - t]; // oldest to newest
            let obs: Vec<usize> = pool
                .numbers_from(draw)
                .iter()
                .map(|&x| (x - 1) as usize)
                .collect();

            let t_max = max_run.min(t);

            // 1. Compute predictive probability P(x_t | r_{t-1} = r) for each active run length
            for r in active_min..=active_max.min(t_max) {
                if run_length_probs[r] < 1e-20 {
                    pred_probs[r] = 0.0;
                    continue;
                }
                let total_count = totals[r];
                let denom = total_count + alpha * n as f64;
                if denom > 0.0 {
                    let mut p = 1.0f64;
                    for &idx in &obs {
                        if idx < n {
                            p *= (counts[r][idx] + alpha) / denom;
                        }
                    }
                    pred_probs[r] = p.max(1e-300);
                } else {
                    pred_probs[r] = (1.0 / n as f64).powi(k as i32);
                }
            }

            // 2. Growth probabilities: P(r_t = r+1) += P(r_{t-1}=r) * pred * (1 - hazard)
            // Zero only the part of new_probs we'll use
            let new_active_max = (active_max + 1).min(max_run);
            for r in 0..=new_active_max {
                new_probs[r] = 0.0;
            }
            let mut changepoint_mass = 0.0f64;

            for r in active_min..max_run.min(t + 1).min(active_max + 1) {
                if run_length_probs[r] < 1e-20 {
                    continue;
                }
                let rp = run_length_probs[r] * pred_probs[r];
                let growth = rp * (1.0 - hazard);
                if r + 1 <= max_run {
                    new_probs[r + 1] += growth;
                }
                changepoint_mass += rp * hazard;
            }

            // 3. Changepoint: r_t = 0 gets the accumulated changepoint mass
            new_probs[0] += changepoint_mass;

            // 4. Normalize
            let total: f64 = new_probs[0..=new_active_max].iter().sum();
            if total > 1e-300 {
                for r in 0..=new_active_max {
                    new_probs[r] /= total;
                }
            }

            // 5. Update sufficient statistics using double buffer
            // Zero only the active part of counts_b/totals_b
            for r in 0..=new_active_max {
                for num in 0..n {
                    counts_b[r][num] = 0.0;
                }
                totals_b[r] = 0.0;
            }

            // For r=0 (new segment after changepoint): start fresh with just this observation
            for &idx in &obs {
                if idx < n {
                    counts_b[0][idx] += 1.0;
                }
            }
            totals_b[0] = obs.len() as f64;

            // For r>0 (growth from r-1): inherit counts from r-1 + add observation
            let copy_max = new_active_max.min(t + 1);
            for r in 1..=copy_max {
                if r >= 1 && (r - 1) >= active_min && (r - 1) <= active_max && new_probs[r] > 1e-20 {
                    counts_b[r][..n].copy_from_slice(&counts[r - 1][..n]);
                    totals_b[r] = totals[r - 1];
                    for &idx in &obs {
                        if idx < n {
                            counts_b[r][idx] += 1.0;
                        }
                    }
                    totals_b[r] += obs.len() as f64;
                }
            }

            // Swap buffers
            std::mem::swap(&mut counts, &mut counts_b);
            std::mem::swap(&mut totals, &mut totals_b);
            std::mem::swap(&mut run_length_probs, &mut new_probs);

            // Update active range based on non-negligible probabilities
            active_min = 0;
            active_max = 0;
            for r in 0..=new_active_max {
                if run_length_probs[r] > 1e-20 {
                    active_max = r;
                }
            }
            // Find first non-negligible
            for r in 0..=active_max {
                if run_length_probs[r] > 1e-20 {
                    active_min = r;
                    break;
                }
            }
        }

        // 6. Final predictive distribution: average over run lengths
        //    P(num) = sum_r P(r_t = r) * P(num | data in segment of length r)
        let mut probs = vec![0.0f64; n];
        for r in 0..=max_run.min(t_len) {
            if run_length_probs[r] < 1e-20 {
                continue;
            }
            let total_count = totals[r];
            let denom = total_count + alpha * n as f64;
            if denom > 0.0 {
                for num in 0..n {
                    let pred = (counts[r][num] + alpha) / denom;
                    probs[num] += run_length_probs[r] * pred;
                }
            }
        }

        probs
    }
}

impl ForecastModel for BocpdModel {
    fn name(&self) -> &str {
        "BOCPD"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        if draws.len() < self.min_draws {
            return vec![1.0 / n as f64; n];
        }

        let mut probs = self.run_bocpd(draws, pool);

        // Smooth towards uniform
        let uniform_val = 1.0 / n as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = if pool == Pool::Balls {
            PROB_FLOOR_BALLS
        } else {
            PROB_FLOOR_STARS
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("expected_run_length".into(), self.expected_run_length),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
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
    fn test_bocpd_valid_distribution() {
        let draws = make_test_draws(100);
        let model = BocpdModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_bocpd_stars() {
        let draws = make_test_draws(100);
        let model = BocpdModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_bocpd_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = BocpdModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bocpd_no_negative() {
        let draws = make_test_draws(100);
        let model = BocpdModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_bocpd_deterministic() {
        let draws = make_test_draws(50);
        let model = BocpdModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-10, "BOCPD should be deterministic");
        }
    }

    #[test]
    fn test_bocpd_empty_draws() {
        let model = BocpdModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bocpd_large_draws() {
        let draws = make_test_draws(200);
        let model = BocpdModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_bocpd_changepoint_sensitivity() {
        // Create draws with a clear changepoint: first 50 draws use low numbers,
        // next 50 use high numbers. BOCPD should detect the shift.
        let draws: Vec<Draw> = (0..100)
            .map(|i| {
                let (b, s) = if i < 50 {
                    // Recent regime: high numbers
                    ([40, 42, 44, 46, 48], [10, 12])
                } else {
                    // Old regime: low numbers
                    ([1, 3, 5, 7, 9], [1, 2])
                };
                Draw {
                    draw_id: format!("{:03}", i),
                    day: "MARDI".to_string(),
                    date: format!("2024-01-{:02}", (i % 28) + 1),
                    balls: b,
                    stars: s,
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: None,
                    star_order: None,
                    cycle_number: None,
        prize_tiers: None,
                }
            })
            .collect();

        let model = BocpdModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));

        // The recent regime uses high numbers, so high numbers should have
        // higher probability than low numbers
        let high_prob: f64 = dist[39..50].iter().sum(); // numbers 40-50
        let low_prob: f64 = dist[0..10].iter().sum(); // numbers 1-10
        assert!(
            high_prob > low_prob,
            "High: {}, Low: {} — BOCPD should favor recent regime",
            high_prob,
            low_prob
        );
    }
}
