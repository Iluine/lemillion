use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// GapModel — simplified gap-based forecaster exploiting inter-appearance signals.
///
/// Research (`research/mathematical.rs`) detects:
/// - Gap distributions deviating from geometric (K-S test)
/// - Trend regression on gaps
/// - Lag-1 autocorrelation on gaps
///
/// For each number k:
/// 1. Compute current_gap (draws since last appearance), mean_gap, std_gap
/// 2. Score via geometric survival P(gap >= current_gap) with overdue/recent adjustments
/// 3. Apply lag-1 autocorrelation correction (momentum vs mean-reversion)
/// 4. Blend 50/50 with EWMA frequency
pub struct GapModel {
    smoothing: f64,
    min_draws: usize,
    alpha: f64,
    overdue_strength: f64,
}

impl Default for GapModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            min_draws: 25,
            alpha: 0.06,
            overdue_strength: 1.5,
        }
    }
}

/// Gap statistics for a single number.
struct GapStats {
    current_gap: usize,
    mean_gap: f64,
    std_gap: f64,
    lag1_autocorr: f64,
    previous_gap: Option<usize>,
}

/// Collect gap statistics for a given number across the draw history.
/// draws[0] = most recent draw.
fn compute_gap_stats(draws: &[Draw], pool: Pool, num: u8) -> GapStats {
    let mut gaps: Vec<usize> = Vec::new();
    let mut current_gap = 0usize;
    let mut found_first = false;

    // Iterate from most recent to oldest
    for draw in draws {
        let numbers = pool.numbers_from(draw);
        if numbers.contains(&num) {
            if found_first {
                gaps.push(current_gap);
            }
            found_first = true;
            current_gap = 0;
        } else {
            current_gap += 1;
        }
    }

    // current_gap = draws since last appearance
    let mut current = 0usize;
    for draw in draws {
        if pool.numbers_from(draw).contains(&num) {
            break;
        }
        current += 1;
    }

    let (mean_gap, std_gap) = if gaps.is_empty() {
        let p = pool.pick_count() as f64 / pool.size() as f64;
        let theoretical_mean = (1.0 - p) / p;
        (theoretical_mean, theoretical_mean)
    } else {
        let n = gaps.len() as f64;
        let mean = gaps.iter().sum::<usize>() as f64 / n;
        let variance = gaps.iter().map(|&g| (g as f64 - mean).powi(2)).sum::<f64>() / n;
        (mean, variance.sqrt())
    };

    let lag1_autocorr = compute_lag1_autocorr(&gaps);
    let previous_gap = gaps.first().copied();

    GapStats {
        current_gap: current,
        mean_gap,
        std_gap,
        lag1_autocorr,
        previous_gap,
    }
}

/// Lag-1 autocorrelation of a gap series.
fn compute_lag1_autocorr(gaps: &[usize]) -> f64 {
    if gaps.len() < 3 {
        return 0.0;
    }

    let n = gaps.len();
    let mean = gaps.iter().sum::<usize>() as f64 / n as f64;
    let var: f64 = gaps.iter().map(|&g| (g as f64 - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-10 {
        return 0.0;
    }

    let mut cov = 0.0;
    for i in 0..n - 1 {
        cov += (gaps[i] as f64 - mean) * (gaps[i + 1] as f64 - mean);
    }
    cov /= (n - 1) as f64;

    (cov / var).clamp(-1.0, 1.0)
}

impl GapModel {
    /// Compute EWMA frequencies for each number in the pool.
    /// Iterates chronologically (oldest to newest).
    fn compute_ewma_freq(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let mut freq = vec![1.0 / size as f64; size];

        for draw in draws.iter().rev() {
            let numbers = pool.numbers_from(draw);
            for (idx, f) in freq.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if numbers.contains(&num) { 1.0 } else { 0.0 };
                *f = self.alpha * present + (1.0 - self.alpha) * *f;
            }
        }

        // Normalize
        let sum: f64 = freq.iter().sum();
        if sum > 0.0 {
            for f in &mut freq {
                *f /= sum;
            }
        }

        freq
    }

    /// Compute gap-based score for a single number.
    fn gap_score(&self, stats: &GapStats, p_base: f64) -> f64 {
        // Geometric survival: P(gap >= current_gap) = (1 - p)^current_gap
        let survival = (1.0 - p_base).powi(stats.current_gap as i32);

        // Overdue/recent factor based on deviation from mean gap
        let overdue_factor = if stats.std_gap > 1e-10 {
            let deviation = (stats.current_gap as f64 - stats.mean_gap) / stats.std_gap;
            if deviation > 0.5 {
                // Overdue: boost proportional to deviation
                1.0 + self.overdue_strength * (deviation - 0.5).min(3.0) / 3.0
            } else if deviation < -0.5 {
                // Too recent: reduce
                (1.0 + deviation * 0.3).max(0.3)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Autocorrelation adjustment
        let autocorr_adj = if let Some(prev_gap) = stats.previous_gap {
            let prev_short = (prev_gap as f64) < stats.mean_gap;
            if stats.lag1_autocorr > 0.05 {
                // Momentum: previous gap pattern tends to repeat
                if prev_short { 1.15 } else { 0.90 }
            } else if stats.lag1_autocorr < -0.05 {
                // Mean-reversion: previous gap pattern tends to invert
                if prev_short { 0.90 } else { 1.15 }
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Combine: inverse survival (higher gap = more likely to appear) * adjustments
        let base_score = 1.0 - survival;
        base_score * overdue_factor * autocorr_adj
    }
}

impl ForecastModel for GapModel {
    fn name(&self) -> &str {
        "Gap"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let p_base = pool.pick_count() as f64 / pool.size() as f64;

        // Compute gap-based scores for each number
        let mut gap_scores = Vec::with_capacity(size);
        for num in 1..=size as u8 {
            let stats = compute_gap_stats(draws, pool, num);
            gap_scores.push(self.gap_score(&stats, p_base));
        }

        // Normalize gap scores to a distribution
        let gap_sum: f64 = gap_scores.iter().sum();
        if gap_sum <= 0.0 {
            return uniform;
        }
        for s in &mut gap_scores {
            *s /= gap_sum;
        }

        // Compute EWMA frequency distribution
        let ewma_freq = self.compute_ewma_freq(draws, pool);

        // Blend: 50% gap signal + 50% EWMA frequency
        let mut probs = vec![0.0f64; size];
        for k in 0..size {
            probs[k] = 0.5 * gap_scores[k] + 0.5 * ewma_freq[k];
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smooth towards uniform
        let uniform_val = 1.0 / size as f64;
        for p in &mut probs {
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
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("alpha".into(), self.alpha),
            ("overdue_strength".into(), self.overdue_strength),
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
    fn test_gap_valid_distribution_balls() {
        let draws = make_test_draws(100);
        let model = GapModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_gap_valid_distribution_stars() {
        let draws = make_test_draws(100);
        let model = GapModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_gap_few_draws_returns_uniform() {
        let draws = make_test_draws(10);
        let model = GapModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6, "Few draws should return uniform");
        }
    }

    #[test]
    fn test_gap_empty_draws_returns_uniform() {
        let model = GapModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gap_no_negative() {
        let draws = make_test_draws(100);
        let model = GapModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_gap_deterministic() {
        let draws = make_test_draws(100);
        let model = GapModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Gap model should be deterministic");
        }
    }

    #[test]
    fn test_gap_sampling_strategy() {
        let model = GapModel::default();
        assert!(matches!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        ));
    }

    #[test]
    fn test_gap_large_draws() {
        let draws = make_test_draws(200);
        let model = GapModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_lag1_autocorr_constant_gaps() {
        let gaps = vec![5, 5, 5, 5, 5];
        let ac = compute_lag1_autocorr(&gaps);
        assert!((ac).abs() < 1e-10, "Constant gaps should have zero autocorrelation");
    }

    #[test]
    fn test_lag1_autocorr_too_few() {
        let gaps = vec![3, 5];
        let ac = compute_lag1_autocorr(&gaps);
        assert!((ac).abs() < 1e-10, "Too few gaps should return 0");
    }

    #[test]
    fn test_gap_stats_basic() {
        // Ball 1 appears in draws 0, 3, 6 (gaps of 3)
        let draws: Vec<Draw> = (0..10)
            .map(|i| {
                let b1 = if i % 3 == 0 { 1 } else { 2 };
                Draw {
                    draw_id: format!("{}", i),
                    day: "MARDI".to_string(),
                    date: format!("2024-01-{:02}", (i % 28) + 1),
                    balls: [b1, 10, 20, 30, 40],
                    stars: [1, 2],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: None,
                    star_order: None,
                    cycle_number: None,
                }
            })
            .collect();
        let stats = compute_gap_stats(&draws, Pool::Balls, 1);
        assert_eq!(stats.current_gap, 0, "Ball 1 is in draws[0]");
        assert!(stats.mean_gap > 0.0, "Should have computed mean gap");
    }

    #[test]
    fn test_gap_overdue_boost() {
        // A number with a very high current gap should have a higher score
        // than one that just appeared
        let model = GapModel::default();
        let p_base = 5.0 / 50.0;

        let overdue = GapStats {
            current_gap: 20,
            mean_gap: 9.0,
            std_gap: 3.0,
            lag1_autocorr: 0.0,
            previous_gap: Some(8),
        };
        let recent = GapStats {
            current_gap: 1,
            mean_gap: 9.0,
            std_gap: 3.0,
            lag1_autocorr: 0.0,
            previous_gap: Some(8),
        };

        let score_overdue = model.gap_score(&overdue, p_base);
        let score_recent = model.gap_score(&recent, p_base);
        assert!(
            score_overdue > score_recent,
            "Overdue ({}) should score higher than recent ({})",
            score_overdue,
            score_recent
        );
    }
}
