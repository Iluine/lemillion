//! Conformal prediction sets for adaptive K selection (v17).
//!
//! Uses non-conformity scores from walk-forward validation to
//! determine prediction set sizes with coverage guarantees.

use lemillion_db::models::{Draw, Pool};

/// Conformal prediction set result.
#[derive(Debug, Clone)]
pub struct ConformalSet {
    /// Indices of numbers in the prediction set (0-indexed).
    pub members: Vec<usize>,
    /// Target coverage level.
    pub coverage_target: f64,
    /// Estimated actual coverage from calibration.
    pub estimated_coverage: f64,
}

/// Compute the conformal prediction set for a pool.
///
/// Uses split conformal prediction:
/// 1. On calibration data (walk-forward), compute non-conformity scores = 1 - p_ensemble(correct_number)
/// 2. Compute the (1-alpha) quantile of these scores
/// 3. The prediction set = all numbers where p_ensemble(number) >= 1 - quantile
///
/// Returns a ConformalSet with the selected numbers.
pub fn conformal_prediction_set(
    ensemble_probs: &[f64],
    calibration_scores: &[f64],  // non-conformity scores from walk-forward
    coverage_target: f64,         // e.g. 0.95
) -> ConformalSet {
    if calibration_scores.is_empty() || ensemble_probs.is_empty() {
        return ConformalSet {
            members: (0..ensemble_probs.len()).collect(),
            coverage_target,
            estimated_coverage: 1.0,
        };
    }

    // Compute the quantile threshold
    let mut sorted_scores = calibration_scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Conformal quantile: ceil((n+1)*(1-alpha))/n-th smallest score
    let n = sorted_scores.len();
    let quantile_idx = (((n + 1) as f64 * coverage_target).ceil() as usize).min(n) - 1;
    let threshold = sorted_scores[quantile_idx];

    // Prediction set: numbers whose non-conformity score <= threshold
    // Non-conformity of number i = 1 - p(i)
    // So include number i if 1 - p(i) <= threshold, i.e., p(i) >= 1 - threshold
    let p_threshold = 1.0 - threshold;
    let eps = 1e-12;

    let members: Vec<usize> = (0..ensemble_probs.len())
        .filter(|&i| ensemble_probs[i] >= p_threshold - eps)
        .collect();

    // If the set is empty (extreme case), include at least the top number
    let members = if members.is_empty() {
        let mut indices: Vec<usize> = (0..ensemble_probs.len()).collect();
        indices.sort_by(|&a, &b| {
            ensemble_probs[b]
                .partial_cmp(&ensemble_probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(5); // minimum 5 for balls
        indices
    } else {
        members
    };

    // Estimate coverage
    let estimated_coverage = (quantile_idx + 1) as f64 / n as f64;

    ConformalSet {
        members,
        coverage_target,
        estimated_coverage,
    }
}

/// Collect non-conformity scores from walk-forward validation.
///
/// For each test draw in the calibration set, computes the non-conformity score
/// for each correct number: score = 1 - p_ensemble(correct_number).
pub fn collect_nonconformity_scores(
    draws: &[Draw],
    pool: Pool,
    predict_fn: &dyn Fn(&[Draw], Pool) -> Vec<f64>,
    n_test: usize,
) -> Vec<f64> {
    let n = n_test.min(draws.len().saturating_sub(30));
    let mut scores = Vec::with_capacity(n * pool.pick_count());

    for i in 0..n {
        let test_draw = &draws[i];
        let training_draws = &draws[i + 1..];

        if training_draws.len() < 20 {
            continue;
        }

        let probs = predict_fn(training_draws, pool);
        let correct_nums = pool.numbers_from(test_draw);

        for &num in correct_nums {
            let p = probs[(num - 1) as usize];
            scores.push(1.0 - p);
        }
    }

    scores
}

/// Compute adaptive K from conformal prediction set size.
///
/// Returns the optimal K (subset size) for jackpot enumeration.
/// Uses conformal prediction to determine how many numbers to include
/// in the enumeration subset, with a coverage guarantee.
pub fn conformal_k(
    ensemble_probs: &[f64],
    calibration_scores: &[f64],
    pool: Pool,
) -> usize {
    let pick = pool.pick_count();
    let size = pool.size();

    // Target 95% coverage for balls, 98% for stars (stars have fewer numbers)
    let coverage = if pick >= 5 { 0.95 } else { 0.98 };

    let set = conformal_prediction_set(ensemble_probs, calibration_scores, coverage);

    // K must be at least pick_count + some margin, and at most pool_size
    let min_k = (pick + 5).min(size);
    let max_k = size;

    set.members.len().clamp(min_k, max_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_prediction_set_basic() {
        // Uniform distribution
        let probs = vec![1.0 / 50.0; 50];
        // All scores are the same: 1 - 1/50 = 0.98
        let scores = vec![0.98; 100];

        let set = conformal_prediction_set(&probs, &scores, 0.95);
        // All numbers should be in the set since they all have p = 1/50 >= 1 - 0.98 = 0.02
        assert_eq!(set.members.len(), 50);
    }

    #[test]
    fn test_conformal_prediction_set_concentrated() {
        // Concentrated distribution: 5 numbers have high probability
        let mut probs = vec![0.005; 50];
        for i in 0..5 {
            probs[i] = 0.15;
        }
        // Normalize
        let sum: f64 = probs.iter().sum();
        let probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();

        // Calibration scores from concentrated draws (low non-conformity for top numbers)
        let scores: Vec<f64> = (0..100)
            .map(|i| {
                if i % 5 < 3 {
                    0.85
                } else {
                    0.95
                } // mostly high non-conformity
            })
            .collect();

        let set = conformal_prediction_set(&probs, &scores, 0.95);
        // Should include fewer numbers since distribution is concentrated
        assert!(
            set.members.len() < 50,
            "Expected fewer members, got {}",
            set.members.len()
        );
    }

    #[test]
    fn test_conformal_k_bounds() {
        let probs = vec![1.0 / 50.0; 50];
        let scores = vec![0.98; 100];

        let k = conformal_k(&probs, &scores, Pool::Balls);
        assert!(k >= 10, "K should be at least 10, got {}", k);
        assert!(k <= 50, "K should be at most 50, got {}", k);
    }

    #[test]
    fn test_conformal_empty_calibration() {
        let probs = vec![1.0 / 50.0; 50];
        let scores: Vec<f64> = vec![];

        let set = conformal_prediction_set(&probs, &scores, 0.95);
        assert_eq!(set.members.len(), 50); // All numbers when no calibration data
    }
}
