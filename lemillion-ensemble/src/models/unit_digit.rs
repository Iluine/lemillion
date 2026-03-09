use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// UnitDigitModel — captures unit-digit (column) biases from the Stresa machine.
///
/// The Stresa machine has 10 columns (units 0-9). Research detects rack-position
/// bias in columns, but no model captures the unit-digit signal directly.
///
/// For balls: EWMA frequency per unit digit (ball % 10), modulated by a drift
/// factor comparing recent window vs full history. Each ball's probability is
/// boosted or dampened by its column's drift.
///
/// For stars (1-12): simple EWMA frequency with higher smoothing.
pub struct UnitDigitModel {
    smoothing: f64,
    star_smoothing: f64,
    min_draws: usize,
    alpha: f64,
    drift_window: usize,
}

impl Default for UnitDigitModel {
    fn default() -> Self {
        Self {
            smoothing: 0.20,
            star_smoothing: 0.45,
            min_draws: 20,
            alpha: 0.06,
            drift_window: 30,
        }
    }
}

impl ForecastModel for UnitDigitModel {
    fn name(&self) -> &str {
        "UnitDigit"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        match pool {
            Pool::Balls => self.predict_balls(draws, size),
            Pool::Stars => self.predict_stars(draws, size),
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("star_smoothing".into(), self.star_smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("alpha".into(), self.alpha),
            ("drift_window".into(), self.drift_window as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

impl UnitDigitModel {
    fn predict_balls(&self, draws: &[Draw], size: usize) -> Vec<f64> {
        let uniform = vec![1.0 / size as f64; size];

        // EWMA frequency per ball number (iterate chronologically, draws reversed)
        let mut freq_ewma = vec![1.0 / size as f64; size];

        for draw in draws.iter().rev() {
            for (idx, freq) in freq_ewma.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if draw.balls.contains(&num) { 1.0 } else { 0.0 };
                *freq = self.alpha * present + (1.0 - self.alpha) * *freq;
            }
        }

        // Compute overall unit-digit frequencies from EWMA
        let mut digit_freq_overall = [0.0f64; 10];
        let mut digit_count = [0usize; 10];
        for k in 0..size {
            let digit = (k + 1) % 10;
            digit_freq_overall[digit] += freq_ewma[k];
            digit_count[digit] += 1;
        }
        // Normalize per digit (average EWMA frequency for numbers in that column)
        for d in 0..10 {
            if digit_count[d] > 0 {
                digit_freq_overall[d] /= digit_count[d] as f64;
            }
        }

        // Compute recent unit-digit frequencies from the last drift_window draws
        let recent_draws = draws.len().min(self.drift_window);
        let mut digit_freq_recent = [0.0f64; 10];
        let mut digit_total_recent = [0usize; 10];
        for draw in &draws[..recent_draws] {
            for &ball in &draw.balls {
                let digit = ball as usize % 10;
                digit_freq_recent[digit] += 1.0;
                digit_total_recent[digit] += 1;
            }
        }
        // Normalize recent digit frequencies
        let total_balls_recent = (recent_draws * 5) as f64;
        if total_balls_recent > 0.0 {
            for d in 0..10 {
                digit_freq_recent[d] /= total_balls_recent;
            }
        }

        // Drift factor: recent vs overall, clamped to [0.5, 2.0]
        let mut drift_factor = [1.0f64; 10];
        for d in 0..10 {
            if digit_freq_overall[d] > 1e-15 {
                drift_factor[d] = (digit_freq_recent[d] / digit_freq_overall[d]).clamp(0.5, 2.0);
            }
        }

        // Final probability: EWMA frequency modulated by drift
        let mut probs = vec![0.0f64; size];
        for k in 0..size {
            let digit = (k + 1) % 10;
            probs[k] = freq_ewma[k] * drift_factor[digit];
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

        // Smoothing with uniform
        let uniform_val = 1.0 / size as f64;
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    fn predict_stars(&self, draws: &[Draw], size: usize) -> Vec<f64> {
        let uniform = vec![1.0 / size as f64; size];

        // EWMA frequency for stars (iterate chronologically)
        let mut freq_ewma = vec![1.0 / size as f64; size];

        for draw in draws.iter().rev() {
            for (idx, freq) in freq_ewma.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if draw.stars.contains(&num) { 1.0 } else { 0.0 };
                *freq = self.alpha * present + (1.0 - self.alpha) * *freq;
            }
        }

        // Normalize
        let sum: f64 = freq_ewma.iter().sum();
        if sum > 0.0 {
            for p in &mut freq_ewma {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Higher smoothing for stars
        let uniform_val = 1.0 / size as f64;
        for p in &mut freq_ewma {
            *p = (1.0 - self.star_smoothing) * *p + self.star_smoothing * uniform_val;
        }

        floor_only(&mut freq_ewma, PROB_FLOOR_STARS);
        freq_ewma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_unit_digit_balls_valid() {
        let model = UnitDigitModel::default();
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
    fn test_unit_digit_stars_valid() {
        let model = UnitDigitModel::default();
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
    fn test_unit_digit_few_draws() {
        let model = UnitDigitModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Too few draws → uniform");
        }
    }

    #[test]
    fn test_unit_digit_no_negative() {
        let model = UnitDigitModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_unit_digit_deterministic() {
        let model = UnitDigitModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_unit_digit_sparse_strategy() {
        let model = UnitDigitModel::default();
        assert!(matches!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        ));
    }

    #[test]
    fn test_unit_digit_balls_not_uniform() {
        let model = UnitDigitModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        let all_uniform = dist.iter().all(|&p| (p - expected).abs() < 1e-6);
        assert!(
            !all_uniform || draws.len() < model.min_draws,
            "Should have some signal"
        );
    }
}
