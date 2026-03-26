use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// DeFinettiModel — Bayesian predictive distribution via Dirichlet-Multinomial mixture.
///
/// Implements de Finetti's theorem for exchangeable sequences: if the draws are
/// exchangeable (a weaker assumption than i.i.d.), then the predictive distribution
/// for x_{n+1} given x_1..x_n is:
///
///   P(x_{n+1} = k | x_1..x_n) = (n_k + α) / (n + K·α)
///
/// where:
///   - n_k = weighted count of number k in the history
///   - n = total weighted count
///   - α = Dirichlet concentration parameter (prior strength)
///   - K = pool size (50 for balls, 12 for stars)
///   - G0 = uniform base measure
///
/// This is the posterior predictive of a Dirichlet-Multinomial model with
/// symmetric Dirichlet prior Dir(α, α, ..., α).
///
/// Key advantage: provides principled uncertainty quantification. When α is
/// large relative to n, predictions stay close to uniform (high uncertainty).
/// When n ≫ α, predictions track observed frequencies (low uncertainty).
///
/// Temporal weighting: more recent draws receive higher weight via exponential
/// decay, so the model adapts to distributional drift while maintaining the
/// Bayesian framework.
pub struct DeFinettiModel {
    smoothing: f64,
    alpha: f64,          // Dirichlet concentration parameter
    temporal_decay: f64, // Weight decay per draw (0.995 = recent draws ~2x weight of 140-draw-old)
    min_draws: usize,
}

impl Default for DeFinettiModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            alpha: 1.0,
            temporal_decay: 0.995,
            min_draws: 30,
        }
    }
}

impl ForecastModel for DeFinettiModel {
    fn name(&self) -> &str {
        "DeFinetti"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // 1. Compute temporally-weighted counts
        // draws[0] = most recent, draws[n-1] = oldest
        // Weight of draw at index t: decay^t (most recent = 1.0)
        let mut weighted_counts = vec![0.0f64; size];
        let mut total_weight = 0.0f64;

        for (t, d) in draws.iter().enumerate() {
            let weight = self.temporal_decay.powi(t as i32);
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < size {
                    weighted_counts[idx] += weight;
                }
            }
            total_weight += weight * pool.pick_count() as f64;
        }

        // 2. Dirichlet-Multinomial posterior predictive
        // P(k) = (n_k + α) / (n + K·α)
        let k_alpha = size as f64 * self.alpha;
        let denom = total_weight + k_alpha;

        let mut probs = vec![0.0f64; size];
        if denom > 0.0 {
            for i in 0..size {
                probs[i] = (weighted_counts[i] + self.alpha) / denom;
            }
        } else {
            return uniform;
        }

        // 3. Compute predictive variance for each number (diagnostic, affects smoothing)
        // Var(p_k | data) ∝ (n_k + α)(n + K·α - n_k - α) / ((n + K·α)^2 · (n + K·α + 1))
        // High variance → more uncertainty → stronger smoothing would help
        // But we keep smoothing fixed and let the Dirichlet prior handle uncertainty
        let effective_n = total_weight / pool.pick_count() as f64;
        let uncertainty_ratio = k_alpha / (effective_n + k_alpha);

        // 4. Adaptive smoothing: when uncertainty is high, smooth more
        // Base smoothing is self.smoothing, boosted by uncertainty ratio
        let adaptive_smoothing = self.smoothing + (1.0 - self.smoothing) * uncertainty_ratio * 0.3;
        let adaptive_smoothing = adaptive_smoothing.min(0.90); // cap at 90%

        // 5. Smooth towards uniform
        let uniform_val = 1.0 / size as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - adaptive_smoothing) * *p + adaptive_smoothing * uniform_val;
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
            ("alpha".into(), self.alpha),
            ("temporal_decay".into(), self.temporal_decay),
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
    fn test_definetti_valid_ball_distribution() {
        let draws = make_test_draws(100);
        let model = DeFinettiModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_definetti_valid_star_distribution() {
        let draws = make_test_draws(100);
        let model = DeFinettiModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_definetti_few_draws_uniform() {
        let draws = make_test_draws(10);
        let model = DeFinettiModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!(
                (p - expected).abs() < 1e-10,
                "Few draws should return uniform, got {}",
                p
            );
        }
    }

    #[test]
    fn test_definetti_no_negative() {
        let draws = make_test_draws(100);
        let model = DeFinettiModel::default();
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            for &p in &dist {
                assert!(p >= 0.0, "Negative probability: {} for {:?}", p, pool);
            }
        }
    }

    #[test]
    fn test_definetti_deterministic() {
        let draws = make_test_draws(100);
        let model = DeFinettiModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "DeFinetti should be deterministic");
        }
    }

    #[test]
    fn test_definetti_high_alpha_near_uniform() {
        // With very high alpha, prior dominates → near uniform
        let model = DeFinettiModel {
            alpha: 1000.0,
            ..DeFinettiModel::default()
        };
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform_val = 1.0 / 50.0;
        for &p in &dist {
            assert!(
                (p - uniform_val).abs() < 0.005,
                "High alpha should give near-uniform, got {} (expected ~{})",
                p,
                uniform_val
            );
        }
    }

    #[test]
    fn test_definetti_low_alpha_tracks_frequency() {
        // With very low alpha, data dominates → tracks observed frequency
        let model = DeFinettiModel {
            alpha: 0.001,
            smoothing: 0.0, // disable smoothing to see raw effect
            ..DeFinettiModel::default()
        };
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        // Numbers that appear should have higher probability than those that don't
        // make_test_draws cycles balls through base*5+1..base*5+5 with base=0..9
        // So balls 1-50 appear (all 50 balls appear)
        assert!(validate_distribution(&dist, Pool::Balls));
        // Probabilities should NOT be uniform with low alpha
        let max_p = dist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_p = dist.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_p - min_p > 1e-6,
            "Low alpha should produce non-uniform distribution"
        );
    }

    #[test]
    fn test_definetti_temporal_decay_recency() {
        // Verify that temporal decay gives more weight to recent draws
        let model = DeFinettiModel {
            temporal_decay: 0.90, // strong decay
            smoothing: 0.05,
            ..DeFinettiModel::default()
        };
        // Create draws where recent draws have ball 1 and older draws don't
        let mut draws = Vec::new();
        // Recent draws: all have ball 1
        for i in 0..50 {
            draws.push(Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-06-{:02}", (i % 28) + 1),
                balls: [1, 10, 20, 30, 40],
                stars: [1, 2],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
        prize_tiers: None,
            });
        }
        // Older draws: never have ball 1
        for i in 50..100 {
            draws.push(Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [2, 11, 21, 31, 41],
                stars: [1, 2],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
        prize_tiers: None,
            });
        }
        let dist = model.predict(&draws, Pool::Balls);
        // Ball 1 (index 0) should have higher probability than ball 2 (index 1)
        // because recent draws favor ball 1 and decay de-weights older draws
        assert!(
            dist[0] > dist[1],
            "Ball 1 ({}) should be higher than ball 2 ({}) with temporal decay",
            dist[0],
            dist[1]
        );
    }

    #[test]
    fn test_definetti_empty_draws() {
        let model = DeFinettiModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }
}
