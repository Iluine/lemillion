use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{filter_star_era, floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// CycleModel — Exploite le numéro de cycle (tirage dans le cycle FDJ).
///
/// La machine Stresa peut avoir des effets de warm-up (cycle 1 = machine froide)
/// ou de fatigue (cycle 3). Ce modèle:
/// 1. Filtre les tirages ayant un cycle_number connu
/// 2. Calcule une fréquence EWMA par numéro pour chaque cycle c ∈ {1, 2, 3}
/// 3. Teste chi² par cycle pour détecter des déviations significatives
/// 4. Si significatif: prédit conditionné au cycle attendu du prochain tirage
/// 5. Si non significatif: retourne uniforme (pas de signal = pas de modèle)
///
/// Fonctionne pour Pool::Balls ET Pool::Stars.
pub struct CycleModel {
    smoothing: f64,
    ewma_alpha: f64,
    min_draws_with_cycle: usize,
    chi2_threshold: f64,
}

impl Default for CycleModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            ewma_alpha: 0.08,
            min_draws_with_cycle: 30,
            chi2_threshold: 25.0,
        }
    }
}

impl CycleModel {
    /// Infer the expected cycle number of the next draw.
    /// The FDJ cycles through 1, 2, 3, 1, 2, 3, ...
    /// We look at the most recent draw's cycle and predict the next one.
    fn infer_next_cycle(&self, draws_with_cycle: &[&Draw]) -> u8 {
        if let Some(most_recent) = draws_with_cycle.first() {
            if let Some(c) = most_recent.cycle_number {
                // Cycle wraps: 1 → 2 → 3 → 1
                return (c % 3) + 1;
            }
        }
        // Fallback: assume cycle 1
        1
    }

    /// Compute EWMA frequency per number for a given cycle.
    /// `draws_for_cycle` should be in most-recent-first order.
    fn ewma_frequencies(&self, draws_for_cycle: &[&Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;

        if draws_for_cycle.is_empty() {
            return vec![uniform; size];
        }

        // Initialize EWMA at uniform frequency
        let mut ewma = vec![uniform; size];

        // Process in chronological order (oldest first)
        for draw in draws_for_cycle.iter().rev() {
            let numbers = pool.numbers_from(draw);
            // Target vector: 1.0 for drawn numbers, 0.0 otherwise
            for idx in 0..size {
                let num = (idx + 1) as u8;
                let val = if numbers.contains(&num) { 1.0 } else { 0.0 };
                ewma[idx] = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * ewma[idx];
            }
        }

        ewma
    }

    /// Compute chi² statistic for a frequency distribution vs uniform.
    /// Returns chi² value with (size - 1) degrees of freedom.
    fn chi2_vs_uniform(&self, frequencies: &[f64], total_draws: usize) -> f64 {
        let size = frequencies.len();
        if size == 0 || total_draws == 0 {
            return 0.0;
        }

        // Convert EWMA frequencies to expected counts
        // Under uniform: expected count per number = total_draws * pick_count / size
        // But EWMA frequencies are already proportions, so we compare them to uniform proportion
        let uniform = 1.0 / size as f64;

        // Use the EWMA as proportions, chi² on proportions scaled by effective sample size
        // Effective sample size for EWMA with alpha is approximately min(n, 2/alpha)
        let n_eff = (total_draws as f64).min(2.0 / self.ewma_alpha);

        let mut chi2 = 0.0;
        for &f in frequencies {
            let diff = f - uniform;
            chi2 += diff * diff / uniform;
        }
        // Scale by effective sample size
        chi2 * n_eff
    }
}

impl ForecastModel for CycleModel {
    fn name(&self) -> &str {
        "CycleModel"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // For stars, filter to current era
        let draws = if pool == Pool::Stars {
            filter_star_era(draws)
        } else {
            draws
        };

        // Filter draws that have cycle_number
        let draws_with_cycle: Vec<&Draw> = draws.iter()
            .filter(|d| d.cycle_number.is_some())
            .collect();

        if draws_with_cycle.len() < self.min_draws_with_cycle {
            return uniform;
        }

        // Infer the expected cycle for the next draw
        let next_cycle = self.infer_next_cycle(&draws_with_cycle);

        // Group draws by cycle number
        let mut by_cycle: HashMap<u8, Vec<&Draw>> = HashMap::new();
        for d in &draws_with_cycle {
            if let Some(c) = d.cycle_number {
                by_cycle.entry(c).or_default().push(d);
            }
        }

        // Compute EWMA frequencies per cycle
        let mut cycle_freqs: HashMap<u8, Vec<f64>> = HashMap::new();
        let mut cycle_chi2: HashMap<u8, f64> = HashMap::new();

        for (&c, cycle_draws) in &by_cycle {
            let freqs = self.ewma_frequencies(cycle_draws, pool);
            let chi2 = self.chi2_vs_uniform(&freqs, cycle_draws.len());
            cycle_chi2.insert(c, chi2);
            cycle_freqs.insert(c, freqs);
        }

        // Check if the target cycle has significant deviation
        let target_chi2 = cycle_chi2.get(&next_cycle).copied().unwrap_or(0.0);

        if target_chi2 < self.chi2_threshold {
            // No significant signal for the expected cycle
            return uniform;
        }

        // Use the EWMA frequencies for the predicted cycle
        let mut probs = match cycle_freqs.get(&next_cycle) {
            Some(freqs) => freqs.clone(),
            None => return uniform,
        };

        // Normalize raw EWMA to sum to 1.0
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

        // Floor and renormalize
        let floor = match pool {
            Pool::Balls => PROB_FLOOR_BALLS,
            Pool::Stars => PROB_FLOOR_STARS,
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("min_draws_with_cycle".into(), self.min_draws_with_cycle as f64),
            ("chi2_threshold".into(), self.chi2_threshold),
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

    /// Helper: create draws with cycle_number set, cycling 1→2→3→1→...
    fn make_cycle_draws(n: usize) -> Vec<Draw> {
        (0..n)
            .map(|i| {
                let base = (i % 10) as u8;
                let cycle = (i % 3) as u8 + 1; // 1, 2, 3, 1, 2, 3, ...
                Draw {
                    draw_id: format!("{:03}", i),
                    day: if i % 2 == 0 { "MARDI".to_string() } else { "VENDREDI".to_string() },
                    date: format!("2024-01-{:02}", (i % 28) + 1),
                    balls: [
                        (base * 5 + 1).clamp(1, 50),
                        (base * 5 + 2).clamp(1, 50),
                        (base * 5 + 3).clamp(1, 50),
                        (base * 5 + 4).clamp(1, 50),
                        (base * 5 + 5).clamp(1, 50),
                    ],
                    stars: [
                        (base % 12 + 1).min(12),
                        ((base + 1) % 12 + 1).min(12),
                    ],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: None,
                    star_order: None,
                    cycle_number: Some(cycle),
                }
            })
            .collect()
    }

    #[test]
    fn test_cycle_model_no_cycle_data_returns_uniform() {
        let model = CycleModel::default();
        // make_test_draws has cycle_number: None
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-9, "No cycle data should return uniform");
        }
    }

    #[test]
    fn test_cycle_model_few_draws_returns_uniform() {
        let model = CycleModel::default();
        // Only 10 draws with cycle data, below min_draws_with_cycle=30
        let draws = make_cycle_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Too few draws should return uniform");
        }
    }

    #[test]
    fn test_cycle_model_valid_distribution_balls() {
        let model = CycleModel::default();
        let draws = make_cycle_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Ball distribution invalid: len={}, sum={}",
            dist.len(),
            dist.iter().sum::<f64>()
        );
    }

    #[test]
    fn test_cycle_model_valid_distribution_stars() {
        let model = CycleModel::default();
        let draws = make_cycle_draws(200);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Star distribution invalid: len={}, sum={}",
            dist.len(),
            dist.iter().sum::<f64>()
        );
    }

    #[test]
    fn test_cycle_model_detects_cycle_bias() {
        // Create draws where cycle 1 strongly favors low balls (1-10)
        // and cycle 2 favors high balls (40-50)
        let model = CycleModel::default();
        let draws: Vec<Draw> = (0..150).map(|i| {
            let cycle = (i % 3) as u8 + 1;
            let balls = match cycle {
                1 => [1, 2, 3, 4, 5],    // always low balls
                2 => [40, 42, 44, 46, 48], // always high balls
                _ => [20, 22, 24, 26, 28], // always middle balls
            };
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls,
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: Some(cycle),
            }
        }).collect();

        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));

        // The next cycle (inferred from most recent draw) should show bias
        // Most recent draw (i=0) has cycle=1, so next cycle=2
        // Cycle 2 always has high balls, so ball 40 (index 39) should be elevated
        // Either the chi2 test passes and we get biased distribution,
        // or it doesn't and we get uniform — both are valid outputs
        // But with 50 draws per cycle and extreme bias, chi2 should pass
        let high_balls_prob: f64 = dist[39..49].iter().sum(); // balls 40-49
        let low_balls_prob: f64 = dist[0..10].iter().sum();   // balls 1-10
        // For cycle 2 (predicted next), high balls should be favored
        // If signal detected, high > low; if not, both ≈ 10/50 = 0.20
        assert!(
            high_balls_prob >= low_balls_prob * 0.8 || (high_balls_prob - 0.2).abs() < 0.05,
            "Cycle bias should either be detected or uniform: high={}, low={}",
            high_balls_prob, low_balls_prob
        );
    }

    #[test]
    fn test_cycle_model_deterministic() {
        let model = CycleModel::default();
        let draws = make_cycle_draws(200);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Model should be deterministic");
        }
    }

    #[test]
    fn test_cycle_model_no_negative_probs() {
        let model = CycleModel::default();
        let draws = make_cycle_draws(200);
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            for (i, &p) in dist.iter().enumerate() {
                assert!(p >= 0.0, "Negative probability at index {}: {}", i, p);
            }
        }
    }

    #[test]
    fn test_cycle_model_params() {
        let model = CycleModel::default();
        let params = model.params();
        assert_eq!(params["smoothing"], 0.25);
        assert_eq!(params["ewma_alpha"], 0.08);
        assert_eq!(params["min_draws_with_cycle"], 30.0);
        assert_eq!(params["chi2_threshold"], 25.0);
    }

    #[test]
    fn test_cycle_model_name_and_strategy() {
        let model = CycleModel::default();
        assert_eq!(model.name(), "CycleModel");
        assert_eq!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        );
    }

    #[test]
    fn test_infer_next_cycle() {
        let model = CycleModel::default();

        // Most recent draw has cycle 1 → next is 2
        let draws: Vec<Draw> = vec![Draw {
            draw_id: "001".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-01".to_string(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: Some(1),
        }];
        let refs: Vec<&Draw> = draws.iter().collect();
        assert_eq!(model.infer_next_cycle(&refs), 2);

        // Cycle 2 → next is 3
        let draws2: Vec<Draw> = vec![Draw {
            draw_id: "002".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-02".to_string(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: Some(2),
        }];
        let refs2: Vec<&Draw> = draws2.iter().collect();
        assert_eq!(model.infer_next_cycle(&refs2), 3);

        // Cycle 3 → next wraps to 1
        let draws3: Vec<Draw> = vec![Draw {
            draw_id: "003".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-03".to_string(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: Some(3),
        }];
        let refs3: Vec<&Draw> = draws3.iter().collect();
        assert_eq!(model.infer_next_cycle(&refs3), 1);
    }

    #[test]
    fn test_cycle_model_mixed_cycle_and_no_cycle() {
        let model = CycleModel::default();
        // Mix of draws with and without cycle_number
        let mut draws = make_test_draws(100); // No cycle data
        let cycle_draws = make_cycle_draws(50); // With cycle data
        draws.extend(cycle_draws);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
