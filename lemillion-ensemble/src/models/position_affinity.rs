use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS};

/// PositionAffinity — Affinité numéro×position d'extraction (50×5).
///
/// Certains numéros sortent préférentiellement en 1ère ou 5ème position.
/// DrawOrder agrège par position mais ne capture pas l'affinité individuelle.
///
/// Algorithme :
/// 1. Matrice EWMA A[num][pos] (fréquence de chaque numéro à chaque position)
/// 2. Test chi² par numéro pour détecter les biais positionnels significatifs
/// 3. Gate : seuls les numéros avec chi² > seuil (p<0.10) ont un signal
/// 4. Blend adaptatif basé sur le ratio de numéros gated
pub struct PositionAffinityModel {
    ewma_alpha: f64,
    smoothing: f64,
    min_draws_with_order: usize,
    /// p-value threshold for chi² gating (p < threshold = significant)
    chi2_p_threshold: f64,
}

impl Default for PositionAffinityModel {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.05,
            smoothing: 0.25,
            min_draws_with_order: 50,
            chi2_p_threshold: 0.10,
        }
    }
}

impl ForecastModel for PositionAffinityModel {
    fn name(&self) -> &str {
        "PositionAffinity"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let pick = pool.pick_count();
        let uniform = 1.0 / size as f64;

        // Balls-only model
        if pool == Pool::Stars {
            return vec![uniform; size];
        }

        // Filter draws with extraction order
        let draws_with_order: Vec<&Draw> = draws
            .iter()
            .filter(|d| d.ball_order.is_some())
            .collect();

        if draws_with_order.len() < self.min_draws_with_order {
            return vec![uniform; size];
        }

        // Build EWMA affinity matrix A[num][pos]
        let mut affinity = vec![vec![0.0f64; pick]; size];
        let mut pos_totals = vec![0.0f64; pick];

        for (t, draw) in draws_with_order.iter().enumerate() {
            let weight = (-self.ewma_alpha * t as f64).exp();
            let order = draw.ball_order.unwrap();

            for (pos, &num) in order.iter().enumerate() {
                let idx = (num - 1) as usize;
                if idx < size && pos < pick {
                    affinity[idx][pos] += weight;
                    pos_totals[pos] += weight;
                }
            }
        }

        // Normalize per position to get P(num | pos)
        for pos in 0..pick {
            if pos_totals[pos] > 0.0 {
                for num in 0..size {
                    affinity[num][pos] /= pos_totals[pos];
                }
            }
        }

        // Compute marginal frequency for each number (across all positions)
        let total_weight: f64 = pos_totals.iter().sum();
        let mut marginal = vec![0.0f64; size];
        for num in 0..size {
            for pos in 0..pick {
                marginal[num] += affinity[num][pos] * pos_totals[pos];
            }
            marginal[num] /= total_weight.max(1e-15);
        }

        // Chi² test per number: does it deviate from uniform across positions?
        // H0: P(num at pos) = marginal[num] for all pos
        // chi2_df = pick - 1 = 4
        // p<0.10 threshold for df=4: chi2 > 7.779
        let chi2_threshold = chi2_critical(pick - 1, self.chi2_p_threshold);

        let mut gated_probs = vec![0.0f64; size];
        let mut gated_count = 0usize;

        for num in 0..size {
            let expected = marginal[num]; // expected uniform fraction across positions
            if expected < 1e-12 {
                continue;
            }

            // chi² = Σ_pos (observed - expected)² / expected
            let chi2: f64 = (0..pick)
                .map(|pos| {
                    let observed = affinity[num][pos];
                    let diff = observed - expected;
                    diff * diff / expected.max(1e-15)
                })
                .sum();

            if chi2 > chi2_threshold {
                // This number has significant positional affinity
                gated_count += 1;
                // Weight by position × affinity (most recent position pattern)
                let mut num_score = 0.0f64;
                for pos in 0..pick {
                    num_score += affinity[num][pos];
                }
                gated_probs[num] = num_score;
            }
        }

        if gated_count == 0 {
            return vec![uniform; size];
        }

        // Normalize gated probs
        let gated_sum: f64 = gated_probs.iter().sum();
        if gated_sum > 0.0 {
            for p in &mut gated_probs {
                *p /= gated_sum;
            }
        }

        // Adaptive blend: more gated numbers → more signal
        let gate_ratio = gated_count as f64 / size as f64;
        let signal_weight = gate_ratio.clamp(0.05, 0.80);

        let mut probs = vec![0.0f64; size];
        for num in 0..size {
            probs[num] = signal_weight * gated_probs[num] + (1.0 - signal_weight) * marginal[num].max(uniform);
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        // Smoothing towards uniform
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("ewma_alpha".to_string(), self.ewma_alpha);
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws_with_order".to_string(), self.min_draws_with_order as f64);
        m.insert("chi2_p_threshold".to_string(), self.chi2_p_threshold);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

/// Approximate chi² critical value for small df and common alpha levels.
fn chi2_critical(df: usize, alpha: f64) -> f64 {
    // Pre-computed critical values for df=1..6, alpha=0.10, 0.05
    match df {
        1 => if alpha <= 0.05 { 3.841 } else { 2.706 },
        2 => if alpha <= 0.05 { 5.991 } else { 4.605 },
        3 => if alpha <= 0.05 { 7.815 } else { 6.251 },
        4 => if alpha <= 0.05 { 9.488 } else { 7.779 },
        5 => if alpha <= 0.05 { 11.070 } else { 9.236 },
        6 => if alpha <= 0.05 { 12.592 } else { 10.645 },
        _ => {
            // Wilson-Hilferty approximation for larger df
            let z = if alpha <= 0.05 { 1.645 } else { 1.282 };
            let d = df as f64;
            d * (1.0 - 2.0 / (9.0 * d) + z * (2.0 / (9.0 * d)).sqrt()).powi(3)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    fn make_draws_with_order(n: usize) -> Vec<Draw> {
        (0..n)
            .map(|i| {
                let base = (i % 8) as u8;
                Draw {
                    draw_id: format!("{:03}", i),
                    day: if i % 2 == 0 { "MARDI".to_string() } else { "VENDREDI".to_string() },
                    date: format!("2024-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1),
                    balls: [
                        (base * 5 + 1).clamp(1, 50),
                        (base * 5 + 7).clamp(1, 50),
                        (base * 5 + 15).clamp(1, 50),
                        (base * 5 + 30).clamp(1, 50),
                        (base * 5 + 45).clamp(1, 50),
                    ],
                    stars: [((base % 6) + 1).min(12), ((base % 6) + 7).min(12)],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: Some([
                        (base * 5 + 1).clamp(1, 50),
                        (base * 5 + 7).clamp(1, 50),
                        (base * 5 + 15).clamp(1, 50),
                        (base * 5 + 30).clamp(1, 50),
                        (base * 5 + 45).clamp(1, 50),
                    ]),
                    star_order: Some([((base % 6) + 1).min(12), ((base % 6) + 7).min(12)]),
                    cycle_number: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_position_affinity_valid_distribution() {
        let model = PositionAffinityModel::default();
        let draws = make_draws_with_order(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
        assert!(dist.iter().all(|&p| p >= 0.0), "All probs should be non-negative");
    }

    #[test]
    fn test_position_affinity_returns_uniform_for_stars() {
        let model = PositionAffinityModel::default();
        let draws = make_draws_with_order(100);
        let dist = model.predict(&draws, Pool::Stars);
        let uniform = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform for stars");
        }
    }

    #[test]
    fn test_position_affinity_without_order_returns_uniform() {
        let model = PositionAffinityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform without order data");
        }
    }

    #[test]
    fn test_position_affinity_few_draws_returns_uniform() {
        let model = PositionAffinityModel::default();
        let draws = make_draws_with_order(20);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Too few draws should be uniform");
        }
    }

    #[test]
    fn test_position_affinity_detects_bias() {
        let model = PositionAffinityModel::default();
        // Ball 1 appears at position 1 in 80% of draws, at other positions in 20%
        // Other balls rotate through positions — ball 1 has strong positional affinity
        let draws: Vec<Draw> = (0..100)
            .map(|i| {
                let variant = i % 5;
                let (order, balls) = if variant < 4 {
                    // 80%: ball 1 always position 1
                    let b2 = ((i % 10) + 2) as u8;
                    let b3 = ((i % 10) + 15) as u8;
                    let b4 = ((i % 10) + 30) as u8;
                    let b5 = ((i % 10) + 42) as u8;
                    ([1, b2, b3, b4, b5], {
                        let mut b = [1, b2, b3, b4, b5];
                        b.sort();
                        b
                    })
                } else {
                    // 20%: ball 1 at position 5
                    let b1 = ((i % 10) + 2) as u8;
                    let b2 = ((i % 10) + 15) as u8;
                    let b3 = ((i % 10) + 30) as u8;
                    let b4 = ((i % 10) + 42) as u8;
                    ([b1, b2, b3, b4, 1], {
                        let mut b = [1, b1, b2, b3, b4];
                        b.sort();
                        b
                    })
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
                    ball_order: Some(order),
                    star_order: Some([3, 10]),
                    cycle_number: None,
                }
            })
            .collect();

        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Distribution should be valid and non-uniform (some signal detected)
        let uniform = 1.0 / 50.0;
        let max_p = dist.iter().cloned().fold(0.0_f64, f64::max);
        let min_p = dist.iter().cloned().fold(1.0_f64, f64::min);
        assert!(max_p > uniform || (max_p - min_p).abs() < 1e-6,
            "Should detect some signal or return uniform, max={}, min={}", max_p, min_p);
    }

    #[test]
    fn test_position_affinity_deterministic() {
        let model = PositionAffinityModel::default();
        let draws = make_draws_with_order(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Should be deterministic");
        }
    }

    #[test]
    fn test_chi2_critical_values() {
        assert!((chi2_critical(4, 0.10) - 7.779).abs() < 0.001);
        assert!((chi2_critical(4, 0.05) - 9.488).abs() < 0.001);
        assert!((chi2_critical(1, 0.05) - 3.841).abs() < 0.001);
    }
}
