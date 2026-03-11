use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// PositionMod8 — Position d'extraction × Mod-8 pour les boules.
///
/// La machine Stresa a 8 barres annulaires créant une symétrie mod-8.
/// `ball_order` donne l'ordre d'extraction physique (1ère à 5ème boule extraite).
///
/// Signal: Si la barre #k alimente préférentiellement la position d'extraction #p,
/// alors P(ball mod 8 = k | position p) ≠ uniforme.
///
/// Algorithme:
/// 1. Pour chaque position p ∈ {1..5}: distribution EWMA de classes mod-8
/// 2. Transition intra-tirage: P(mod8 à pos p+1 | mod8 à pos p)
/// 3. Redistribution intra-classe via fréquence historique
/// 4. Moyenne pondérée par informativeness (entropie inverse)
pub struct PositionMod8Model {
    ewma_alpha: f64,
    smoothing: f64,
    min_draws_with_order: usize,
}

impl Default for PositionMod8Model {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.05,
            smoothing: 0.25,
            min_draws_with_order: 50,
        }
    }
}

const N_MOD: usize = 8;
const N_POS: usize = 5;
const BALL_SIZE: usize = 50;

/// Ball mod-8 class: (ball - 1) % 8
fn ball_mod8(ball: u8) -> usize {
    ((ball - 1) % N_MOD as u8) as usize
}

/// Shannon entropy of a distribution (nats)
fn entropy(probs: &[f64]) -> f64 {
    probs.iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| -p * p.ln())
        .sum()
}

impl ForecastModel for PositionMod8Model {
    fn name(&self) -> &str {
        "PositionMod8"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Only for balls
        if pool == Pool::Stars {
            return uniform;
        }

        // Filter draws with ball_order
        let draws_with_order: Vec<&Draw> = draws.iter()
            .filter(|d| d.ball_order.is_some())
            .collect();

        if draws_with_order.len() < self.min_draws_with_order {
            return uniform;
        }

        // ── Signal 1: Positional EWMA mod-8 distribution ──
        // For each extraction position, track EWMA of mod-8 class distribution
        let mut pos_mod8 = vec![vec![1.0 / N_MOD as f64; N_MOD]; N_POS];

        // Process chronologically (oldest first)
        for &draw in draws_with_order.iter().rev() {
            let order = draw.ball_order.as_ref().unwrap();
            for (pos, &ball) in order.iter().enumerate() {
                if pos < N_POS && ball >= 1 && ball <= BALL_SIZE as u8 {
                    let cls = ball_mod8(ball);
                    for c in 0..N_MOD {
                        if c == cls {
                            pos_mod8[pos][c] = (1.0 - self.ewma_alpha) * pos_mod8[pos][c]
                                + self.ewma_alpha * 1.0;
                        } else {
                            pos_mod8[pos][c] = (1.0 - self.ewma_alpha) * pos_mod8[pos][c];
                        }
                    }
                    // Renormalize
                    let sum: f64 = pos_mod8[pos].iter().sum();
                    if sum > 0.0 {
                        for p in &mut pos_mod8[pos] {
                            *p /= sum;
                        }
                    }
                }
            }
        }

        // ── Signal 2: Intra-draw mod-8 transition ──
        // P(mod8 at pos p+1 | mod8 at pos p)
        let laplace = 0.5;
        let mut transition = vec![vec![laplace; N_MOD]; N_MOD];
        let mut trans_totals = vec![laplace * N_MOD as f64; N_MOD];

        for &draw in &draws_with_order {
            let order = draw.ball_order.as_ref().unwrap();
            for w in order.windows(2) {
                if w[0] >= 1 && w[1] >= 1 && w[0] <= BALL_SIZE as u8 && w[1] <= BALL_SIZE as u8 {
                    let from = ball_mod8(w[0]);
                    let to = ball_mod8(w[1]);
                    transition[from][to] += 1.0;
                    trans_totals[from] += 1.0;
                }
            }
        }

        // Normalize transition
        for c in 0..N_MOD {
            if trans_totals[c] > 0.0 {
                for t in &mut transition[c] {
                    *t /= trans_totals[c];
                }
            }
        }

        // ── Combine: weighted by informativeness ──
        // Weight each position by inverse entropy (more concentrated = more informative)
        let max_entropy = (N_MOD as f64).ln();
        let mut pos_weights = vec![0.0f64; N_POS];
        for pos in 0..N_POS {
            let h = entropy(&pos_mod8[pos]);
            pos_weights[pos] = (max_entropy - h).max(0.01); // inverse entropy, min floor
        }
        let pw_sum: f64 = pos_weights.iter().sum();
        if pw_sum > 0.0 {
            for w in &mut pos_weights {
                *w /= pw_sum;
            }
        }

        // Combined mod-8 prediction: weighted average of positional distributions
        let mut mod8_pred = vec![0.0f64; N_MOD];
        for pos in 0..N_POS {
            for c in 0..N_MOD {
                mod8_pred[c] += pos_weights[pos] * pos_mod8[pos][c];
            }
        }

        // Apply transition from last draw's last extracted ball
        let last_draw = &draws_with_order[0];
        let last_order = last_draw.ball_order.as_ref().unwrap();
        if let Some(&last_ball) = last_order.last() {
            if last_ball >= 1 && last_ball <= BALL_SIZE as u8 {
                let last_class = ball_mod8(last_ball);
                let mut trans_pred = vec![0.0f64; N_MOD];
                for c in 0..N_MOD {
                    trans_pred[c] = transition[last_class][c];
                }
                // Blend: 70% positional + 30% transition
                for c in 0..N_MOD {
                    mod8_pred[c] = 0.7 * mod8_pred[c] + 0.3 * trans_pred[c];
                }
            }
        }

        // Normalize mod8_pred
        let sum: f64 = mod8_pred.iter().sum();
        if sum > 0.0 {
            for p in &mut mod8_pred {
                *p /= sum;
            }
        }

        // ── Redistribute to individual balls ──
        // Historical frequency per ball
        let mut freq = vec![1.0f64; BALL_SIZE]; // Laplace
        for &draw in &draws_with_order {
            for &b in &draw.balls {
                if b >= 1 && b <= BALL_SIZE as u8 {
                    freq[(b - 1) as usize] += 1.0;
                }
            }
        }

        // Class members and frequency sums
        let mut class_sum = vec![0.0f64; N_MOD];
        for (k, &f) in freq.iter().enumerate() {
            class_sum[k % N_MOD] += f;
        }

        // Redistribute
        let mut prob = vec![0.0f64; BALL_SIZE];
        for (k, p) in prob.iter_mut().enumerate() {
            let c = k % N_MOD;
            if class_sum[c] > 0.0 {
                *p = mod8_pred[c] * freq[k] / class_sum[c];
            }
        }

        // Normalize
        let sum: f64 = prob.iter().sum();
        if sum > 0.0 {
            for p in &mut prob {
                *p /= sum;
            }
        }

        // Smooth with uniform
        let uniform_val = 1.0 / BALL_SIZE as f64;
        for p in &mut prob {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Final normalize
        let sum: f64 = prob.iter().sum();
        if sum > 0.0 {
            for p in &mut prob {
                *p /= sum;
            }
        }

        prob
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("ewma_alpha".into(), self.ewma_alpha),
            ("smoothing".into(), self.smoothing),
            ("min_draws_with_order".into(), self.min_draws_with_order as f64),
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

    fn make_test_draws_with_ball_order(n: usize) -> Vec<Draw> {
        (0..n)
            .map(|i| {
                let base = (i % 10) as u8;
                let balls = [
                    (base * 5 + 1).clamp(1, 50),
                    (base * 5 + 2).clamp(1, 50),
                    (base * 5 + 3).clamp(1, 50),
                    (base * 5 + 4).clamp(1, 50),
                    (base * 5 + 5).clamp(1, 50),
                ];
                Draw {
                    draw_id: format!("{:03}", i),
                    day: if i % 2 == 0 { "MARDI".to_string() } else { "VENDREDI".to_string() },
                    date: format!("2024-01-{:02}", (i % 28) + 1),
                    balls,
                    stars: [
                        (base % 12 + 1).min(12),
                        ((base + 1) % 12 + 1).min(12),
                    ],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: Some(balls),
                    star_order: None,
                    cycle_number: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_position_mod8_valid_distribution() {
        let model = PositionMod8Model::default();
        let draws = make_test_draws_with_ball_order(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_position_mod8_stars_uniform() {
        let model = PositionMod8Model::default();
        let draws = make_test_draws_with_ball_order(100);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_position_mod8_no_order_uniform() {
        let model = PositionMod8Model::default();
        let draws = make_test_draws(100); // No ball_order
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_position_mod8_few_draws_uniform() {
        let model = PositionMod8Model::default();
        let draws = make_test_draws_with_ball_order(10);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_position_mod8_no_negative() {
        let model = PositionMod8Model::default();
        let draws = make_test_draws_with_ball_order(100);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_position_mod8_deterministic() {
        let model = PositionMod8Model::default();
        let draws = make_test_draws_with_ball_order(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_ball_mod8_classes() {
        assert_eq!(ball_mod8(1), 0); // (1-1)%8 = 0
        assert_eq!(ball_mod8(9), 0); // (9-1)%8 = 0
        assert_eq!(ball_mod8(2), 1); // (2-1)%8 = 1
        assert_eq!(ball_mod8(50), 1); // (50-1)%8 = 1
    }
}
