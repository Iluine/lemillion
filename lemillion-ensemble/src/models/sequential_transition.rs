use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS};

/// SequentialTransition — Matrice de transition séquentielle 50×50.
///
/// Chaîne de Markov positionnelle : P(num en pos_{k+1} | num en pos_k).
/// DrawOrder utilise les positions indépendamment, pas les transitions.
///
/// Algorithme :
/// 1. Matrice de transition T[a][b] Laplace-smoothed
/// 2. 4 transitions intra-tirage (pos 1→2, 2→3, 3→4, 4→5)
/// 3. Transition inter-tirage : dernier extrait → prochain tirage pos 1
/// 4. Combinaison pondérée par entropie inverse
pub struct SequentialTransitionModel {
    smoothing: f64,
    min_draws_with_order: usize,
    laplace_alpha: f64,
}

impl Default for SequentialTransitionModel {
    fn default() -> Self {
        Self {
            smoothing: 0.30,
            min_draws_with_order: 50,
            laplace_alpha: 0.5,
        }
    }
}

impl ForecastModel for SequentialTransitionModel {
    fn name(&self) -> &str {
        "SeqTransition"
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

        // Build transition matrix T[from][to] from intra-draw sequences
        // Laplace-smoothed: T[a][b] = (count[a][b] + alpha) / (total[a] + alpha * size)
        let mut counts = vec![vec![0.0f64; size]; size];

        // Process chronologically (oldest first)
        for draw in draws_with_order.iter().rev() {
            let order = draw.ball_order.unwrap();

            // Intra-draw transitions: pos k → pos k+1
            for k in 0..(pick - 1) {
                let from = (order[k] - 1) as usize;
                let to = (order[k + 1] - 1) as usize;
                if from < size && to < size {
                    counts[from][to] += 1.0;
                }
            }
        }

        // Also build inter-draw transition: last extracted → next draw first extracted
        let mut inter_counts = vec![vec![0.0f64; size]; size];
        for w in draws_with_order.windows(2).rev() {
            // w[0] is more recent, w[1] is older
            // Inter-draw: older.last → newer.first
            let older = w[1];
            let newer = w[0];
            let older_order = older.ball_order.unwrap();
            let newer_order = newer.ball_order.unwrap();
            let from = (older_order[pick - 1] - 1) as usize;
            let to = (newer_order[0] - 1) as usize;
            if from < size && to < size {
                inter_counts[from][to] += 1.0;
            }
        }

        // Normalize transition matrices with Laplace smoothing
        let normalize_row = |row: &[f64], alpha: f64, n: usize| -> Vec<f64> {
            let total: f64 = row.iter().sum();
            let denom = total + alpha * n as f64;
            if denom < 1e-15 {
                return vec![1.0 / n as f64; n];
            }
            row.iter().map(|&c| (c + alpha) / denom).collect()
        };

        // Build normalized transition matrices
        let trans: Vec<Vec<f64>> = (0..size)
            .map(|from| normalize_row(&counts[from], self.laplace_alpha, size))
            .collect();
        let inter_trans: Vec<Vec<f64>> = (0..size)
            .map(|from| normalize_row(&inter_counts[from], self.laplace_alpha, size))
            .collect();

        // Use the most recent draw to predict next draw
        let last_draw = draws_with_order[0]; // most recent
        let last_order = last_draw.ball_order.unwrap();

        // Collect distributions from each transition source
        let mut distributions: Vec<Vec<f64>> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();

        // 4 intra-draw transition distributions (from the last draw)
        // What follows each extracted ball? → prediction for next draw
        for k in 0..(pick - 1) {
            let from = (last_order[k] - 1) as usize;
            if from < size {
                distributions.push(trans[from].clone());
                // Weight by inverse entropy (more concentrated = more informative)
                let h = entropy(&trans[from]);
                let h_max = (size as f64).ln();
                let w = 1.0 / (0.5 + h / h_max);
                weights.push(w);
            }
        }

        // Inter-draw transition: last extracted (pos 5) → next draw pos 1
        let last_extracted = (last_order[pick - 1] - 1) as usize;
        if last_extracted < size {
            distributions.push(inter_trans[last_extracted].clone());
            let h = entropy(&inter_trans[last_extracted]);
            let h_max = (size as f64).ln();
            let w = 1.0 / (0.5 + h / h_max);
            weights.push(w * 1.5); // Slightly higher weight for inter-draw
        }

        if distributions.is_empty() {
            return vec![uniform; size];
        }

        // Weighted combination
        let w_total: f64 = weights.iter().sum();
        let mut probs = vec![0.0f64; size];
        for (dist, w) in distributions.iter().zip(weights.iter()) {
            let nw = w / w_total;
            for (num, &p) in dist.iter().enumerate() {
                probs[num] += nw * p;
            }
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
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws_with_order".to_string(), self.min_draws_with_order as f64);
        m.insert("laplace_alpha".to_string(), self.laplace_alpha);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

/// Shannon entropy of a probability distribution.
fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| -p * p.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    fn make_draws_with_order(n: usize) -> Vec<Draw> {
        (0..n)
            .map(|i| {
                let base = (i % 8) as u8;
                let b1 = (base * 5 + 1).clamp(1, 50);
                let b2 = (base * 5 + 7).clamp(1, 50);
                let b3 = (base * 5 + 15).clamp(1, 50);
                let b4 = (base * 5 + 30).clamp(1, 50);
                let b5 = (base * 5 + 45).clamp(1, 50);
                Draw {
                    draw_id: format!("{:03}", i),
                    day: if i % 2 == 0 { "MARDI".to_string() } else { "VENDREDI".to_string() },
                    date: format!("2024-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1),
                    balls: {
                        let mut b = [b1, b2, b3, b4, b5];
                        b.sort();
                        b
                    },
                    stars: [((base % 6) + 1).min(12), ((base % 6) + 7).min(12)],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: Some([b1, b2, b3, b4, b5]),
                    star_order: Some([((base % 6) + 1).min(12), ((base % 6) + 7).min(12)]),
                    cycle_number: None,
        prize_tiers: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_seq_transition_valid_distribution() {
        let model = SequentialTransitionModel::default();
        let draws = make_draws_with_order(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
        assert!(dist.iter().all(|&p| p >= 0.0), "All probs should be non-negative");
    }

    #[test]
    fn test_seq_transition_returns_uniform_for_stars() {
        let model = SequentialTransitionModel::default();
        let draws = make_draws_with_order(100);
        let dist = model.predict(&draws, Pool::Stars);
        let uniform = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform for stars");
        }
    }

    #[test]
    fn test_seq_transition_without_order_returns_uniform() {
        let model = SequentialTransitionModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform without order data");
        }
    }

    #[test]
    fn test_seq_transition_few_draws_returns_uniform() {
        let model = SequentialTransitionModel::default();
        let draws = make_draws_with_order(20);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Too few draws should be uniform");
        }
    }

    #[test]
    fn test_seq_transition_deterministic() {
        let model = SequentialTransitionModel::default();
        let draws = make_draws_with_order(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Should be deterministic");
        }
    }

    #[test]
    fn test_seq_transition_with_consistent_pattern() {
        // Same sequence every draw: 1→10→20→30→40
        // Transition matrix should show strong 1→10, 10→20, etc.
        let model = SequentialTransitionModel::default();
        let draws: Vec<Draw> = (0..100)
            .map(|i| Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 10, 20, 30, 40],
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: Some([1, 10, 20, 30, 40]),
                star_order: Some([3, 10]),
                cycle_number: None,
        prize_tiers: None,
            })
            .collect();

        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_entropy_uniform() {
        let n = 50;
        let uniform = vec![1.0 / n as f64; n];
        let h = entropy(&uniform);
        let expected = (n as f64).ln();
        assert!((h - expected).abs() < 1e-9);
    }
}
