use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// DrawOrderModel — Exploite l'ordre d'extraction physique des boules/étoiles.
///
/// La machine Stresa a une trappe gravitaire à la base. Les boules physiquement proches
/// de la trappe sont extraites en premier. Le brevet US6145836A montre 3 barres centrales
/// + 8 barres annulaires créant des zones de mélange asymétriques.
///
/// v11: Matrice positionnelle complète N×pick au lieu du split binaire early/late.
/// P(ball k) = somme pondérée sur les positions, chaque position pondérée par
/// son information (inverse entropie normalisée).
pub struct DrawOrderModel {
    ewma_alpha: f64,
    smoothing: f64,
    min_draws_with_order: usize,
}

impl Default for DrawOrderModel {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.05,
            smoothing: 0.25,
            min_draws_with_order: 50,
        }
    }
}

impl ForecastModel for DrawOrderModel {
    fn name(&self) -> &str {
        "DrawOrder"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let pick = pool.pick_count();
        let uniform = 1.0 / size as f64;

        // Filtrer les draws qui ont des données d'ordre
        let draws_with_order: Vec<&Draw> = draws.iter()
            .filter(|d| match pool {
                Pool::Balls => d.ball_order.is_some(),
                Pool::Stars => d.star_order.is_some(),
            })
            .collect();

        if draws_with_order.len() < self.min_draws_with_order {
            return vec![uniform; size];
        }

        // v11: Matrice positionnelle complète freq[number][position]
        let mut freq = vec![vec![0.0f64; pick]; size];
        let mut totals = vec![0.0f64; pick];

        for (t, draw) in draws_with_order.iter().enumerate() {
            let weight = (-self.ewma_alpha * t as f64).exp();

            let order = match pool {
                Pool::Balls => {
                    let order = draw.ball_order.unwrap();
                    order.to_vec()
                }
                Pool::Stars => {
                    let order = draw.star_order.unwrap();
                    order.to_vec()
                }
            };

            // order[pos] = numéro extrait en position pos
            for (pos, &num) in order.iter().enumerate() {
                let idx = (num - 1) as usize;
                if idx < size && pos < pick {
                    freq[idx][pos] += weight;
                    totals[pos] += weight;
                }
            }
        }

        // Poids par position basé sur l'information (inverse entropie normalisée)
        // Position plus concentrée = plus informative = poids plus fort
        let h_max = (size as f64).ln();
        let position_weights: Vec<f64> = (0..pick).map(|pos| {
            if totals[pos] < 5.0 { return 1.0; }
            let h: f64 = (0..size).map(|k| {
                let p = freq[k][pos] / totals[pos];
                if p > 1e-15 { -p * p.ln() } else { 0.0 }
            }).sum();
            1.0 / (0.5 + h / h_max)  // Position concentrée = poids fort
        }).collect();
        let pw_total: f64 = position_weights.iter().sum();

        // P(ball k) = somme pondérée sur les positions
        let mut probs = vec![0.0f64; size];
        for num in 0..size {
            for pos in 0..pick {
                if totals[pos] > 1.0 {
                    probs[num] += position_weights[pos] / pw_total
                        * freq[num][pos] / totals[pos];
                }
            }
        }

        // Smoothing vers uniforme
        for p in &mut probs {
            *p = *p * (1.0 - self.smoothing) + uniform * self.smoothing;
        }

        let floor = match pool {
            Pool::Balls => PROB_FLOOR_BALLS,
            Pool::Stars => PROB_FLOOR_STARS,
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("ewma_alpha".to_string(), self.ewma_alpha);
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws_with_order".to_string(), self.min_draws_with_order as f64);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_draw_order_valid_distribution() {
        let model = DrawOrderModel::default();
        let draws = make_test_draws(100);
        // make_test_draws has ball_order: None, so should return uniform
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_draw_order_without_order_returns_uniform() {
        let model = DrawOrderModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform without order data");
        }
    }

    #[test]
    fn test_draw_order_detects_bias() {
        let model = DrawOrderModel::default();
        let draws: Vec<Draw> = (0..100).map(|i| {
            let base = (i % 10) as u8;
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 7, 15, 30, 45],
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                // Ball 7 toujours en position 1 (précoce)
                ball_order: Some([7, 1, 15, 30, 45]),
                star_order: Some([3, 10]),
                cycle_number: Some((base + 1).min(5)),
            }
        }).collect();

        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);

        // Ball 7 (index 6) should have elevated probability due to consistent early extraction
        let uniform = 1.0 / 50.0;
        let p7 = dist[6]; // ball 7 = index 6
        assert!(p7 > uniform, "Ball 7 (always position 1) should have P > uniform: {} vs {}", p7, uniform);
    }

    #[test]
    fn test_draw_order_few_draws_returns_uniform() {
        let model = DrawOrderModel::default();
        let draws: Vec<Draw> = (0..20).map(|i| {
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 7, 15, 30, 45],
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: Some([7, 1, 15, 30, 45]),
                star_order: Some([3, 10]),
                cycle_number: None,
            }
        }).collect();

        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Too few draws, should be uniform");
        }
    }

    #[test]
    fn test_draw_order_position_conditional() {
        // Ball 1 always extracted first → should have highest probability
        let model = DrawOrderModel::default();
        let draws: Vec<Draw> = (0..100).map(|i| {
            Draw {
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
            }
        }).collect();

        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // All 5 balls should have elevated probability
        assert!(dist[0] > 1.0 / 50.0); // ball 1
        assert!(dist[9] > 1.0 / 50.0); // ball 10
    }

    #[test]
    fn test_draw_order_position_weights() {
        // With varied extraction orders, position 1 should be more informative
        // when it consistently selects from a smaller set
        let model = DrawOrderModel::default();
        let draws: Vec<Draw> = (0..100).map(|i| {
            // Position 1 always ball 1, but later positions rotate
            let base = (i % 5) as u8;
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 10 + base, 20, 30, 40 + base],
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: Some([1, 10 + base, 20, 30, 40 + base]),
                star_order: Some([3, 10]),
                cycle_number: None,
            }
        }).collect();

        let dist = model.predict(&draws, Pool::Balls);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Ball 1 (always position 1) should dominate
        let p1 = dist[0];
        let uniform = 1.0 / 50.0;
        assert!(p1 > uniform, "Ball 1 (always first) should be above uniform");
    }

    #[test]
    fn test_draw_order_deterministic() {
        let model = DrawOrderModel::default();
        let draws: Vec<Draw> = (0..80).map(|i| {
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 7, 15, 30, 45],
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: Some([7, 1, 15, 30, 45]),
                star_order: Some([3, 10]),
                cycle_number: None,
            }
        }).collect();

        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Should be deterministic");
        }
    }

    #[test]
    fn test_draw_order_stars() {
        let model = DrawOrderModel::default();
        let draws: Vec<Draw> = (0..80).map(|i| {
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 7, 15, 30, 45],
                stars: [3, 10],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: Some([7, 1, 15, 30, 45]),
                star_order: Some([10, 3]),  // star 10 always first
                cycle_number: None,
            }
        }).collect();

        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
