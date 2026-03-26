use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::stresa::{encode_all_states, chaos_takens_embed, euclidean_dist};
use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// TlrModel — Time-Lagged Recurrence (PNAS 2025).
///
/// Estime la prédictibilité locale de chaque état du système dynamique.
/// Encode les tirages en vecteurs d'état, effectue un embedding de Takens,
/// trouve les K plus proches voisins du point courant, et pondère les
/// prédictions par la qualité TLR (les voisins dont les trajectoires
/// convergent sont plus fiables).
pub struct TlrModel {
    embedding_dim: usize,
    delay: usize,
    k_neighbors: usize,
    smoothing: f64,
    min_draws: usize,
    theiler_window: usize,
}

impl Default for TlrModel {
    fn default() -> Self {
        Self {
            embedding_dim: 3,
            delay: 1,
            k_neighbors: 20,
            smoothing: 0.30,
            min_draws: 60,
            theiler_window: 3,
        }
    }
}

impl ForecastModel for TlrModel {
    fn name(&self) -> &str {
        "TLR"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;
        if draws.len() < self.min_draws {
            return vec![uniform; size];
        }

        // 1. Encoder tous les draws (oldest-first)
        let states = encode_all_states(draws, pool);
        let n = states.len();

        // 2. Takens embedding
        let embedded = chaos_takens_embed(&states, self.delay, self.embedding_dim);
        if embedded.len() < self.k_neighbors + 2 {
            return vec![uniform; size];
        }

        // 3. Query = dernier point (most recent in chronological order)
        let query_idx = embedded.len() - 1;
        let query = &embedded[query_idx];

        // 4. Trouver K voisins les plus proches (excluant fenêtre de Theiler)
        let mut distances: Vec<(usize, f64)> = (0..embedded.len())
            .filter(|&i| query_idx.abs_diff(i) > self.theiler_window)
            .map(|i| (i, euclidean_dist(&embedded[i], query)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors: Vec<(usize, f64)> = distances.into_iter()
            .take(self.k_neighbors).collect();

        if neighbors.is_empty() {
            return vec![uniform; size];
        }
        let max_dist = neighbors.last().unwrap().1.max(1e-15);

        // 5. TLR weight: vérifier si les trajectoires voisines convergent
        let mut probs = vec![0.0f64; size];
        let mut total_weight = 0.0f64;

        for &(idx, dist) in &neighbors {
            // Distance kernel (Gaussian-like)
            let proximity_weight = (-dist / max_dist).exp();

            // TLR: vérifier la cohérence temporelle pour lags 1-2
            let mut tlr_weight = 1.0f64;
            for lag in 1..=2usize {
                if idx + lag < embedded.len() && query_idx >= lag {
                    let forward_dist = euclidean_dist(
                        &embedded[idx + lag],
                        &embedded[query_idx - lag + 1],
                    );
                    tlr_weight *= (-forward_dist / (max_dist * 2.0)).exp();
                }
            }

            let weight = proximity_weight * tlr_weight;

            // Le draw qui SUIT le voisin est notre prédiction
            // embedded indices map to chronological order (oldest-first)
            // embedded[idx] corresponds to the state at chronological time idx
            // The successor draw is at chronological index idx+1
            let chrono_succ = idx + 1;
            if chrono_succ < n {
                // Convert chronological index to draws[] index (reverse order)
                let draws_idx = n - 1 - chrono_succ;
                if draws_idx < draws.len() {
                    let nums = pool.numbers_from(&draws[draws_idx]);
                    for &num in nums {
                        probs[(num - 1) as usize] += weight;
                    }
                    total_weight += weight;
                }
            }
        }

        // Normaliser + smooth
        if total_weight > 0.0 {
            for p in &mut probs {
                *p /= total_weight;
            }
        } else {
            return vec![uniform; size];
        }
        for p in &mut probs {
            *p = *p * (1.0 - self.smoothing) + uniform * self.smoothing;
        }

        let floor = if pool == Pool::Balls { PROB_FLOOR_BALLS } else { PROB_FLOOR_STARS };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("embedding_dim".to_string(), self.embedding_dim as f64);
        m.insert("delay".to_string(), self.delay as f64);
        m.insert("k_neighbors".to_string(), self.k_neighbors as f64);
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws".to_string(), self.min_draws as f64);
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
    fn test_tlr_valid_distribution() {
        let model = TlrModel::default();
        let draws = make_test_draws(100);
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            assert_eq!(dist.len(), pool.size());
            let sum: f64 = dist.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0 for {:?}, got {}", pool, sum);
            assert!(dist.iter().all(|&p| p >= 0.0), "All probs should be non-negative");
        }
    }

    #[test]
    fn test_tlr_insufficient_data_returns_uniform() {
        let model = TlrModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform with few draws");
        }
    }

    #[test]
    fn test_tlr_deterministic() {
        let model = TlrModel::default();
        let draws = make_test_draws(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Should be deterministic");
        }
    }

    #[test]
    fn test_tlr_with_real_draws() {
        // Varied data to exercise TLR neighbor search
        let draws: Vec<Draw> = (0..120).map(|i| {
            let base = ((i * 7 + 3) % 10) as u8;
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [
                    (base * 5 + 1).clamp(1, 50),
                    (base * 5 + 2).clamp(1, 50),
                    (base * 5 + 3).clamp(1, 50),
                    (base * 5 + 4).clamp(1, 50),
                    (base * 5 + 5).clamp(1, 50),
                ],
                stars: [(base % 12 + 1).min(12), ((base + 1) % 12 + 1).min(12)],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
        prize_tiers: None,
            }
        }).collect();

        let model = TlrModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "TLR should produce valid distribution");
        assert!(dist.iter().all(|&p| p >= 0.0), "All probs should be non-negative");
    }

    #[test]
    fn test_tlr_sampling_strategy() {
        let model = TlrModel::default();
        assert!(matches!(model.sampling_strategy(), SamplingStrategy::Sparse { .. }));
    }

    #[test]
    fn test_tlr_stars() {
        let model = TlrModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
