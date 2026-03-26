use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, filter_star_era};

/// PaqueretteMod4 — Position d'extraction × Mod-4 pour les étoiles.
///
/// La Pâquerette utilise 4 pales (paddles), créant une symétrie mod-4 pour les étoiles.
/// Ce modèle combine:
/// 1. Fréquence positionnelle EWMA: P(star | position d'extraction 1ère ou 2ème)
/// 2. Transition mod-4 intra-tirage: P(star mod 4 à pos 2 | star mod 4 à pos 1)
///
/// Retourne uniforme pour Pool::Balls (star-only model).
pub struct PaqueretteMod4Model {
    ewma_alpha: f64,
    smoothing: f64,
    min_draws_with_order: usize,
}

impl Default for PaqueretteMod4Model {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.08,
            smoothing: 0.30,
            min_draws_with_order: 30,
        }
    }
}

impl ForecastModel for PaqueretteMod4Model {
    fn name(&self) -> &str {
        "PaqueretteMod4"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if pool == Pool::Balls {
            return uniform;
        }

        // Filter to current star era
        let era_draws = filter_star_era(draws);

        // Only keep draws with star_order data
        let draws_with_order: Vec<&Draw> = era_draws.iter()
            .filter(|d| d.star_order.is_some())
            .collect();

        if draws_with_order.len() < self.min_draws_with_order {
            return uniform;
        }

        // ── Signal 1: Positional EWMA frequency ──
        // For each position (1st extracted, 2nd extracted), track EWMA of each star
        let n_pos = 2usize;
        let mut pos_freq = vec![vec![1.0 / size as f64; size]; n_pos];

        // Process chronologically (oldest first)
        for &draw in draws_with_order.iter().rev() {
            let order = draw.star_order.as_ref().unwrap();
            for (pos, &star) in order.iter().enumerate() {
                if pos < n_pos && star >= 1 && star <= size as u8 {
                    let idx = (star - 1) as usize;
                    for j in 0..size {
                        if j == idx {
                            pos_freq[pos][j] = (1.0 - self.ewma_alpha) * pos_freq[pos][j]
                                + self.ewma_alpha * 1.0;
                        } else {
                            pos_freq[pos][j] = (1.0 - self.ewma_alpha) * pos_freq[pos][j];
                        }
                    }
                    // Renormalize
                    let sum: f64 = pos_freq[pos].iter().sum();
                    if sum > 0.0 {
                        for p in &mut pos_freq[pos] {
                            *p /= sum;
                        }
                    }
                }
            }
        }

        // Combine positional frequencies: average over positions
        let mut pos_scores = vec![0.0f64; size];
        for j in 0..size {
            for pos in 0..n_pos {
                pos_scores[j] += pos_freq[pos][j];
            }
            pos_scores[j] /= n_pos as f64;
        }

        // ── Signal 2: KL-weighted multi-modular analysis (v19 C3) ──
        // Test mod-2 (even/odd), mod-3, mod-4 (Pâquerette blades), mod-6
        // Weight each by KL-divergence from uniform (more informative = higher weight)

        // Historical frequency for redistribution
        let mut freq = vec![1.0f64; size]; // Laplace prior
        for &draw in &draws_with_order {
            for &s in &draw.stars {
                if s >= 1 && s <= size as u8 {
                    freq[(s - 1) as usize] += 1.0;
                }
            }
        }

        let last_draw = &draws_with_order[0];
        let last_order = last_draw.star_order.as_ref().unwrap();

        let moduli = [2usize, 3, 4, 6];
        let mut mod_scores_all: Vec<(Vec<f64>, f64)> = Vec::new(); // (scores, kl_divergence)

        for &modulus in &moduli {
            let laplace = 0.5;
            let mut transition = vec![vec![laplace; modulus]; modulus];
            let mut totals = vec![laplace * modulus as f64; modulus];

            for &draw in &draws_with_order {
                let order = draw.star_order.as_ref().unwrap();
                if order.len() >= 2 && order[0] >= 1 && order[1] >= 1 {
                    let from_class = ((order[0] - 1) as usize) % modulus;
                    let to_class = ((order[1] - 1) as usize) % modulus;
                    transition[from_class][to_class] += 1.0;
                    totals[from_class] += 1.0;
                }
            }

            // Normalize transition rows
            for c in 0..modulus {
                if totals[c] > 0.0 {
                    for t in &mut transition[c] {
                        *t /= totals[c];
                    }
                }
            }

            // Predict class distribution based on last draw
            let mut class_pred = vec![0.0f64; modulus];
            if !last_order.is_empty() && last_order[0] >= 1 {
                let last_class = ((last_order[0] - 1) as usize) % modulus;
                for c in 0..modulus {
                    class_pred[c] = transition[last_class][c];
                }
            } else {
                for c in 0..modulus {
                    class_pred[c] = 1.0 / modulus as f64;
                }
            }

            // KL divergence from uniform
            let uniform_class = 1.0 / modulus as f64;
            let kl: f64 = class_pred.iter()
                .map(|&p| {
                    let p = p.max(1e-15);
                    p * (p / uniform_class).ln()
                })
                .sum();

            // Redistribute to individual stars using within-class frequency
            let mut class_members: Vec<Vec<usize>> = vec![vec![]; modulus];
            for s in 0..size {
                let c = s % modulus;
                class_members[c].push(s);
            }
            let mut class_sum = vec![0.0f64; modulus];
            for c in 0..modulus {
                for &idx in &class_members[c] {
                    class_sum[c] += freq[idx];
                }
            }

            let mut scores = vec![0.0f64; size];
            for c in 0..modulus {
                if class_sum[c] > 0.0 {
                    for &idx in &class_members[c] {
                        scores[idx] = class_pred[c] * freq[idx] / class_sum[c];
                    }
                }
            }

            // Normalize
            let sum: f64 = scores.iter().sum();
            if sum > 0.0 {
                for p in &mut scores {
                    *p /= sum;
                }
            }

            mod_scores_all.push((scores, kl.max(1e-10)));
        }

        // KL-weighted blend of modular scores
        let total_kl: f64 = mod_scores_all.iter().map(|(_, kl)| kl).sum();
        let mut mod_combined = vec![0.0f64; size];
        if total_kl > 0.0 {
            for (scores, kl) in &mod_scores_all {
                let w = kl / total_kl;
                for j in 0..size {
                    mod_combined[j] += w * scores[j];
                }
            }
        } else {
            for j in 0..size {
                mod_combined[j] = 1.0 / size as f64;
            }
        }

        // Normalize mod_combined
        let sum: f64 = mod_combined.iter().sum();
        if sum > 0.0 {
            for p in &mut mod_combined {
                *p /= sum;
            }
        }

        // ── Blend: 60% positional + 40% KL-weighted modular ──
        let blend_pos = 0.6;
        let blend_mod = 0.4;
        let mut combined = vec![0.0f64; size];
        for j in 0..size {
            combined[j] = blend_pos * pos_scores[j] + blend_mod * mod_combined[j];
        }

        // Smooth with uniform
        let uniform_val = 1.0 / size as f64;
        for p in &mut combined {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Normalize
        let sum: f64 = combined.iter().sum();
        if sum > 0.0 {
            for p in &mut combined {
                *p /= sum;
            }
        }

        combined
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("ewma_alpha".into(), self.ewma_alpha),
            ("smoothing".into(), self.smoothing),
            ("min_draws_with_order".into(), self.min_draws_with_order as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn is_stars_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    fn make_test_draws_with_star_order(n: usize) -> Vec<Draw> {
        (0..n)
            .map(|i| {
                let base = (i % 10) as u8;
                let s1 = (base % 12 + 1).min(12);
                let s2 = ((base + 1) % 12 + 1).min(12);
                let (s1_o, s2_o) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
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
                    stars: [s1_o, s2_o],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: None,
                    star_order: Some([s1, s2]),
                    cycle_number: None,
        prize_tiers: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_paquerette_balls_uniform() {
        let model = PaqueretteMod4Model::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_paquerette_stars_valid() {
        let model = PaqueretteMod4Model::default();
        let draws = make_test_draws_with_star_order(80);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_paquerette_few_draws_uniform() {
        let model = PaqueretteMod4Model::default();
        let draws = make_test_draws_with_star_order(10);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_paquerette_no_order_uniform() {
        let model = PaqueretteMod4Model::default();
        let draws = make_test_draws(80); // No star_order
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_paquerette_no_negative() {
        let model = PaqueretteMod4Model::default();
        let draws = make_test_draws_with_star_order(80);
        let dist = model.predict(&draws, Pool::Stars);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_paquerette_deterministic() {
        let model = PaqueretteMod4Model::default();
        let draws = make_test_draws_with_star_order(80);
        let d1 = model.predict(&draws, Pool::Stars);
        let d2 = model.predict(&draws, Pool::Stars);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_star_mod_classes() {
        // (star-1) % modulus
        assert_eq!((1u8 - 1) as usize % 4, 0); // star 1 → mod4 class 0
        assert_eq!((2u8 - 1) as usize % 4, 1); // star 2 → mod4 class 1
        assert_eq!((5u8 - 1) as usize % 4, 0); // star 5 → mod4 class 0
        assert_eq!((12u8 - 1) as usize % 4, 3); // star 12 → mod4 class 3
        // mod-2 (even/odd)
        assert_eq!((1u8 - 1) as usize % 2, 0);
        assert_eq!((2u8 - 1) as usize % 2, 1);
        // mod-6
        assert_eq!((7u8 - 1) as usize % 6, 0);
        assert_eq!((12u8 - 1) as usize % 6, 5);
    }
}
