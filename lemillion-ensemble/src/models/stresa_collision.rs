//! StresaCollisionModel — v22: Exploits the 3+8 bar counter-rotation structure
//! from patent US6145836A.
//!
//! The Stresa machine has:
//! - 3 central bars (inner rotor)
//! - 8 annular bars (outer ring), rotating in opposite direction
//! - Extraction via central tube (radial bias)
//!
//! Collision periodicity: lcm(3, 8) = 24 rotation cycles.
//! This model captures the mod-24 interaction pattern and the radial
//! extraction bias that existing mod-8 models miss.

use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};
use super::{ForecastModel, SamplingStrategy};

pub struct StresaCollisionModel {
    ewma_alpha: f64,
    smoothing: f64,
    mod24_weight: f64,
    radial_weight: f64,
}

impl Default for StresaCollisionModel {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.05,
            smoothing: 0.25,
            mod24_weight: 0.50,
            radial_weight: 0.30,
        }
    }
}

impl StresaCollisionModel {
    /// Compute the collision frequency profile under the 3+8 bar model.
    /// For a ball at position `num` (1-indexed), the collision rate depends on:
    /// - num % 3 (interaction with 3 central bars)
    /// - num % 8 (interaction with 8 annular bars)
    /// - The combined mod-24 phase = (num-1) % 24
    fn collision_profile(num: u8) -> (usize, usize, usize) {
        let n = (num - 1) as usize;
        (n % 3, n % 8, n % 24)
    }

    /// Compute radial zone: center (1), intermediate (2), or periphery (3)
    /// Based on decade: balls 1-10 and 41-50 are at the edges of the rack,
    /// balls 21-30 are at the center. This models the extraction bias.
    fn radial_zone(num: u8) -> usize {
        let decade = ((num - 1) / 10) as usize; // 0-4
        match decade {
            2 => 0,        // center (21-30): closest to extraction tube
            1 | 3 => 1,    // intermediate (11-20, 31-40)
            0 | 4 => 2,    // periphery (1-10, 41-50)
            _ => 1,
        }
    }
}

impl ForecastModel for StresaCollisionModel {
    fn name(&self) -> &str {
        "StresaCollision"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;
        let mut probs = vec![uniform; size];

        if draws.len() < 50 || pool == Pool::Stars {
            return probs;
        }

        // 1. Compute mod-24 frequency profile via EWMA
        let mut mod24_freq = [0.0f64; 24];
        let mut mod3_freq = [0.0f64; 3];
        let mut mod8_freq = [0.0f64; 8];
        let mut radial_freq = [0.0f64; 3]; // center, intermediate, periphery
        let mut total_weight = 0.0f64;

        for (i, draw) in draws.iter().enumerate() {
            let weight = (-self.ewma_alpha * i as f64).exp();
            let nums = pool.numbers_from(draw);
            for &num in nums {
                let (m3, m8, m24) = Self::collision_profile(num);
                let rz = Self::radial_zone(num);
                mod24_freq[m24] += weight;
                mod3_freq[m3] += weight;
                mod8_freq[m8] += weight;
                radial_freq[rz] += weight;
            }
            total_weight += weight * nums.len() as f64;
        }

        // Normalize to probability
        if total_weight > 0.0 {
            for v in &mut mod24_freq { *v /= total_weight; }
            for v in &mut mod3_freq { *v /= total_weight; }
            for v in &mut mod8_freq { *v /= total_weight; }
            for v in &mut radial_freq { *v /= total_weight; }
        }

        // Expected frequencies under uniform
        let expected_mod24 = 1.0 / 24.0;
        let expected_mod3 = 1.0 / 3.0;
        let expected_mod8 = 1.0 / 8.0;
        let expected_radial = [10.0 / 50.0, 20.0 / 50.0, 20.0 / 50.0]; // center=10, inter=20, peri=20

        // 2. Compute per-number bias from collision dynamics
        for num_idx in 0..size {
            let num = (num_idx + 1) as u8;
            let (m3, m8, m24) = Self::collision_profile(num);
            let rz = Self::radial_zone(num);

            // Mod-24 signal: ratio of observed to expected (combined 3×8 pattern)
            let mod24_ratio = if expected_mod24 > 0.0 {
                mod24_freq[m24] / expected_mod24
            } else { 1.0 };

            // Mod-3 signal (central bars only)
            let mod3_ratio = if expected_mod3 > 0.0 {
                mod3_freq[m3] / expected_mod3
            } else { 1.0 };

            // Mod-8 signal (annular bars only)
            let mod8_ratio = if expected_mod8 > 0.0 {
                mod8_freq[m8] / expected_mod8
            } else { 1.0 };

            // Radial extraction bias
            let radial_ratio = if expected_radial[rz] > 0.0 {
                radial_freq[rz] / expected_radial[rz]
            } else { 1.0 };

            // Combined score: weighted geometric mean of ratios
            // mod-24 captures the INTERACTION between 3 and 8 bars (the novel signal)
            // radial captures the extraction tube bias
            let log_score =
                self.mod24_weight * (mod24_ratio.max(0.1).ln())
                + (1.0 - self.mod24_weight - self.radial_weight) * 0.5 * (mod3_ratio.max(0.1).ln() + mod8_ratio.max(0.1).ln())
                + self.radial_weight * (radial_ratio.max(0.1).ln());

            // Bayesian shrinkage: blend toward 0 (uniform) when signal is weak
            let shrunk_score = log_score.clamp(-1.0, 1.0);
            probs[num_idx] = uniform * shrunk_score.exp();
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs { *p /= sum; }
        }

        // Smooth with uniform
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        // Final normalization
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs { *p /= sum; }
        }

        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("ewma_alpha".into(), self.ewma_alpha);
        m.insert("smoothing".into(), self.smoothing);
        m.insert("mod24_weight".into(), self.mod24_weight);
        m.insert("radial_weight".into(), self.radial_weight);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lemillion_db::models::Draw;

    fn make_draw(id: u32, balls: [u8; 5], stars: [u8; 2]) -> Draw {
        Draw {
            draw_id: id.to_string(),
            day: "MARDI".into(),
            date: "2025-01-01".into(),
            balls,
            stars,
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        prize_tiers: None,
        }
    }

    #[test]
    fn test_stresa_collision_output_valid() {
        let model = StresaCollisionModel::default();
        let draws: Vec<Draw> = (0..100).map(|i| {
            make_draw(i, [1 + (i % 46) as u8, 2 + (i % 46) as u8, 3 + (i % 46) as u8,
                          4 + (i % 46) as u8, 5 + (i % 46) as u8], [1, 2])
        }).collect();

        let probs = model.predict(&draws, Pool::Balls);
        assert_eq!(probs.len(), 50);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum should be 1.0, got {}", sum);
        assert!(probs.iter().all(|&p| p > 0.0), "All probabilities should be positive");
    }

    #[test]
    fn test_stresa_collision_stars_uniform() {
        let model = StresaCollisionModel::default();
        let draws: Vec<Draw> = (0..100).map(|i| {
            make_draw(i, [1, 2, 3, 4, 5], [1 + (i % 11) as u8, 2 + (i % 11) as u8])
        }).collect();

        let probs = model.predict(&draws, Pool::Stars);
        assert_eq!(probs.len(), 12);
        // Should be uniform for stars (model is balls-only)
        let expected = 1.0 / 12.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_collision_profile() {
        let (m3, m8, m24) = StresaCollisionModel::collision_profile(1);
        assert_eq!(m3, 0);
        assert_eq!(m8, 0);
        assert_eq!(m24, 0);

        let (m3, m8, m24) = StresaCollisionModel::collision_profile(25);
        assert_eq!(m3, 0); // (25-1) % 3 = 0
        assert_eq!(m8, 0); // (25-1) % 8 = 0
        assert_eq!(m24, 0); // (25-1) % 24 = 0
    }

    #[test]
    fn test_radial_zone() {
        assert_eq!(StresaCollisionModel::radial_zone(5), 2);  // 1-10: periphery
        assert_eq!(StresaCollisionModel::radial_zone(15), 1); // 11-20: intermediate
        assert_eq!(StresaCollisionModel::radial_zone(25), 0); // 21-30: center
        assert_eq!(StresaCollisionModel::radial_zone(35), 1); // 31-40: intermediate
        assert_eq!(StresaCollisionModel::radial_zone(45), 2); // 41-50: periphery
    }
}
