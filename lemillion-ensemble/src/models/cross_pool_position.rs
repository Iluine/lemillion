use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, filter_star_era, floor_only, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// CrossPoolPosition — Couplage cross-pool par position d'extraction.
///
/// Corrélation entre l'ordre d'extraction des boules et le résultat des étoiles.
/// Par exemple, si la somme des positions d'extraction des boules est élevée,
/// cela pourrait influencer l'état de la machine pour l'extraction étoiles.
///
/// Pour les étoiles : contexte basé sur sum_extraction_order des boules → 5 bins
/// Pour les boules : contexte basé sur les étoiles du tirage précédent → 4 bins
pub struct CrossPoolPositionModel {
    ewma_alpha: f64,
    smoothing: f64,
    min_draws_with_order: usize,
    n_bins: usize,
}

impl Default for CrossPoolPositionModel {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.05,
            smoothing: 0.30,
            min_draws_with_order: 50,
            n_bins: 5,
        }
    }
}

impl ForecastModel for CrossPoolPositionModel {
    fn name(&self) -> &str {
        "CrossPoolPos"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;

        match pool {
            Pool::Stars => self.predict_stars(draws, size, uniform),
            Pool::Balls => self.predict_balls(draws, size, uniform),
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("ewma_alpha".to_string(), self.ewma_alpha);
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws_with_order".to_string(), self.min_draws_with_order as f64);
        m.insert("n_bins".to_string(), self.n_bins as f64);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

impl CrossPoolPositionModel {
    /// Predict stars using ball extraction order context.
    fn predict_stars(&self, draws: &[Draw], size: usize, uniform: f64) -> Vec<f64> {
        // Filter for star era and draws with ball order
        let era_draws = filter_star_era(draws);
        let draws_with_order: Vec<&Draw> = era_draws
            .iter()
            .filter(|d| d.ball_order.is_some())
            .collect();

        if draws_with_order.len() < self.min_draws_with_order {
            return vec![uniform; size];
        }

        // Compute ball extraction order features for binning
        // Feature: sum of extraction positions weighted by ball value
        // This captures "high balls early" vs "low balls early"
        let features: Vec<f64> = draws_with_order
            .iter()
            .map(|d| {
                let order = d.ball_order.unwrap();
                // Sum of (position+1) × ball_value — captures extraction pattern
                order
                    .iter()
                    .enumerate()
                    .map(|(pos, &ball)| (pos + 1) as f64 * ball as f64)
                    .sum::<f64>()
            })
            .collect();

        // Compute bin boundaries (quantile-based)
        let mut sorted_features = features.clone();
        sorted_features.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bin_boundaries: Vec<f64> = (1..self.n_bins)
            .map(|i| {
                let idx = (i * sorted_features.len() / self.n_bins).min(sorted_features.len() - 1);
                sorted_features[idx]
            })
            .collect();

        // Build EWMA distributions per bin
        let mut bin_dists = vec![vec![0.0f64; size]; self.n_bins];
        let mut bin_totals = vec![0.0f64; self.n_bins];

        for (t, (draw, feature)) in draws_with_order.iter().zip(features.iter()).enumerate() {
            let weight = (-self.ewma_alpha * t as f64).exp();
            let bin = feature_to_bin(*feature, &bin_boundaries, self.n_bins);

            for &star in &draw.stars {
                let idx = (star - 1) as usize;
                if idx < size {
                    bin_dists[bin][idx] += weight;
                    bin_totals[bin] += weight;
                }
            }
        }

        // Normalize each bin
        for bin in 0..self.n_bins {
            if bin_totals[bin] > 0.0 {
                for p in &mut bin_dists[bin] {
                    *p /= bin_totals[bin];
                }
            } else {
                bin_dists[bin] = vec![uniform; size];
            }
        }

        // Determine current context from most recent draw
        let current_feature = features[0];
        let current_bin = feature_to_bin(current_feature, &bin_boundaries, self.n_bins);

        // Adaptive blend: more observations in this bin → more weight to conditional
        let local_weight = bin_totals[current_bin] / (bin_totals[current_bin] + 10.0);

        // Global average distribution
        let total_all: f64 = bin_totals.iter().sum();
        let mut global = vec![0.0f64; size];
        for bin in 0..self.n_bins {
            let w = bin_totals[bin] / total_all.max(1e-15);
            for i in 0..size {
                global[i] += w * bin_dists[bin][i];
            }
        }

        // Blend local and global
        let mut probs = vec![0.0f64; size];
        for i in 0..size {
            probs[i] = local_weight * bin_dists[current_bin][i] + (1.0 - local_weight) * global[i];
        }

        // Smoothing towards uniform
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        floor_only(&mut probs, PROB_FLOOR_STARS);
        probs
    }

    /// Predict balls using star context from previous draw.
    fn predict_balls(&self, draws: &[Draw], size: usize, uniform: f64) -> Vec<f64> {
        let draws_with_order: Vec<&Draw> = draws
            .iter()
            .filter(|d| d.star_order.is_some())
            .collect();

        if draws_with_order.len() < self.min_draws_with_order {
            return vec![uniform; size];
        }

        // Feature: star sum × mod-4 pattern of stars from previous draw
        // We use star_sum as the binning feature (simpler, more data per bin)
        let n_bins = 4; // 4 bins for star context

        // Build pairs: (star_context from draw t, balls from draw t)
        // star_context comes from same draw (since stars and balls are drawn together)
        let features: Vec<f64> = draws_with_order
            .iter()
            .map(|d| {
                let star_order = d.star_order.unwrap();
                // Feature: first_star_extracted × 12 + second_star_extracted
                // This creates a simple ordering of star extraction contexts
                star_order[0] as f64 * 13.0 + star_order[1] as f64
            })
            .collect();

        // Compute bin boundaries
        let mut sorted_features = features.clone();
        sorted_features.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bin_boundaries: Vec<f64> = (1..n_bins)
            .map(|i| {
                let idx = (i * sorted_features.len() / n_bins).min(sorted_features.len() - 1);
                sorted_features[idx]
            })
            .collect();

        // Build EWMA distributions per bin
        let mut bin_dists = vec![vec![0.0f64; size]; n_bins];
        let mut bin_totals = vec![0.0f64; n_bins];

        for (t, (draw, feature)) in draws_with_order.iter().zip(features.iter()).enumerate() {
            let weight = (-self.ewma_alpha * t as f64).exp();
            let bin = feature_to_bin(*feature, &bin_boundaries, n_bins);

            for &ball in &draw.balls {
                let idx = (ball - 1) as usize;
                if idx < size {
                    bin_dists[bin][idx] += weight;
                    bin_totals[bin] += weight;
                }
            }
        }

        // Normalize each bin
        for bin in 0..n_bins {
            if bin_totals[bin] > 0.0 {
                for p in &mut bin_dists[bin] {
                    *p /= bin_totals[bin];
                }
            } else {
                bin_dists[bin] = vec![uniform; size];
            }
        }

        // Current context: most recent draw's star extraction
        let current_feature = features[0];
        let current_bin = feature_to_bin(current_feature, &bin_boundaries, n_bins);

        let local_weight = bin_totals[current_bin] / (bin_totals[current_bin] + 10.0);

        // Global average
        let total_all: f64 = bin_totals.iter().sum();
        let mut global = vec![0.0f64; size];
        for bin in 0..n_bins {
            let w = bin_totals[bin] / total_all.max(1e-15);
            for i in 0..size {
                global[i] += w * bin_dists[bin][i];
            }
        }

        let mut probs = vec![0.0f64; size];
        for i in 0..size {
            probs[i] = local_weight * bin_dists[current_bin][i] + (1.0 - local_weight) * global[i];
        }

        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }
}

/// Map a feature value to a bin index using quantile boundaries.
fn feature_to_bin(value: f64, boundaries: &[f64], n_bins: usize) -> usize {
    for (i, &b) in boundaries.iter().enumerate() {
        if value <= b {
            return i;
        }
    }
    n_bins - 1
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
                }
            })
            .collect()
    }

    #[test]
    fn test_cross_pool_balls_valid_distribution() {
        let model = CrossPoolPositionModel::default();
        let draws = make_draws_with_order(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
        assert!(dist.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_cross_pool_stars_valid_distribution() {
        let model = CrossPoolPositionModel::default();
        let draws = make_draws_with_order(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
        assert!(dist.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_cross_pool_without_order_returns_uniform_balls() {
        let model = CrossPoolPositionModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform without order data");
        }
    }

    #[test]
    fn test_cross_pool_without_order_returns_uniform_stars() {
        let model = CrossPoolPositionModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        let uniform = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Should be uniform without order data");
        }
    }

    #[test]
    fn test_cross_pool_few_draws_returns_uniform() {
        let model = CrossPoolPositionModel::default();
        let draws = make_draws_with_order(20);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9, "Too few draws should be uniform");
        }
    }

    #[test]
    fn test_cross_pool_deterministic() {
        let model = CrossPoolPositionModel::default();
        let draws = make_draws_with_order(100);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Should be deterministic");
        }
        let d3 = model.predict(&draws, Pool::Stars);
        let d4 = model.predict(&draws, Pool::Stars);
        for (a, b) in d3.iter().zip(d4.iter()) {
            assert!((a - b).abs() < 1e-15, "Stars should be deterministic");
        }
    }

    #[test]
    fn test_feature_to_bin() {
        let boundaries = vec![10.0, 20.0, 30.0];
        assert_eq!(feature_to_bin(5.0, &boundaries, 4), 0);
        assert_eq!(feature_to_bin(10.0, &boundaries, 4), 0);
        assert_eq!(feature_to_bin(15.0, &boundaries, 4), 1);
        assert_eq!(feature_to_bin(25.0, &boundaries, 4), 2);
        assert_eq!(feature_to_bin(35.0, &boundaries, 4), 3);
    }

    #[test]
    fn test_cross_pool_with_consistent_pattern() {
        // Stars always [3, 10] when balls extracted in order [1, 10, 20, 30, 40]
        let model = CrossPoolPositionModel::default();
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
            })
            .collect();

        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Stars 3 and 10 should have elevated probability
        let uniform = 1.0 / 12.0;
        assert!(dist[2] > uniform, "Star 3 should be above uniform");
        assert!(dist[9] > uniform, "Star 10 should be above uniform");
    }
}
