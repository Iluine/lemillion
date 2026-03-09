use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Community — co-occurrence graph community transition model.
///
/// The research module (mathematical.rs) shows that the co-occurrence graph of
/// EuroMillions balls has significant modularity when partitioned by decades.
/// This model exploits community-level transition patterns:
///
/// 1. Partition balls into 5 communities by decade (1-10, 11-20, ..., 41-50),
///    matching the 5 physical rows of the Stresa machine rack.
/// 2. Build a community-level transition matrix: for each community pair (c_from, c_to),
///    estimate P(ball drawn from c_to | ball drawn from c_from) across consecutive draws.
/// 3. From the most recent draw's community profile, predict the expected community
///    distribution for the next draw.
/// 4. Redistribute community-level probabilities to individual balls using EWMA frequencies
///    within each community.
/// 5. Blend community-based prediction with global EWMA for robustness.
///
/// For stars (only 12 numbers): communities are trivial, so use simple EWMA frequency.
pub struct CommunityModel {
    smoothing: f64,
    min_draws: usize,
    window: usize,
    n_communities: usize,
    community_blend: f64,
    ewma_alpha: f64,
}

impl Default for CommunityModel {
    fn default() -> Self {
        Self {
            smoothing: 0.20,
            min_draws: 25,
            window: 100,
            n_communities: 5,
            community_blend: 0.5,
            ewma_alpha: 0.05,
        }
    }
}

/// Returns the community index (0-4) for a ball number (1-50).
/// Community 0 = balls 1-10, community 1 = balls 11-20, etc.
fn ball_community(ball: u8) -> usize {
    ((ball - 1) / 10) as usize
}

/// Compute a community profile for a set of balls: count per community.
fn community_profile(balls: &[u8], n_communities: usize) -> Vec<f64> {
    let mut profile = vec![0.0; n_communities];
    for &b in balls {
        let c = ball_community(b);
        if c < n_communities {
            profile[c] += 1.0;
        }
    }
    profile
}

/// Compute EWMA frequencies for a pool from draws (draws[0] = most recent).
/// Returns a Vec of size pool.size() summing to 1.0.
fn ewma_frequencies(draws: &[Draw], pool: Pool, alpha: f64) -> Vec<f64> {
    let size = pool.size();
    let mut freq = vec![0.0f64; size];

    // Iterate from oldest to newest for proper EWMA accumulation
    for draw in draws.iter().rev() {
        let numbers = pool.numbers_from(draw);
        // Decay all
        for f in freq.iter_mut() {
            *f *= 1.0 - alpha;
        }
        // Increment drawn numbers
        for &num in numbers {
            let idx = (num - 1) as usize;
            if idx < size {
                freq[idx] += alpha;
            }
        }
    }

    // Normalize
    let sum: f64 = freq.iter().sum();
    if sum > 0.0 {
        for f in &mut freq {
            *f /= sum;
        }
    } else {
        let uniform_val = 1.0 / size as f64;
        for f in &mut freq {
            *f = uniform_val;
        }
    }

    freq
}

impl ForecastModel for CommunityModel {
    fn name(&self) -> &str {
        "Community"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Stars: simple EWMA (communities are trivial for 12 numbers)
        if pool == Pool::Stars {
            let mut probs = ewma_frequencies(draws, pool, self.ewma_alpha);

            let uniform_val = 1.0 / size as f64;
            for p in probs.iter_mut() {
                *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
            }

            floor_only(&mut probs, PROB_FLOOR_STARS);
            return probs;
        }

        // Balls: community transition model
        let nc = self.n_communities;
        let window = self.window.min(draws.len());
        let recent = &draws[..window];

        // Build community-level transition matrix T[from][to] with Laplace smoothing.
        // T[c_from][c_to] counts how many balls from community c_to appear in draw[t]
        // given that balls from community c_from appeared in draw[t+1] (the previous draw).
        let mut transition = vec![vec![1.0f64; nc]; nc];

        for t in 0..recent.len() - 1 {
            let from_profile = community_profile(&recent[t + 1].balls, nc);
            let to_profile = community_profile(&recent[t].balls, nc);

            for (c_from, &count_from) in from_profile.iter().enumerate() {
                if count_from > 0.0 {
                    for (c_to, &count_to) in to_profile.iter().enumerate() {
                        transition[c_from][c_to] += count_from * count_to;
                    }
                }
            }
        }

        // Normalize rows
        for row in &mut transition {
            let row_sum: f64 = row.iter().sum();
            if row_sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= row_sum;
                }
            }
        }

        // Predict next community distribution from the most recent draw's profile.
        // For each community represented in the current draw, accumulate its
        // transition row, weighted by how many balls came from that community.
        let current_profile = community_profile(&recent[0].balls, nc);
        let total_balls: f64 = current_profile.iter().sum();
        let mut predicted_community = vec![0.0f64; nc];

        if total_balls > 0.0 {
            for (c_from, &count) in current_profile.iter().enumerate() {
                if count > 0.0 {
                    let weight = count / total_balls;
                    for (c_to, pred) in predicted_community.iter_mut().enumerate() {
                        *pred += weight * transition[c_from][c_to];
                    }
                }
            }
        }

        // Normalize predicted community distribution
        let comm_sum: f64 = predicted_community.iter().sum();
        if comm_sum > 0.0 {
            for p in &mut predicted_community {
                *p /= comm_sum;
            }
        } else {
            let comm_uniform = 1.0 / nc as f64;
            for p in &mut predicted_community {
                *p = comm_uniform;
            }
        }

        // Compute EWMA frequencies per ball for intra-community redistribution
        let ewma = ewma_frequencies(recent, pool, self.ewma_alpha);

        // Compute EWMA sum per community for redistribution
        let mut community_ewma_sum = vec![0.0f64; nc];
        for (k, &freq) in ewma.iter().enumerate() {
            let c = k / 10;
            if c < nc {
                community_ewma_sum[c] += freq;
            }
        }

        // Community-based probability: P(ball_k) = P(community_c) * ewma[k] / ewma_sum[c]
        let mut community_probs = vec![0.0f64; size];
        for (k, prob) in community_probs.iter_mut().enumerate() {
            let c = k / 10;
            if c < nc && community_ewma_sum[c] > 0.0 {
                *prob = predicted_community[c] * ewma[k] / community_ewma_sum[c];
            }
        }

        // Normalize community probs
        let cp_sum: f64 = community_probs.iter().sum();
        if cp_sum > 0.0 {
            for p in &mut community_probs {
                *p /= cp_sum;
            }
        } else {
            return uniform;
        }

        // Blend community-based prediction with global EWMA
        let mut probs = vec![0.0f64; size];
        for (k, p) in probs.iter_mut().enumerate() {
            *p = self.community_blend * community_probs[k]
                + (1.0 - self.community_blend) * ewma[k];
        }

        // Smooth towards uniform
        let uniform_val = 1.0 / size as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("window".into(), self.window as f64),
            ("n_communities".into(), self.n_communities as f64),
            ("community_blend".into(), self.community_blend),
            ("ewma_alpha".into(), self.ewma_alpha),
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
    fn test_community_valid_distribution_balls() {
        let draws = make_test_draws(100);
        let model = CommunityModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_community_valid_distribution_stars() {
        let draws = make_test_draws(100);
        let model = CommunityModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_community_few_draws_returns_uniform() {
        let model = CommunityModel::default();
        let draws = make_test_draws(10);

        let dist_balls = model.predict(&draws, Pool::Balls);
        let expected_balls = 1.0 / 50.0;
        for &p in &dist_balls {
            assert!(
                (p - expected_balls).abs() < 1e-10,
                "Expected uniform for balls with few draws"
            );
        }

        let dist_stars = model.predict(&draws, Pool::Stars);
        let expected_stars = 1.0 / 12.0;
        for &p in &dist_stars {
            assert!(
                (p - expected_stars).abs() < 1e-10,
                "Expected uniform for stars with few draws"
            );
        }
    }

    #[test]
    fn test_community_no_negative() {
        let draws = make_test_draws(100);
        let model = CommunityModel::default();

        let dist_balls = model.predict(&draws, Pool::Balls);
        for &p in &dist_balls {
            assert!(p >= 0.0, "Negative ball probability: {}", p);
        }

        let dist_stars = model.predict(&draws, Pool::Stars);
        for &p in &dist_stars {
            assert!(p >= 0.0, "Negative star probability: {}", p);
        }
    }

    #[test]
    fn test_community_deterministic() {
        let draws = make_test_draws(100);
        let model = CommunityModel::default();

        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Community should be deterministic");
        }

        let s1 = model.predict(&draws, Pool::Stars);
        let s2 = model.predict(&draws, Pool::Stars);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-15, "Community stars should be deterministic");
        }
    }

    #[test]
    fn test_community_sampling_strategy() {
        let model = CommunityModel::default();
        assert_eq!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        );
    }

    #[test]
    fn test_community_empty_draws() {
        let model = CommunityModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_community_large_draws() {
        let draws = make_test_draws(300);
        let model = CommunityModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_ball_community_mapping() {
        assert_eq!(ball_community(1), 0);
        assert_eq!(ball_community(10), 0);
        assert_eq!(ball_community(11), 1);
        assert_eq!(ball_community(20), 1);
        assert_eq!(ball_community(21), 2);
        assert_eq!(ball_community(30), 2);
        assert_eq!(ball_community(31), 3);
        assert_eq!(ball_community(40), 3);
        assert_eq!(ball_community(41), 4);
        assert_eq!(ball_community(50), 4);
    }

    #[test]
    fn test_community_profile_uniform() {
        let balls = [5, 15, 25, 35, 45];
        let profile = community_profile(&balls, 5);
        assert_eq!(profile, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_community_profile_concentrated() {
        let balls = [1, 2, 3, 4, 5];
        let profile = community_profile(&balls, 5);
        assert_eq!(profile, vec![5.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ewma_frequencies_normalize() {
        let draws = make_test_draws(50);
        let freq = ewma_frequencies(&draws, Pool::Balls, 0.05);
        let sum: f64 = freq.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "EWMA frequencies should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_community_params() {
        let model = CommunityModel::default();
        let params = model.params();
        assert_eq!(params["smoothing"], 0.20);
        assert_eq!(params["min_draws"], 25.0);
        assert_eq!(params["window"], 100.0);
        assert_eq!(params["n_communities"], 5.0);
        assert_eq!(params["community_blend"], 0.5);
        assert_eq!(params["ewma_alpha"], 0.05);
    }

    #[test]
    fn test_community_name() {
        let model = CommunityModel::default();
        assert_eq!(model.name(), "Community");
    }
}
