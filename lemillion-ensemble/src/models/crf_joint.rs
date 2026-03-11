use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Conditional Random Field for joint ball-star distribution.
///
/// Unifies three ad-hoc components (JointConditionalModel, BallStarConditioner,
/// CoherenceScorer) into a single principled CRF framework:
///
/// ```text
/// P(b1..b5, s1, s2) proportional to exp(
///     sum_i phi_i(bi)          -- unary ball potentials (EWMA frequency)
///   + sum_{i<j} psi_ij(bi,bj)  -- pairwise ball-ball potentials (co-occurrence log-ratio)
///   + sum_k chi_k(sk)          -- unary star potentials (EWMA frequency)
///   + sum_{i,k} omega(bi,sk)   -- cross-pool ball-star potentials (sum_bin x star freq)
/// )
/// ```
///
/// Training via pseudo-likelihood avoids computing the partition function Z
/// (which has ~140M terms for C(50,5)*C(12,2)). Marginal prediction aggregates
/// each number's contribution across all potential types.
pub struct CrfJointModel {
    smoothing: f64,
    ewma_alpha: f64,
    min_draws: usize,
}

impl Default for CrfJointModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            ewma_alpha: 0.10,
            min_draws: 50,
        }
    }
}

/// Learned CRF potentials from historical draws.
struct CrfPotentials {
    /// Unary ball potentials phi[0..50]: EWMA log-frequency ratio vs uniform.
    phi_balls: Vec<f64>,
    /// Pairwise ball-ball potentials psi[i*50+j]: co-occurrence log-ratio.
    /// Symmetric: psi[i][j] == psi[j][i]. Stored as flat n_balls x n_balls.
    psi_ball_pairs: Vec<f64>,
    /// Unary star potentials chi[0..12]: EWMA log-frequency ratio vs uniform.
    chi_stars: Vec<f64>,
    /// Cross-pool potentials omega[sum_bin][star_idx]: ball-sum-conditioned star bias.
    /// 5 sum bins x 12 stars.
    omega_cross: Vec<Vec<f64>>,
}

impl CrfJointModel {
    /// Learn CRF potentials from historical draws via pseudo-likelihood.
    fn learn_potentials(&self, draws: &[Draw]) -> CrfPotentials {
        let n_balls = 50usize;
        let n_stars = 12usize;

        // --- Unary ball potentials: EWMA frequency -> log-ratio vs uniform ---
        let uniform_ball = 5.0 / n_balls as f64; // E[freq] = pick_count / pool_size
        let mut ewma_balls = vec![uniform_ball; n_balls];
        // Iterate from oldest to newest (draws[0] is most recent)
        for draw in draws.iter().rev() {
            let mut indicator = vec![0.0f64; n_balls];
            for &b in &draw.balls {
                indicator[(b - 1) as usize] = 1.0;
            }
            for i in 0..n_balls {
                ewma_balls[i] = (1.0 - self.ewma_alpha) * ewma_balls[i]
                    + self.ewma_alpha * indicator[i];
            }
        }
        let phi_balls: Vec<f64> = ewma_balls
            .iter()
            .map(|&f| (f.max(1e-10) / uniform_ball).ln())
            .collect();

        // --- Pairwise ball-ball potentials: co-occurrence log-ratio ---
        let n_draws = draws.len() as f64;
        let mut pair_counts = vec![0u32; n_balls * n_balls];
        for draw in draws {
            let idxs: Vec<usize> = draw.balls.iter().map(|&b| (b - 1) as usize).collect();
            for a in 0..idxs.len() {
                for b in (a + 1)..idxs.len() {
                    pair_counts[idxs[a] * n_balls + idxs[b]] += 1;
                    pair_counts[idxs[b] * n_balls + idxs[a]] += 1;
                }
            }
        }
        // Expected pair rate under independence: C(48,3)/C(50,5) = 5*4/(50*49)
        let p_pair_expected = (5.0 * 4.0) / (n_balls as f64 * (n_balls as f64 - 1.0));
        let expected_count = n_draws * p_pair_expected;
        let mut psi_ball_pairs = vec![0.0f64; n_balls * n_balls];
        for i in 0..n_balls {
            for j in (i + 1)..n_balls {
                let obs = pair_counts[i * n_balls + j] as f64;
                // Log-ratio with Laplace smoothing to avoid log(0)
                let ratio = (obs + 1.0) / (expected_count + 1.0);
                let psi = ratio.ln().clamp(-3.0, 3.0);
                psi_ball_pairs[i * n_balls + j] = psi;
                psi_ball_pairs[j * n_balls + i] = psi;
            }
        }

        // --- Unary star potentials: EWMA frequency -> log-ratio vs uniform ---
        let uniform_star = 2.0 / n_stars as f64;
        let mut ewma_stars = vec![uniform_star; n_stars];
        for draw in draws.iter().rev() {
            let mut indicator = vec![0.0f64; n_stars];
            for &s in &draw.stars {
                indicator[(s - 1) as usize] = 1.0;
            }
            for i in 0..n_stars {
                ewma_stars[i] = (1.0 - self.ewma_alpha) * ewma_stars[i]
                    + self.ewma_alpha * indicator[i];
            }
        }
        let chi_stars: Vec<f64> = ewma_stars
            .iter()
            .map(|&f| (f.max(1e-10) / uniform_star).ln())
            .collect();

        // --- Cross-pool potentials: P(star | ball_sum_bin) log-ratio ---
        let n_bins = 5usize;
        let mut bin_star_counts = vec![vec![0.0f64; n_stars]; n_bins];
        let mut bin_totals = vec![0.0f64; n_bins];
        for draw in draws {
            let ball_sum: u16 = draw.balls.iter().map(|&b| b as u16).sum();
            let bin = ball_sum_bin(ball_sum);
            for &s in &draw.stars {
                bin_star_counts[bin][(s - 1) as usize] += 1.0;
            }
            bin_totals[bin] += 1.0;
        }
        let omega_cross: Vec<Vec<f64>> = (0..n_bins)
            .map(|bin| {
                let total_stars = bin_totals[bin] * 2.0; // 2 stars per draw
                if total_stars < 5.0 {
                    return vec![0.0; n_stars];
                }
                let expected_per_star = total_stars / n_stars as f64;
                bin_star_counts[bin]
                    .iter()
                    .map(|&c| {
                        let ratio = (c + 0.5) / (expected_per_star + 0.5);
                        ratio.ln().clamp(-2.0, 2.0)
                    })
                    .collect()
            })
            .collect();

        CrfPotentials {
            phi_balls,
            psi_ball_pairs,
            chi_stars,
            omega_cross,
        }
    }

    /// Marginalize CRF potentials for a given pool.
    ///
    /// For balls: phi_i + mean-field sum_j psi_ij * m_j + mean-field cross omega.
    /// For stars: chi_k + mean-field cross omega.
    fn marginalize(&self, potentials: &CrfPotentials, pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let mut scores = vec![0.0f64; n];

        match pool {
            Pool::Balls => {
                let n_balls = 50;
                // Mean activation: uniform 5/50 = 0.1 per ball
                let m_j = 5.0 / n_balls as f64;

                for i in 0..n_balls {
                    // Unary potential
                    scores[i] += potentials.phi_balls[i];

                    // Mean-field pairwise: sum_j psi_ij * m_j
                    let row_offset = i * n_balls;
                    let mut pair_sum = 0.0f64;
                    for j in 0..n_balls {
                        if j != i {
                            pair_sum += potentials.psi_ball_pairs[row_offset + j];
                        }
                    }
                    scores[i] += pair_sum * m_j;

                    // Mean-field cross-pool contribution (averaged over sum bins)
                    // Each ball's contribution to sum modulates expected star outcome;
                    // here we average the cross-potential across bins weighted by bin frequency.
                    let cross_avg: f64 = potentials
                        .omega_cross
                        .iter()
                        .map(|bin| bin.iter().sum::<f64>() / bin.len() as f64)
                        .sum::<f64>()
                        / potentials.omega_cross.len() as f64;
                    scores[i] += cross_avg * 0.1; // mild cross-pool influence
                }
            }
            Pool::Stars => {
                let n_stars = 12;
                for k in 0..n_stars {
                    // Unary star potential
                    scores[k] += potentials.chi_stars[k];

                    // Mean-field cross-pool: average omega across sum bins for this star
                    let cross_sum: f64 = potentials
                        .omega_cross
                        .iter()
                        .map(|bin| bin[k])
                        .sum::<f64>();
                    scores[k] += cross_sum / potentials.omega_cross.len() as f64;
                }
            }
        }

        scores
    }
}

/// Score a complete grid (5 balls + 2 stars) using CRF potentials.
///
/// Returns the unnormalized log-potential:
///   sum_i phi(bi) + sum_{i<j} psi(bi,bj) + sum_k chi(sk) + sum_{i,k} omega(bi,sk)
///
/// This is O(C(5,2) + 5*2) = O(20) operations after potential lookup, making it
/// extremely fast for scoring large candidate sets.
pub fn score_grid(balls: &[u8; 5], stars: &[u8; 2], draws: &[Draw]) -> f64 {
    if draws.len() < 50 {
        return 0.0;
    }

    let model = CrfJointModel::default();
    let potentials = model.learn_potentials(draws);

    let mut score = 0.0f64;

    // Unary ball potentials
    for &b in balls {
        let idx = (b - 1) as usize;
        if idx < 50 {
            score += potentials.phi_balls[idx];
        }
    }

    // Pairwise ball-ball potentials: C(5,2) = 10 pairs
    for a in 0..balls.len() {
        for b in (a + 1)..balls.len() {
            let i = (balls[a] - 1) as usize;
            let j = (balls[b] - 1) as usize;
            if i < 50 && j < 50 {
                score += potentials.psi_ball_pairs[i * 50 + j];
            }
        }
    }

    // Unary star potentials
    for &s in stars {
        let idx = (s - 1) as usize;
        if idx < 12 {
            score += potentials.chi_stars[idx];
        }
    }

    // Cross-pool potentials: 5 balls x 2 stars = 10 terms
    let ball_sum: u16 = balls.iter().map(|&b| b as u16).sum();
    let bin = ball_sum_bin(ball_sum);
    for &s in stars {
        let s_idx = (s - 1) as usize;
        if s_idx < 12 {
            score += potentials.omega_cross[bin][s_idx];
        }
    }

    score
}

impl ForecastModel for CrfJointModel {
    fn name(&self) -> &str {
        "CRFJoint"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let uniform = vec![1.0 / n as f64; n];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let potentials = self.learn_potentials(draws);
        let scores = self.marginalize(&potentials, pool);

        // Convert scores to probabilities via softmax
        let max_score = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();

        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            return uniform;
        }
        for p in &mut probs {
            *p /= sum;
        }

        // Smooth towards uniform
        let uniform_val = 1.0 / n as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = if pool == Pool::Balls {
            PROB_FLOOR_BALLS
        } else {
            PROB_FLOOR_STARS
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::FullHistory
    }
}

/// Bin the ball sum into 5 quintiles.
/// Theoretical range: 15 (1+2+3+4+5) to 240 (46+47+48+49+50).
/// Practical range clusters around ~80-180.
fn ball_sum_bin(sum: u16) -> usize {
    match sum {
        0..=99 => 0,
        100..=119 => 1,
        120..=139 => 2,
        140..=159 => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_crf_valid_ball_distribution() {
        let draws = make_test_draws(100);
        let model = CrfJointModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Ball distribution invalid: sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_crf_valid_star_distribution() {
        let draws = make_test_draws(100);
        let model = CrfJointModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Star distribution invalid: sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_crf_few_draws_returns_uniform() {
        let draws = make_test_draws(10);
        let model = CrfJointModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!(
                (p - uniform).abs() < 1e-6,
                "Expected uniform {}, got {}",
                uniform,
                p
            );
        }
    }

    #[test]
    fn test_crf_empty_draws() {
        let model = CrfJointModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crf_no_negative_probabilities() {
        let draws = make_test_draws(100);
        let model = CrfJointModel::default();
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            for (i, &p) in dist.iter().enumerate() {
                assert!(p >= 0.0, "Negative probability at index {}: {}", i, p);
            }
        }
    }

    #[test]
    fn test_crf_deterministic() {
        let draws = make_test_draws(100);
        let model = CrfJointModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "CRF should be deterministic");
        }
    }

    #[test]
    fn test_score_grid_returns_finite() {
        let draws = make_test_draws(100);
        let s = score_grid(&[1, 10, 20, 30, 40], &[3, 7], &draws);
        assert!(s.is_finite(), "Grid score should be finite, got {}", s);
    }

    #[test]
    fn test_score_grid_few_draws_returns_zero() {
        let draws = make_test_draws(10);
        let s = score_grid(&[1, 2, 3, 4, 5], &[1, 2], &draws);
        assert_eq!(s, 0.0, "score_grid with <50 draws should return 0.0");
    }

    #[test]
    fn test_score_grid_deterministic() {
        let draws = make_test_draws(100);
        let s1 = score_grid(&[5, 15, 25, 35, 45], &[2, 8], &draws);
        let s2 = score_grid(&[5, 15, 25, 35, 45], &[2, 8], &draws);
        assert!(
            (s1 - s2).abs() < 1e-15,
            "score_grid should be deterministic"
        );
    }

    #[test]
    fn test_crf_large_draws() {
        let draws = make_test_draws(300);
        let model = CrfJointModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars));
    }

    #[test]
    fn test_crf_params() {
        let model = CrfJointModel::default();
        let params = model.params();
        assert_eq!(params["smoothing"], 0.25);
        assert_eq!(params["ewma_alpha"], 0.10);
        assert_eq!(params["min_draws"], 50.0);
    }

    #[test]
    fn test_crf_name_and_strategy() {
        let model = CrfJointModel::default();
        assert_eq!(model.name(), "CRFJoint");
        assert_eq!(model.sampling_strategy(), SamplingStrategy::FullHistory);
    }

    #[test]
    fn test_ball_sum_bin_boundaries() {
        assert_eq!(ball_sum_bin(15), 0);   // minimum possible sum
        assert_eq!(ball_sum_bin(99), 0);
        assert_eq!(ball_sum_bin(100), 1);
        assert_eq!(ball_sum_bin(119), 1);
        assert_eq!(ball_sum_bin(120), 2);
        assert_eq!(ball_sum_bin(139), 2);
        assert_eq!(ball_sum_bin(140), 3);
        assert_eq!(ball_sum_bin(159), 3);
        assert_eq!(ball_sum_bin(160), 4);
        assert_eq!(ball_sum_bin(240), 4);  // maximum possible sum
    }

    #[test]
    fn test_different_grids_get_different_scores() {
        let draws = make_test_draws(100);
        let s1 = score_grid(&[1, 2, 3, 4, 5], &[1, 2], &draws);
        let s2 = score_grid(&[46, 47, 48, 49, 50], &[11, 12], &draws);
        // With non-uniform history, different grids should get different scores
        assert!(
            (s1 - s2).abs() > 1e-10,
            "Different grids should have different scores: {} vs {}",
            s1,
            s2
        );
    }
}
