use lemillion_db::models::Pool;

/// Wasserstein barycenter via Sinkhorn iterations (entropic regularization).
///
/// Computes the optimal transport barycenter of multiple probability distributions,
/// where the cost matrix encodes physical proximity on the Stresa machine (balls)
/// or mod-4 blade proximity on the Paquerette (stars).
///
/// The Sinkhorn algorithm alternates row/column normalizations of the Gibbs kernel
/// K = exp(-C/epsilon) to converge to the optimal barycenter. Entropic regularization
/// (epsilon > 0) makes the problem strongly convex and solvable via matrix scaling.

/// Build cost matrix for the given pool based on physical machine proximity.
///
/// Balls (50): Stresa machine has 5 rack rows (decades). row = (ball-1)/10.
///   - Same row: |i-j| / 10 (normalized within-row distance)
///   - Adjacent rows (|row_i - row_j| == 1): 1 + |i-j| / 10
///   - Far rows (|row_i - row_j| >= 2): 2 + |i-j| / 10
///
/// Stars (12): Paquerette has 4 blades. blade = (star-1) % 4.
///   - Same blade: 0.5
///   - Adjacent blades (|blade_i - blade_j| in {1,3}): 1.0
///   - Opposite blades (|blade_i - blade_j| == 2): 1.5
fn build_cost_matrix(pool: Pool) -> Vec<Vec<f64>> {
    let n = pool.size();
    let mut cost = vec![vec![0.0; n]; n];

    match pool {
        Pool::Balls => {
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let row_i = i / 10;
                    let row_j = j / 10;
                    let row_diff = (row_i as isize - row_j as isize).unsigned_abs();
                    let abs_diff = (i as isize - j as isize).unsigned_abs() as f64;
                    let within = abs_diff / 10.0;
                    cost[i][j] = match row_diff {
                        0 => within,
                        1 => 1.0 + within,
                        _ => 2.0 + within,
                    };
                }
            }
        }
        Pool::Stars => {
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let blade_i = i % 4;
                    let blade_j = j % 4;
                    let blade_diff_raw = (blade_i as isize - blade_j as isize).unsigned_abs();
                    // Circular distance on 4 blades
                    let blade_diff = blade_diff_raw.min(4 - blade_diff_raw);
                    cost[i][j] = match blade_diff {
                        0 => 0.5,
                        1 => 1.0,
                        _ => 1.5, // blade_diff == 2 (opposite)
                    };
                }
            }
        }
    }

    cost
}

/// Compute the Gibbs kernel K[i][j] = exp(-C[i][j] / epsilon).
fn gibbs_kernel(cost: &[Vec<f64>], epsilon: f64) -> Vec<Vec<f64>> {
    let n = cost.len();
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            k[i][j] = (-cost[i][j] / epsilon).exp();
        }
    }
    k
}

/// Compute Wasserstein barycenter of weighted distributions.
///
/// Uses the Sinkhorn iterative algorithm (Cuturi & Doucet, 2014) with entropic
/// regularization. The barycenter minimizes the weighted sum of Wasserstein-2
/// distances to each input distribution.
///
/// Returns a distribution of size `pool.size()` summing to 1.0.
///
/// # Arguments
/// * `distributions` - One probability distribution per model (each sums to ~1.0)
/// * `weights` - Per-model weights (should sum to ~1.0)
/// * `pool` - `Pool::Balls` (50) or `Pool::Stars` (12)
/// * `epsilon` - Entropic regularization parameter (recommended: 0.1)
/// * `max_iter` - Number of Sinkhorn iterations (recommended: 50)
pub fn wasserstein_barycenter(
    distributions: &[Vec<f64>],
    weights: &[f64],
    pool: Pool,
    epsilon: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = pool.size();

    // Edge cases: no distributions or empty weights
    if distributions.is_empty() || weights.is_empty() {
        return vec![1.0 / n as f64; n];
    }

    let m = distributions.len();

    // Normalize weights to sum to 1
    let w_sum: f64 = weights.iter().sum();
    let norm_weights: Vec<f64> = if w_sum > 1e-15 {
        weights.iter().map(|&w| w / w_sum).collect()
    } else {
        vec![1.0 / m as f64; m]
    };

    // Validate and clamp distributions (ensure no zeros for log-stability)
    let floor = 1e-10;
    let dists: Vec<Vec<f64>> = distributions
        .iter()
        .map(|d| {
            let mut clamped: Vec<f64> = d.iter().map(|&p| p.max(floor)).collect();
            // Pad or truncate to pool size
            clamped.resize(n, floor);
            let s: f64 = clamped.iter().sum();
            if s > 0.0 {
                for p in clamped.iter_mut() {
                    *p /= s;
                }
            }
            clamped
        })
        .collect();

    // Single distribution: return it directly
    if m == 1 {
        return dists[0].clone();
    }

    // Build cost matrix and Gibbs kernel
    let cost = build_cost_matrix(pool);
    let kernel = gibbs_kernel(&cost, epsilon);

    // Sinkhorn fixed-point iteration for barycenter.
    // For each source distribution k, maintain dual variable v_k.
    // The barycenter q is updated as the weighted geometric mean of K^T * v_k.
    //
    // Algorithm (simplified Sinkhorn barycenter):
    //   Initialize v_k = 1 for all k
    //   For each iteration:
    //     For each k: u_k = p_k ./ (K * v_k)
    //     q = prod_k (K^T * u_k)^{w_k}   (weighted geometric mean)
    //     For each k: v_k = q ./ (K^T * u_k)  ... but we fold this into next iteration

    // Dual variables: one per source distribution, each of size n
    let mut v: Vec<Vec<f64>> = vec![vec![1.0; n]; m];

    // Barycenter (initialized uniform)
    let mut bary = vec![1.0 / n as f64; n];

    for _iter in 0..max_iter {
        // Weighted geometric mean in log-space for numerical stability
        let mut log_bary = vec![0.0f64; n];

        for k in 0..m {
            // Compute K * v_k (matrix-vector product)
            let kv = mat_vec_mul(&kernel, &v[k]);

            // u_k = p_k ./ (K * v_k)
            let u_k: Vec<f64> = (0..n)
                .map(|i| dists[k][i] / kv[i].max(1e-300))
                .collect();

            // Compute K^T * u_k
            let ktu = mat_t_vec_mul(&kernel, &u_k);

            // Accumulate log(K^T * u_k) weighted by w_k
            for i in 0..n {
                log_bary[i] += norm_weights[k] * ktu[i].max(1e-300).ln();
            }
        }

        // Exponentiate with max-subtraction for stability
        let max_lb = log_bary.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        bary = log_bary.iter().map(|&lb| (lb - max_lb).exp()).collect();

        // Normalize barycenter
        let total: f64 = bary.iter().sum();
        if total > 0.0 {
            for p in bary.iter_mut() {
                *p /= total;
            }
        }

        // Update dual variables: v_k = bary ./ (K^T * u_k)
        // Recompute K^T * u_k for the update
        for k in 0..m {
            let kv = mat_vec_mul(&kernel, &v[k]);
            let u_k: Vec<f64> = (0..n)
                .map(|i| dists[k][i] / kv[i].max(1e-300))
                .collect();
            let ktu = mat_t_vec_mul(&kernel, &u_k);
            for i in 0..n {
                v[k][i] = bary[i] / ktu[i].max(1e-300);
            }
        }
    }

    // Final normalization (defensive)
    let total: f64 = bary.iter().sum();
    if total > 0.0 {
        for p in bary.iter_mut() {
            *p /= total;
        }
    } else {
        // Fallback to uniform
        bary = vec![1.0 / n as f64; n];
    }

    bary
}

/// Matrix-vector multiplication: result[i] = sum_j M[i][j] * v[j]
#[inline]
fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// Transpose matrix-vector multiplication: result[j] = sum_i M[i][j] * v[i]
#[inline]
fn mat_t_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = if m.is_empty() { 0 } else { m[0].len() };
    let mut result = vec![0.0; n];
    for (i, row) in m.iter().enumerate() {
        let vi = v[i];
        for (j, &mij) in row.iter().enumerate() {
            result[j] += mij * vi;
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// v19 E4: Wasserstein-1 Drift Detection
// ═══════════════════════════════════════════════════════════════════

/// Compute Wasserstein-1 (Earth Mover's Distance) between two discrete distributions.
/// For 1D discrete distributions: W1 = Σ |CDF_P(i) - CDF_Q(i)|
pub fn wasserstein_1(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().min(q.len());
    let mut cdf_p = 0.0f64;
    let mut cdf_q = 0.0f64;
    let mut w1 = 0.0f64;
    for i in 0..n {
        cdf_p += p[i];
        cdf_q += q[i];
        w1 += (cdf_p - cdf_q).abs();
    }
    w1
}

/// Detect distribution drift by computing W1 between predicted and empirical
/// distributions over a sliding window.
/// Returns (w1_recent, w1_baseline, is_drifting).
pub fn detect_drift(
    predicted: &[f64],
    draws: &[lemillion_db::models::Draw],
    pool: Pool,
    window: usize,
) -> (f64, f64, bool) {
    let size = pool.size();
    let w = window.min(draws.len());
    if w == 0 {
        return (0.0, 0.0, false);
    }

    // Empirical distribution from recent draws
    let mut empirical = vec![0.0f64; size];
    for draw in draws.iter().take(w) {
        let numbers = pool.numbers_from(draw);
        for &n in numbers {
            let idx = (n - 1) as usize;
            if idx < size {
                empirical[idx] += 1.0;
            }
        }
    }
    let total: f64 = empirical.iter().sum();
    if total > 0.0 {
        for p in &mut empirical { *p /= total; }
    }

    let w1 = wasserstein_1(predicted, &empirical);

    // Baseline: W1 between uniform and empirical
    let uniform = vec![1.0 / size as f64; size];
    let w1_baseline = wasserstein_1(&uniform, &empirical);

    let is_drifting = w1 > 2.0 * w1_baseline;

    (w1, w1_baseline, is_drifting)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: check that a distribution sums to ~1.0 and has correct length.
    fn assert_valid_distribution(dist: &[f64], pool: Pool, tol: f64) {
        assert_eq!(dist.len(), pool.size());
        let sum: f64 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < tol,
            "Distribution sum = {}, expected ~1.0",
            sum
        );
        for (i, &p) in dist.iter().enumerate() {
            assert!(p >= 0.0, "Negative probability at index {}: {}", i, p);
        }
    }

    #[test]
    fn test_uniform_inputs_yield_uniform_barycenter() {
        // When all input distributions are uniform, the barycenter should also be uniform.
        let n = Pool::Balls.size();
        let uniform = vec![1.0 / n as f64; n];
        let distributions = vec![uniform.clone(), uniform.clone(), uniform.clone()];
        let weights = vec![1.0 / 3.0; 3];

        let bary = wasserstein_barycenter(&distributions, &weights, Pool::Balls, 0.1, 50);
        assert_valid_distribution(&bary, Pool::Balls, 1e-6);

        // Every element should be close to 1/50.
        // With entropic regularization (epsilon=0.1), the Gibbs kernel introduces
        // slight asymmetry due to the non-uniform cost matrix, so we use a
        // tolerance of 5e-3 (25% of uniform = 0.02).
        let expected = 1.0 / n as f64;
        for &p in &bary {
            assert!(
                (p - expected).abs() < 5e-3,
                "Expected ~{:.6}, got {:.6}",
                expected,
                p
            );
        }
    }

    #[test]
    fn test_single_distribution_returned_as_is() {
        // A single input distribution should be returned (approximately) unchanged.
        let mut dist = vec![0.0; 12];
        // Concentrated on stars 1-4
        dist[0] = 0.25;
        dist[1] = 0.25;
        dist[2] = 0.25;
        dist[3] = 0.25;

        let bary = wasserstein_barycenter(&[dist.clone()], &[1.0], Pool::Stars, 0.1, 50);
        assert_valid_distribution(&bary, Pool::Stars, 1e-6);

        // Should be close to the input
        for i in 0..4 {
            assert!(
                bary[i] > 0.15,
                "Star {} should have significant mass, got {:.6}",
                i + 1,
                bary[i]
            );
        }
    }

    #[test]
    fn test_barycenter_interpolates_between_two_distributions() {
        // Two point-like distributions on opposite sides should produce a barycenter
        // with mass spread between them, influenced by the cost matrix.
        let n = Pool::Stars.size(); // 12

        // Distribution A: concentrated on star 1 (blade 0)
        let mut dist_a = vec![1e-10; n];
        dist_a[0] = 1.0;
        let sum_a: f64 = dist_a.iter().sum();
        for p in dist_a.iter_mut() {
            *p /= sum_a;
        }

        // Distribution B: concentrated on star 3 (blade 2 — opposite blade)
        let mut dist_b = vec![1e-10; n];
        dist_b[2] = 1.0;
        let sum_b: f64 = dist_b.iter().sum();
        for p in dist_b.iter_mut() {
            *p /= sum_b;
        }

        let bary = wasserstein_barycenter(
            &[dist_a, dist_b],
            &[0.5, 0.5],
            Pool::Stars,
            0.1,
            50,
        );
        assert_valid_distribution(&bary, Pool::Stars, 1e-6);

        // The barycenter should have mass on both star 1 and star 3 regions,
        // and more total mass near those two than on far-away stars.
        let mass_near = bary[0] + bary[2]; // stars 1 and 3
        let mass_far = bary[5] + bary[9]; // stars 6 and 10 (different blades)
        assert!(
            mass_near > mass_far,
            "Mass near sources ({:.4}) should exceed mass far ({:.4})",
            mass_near,
            mass_far
        );
    }

    #[test]
    fn test_cost_matrix_balls_same_row_cheaper() {
        // Balls in the same decade row should have lower cost than balls in different rows.
        let cost = build_cost_matrix(Pool::Balls);

        // Ball 1 (idx 0, row 0) to ball 5 (idx 4, row 0): same row
        let same_row_cost = cost[0][4];
        // Ball 1 (idx 0, row 0) to ball 15 (idx 14, row 1): adjacent row
        let adj_row_cost = cost[0][14];
        // Ball 1 (idx 0, row 0) to ball 45 (idx 44, row 4): far row
        let far_row_cost = cost[0][44];

        assert!(
            same_row_cost < adj_row_cost,
            "Same row cost ({:.4}) should be less than adjacent row ({:.4})",
            same_row_cost,
            adj_row_cost,
        );
        assert!(
            adj_row_cost < far_row_cost,
            "Adjacent row cost ({:.4}) should be less than far row ({:.4})",
            adj_row_cost,
            far_row_cost,
        );
    }

    #[test]
    fn test_cost_matrix_stars_blade_proximity() {
        // Stars on the same Paquerette blade should be cheapest, opposite blades most expensive.
        let cost = build_cost_matrix(Pool::Stars);

        // Star 1 (idx 0, blade 0) and star 5 (idx 4, blade 0): same blade
        let same_blade = cost[0][4];
        // Star 1 (idx 0, blade 0) and star 2 (idx 1, blade 1): adjacent blade
        let adj_blade = cost[0][1];
        // Star 1 (idx 0, blade 0) and star 3 (idx 2, blade 2): opposite blade
        let opp_blade = cost[0][2];

        assert!(
            (same_blade - 0.5).abs() < 1e-9,
            "Same blade cost should be 0.5, got {}",
            same_blade
        );
        assert!(
            (adj_blade - 1.0).abs() < 1e-9,
            "Adjacent blade cost should be 1.0, got {}",
            adj_blade
        );
        assert!(
            (opp_blade - 1.5).abs() < 1e-9,
            "Opposite blade cost should be 1.5, got {}",
            opp_blade
        );
    }

    #[test]
    fn test_cost_matrix_diagonal_zero() {
        // Diagonal of cost matrix should be zero (cost of transporting to self).
        for pool in [Pool::Balls, Pool::Stars] {
            let cost = build_cost_matrix(pool);
            for i in 0..pool.size() {
                assert!(
                    cost[i][i].abs() < 1e-15,
                    "Cost[{0}][{0}] should be 0, got {1} for {2:?}",
                    i,
                    cost[i][i],
                    pool,
                );
            }
        }
    }

    #[test]
    fn test_weight_sensitivity() {
        // Heavier weight on one distribution should pull the barycenter towards it.
        let n = Pool::Stars.size();

        // Distribution A: concentrated on star 1
        let mut dist_a = vec![1e-10; n];
        dist_a[0] = 0.9;
        dist_a[1] = 0.1;
        let sum_a: f64 = dist_a.iter().sum();
        for p in dist_a.iter_mut() {
            *p /= sum_a;
        }

        // Distribution B: concentrated on star 7
        let mut dist_b = vec![1e-10; n];
        dist_b[6] = 0.9;
        dist_b[7] = 0.1;
        let sum_b: f64 = dist_b.iter().sum();
        for p in dist_b.iter_mut() {
            *p /= sum_b;
        }

        // Heavy weight on A
        let bary_a = wasserstein_barycenter(
            &[dist_a.clone(), dist_b.clone()],
            &[0.9, 0.1],
            Pool::Stars,
            0.1,
            50,
        );
        // Heavy weight on B
        let bary_b = wasserstein_barycenter(
            &[dist_a, dist_b],
            &[0.1, 0.9],
            Pool::Stars,
            0.1,
            50,
        );

        assert_valid_distribution(&bary_a, Pool::Stars, 1e-6);
        assert_valid_distribution(&bary_b, Pool::Stars, 1e-6);

        // bary_a should have more mass on star 1 (idx 0) than bary_b
        assert!(
            bary_a[0] > bary_b[0],
            "Heavy-A barycenter should have more mass on star 1: {:.6} vs {:.6}",
            bary_a[0],
            bary_b[0],
        );
        // bary_b should have more mass on star 7 (idx 6) than bary_a
        assert!(
            bary_b[6] > bary_a[6],
            "Heavy-B barycenter should have more mass on star 7: {:.6} vs {:.6}",
            bary_b[6],
            bary_a[6],
        );
    }

    #[test]
    fn test_empty_inputs_return_uniform() {
        let bary = wasserstein_barycenter(&[], &[], Pool::Balls, 0.1, 50);
        assert_valid_distribution(&bary, Pool::Balls, 1e-6);
        let expected = 1.0 / 50.0;
        for &p in &bary {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_balls_barycenter_respects_row_structure() {
        // Two distributions concentrated on different rows should produce a barycenter
        // that respects the physical cost structure (not just a pointwise average).
        let n = Pool::Balls.size();

        // Distribution A: uniform over row 0 (balls 1-10)
        let mut dist_a = vec![1e-10; n];
        for i in 0..10 {
            dist_a[i] = 0.1;
        }
        let sum_a: f64 = dist_a.iter().sum();
        for p in dist_a.iter_mut() {
            *p /= sum_a;
        }

        // Distribution B: uniform over row 4 (balls 41-50)
        let mut dist_b = vec![1e-10; n];
        for i in 40..50 {
            dist_b[i] = 0.1;
        }
        let sum_b: f64 = dist_b.iter().sum();
        for p in dist_b.iter_mut() {
            *p /= sum_b;
        }

        // Use higher epsilon (0.5) to allow more mass spreading across rows.
        // At epsilon=0.1 the kernel is very peaked and nearly all mass stays on source rows.
        let bary = wasserstein_barycenter(
            &[dist_a.clone(), dist_b.clone()],
            &[0.5, 0.5],
            Pool::Balls,
            0.5,
            50,
        );
        assert_valid_distribution(&bary, Pool::Balls, 1e-6);

        // The Wasserstein barycenter should spread mass more towards intermediate rows
        // compared to a naive pointwise average (which would only have mass on rows 0 and 4).
        // Check that middle rows (2, 3) have non-trivial mass.
        let row2_mass: f64 = bary[20..30].iter().sum();
        assert!(
            row2_mass > 0.01,
            "Middle row should have some mass in Wasserstein barycenter, got {:.6}",
            row2_mass,
        );

        // Also verify that with epsilon=0.1 (production setting), the barycenter still
        // has more mass on the source rows than the middle, demonstrating transport cost
        // sensitivity. This is the expected behavior: low epsilon = less blurring.
        let bary_tight = wasserstein_barycenter(
            &[dist_a, dist_b],
            &[0.5, 0.5],
            Pool::Balls,
            0.1,
            50,
        );
        assert_valid_distribution(&bary_tight, Pool::Balls, 1e-6);
        let source_mass: f64 = bary_tight[0..10].iter().copied().sum::<f64>()
            + bary_tight[40..50].iter().copied().sum::<f64>();
        let middle_mass: f64 = bary_tight[10..40].iter().copied().sum();
        assert!(
            source_mass > middle_mass,
            "Low-epsilon barycenter should concentrate on source rows: source={:.4}, middle={:.4}",
            source_mass,
            middle_mass,
        );
    }
}
