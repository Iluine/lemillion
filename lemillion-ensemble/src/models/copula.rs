use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Gaussian Copula model for EuroMillions prediction.
///
/// Separates marginal distributions (EWMA frequencies) from the dependency
/// structure (Spearman rank correlation matrix). Prediction uses conditional
/// Gaussian regression: given which numbers appeared in the last draw,
/// estimate the latent z-score for each number and convert back to
/// probability via the Gaussian CDF.
///
/// Algorithm:
/// 1. Compute EWMA frequencies for each number (marginals)
/// 2. Build binary presence vectors over window, compute Spearman rank
///    correlations between all number pairs
/// 3. For numbers present in the last draw, set z_j = Φ⁻¹(freq_j)
/// 4. Predict z_i = Σ_j ρ_{ij} × z_j for each target i
/// 5. Convert to probabilities via Φ(z_i), normalize, smooth, floor
pub struct CopulaModel {
    window: usize,
    ewma_alpha: f64,
    smoothing: f64,
    min_draws: usize,
}

impl Default for CopulaModel {
    fn default() -> Self {
        Self {
            window: 100,
            ewma_alpha: 0.05,
            smoothing: 0.25,
            min_draws: 30,
        }
    }
}

/// Standard normal CDF approximation using Hart's algorithm (1968).
/// Accurate to ~7.5e-8 for all x.
fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 1e-15;
    }
    if x > 8.0 {
        return 1.0 - 1e-15;
    }

    // Use the complementary error function relationship:
    // Φ(x) = 0.5 × erfc(-x / √2)
    // Implement erfc via a high-accuracy rational approximation.
    let abs_x = x.abs();
    let t = abs_x / std::f64::consts::SQRT_2;

    // Rational approximation for erfc(t) based on Horner form
    // From Cody (1969), accurate to ~1e-7
    let tau = 1.0 / (1.0 + 0.5 * t);

    let ans = tau
        * (-t * t - 1.26551223
            + tau
                * (1.00002368
                    + tau
                        * (0.37409196
                            + tau
                                * (0.09678418
                                    + tau
                                        * (-0.18628806
                                            + tau
                                                * (0.27886807
                                                    + tau
                                                        * (-1.13520398
                                                            + tau
                                                                * (1.48851587
                                                                    + tau
                                                                        * (-0.82215223
                                                                            + tau
                                                                                * 0.17087277)))))))))
        .exp();

    if x >= 0.0 {
        1.0 - 0.5 * ans
    } else {
        0.5 * ans
    }
}

/// Inverse standard normal CDF (probit function) via rational approximation.
/// Uses the Beasley-Springer-Moro algorithm.
fn norm_ppf(p: f64) -> f64 {
    // Clamp to avoid infinities
    let p = p.clamp(1e-10, 1.0 - 1e-10);

    // Rational approximation for central region
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let mut result = if p < p_low {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };

    // Newton-Raphson refinement for improved accuracy:
    // x_{n+1} = x_n - (Φ(x_n) - p) / φ(x_n)
    // where φ(x) = (1/√(2π)) exp(-x²/2)
    for _ in 0..5 {
        let phi = (-(result * result) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        if phi > 1e-15 {
            let err = norm_cdf(result) - p;
            if err.abs() < 1e-12 {
                break;
            }
            result -= err / phi;
        } else {
            break;
        }
    }
    result
}

/// Compute Spearman rank of a slice (handling ties via average rank).
fn spearman_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Create (value, original_index) pairs and sort by value
    let mut indexed: Vec<(f64, usize)> = data.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find all elements with the same value (ties)
        let mut j = i;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for tied values (1-based ranks)
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.1] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Compute Pearson correlation between two slices.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }

    let n_f = n as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n_f;
    let mean_y: f64 = y.iter().sum::<f64>() / n_f;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

impl ForecastModel for CopulaModel {
    fn name(&self) -> &str {
        "Copula"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let uniform = vec![1.0 / n as f64; n];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let window = self.window.min(draws.len());

        // 1. Compute EWMA frequencies for each number
        //    draws[0] = most recent, iterate from oldest to newest
        let mut ewma_freq = vec![0.0f64; n];
        let p_base = pool.pick_count() as f64 / n as f64;
        for f in &mut ewma_freq {
            *f = p_base; // initialize to expected frequency
        }

        for t in (0..window).rev() {
            let draw = &draws[t];
            let mut present = vec![0.0f64; n];
            for &num in pool.numbers_from(draw) {
                let idx = (num - 1) as usize;
                if idx < n {
                    present[idx] = 1.0;
                }
            }
            for i in 0..n {
                ewma_freq[i] = self.ewma_alpha * present[i] + (1.0 - self.ewma_alpha) * ewma_freq[i];
            }
        }

        // 2. Build binary presence matrix: [window][n]
        //    Row t = draw at position t (0 = most recent)
        let mut presence = vec![vec![0.0f64; window]; n];
        for t in 0..window {
            for &num in pool.numbers_from(&draws[t]) {
                let idx = (num - 1) as usize;
                if idx < n {
                    presence[idx][t] = 1.0;
                }
            }
        }

        // 3. Compute Spearman rank correlation matrix
        //    Only compute for pairs where both numbers have some variance
        let ranked: Vec<Vec<f64>> = presence.iter().map(|series| spearman_ranks(series)).collect();

        // Flat correlation matrix for cache locality
        let mut corr_matrix = vec![0.0f64; n * n];
        for i in 0..n {
            corr_matrix[i * n + i] = 1.0; // diagonal
            for j in (i + 1)..n {
                let rho = pearson_correlation(&ranked[i], &ranked[j]);
                corr_matrix[i * n + j] = rho;
                corr_matrix[j * n + i] = rho;
            }
        }

        // 4. Transform EWMA frequencies to z-scores via Φ⁻¹
        //    Clamp frequencies to (0, 1) to avoid infinities
        let z_marginal: Vec<f64> = ewma_freq
            .iter()
            .map(|&f| norm_ppf(f.clamp(0.001, 0.999)))
            .collect();

        // 5. Conditional prediction using last draw context
        //    For numbers present in the last draw, use their z-scores as conditioning
        //    E[z_i | z_observed] ≈ Σ_{j in observed} ρ_{ij} × z_j / |observed|
        let last_present: Vec<usize> = if !draws.is_empty() {
            pool.numbers_from(&draws[0])
                .iter()
                .map(|&num| (num - 1) as usize)
                .filter(|&idx| idx < n)
                .collect()
        } else {
            vec![]
        };

        let mut z_predicted = vec![0.0f64; n];

        if last_present.is_empty() {
            // No context: just use marginal z-scores
            z_predicted = z_marginal;
        } else {
            let n_obs = last_present.len() as f64;
            for i in 0..n {
                let mut z_cond = 0.0;
                for &j in &last_present {
                    z_cond += corr_matrix[i * n + j] * z_marginal[j];
                }
                // Blend conditional and marginal: the conditional adjustment is
                // averaged over observed numbers and added to the marginal baseline
                z_predicted[i] = z_marginal[i] * 0.5 + (z_cond / n_obs) * 0.5;
            }
        }

        // 6. Convert z-scores to probabilities via Φ(z)
        let mut probs: Vec<f64> = z_predicted.iter().map(|&z| norm_cdf(z)).collect();

        // 7. Normalize to a probability distribution
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // 8. Smooth towards uniform
        let uniform_val = 1.0 / n as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // 9. Floor and renormalize
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
            ("window".into(), self.window as f64),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
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
    fn test_copula_valid_distribution_balls() {
        let draws = make_test_draws(100);
        let model = CopulaModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_copula_valid_distribution_stars() {
        let draws = make_test_draws(100);
        let model = CopulaModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_copula_deterministic() {
        let draws = make_test_draws(100);
        let model = CopulaModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Copula should be deterministic");
        }
    }

    #[test]
    fn test_copula_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = CopulaModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_copula_empty_draws() {
        let model = CopulaModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_copula_no_negative() {
        let draws = make_test_draws(200);
        let model = CopulaModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_copula_large_draws() {
        let draws = make_test_draws(300);
        let model = CopulaModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_norm_cdf_basic() {
        // Φ(0) = 0.5
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        // Φ(+∞) → 1
        assert!(norm_cdf(5.0) > 0.999);
        // Φ(-∞) → 0
        assert!(norm_cdf(-5.0) < 0.001);
        // Symmetry: Φ(-x) = 1 - Φ(x)
        let x = 1.5;
        assert!((norm_cdf(-x) - (1.0 - norm_cdf(x))).abs() < 1e-6);
    }

    #[test]
    fn test_norm_ppf_basic() {
        // Φ⁻¹(0.5) = 0
        assert!(norm_ppf(0.5).abs() < 1e-6);
        // Round-trip: Φ(Φ⁻¹(p)) ≈ p
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let z = norm_ppf(p);
            let p_back = norm_cdf(z);
            assert!(
                (p - p_back).abs() < 1e-5,
                "Round-trip failed for p={}: got {}",
                p,
                p_back
            );
        }
    }

    #[test]
    fn test_spearman_ranks_basic() {
        let data = vec![3.0, 1.0, 2.0];
        let ranks = spearman_ranks(&data);
        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 is largest -> rank 3
        assert!((ranks[1] - 1.0).abs() < 1e-10); // 1.0 is smallest -> rank 1
        assert!((ranks[2] - 2.0).abs() < 1e-10); // 2.0 is middle -> rank 2
    }

    #[test]
    fn test_spearman_ranks_ties() {
        let data = vec![1.0, 1.0, 3.0];
        let ranks = spearman_ranks(&data);
        // Two tied values at 1.0 get average rank = (1+2)/2 = 1.5
        assert!((ranks[0] - 1.5).abs() < 1e-10);
        assert!((ranks[1] - 1.5).abs() < 1e-10);
        assert!((ranks[2] - 3.0).abs() < 1e-10);
    }
}
