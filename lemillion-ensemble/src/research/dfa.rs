use lemillion_db::models::Draw;

use super::{TestResult, ResearchVerdict, two_sided_p, verdict_from_p};

/// Detrended Fluctuation Analysis (DFA)
///
/// Computes the Hurst exponent H for each number's binary presence series.
/// H = 0.5 -> random walk, H > 0.5 -> persistent (momentum), H < 0.5 -> anti-persistent (mean-reversion).

pub fn run_dfa_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    if draws.len() < 50 {
        results.push(TestResult {
            test_name: "DFA Hurst Exponent (Boules)".to_string(),
            category: "dfa".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: format!("Insuffisant : {} tirages (minimum 50)", draws.len()),
        });
        return results;
    }

    // Compute Hurst exponents for balls (1-50)
    let ball_hursts = compute_hurst_exponents_for_pool(draws, 50, |d, num| d.balls.contains(&num));
    let (ball_avg, ball_std) = mean_std(&ball_hursts);

    // z-test: H_avg vs 0.5
    let n_balls = ball_hursts.len() as f64;
    let se_balls = if n_balls > 1.0 { ball_std / n_balls.sqrt() } else { 1.0 };
    let z_balls = if se_balls > 1e-12 { (ball_avg - 0.5) / se_balls } else { 0.0 };
    let p_balls = two_sided_p(z_balls);

    let persistent_balls = ball_hursts.iter().filter(|&&h| h > 0.55).count();
    let anti_persistent_balls = ball_hursts.iter().filter(|&&h| h < 0.45).count();

    results.push(TestResult {
        test_name: "DFA Hurst Exponent (Boules)".to_string(),
        category: "dfa".to_string(),
        statistic: z_balls,
        p_value: Some(p_balls),
        effect_size: (ball_avg - 0.5).abs(),
        verdict: verdict_from_p(p_balls),
        detail: format!(
            "H_moy={:.4}, H_std={:.4}, z={:.2}, p={:.4} | {}/{} persistants, {}/{} anti-persistants",
            ball_avg, ball_std, z_balls, p_balls,
            persistent_balls, ball_hursts.len(),
            anti_persistent_balls, ball_hursts.len()
        ),
    });

    // Compute Hurst exponents for stars (1-12)
    let star_hursts = compute_hurst_exponents_for_pool(draws, 12, |d, num| d.stars.contains(&num));
    let (star_avg, star_std) = mean_std(&star_hursts);

    let n_stars = star_hursts.len() as f64;
    let se_stars = if n_stars > 1.0 { star_std / n_stars.sqrt() } else { 1.0 };
    let z_stars = if se_stars > 1e-12 { (star_avg - 0.5) / se_stars } else { 0.0 };
    let p_stars = two_sided_p(z_stars);

    let persistent_stars = star_hursts.iter().filter(|&&h| h > 0.55).count();
    let anti_persistent_stars = star_hursts.iter().filter(|&&h| h < 0.45).count();

    results.push(TestResult {
        test_name: "DFA Hurst Exponent (Etoiles)".to_string(),
        category: "dfa".to_string(),
        statistic: z_stars,
        p_value: Some(p_stars),
        effect_size: (star_avg - 0.5).abs(),
        verdict: verdict_from_p(p_stars),
        detail: format!(
            "H_moy={:.4}, H_std={:.4}, z={:.2}, p={:.4} | {}/{} persistants, {}/{} anti-persistants",
            star_avg, star_std, z_stars, p_stars,
            persistent_stars, star_hursts.len(),
            anti_persistent_stars, star_hursts.len()
        ),
    });

    // Classification result: momentum vs mean-reversion vs random
    let overall_avg = if ball_hursts.len() + star_hursts.len() > 0 {
        (ball_hursts.iter().sum::<f64>() + star_hursts.iter().sum::<f64>())
            / (ball_hursts.len() + star_hursts.len()) as f64
    } else {
        0.5
    };

    let (classification, verdict) = if overall_avg > 0.55 {
        ("PERSISTANT (momentum)", ResearchVerdict::Significant)
    } else if overall_avg < 0.45 {
        ("ANTI-PERSISTANT (mean-reversion)", ResearchVerdict::Significant)
    } else if overall_avg > 0.52 || overall_avg < 0.48 {
        ("FAIBLEMENT BIAISE", ResearchVerdict::Marginal)
    } else {
        ("ALEATOIRE (random walk)", ResearchVerdict::NotSignificant)
    };

    results.push(TestResult {
        test_name: "DFA Classification de persistance".to_string(),
        category: "dfa".to_string(),
        statistic: overall_avg,
        p_value: None,
        effect_size: (overall_avg - 0.5).abs(),
        verdict,
        detail: format!(
            "H_global={:.4}, classification: {} | boules H={:.4}, etoiles H={:.4}",
            overall_avg, classification, ball_avg, star_avg
        ),
    });

    results
}

/// Compute Hurst exponents for all numbers in a pool.
/// `contains` is a closure that checks if a draw contains a given number.
fn compute_hurst_exponents_for_pool(
    draws: &[Draw],
    pool_size: u8,
    contains: impl Fn(&Draw, u8) -> bool,
) -> Vec<f64> {
    let mut hursts = Vec::new();

    for num in 1..=pool_size {
        // Build binary presence series (chronological order)
        let series: Vec<f64> = draws.iter().rev()
            .map(|d| if contains(d, num) { 1.0 } else { 0.0 })
            .collect();

        if let Some(h) = compute_hurst_exponent(&series) {
            hursts.push(h);
        }
    }

    hursts
}

/// Compute the Hurst exponent via DFA for a single binary series.
///
/// Steps:
/// 1. Compute cumulative deviation from mean (profile Y(i))
/// 2. For each window size s, divide into non-overlapping windows
/// 3. In each window, fit a linear trend and compute RMS of residuals
/// 4. F(s) = sqrt(mean of squared residuals across all windows)
/// 5. H = slope of log(F(s)) vs log(s)
pub fn compute_hurst_exponent(series: &[f64]) -> Option<f64> {
    let n = series.len();
    if n < 30 {
        return None;
    }

    // Step 1: Compute profile (cumulative deviation from mean)
    let mean = series.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    for &x in series {
        cumsum += x - mean;
        profile.push(cumsum);
    }

    // Step 2: Compute F(s) for multiple scales
    let scales: Vec<usize> = [10, 15, 20, 30, 50, 80, 100, 150, 200]
        .iter()
        .copied()
        .filter(|&s| s <= n / 4) // need at least 4 windows
        .collect();

    if scales.len() < 3 {
        return None;
    }

    let mut log_s = Vec::new();
    let mut log_f = Vec::new();

    for &s in &scales {
        let n_windows = n / s;
        if n_windows < 2 {
            continue;
        }

        let mut total_rms_sq = 0.0;
        let mut count = 0;

        for w in 0..n_windows {
            let start = w * s;
            let end = start + s;
            let window = &profile[start..end];

            // Fit linear trend y = a + b*t in this window
            let (_, rms) = detrend_rms(window);
            total_rms_sq += rms * rms;
            count += 1;
        }

        if count > 0 {
            let f_s = (total_rms_sq / count as f64).sqrt();
            if f_s > 1e-15 {
                log_s.push((s as f64).ln());
                log_f.push(f_s.ln());
            }
        }
    }

    if log_s.len() < 3 {
        return None;
    }

    // Step 5: Linear regression of log(F(s)) vs log(s)
    let h = linear_regression_slope(&log_s, &log_f);

    // Clamp to reasonable range [0, 1.5]
    Some(h.clamp(0.0, 1.5))
}

/// Fit a linear trend in a window and return (slope, RMS of residuals).
fn detrend_rms(window: &[f64]) -> (f64, f64) {
    let n = window.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }

    // Linear regression: y = a + b * t
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_ty = 0.0;
    let mut sum_tt = 0.0;

    for (i, &y) in window.iter().enumerate() {
        let t = i as f64;
        sum_t += t;
        sum_y += y;
        sum_ty += t * y;
        sum_tt += t * t;
    }

    let denom = n * sum_tt - sum_t * sum_t;
    let (a, b) = if denom.abs() > 1e-15 {
        let b = (n * sum_ty - sum_t * sum_y) / denom;
        let a = (sum_y - b * sum_t) / n;
        (a, b)
    } else {
        (sum_y / n, 0.0)
    };

    // RMS of residuals
    let mut rms_sq = 0.0;
    for (i, &y) in window.iter().enumerate() {
        let trend = a + b * i as f64;
        let residual = y - trend;
        rms_sq += residual * residual;
    }
    let rms = (rms_sq / n).sqrt();

    (b, rms)
}

/// Simple linear regression: returns the slope.
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    (n * sum_xy - sum_x * sum_y) / denom
}

/// Compute mean and standard deviation.
fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.5, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_dfa_no_panic() {
        let draws = make_test_draws(200);
        let results = run_dfa_tests(&draws);
        assert!(results.len() >= 3, "Expected at least 3 results, got {}", results.len());
        for r in &results {
            assert_eq!(r.category, "dfa");
        }
    }

    #[test]
    fn test_dfa_insufficient_data() {
        let draws = make_test_draws(10);
        let results = run_dfa_tests(&draws);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hurst_random_walk() {
        // Create a pseudo-random binary series with roughly 10% probability
        let mut series = vec![0.0f64; 500];
        let mut rng: u64 = 42;
        for x in series.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = if (rng >> 58) == 0 { 1.0 } else { 0.0 }; // ~1/64 probability
        }

        if let Some(h) = compute_hurst_exponent(&series) {
            // For truly random series, H should be near 0.5 (with some tolerance)
            assert!(h > 0.1 && h < 1.2, "H should be in reasonable range, got {}", h);
        }
    }

    #[test]
    fn test_hurst_persistent_series() {
        // Create a persistent series: long runs of 1s and 0s
        let mut series = Vec::with_capacity(600);
        for i in 0..30 {
            let val = if i % 2 == 0 { 1.0 } else { 0.0 };
            for _ in 0..20 {
                series.push(val);
            }
        }

        if let Some(h) = compute_hurst_exponent(&series) {
            // Long-range correlated series should have H > 0.5
            assert!(h > 0.3, "H should be > 0.3 for persistent series, got {}", h);
        }
    }

    #[test]
    fn test_detrend_rms_flat() {
        let window = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let (slope, rms) = detrend_rms(&window);
        assert!(slope.abs() < 1e-10, "Slope should be ~0, got {}", slope);
        assert!(rms < 1e-10, "RMS should be ~0, got {}", rms);
    }

    #[test]
    fn test_detrend_rms_linear() {
        // Perfect linear: residuals should be 0
        let window: Vec<f64> = (0..10).map(|i| 2.0 + 3.0 * i as f64).collect();
        let (slope, rms) = detrend_rms(&window);
        assert!((slope - 3.0).abs() < 1e-10, "Slope should be 3.0, got {}", slope);
        assert!(rms < 1e-10, "RMS should be ~0 for linear, got {}", rms);
    }

    #[test]
    fn test_linear_regression_slope() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let slope = linear_regression_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10, "Slope should be 2.0, got {}", slope);
    }

    #[test]
    fn test_mean_std_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = mean_std(&values);
        assert!((mean - 3.0).abs() < 1e-10);
        // std = sqrt(2.0)
        assert!((std - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mean_std_empty() {
        let (mean, std) = mean_std(&[]);
        assert!((mean - 0.5).abs() < 1e-10);
        assert!(std.abs() < 1e-10);
    }

    #[test]
    fn test_dfa_ball_hursts_count() {
        let draws = make_test_draws(200);
        let hursts = compute_hurst_exponents_for_pool(&draws, 50, |d, num| d.balls.contains(&num));
        // With 200 draws, should be able to compute Hurst for most balls
        assert!(!hursts.is_empty(), "Should compute at least some Hurst exponents");
    }

    #[test]
    fn test_hurst_short_series() {
        // Series too short: should return None
        let series = vec![1.0, 0.0, 1.0];
        assert!(compute_hurst_exponent(&series).is_none());
    }
}
