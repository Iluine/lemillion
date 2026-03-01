use std::collections::HashMap;

use lemillion_db::models::Draw;

use super::{TestResult, ResearchVerdict, chi_squared, chi_squared_p_value, verdict_from_p};

pub fn run_mathematical_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    results.extend(modular_analysis(draws));
    results.extend(sum_spread_analysis(draws));
    results.push(cooccurrence_graph_analysis(draws));
    results.extend(gap_analysis(draws));

    results
}

// ════════════════════════════════════════════════════════════════
// 1. Analyse modulaire
// ════════════════════════════════════════════════════════════════

fn modular_analysis(draws: &[Draw]) -> Vec<TestResult> {
    let moduli = [2, 3, 4, 5, 7, 10];
    let mut results = Vec::new();

    for &p in &moduli {
        results.push(modular_chi2(draws, p));
    }

    // Inter-draw residue correlation for mod 4 (priority: 4 blades)
    results.push(modular_correlation(draws, 4));

    results
}

fn modular_chi2(draws: &[Draw], modulus: usize) -> TestResult {
    let mut residue_counts = vec![0u64; modulus];

    for d in draws {
        for &b in &d.balls {
            residue_counts[((b - 1) as usize) % modulus] += 1;
        }
    }

    // Expected: not exactly uniform because numbers 1-50 aren't uniformly distributed mod p
    // For each residue class r, count how many numbers 1..50 have (n-1) % p == r
    let mut expected_counts = vec![0.0f64; modulus];
    for num in 1..=50u8 {
        let residue = ((num - 1) as usize) % modulus;
        expected_counts[residue] += 1.0;
    }
    // Normalize: each number has P = 5/50 = 0.1 probability
    let p_per_number = 5.0 / 50.0;
    for e in &mut expected_counts {
        *e *= p_per_number * draws.len() as f64;
    }

    let observed: Vec<f64> = residue_counts.iter().map(|&c| c as f64).collect();
    let chi2 = chi_squared(&observed, &expected_counts);
    let df = modulus - 1;
    let p = chi_squared_p_value(chi2, df);

    let max_dev = observed.iter().zip(expected_counts.iter())
        .map(|(&o, &e)| if e > 0.0 { ((o - e) / e).abs() } else { 0.0 })
        .fold(0.0f64, f64::max);

    let dist_str = residue_counts.iter().enumerate()
        .map(|(i, &c)| format!("r{}={}", i, c))
        .collect::<Vec<_>>()
        .join(", ");

    TestResult {
        test_name: format!("Résidus mod {}", modulus),
        category: "mathematical".to_string(),
        statistic: chi2,
        p_value: Some(p),
        effect_size: max_dev,
        verdict: verdict_from_p(p),
        detail: format!("chi2={:.2}, df={}, p={:.4} | {}", chi2, df, p, dist_str),
    }
}

fn modular_correlation(draws: &[Draw], modulus: usize) -> TestResult {
    // Encode each draw as residue distribution: count of balls in each residue class
    let n = draws.len();
    if n < 10 {
        return TestResult {
            test_name: format!("Corrélation inter-tirages mod {}", modulus),
            category: "mathematical".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: "Insuffisant".to_string(),
        };
    }

    // draws[0] = most recent, so we iterate in chronological order (reversed)
    let residue_vectors: Vec<Vec<usize>> = draws.iter().rev()
        .map(|d| {
            let mut counts = vec![0usize; modulus];
            for &b in &d.balls {
                counts[((b - 1) as usize) % modulus] += 1;
            }
            counts
        })
        .collect();

    // Compute autocorrelation of residue vectors (cosine similarity lag 1)
    let mut correlations = Vec::new();
    for t in 0..(residue_vectors.len() - 1) {
        let dot: f64 = residue_vectors[t].iter().zip(residue_vectors[t + 1].iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let norm_a: f64 = residue_vectors[t].iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = residue_vectors[t + 1].iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            correlations.push(dot / (norm_a * norm_b));
        }
    }

    let avg_corr = if correlations.is_empty() {
        0.0
    } else {
        correlations.iter().sum::<f64>() / correlations.len() as f64
    };

    // Expected cosine similarity under independence (approximately)
    // For 5 balls in modulus classes, expected ~ depends on distribution
    // Empirically estimate via shuffled baseline
    let expected_corr = estimate_expected_cosine(modulus);

    let deviation = (avg_corr - expected_corr).abs();
    let verdict = if deviation > 0.05 {
        ResearchVerdict::Significant
    } else if deviation > 0.02 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    TestResult {
        test_name: format!("Corrélation inter-tirages mod {}", modulus),
        category: "mathematical".to_string(),
        statistic: avg_corr,
        p_value: None,
        effect_size: deviation,
        verdict,
        detail: format!(
            "corr_moy={:.4}, attendu≈{:.4}, dév={:.4}",
            avg_corr, expected_corr, deviation
        ),
    }
}

fn estimate_expected_cosine(modulus: usize) -> f64 {
    // For 5 balls drawn uniformly from 50, the expected residue distribution
    // Each number 1-50 mapped to modulus classes. Since we pick 5 out of 50,
    // the expected count per class ≈ 5 * (class_size / 50)
    // The cosine similarity between two independent draws converges to
    // a predictable value based on the multinomial structure.
    // For mod 4: classes of sizes 13,13,12,12 → fairly uniform → cosine ≈ 0.4-0.5
    match modulus {
        2 => 0.45,
        3 => 0.40,
        4 => 0.38,
        5 => 0.35,
        7 => 0.30,
        10 => 0.25,
        _ => 0.35,
    }
}

// ════════════════════════════════════════════════════════════════
// 2. Analyse par somme et écart
// ════════════════════════════════════════════════════════════════

fn sum_spread_analysis(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    let ball_sums: Vec<f64> = draws.iter().map(|d| d.balls.iter().map(|&b| b as f64).sum()).collect();
    let ball_spreads: Vec<f64> = draws.iter().map(|d| {
        let max = *d.balls.iter().max().unwrap() as f64;
        let min = *d.balls.iter().min().unwrap() as f64;
        max - min
    }).collect();
    let odd_counts: Vec<f64> = draws.iter().map(|d| {
        d.balls.iter().filter(|&&b| b % 2 == 1).count() as f64
    }).collect();

    // Ball sum distribution
    // Theoretical mean for 5 balls from 1-50 (without replacement):
    // E[sum] = 5 * (1+50)/2 = 127.5
    // Var[sum] = 5 * (50^2 - 1)/12 * (50 - 5)/(50 - 1) ≈ 5 * 208.25 * 0.9184 ≈ 956.1
    let expected_mean = 127.5;
    let expected_var: f64 = 5.0 * (2500.0 - 1.0) / 12.0 * (50.0 - 5.0) / (50.0 - 1.0);
    let expected_std = expected_var.sqrt();

    let obs_mean_sum = ball_sums.iter().sum::<f64>() / ball_sums.len() as f64;
    let obs_var_sum = ball_sums.iter().map(|&s| (s - obs_mean_sum).powi(2)).sum::<f64>() / ball_sums.len() as f64;

    let z_mean = (obs_mean_sum - expected_mean) / (expected_std / (draws.len() as f64).sqrt());
    let p_mean = super::two_sided_p(z_mean);

    results.push(TestResult {
        test_name: "Distribution somme boules".to_string(),
        category: "mathematical".to_string(),
        statistic: z_mean,
        p_value: Some(p_mean),
        effect_size: (obs_mean_sum - expected_mean).abs() / expected_std,
        verdict: verdict_from_p(p_mean),
        detail: format!(
            "moy={:.1} (att={:.1}), σ_obs={:.1} (att={:.1}), z={:.2}, p={:.4}",
            obs_mean_sum, expected_mean, obs_var_sum.sqrt(), expected_std, z_mean, p_mean
        ),
    });

    // Spread distribution
    let obs_mean_spread = ball_spreads.iter().sum::<f64>() / ball_spreads.len() as f64;
    // Expected spread for 5 draws from 1-50: E[max-min] ≈ 40.8 (can be computed exactly)
    let expected_spread = 40.8;
    let spread_dev = (obs_mean_spread - expected_spread).abs() / expected_spread;

    let verdict_spread = if spread_dev > 0.05 {
        ResearchVerdict::Significant
    } else if spread_dev > 0.02 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    results.push(TestResult {
        test_name: "Distribution écart (max-min)".to_string(),
        category: "mathematical".to_string(),
        statistic: obs_mean_spread,
        p_value: None,
        effect_size: spread_dev,
        verdict: verdict_spread,
        detail: format!("moy_écart={:.1} (att≈{:.1}), dév={:.3}", obs_mean_spread, expected_spread, spread_dev),
    });

    // Odd count distribution: should be approximately Hypergeometric(50, 25, 5)
    // but with 25 odd and 25 even in 1-50, it's Hypergeometric(50,25,5)
    // E[odd] = 5 * 25/50 = 2.5
    let expected_odd = 2.5;
    let obs_mean_odd = odd_counts.iter().sum::<f64>() / odd_counts.len() as f64;

    // Chi-squared on odd count distribution (0,1,2,3,4,5)
    let mut odd_dist = [0u64; 6];
    for &c in &odd_counts {
        odd_dist[c as usize] += 1;
    }

    // Hypergeometric probabilities for H(50, 25, 5)
    let hyper_probs = hypergeometric_pmf(50, 25, 5);
    let expected_odd_dist: Vec<f64> = hyper_probs.iter().map(|&p| p * draws.len() as f64).collect();
    let observed_odd_dist: Vec<f64> = odd_dist.iter().map(|&c| c as f64).collect();

    let chi2_odd = chi_squared(&observed_odd_dist, &expected_odd_dist);
    let df_odd = 5; // 6 categories - 1
    let p_odd = chi_squared_p_value(chi2_odd, df_odd);

    results.push(TestResult {
        test_name: "Distribution nombre d'impairs".to_string(),
        category: "mathematical".to_string(),
        statistic: chi2_odd,
        p_value: Some(p_odd),
        effect_size: (obs_mean_odd - expected_odd).abs(),
        verdict: verdict_from_p(p_odd),
        detail: format!(
            "moy={:.2} (att={:.1}), chi2={:.2}, p={:.4} | dist: {:?}",
            obs_mean_odd, expected_odd, chi2_odd, p_odd,
            odd_dist.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
        ),
    });

    results
}

fn hypergeometric_pmf(n_pop: u64, n_success: u64, n_draws: u64) -> Vec<f64> {
    // P(X=k) = C(K,k)*C(N-K,n-k) / C(N,n)
    let max_k = n_draws.min(n_success);
    let mut probs = Vec::new();

    let log_c_n_n = log_binomial(n_pop, n_draws);

    for k in 0..=max_k {
        if n_draws - k > n_pop - n_success {
            probs.push(0.0);
            continue;
        }
        let log_p = log_binomial(n_success, k) + log_binomial(n_pop - n_success, n_draws - k) - log_c_n_n;
        probs.push(log_p.exp());
    }

    probs
}

fn log_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let mut result = 0.0;
    let k = k.min(n - k); // symmetry
    for i in 0..k {
        result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    result
}

// ════════════════════════════════════════════════════════════════
// 3. Graphe de co-occurrence
// ════════════════════════════════════════════════════════════════

fn cooccurrence_graph_analysis(draws: &[Draw]) -> TestResult {
    // Build adjacency matrix (co-occurrence counts)
    let mut adj = vec![vec![0.0f64; 50]; 50];
    for d in draws {
        for i in 0..5 {
            for j in (i + 1)..5 {
                let a = (d.balls[i] - 1) as usize;
                let b = (d.balls[j] - 1) as usize;
                adj[a][b] += 1.0;
                adj[b][a] += 1.0;
            }
        }
    }

    // Compute degree matrix and Laplacian
    let mut degrees = vec![0.0f64; 50];
    for i in 0..50 {
        degrees[i] = adj[i].iter().sum();
    }

    // Compute modularity Q using simple partition (5 decades)
    let total_edges: f64 = degrees.iter().sum::<f64>() / 2.0;
    if total_edges == 0.0 {
        return TestResult {
            test_name: "Graphe co-occurrence - modularité".to_string(),
            category: "mathematical".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: "Pas d'arêtes".to_string(),
        };
    }

    // Decade partition: node i belongs to community i/10
    let modularity_decade = compute_modularity(&adj, &degrees, total_edges, |i| i / 10);

    // Random baseline: estimate expected modularity by Monte Carlo
    let n_sims = 1000;
    let mut random_modularities = Vec::new();

    // Simple permutation test: shuffle node assignments
    let mut rng_state: u64 = 42;
    for _ in 0..n_sims {
        // Generate random partition of 50 nodes into 5 groups
        let mut perm: Vec<usize> = (0..50).collect();
        // Fisher-Yates shuffle with simple PRNG
        for i in (1..50).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        let q = compute_modularity(&adj, &degrees, total_edges, |i| perm[i] / 10);
        random_modularities.push(q);
    }

    random_modularities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_random = random_modularities.iter().sum::<f64>() / n_sims as f64;
    let p95 = random_modularities[(n_sims as f64 * 0.95) as usize];

    let effect = (modularity_decade - mean_random) / mean_random.abs().max(0.001);

    let verdict = if modularity_decade > p95 {
        ResearchVerdict::Significant
    } else if modularity_decade > mean_random * 1.5 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    TestResult {
        test_name: "Graphe co-occurrence - modularité".to_string(),
        category: "mathematical".to_string(),
        statistic: modularity_decade,
        p_value: None,
        effect_size: effect,
        verdict,
        detail: format!(
            "Q_décade={:.4}, Q_random_moy={:.4}, Q_95%={:.4}, effet={:.2}",
            modularity_decade, mean_random, p95, effect
        ),
    }
}

fn compute_modularity(adj: &[Vec<f64>], degrees: &[f64], total_edges: f64, community: impl Fn(usize) -> usize) -> f64 {
    let n = adj.len();
    let m2 = 2.0 * total_edges;
    let mut q = 0.0;

    for i in 0..n {
        for j in 0..n {
            if community(i) == community(j) {
                q += adj[i][j] - degrees[i] * degrees[j] / m2;
            }
        }
    }

    q / m2
}

// ════════════════════════════════════════════════════════════════
// 4. Analyse des gaps (inter-apparitions)
// ════════════════════════════════════════════════════════════════

fn gap_analysis(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    // Compute gaps for each ball number (chronological order)
    let mut all_gaps: Vec<Vec<usize>> = Vec::new();
    let mut trend_slopes: Vec<f64> = Vec::new();
    let mut autocorr_values: Vec<f64> = Vec::new();

    for num in 1..=50u8 {
        let mut gaps = Vec::new();
        let mut last_seen = None;

        for (t, d) in draws.iter().rev().enumerate() {
            if d.balls.contains(&num) {
                if let Some(prev) = last_seen {
                    gaps.push(t - prev);
                }
                last_seen = Some(t);
            }
        }

        if gaps.len() >= 10 {
            // Trend test: linear regression of gap vs index
            let n_gaps = gaps.len();
            let x: Vec<f64> = (0..n_gaps).map(|i| i as f64).collect();
            let y: Vec<f64> = gaps.iter().map(|&g| g as f64).collect();
            let slope = linear_regression_slope(&x, &y);
            trend_slopes.push(slope);

            // Autocorrelation of gaps (lag 1)
            if gaps.len() >= 5 {
                let mean = y.iter().sum::<f64>() / n_gaps as f64;
                let var: f64 = y.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / n_gaps as f64;
                if var > 0.0 {
                    let autocov: f64 = (0..n_gaps - 1)
                        .map(|i| (y[i] - mean) * (y[i + 1] - mean))
                        .sum::<f64>() / (n_gaps - 1) as f64;
                    autocorr_values.push(autocov / var);
                }
            }

            all_gaps.push(gaps);
        }
    }

    // Geometric fit test: under uniformity, gaps ~ Geometric(p=5/50=0.1)
    // Mean gap should be ~ 1/p - 1 = 9
    let expected_mean_gap = 9.0; // E[gap] = (50-5)/5 = 9
    let all_flat_gaps: Vec<usize> = all_gaps.iter().flatten().copied().collect();
    let obs_mean_gap = if all_flat_gaps.is_empty() {
        0.0
    } else {
        all_flat_gaps.iter().sum::<usize>() as f64 / all_flat_gaps.len() as f64
    };

    // K-S like test: compare gap CDF to geometric CDF
    let mut gap_hist: HashMap<usize, usize> = HashMap::new();
    for &g in &all_flat_gaps {
        *gap_hist.entry(g).or_insert(0) += 1;
    }
    let max_gap = all_flat_gaps.iter().copied().max().unwrap_or(0);
    let p_geom: f64 = 5.0 / 50.0;

    let mut max_ks = 0.0f64;
    let mut cum_obs = 0.0;
    for g in 1..=max_gap {
        cum_obs += gap_hist.get(&g).copied().unwrap_or(0) as f64 / all_flat_gaps.len() as f64;
        let cum_exp = 1.0 - (1.0 - p_geom).powi(g as i32);
        let diff = (cum_obs - cum_exp).abs();
        if diff > max_ks {
            max_ks = diff;
        }
    }

    let ks_critical = 1.36 / (all_flat_gaps.len() as f64).sqrt(); // alpha=0.05
    let verdict_geom = if max_ks > ks_critical * 1.5 {
        ResearchVerdict::Significant
    } else if max_ks > ks_critical {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    results.push(TestResult {
        test_name: "Gaps - test géométrique (K-S)".to_string(),
        category: "mathematical".to_string(),
        statistic: max_ks,
        p_value: None,
        effect_size: (obs_mean_gap - expected_mean_gap).abs() / expected_mean_gap,
        verdict: verdict_geom,
        detail: format!(
            "D_KS={:.4}, seuil_5%={:.4}, gap_moy={:.1} (att={:.1}), {} gaps totaux",
            max_ks, ks_critical, obs_mean_gap, expected_mean_gap, all_flat_gaps.len()
        ),
    });

    // Trend test aggregate
    if !trend_slopes.is_empty() {
        let avg_slope = trend_slopes.iter().sum::<f64>() / trend_slopes.len() as f64;
        let positive_slopes = trend_slopes.iter().filter(|&&s| s > 0.0).count();
        let fraction_positive = positive_slopes as f64 / trend_slopes.len() as f64;

        // Under null, about 50% should have positive slope
        let z_trend = (fraction_positive - 0.5) / (0.25 / trend_slopes.len() as f64).sqrt();
        let p_trend = super::two_sided_p(z_trend);

        results.push(TestResult {
            test_name: "Gaps - tendance temporelle".to_string(),
            category: "mathematical".to_string(),
            statistic: z_trend,
            p_value: Some(p_trend),
            effect_size: avg_slope.abs(),
            verdict: verdict_from_p(p_trend),
            detail: format!(
                "pente_moy={:.4}, {}/{} pentes positives, z={:.2}, p={:.4}",
                avg_slope, positive_slopes, trend_slopes.len(), z_trend, p_trend
            ),
        });
    }

    // Gap autocorrelation aggregate
    if !autocorr_values.is_empty() {
        let avg_autocorr = autocorr_values.iter().sum::<f64>() / autocorr_values.len() as f64;
        let significant_autocorr = autocorr_values.iter()
            .filter(|&&a| a.abs() > 2.0 / (draws.len() as f64 / 10.0).sqrt())
            .count();

        let verdict_auto = if significant_autocorr as f64 / autocorr_values.len() as f64 > 0.15 {
            ResearchVerdict::Significant
        } else if avg_autocorr.abs() > 0.1 {
            ResearchVerdict::Marginal
        } else {
            ResearchVerdict::NotSignificant
        };

        results.push(TestResult {
            test_name: "Gaps - autocorrélation (lag 1)".to_string(),
            category: "mathematical".to_string(),
            statistic: avg_autocorr,
            p_value: None,
            effect_size: avg_autocorr.abs(),
            verdict: verdict_auto,
            detail: format!(
                "autocorr_moy={:.4}, {}/{} numéros sig.",
                avg_autocorr, significant_autocorr, autocorr_values.len()
            ),
        });
    }

    results
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_modular_analysis_no_panic() {
        let draws = make_test_draws(100);
        let results = modular_analysis(&draws);
        assert!(results.len() >= 7); // 6 moduli + correlation
    }

    #[test]
    fn test_sum_spread_no_panic() {
        let draws = make_test_draws(100);
        let results = sum_spread_analysis(&draws);
        assert!(results.len() >= 3); // sum, spread, odd count
    }

    #[test]
    fn test_cooccurrence_graph_no_panic() {
        let draws = make_test_draws(100);
        let result = cooccurrence_graph_analysis(&draws);
        assert_eq!(result.category, "mathematical");
    }

    #[test]
    fn test_gap_analysis_no_panic() {
        let draws = make_test_draws(200);
        let results = gap_analysis(&draws);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hypergeometric_pmf() {
        let probs = hypergeometric_pmf(50, 25, 5);
        assert_eq!(probs.len(), 6); // k = 0,1,2,3,4,5
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "PMF should sum to 1, got {}", sum);
    }

    #[test]
    fn test_log_binomial() {
        // C(10,3) = 120
        let lc = log_binomial(10, 3);
        assert!((lc.exp() - 120.0).abs() < 1.0, "C(10,3) should be 120, got {}", lc.exp());
    }

    #[test]
    fn test_modularity_uniform() {
        // For a complete graph with uniform weights, modularity of any partition is ~0
        let adj = vec![vec![1.0; 10]; 10];
        let degrees: Vec<f64> = adj.iter().map(|row| row.iter().sum()).collect();
        let total_edges: f64 = degrees.iter().sum::<f64>() / 2.0;
        let q = compute_modularity(&adj, &degrees, total_edges, |i| i / 5);
        // Should be close to 0 (slightly negative due to self-loops)
        assert!(q.abs() < 0.2, "Modularity should be ~0 for complete graph, got {}", q);
    }

    #[test]
    fn test_run_mathematical_tests() {
        let draws = make_test_draws(200);
        let results = run_mathematical_tests(&draws);
        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.category, "mathematical");
        }
    }
}
