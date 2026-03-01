use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{TestResult, ResearchVerdict, chi_squared, chi_squared_p_value, verdict_from_p, two_sided_p};

pub fn run_physical_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    results.extend(rack_position_tests(draws));
    results.extend(trap_bias_tests(draws));
    results.extend(cooccurrence_tests(draws));
    results.push(balls_stars_independence(draws));

    results
}

// ════════════════════════════════════════════════════════════════
// 1. Analyse par position dans le rack (décade)
// ════════════════════════════════════════════════════════════════

fn rack_position_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    // Balls: 5 rows of 10 (decades 1-10, 11-20, ..., 41-50)
    let ball_numbers: Vec<Vec<u8>> = draws.iter().rev()
        .map(|d| d.balls.to_vec())
        .collect();

    // Row test (decades) for balls
    results.push(row_chi2_test(&ball_numbers, 5, 10, "Boules - rangées rack (décades)", Pool::Balls));

    // Column test (units) for balls
    results.push(column_chi2_test(&ball_numbers, 5, 10, "Boules - colonnes rack (unités)"));

    // Stars: 2 rows of 6 (1-6 high, 7-12 low)
    let star_numbers: Vec<Vec<u8>> = draws.iter().rev()
        .map(|d| d.stars.to_vec())
        .collect();

    results.push(row_chi2_test(&star_numbers, 2, 6, "Étoiles - rangées rack (haut/bas)", Pool::Stars));

    // Windowed analysis for drift detection
    let windows = [100, 200, 500];
    for &w in &windows {
        if draws.len() >= w {
            let recent_balls: Vec<Vec<u8>> = draws[..w].iter().rev()
                .map(|d| d.balls.to_vec())
                .collect();
            results.push(row_chi2_test(
                &recent_balls, 5, 10,
                &format!("Boules décades (w={})", w),
                Pool::Balls,
            ));
        }
    }

    results
}

fn row_chi2_test(numbers: &[Vec<u8>], n_rows: usize, row_size: usize, label: &str, pool: Pool) -> TestResult {
    let pick_count = match pool {
        Pool::Balls => 5,
        Pool::Stars => 2,
    };
    let n_draws = numbers.len();

    let mut row_counts = vec![0u64; n_rows];
    for draw_nums in numbers {
        for &num in draw_nums {
            let row = ((num - 1) as usize) / row_size;
            if row < n_rows {
                row_counts[row] += 1;
            }
        }
    }

    let total_picks = (n_draws * pick_count) as f64;
    let expected_per_row = total_picks * (row_size as f64 / (n_rows * row_size) as f64);

    let observed: Vec<f64> = row_counts.iter().map(|&c| c as f64).collect();
    let expected: Vec<f64> = vec![expected_per_row; n_rows];

    let chi2 = chi_squared(&observed, &expected);
    let df = n_rows - 1;
    let p = chi_squared_p_value(chi2, df);

    let max_dev = observed.iter().zip(expected.iter())
        .map(|(&o, &e)| ((o - e) / e).abs())
        .fold(0.0f64, f64::max);

    let detail = row_counts.iter().enumerate()
        .map(|(i, &c)| format!("R{}={}", i + 1, c))
        .collect::<Vec<_>>()
        .join(", ");

    TestResult {
        test_name: label.to_string(),
        category: "physical".to_string(),
        statistic: chi2,
        p_value: Some(p),
        effect_size: max_dev,
        verdict: verdict_from_p(p),
        detail: format!("chi2={:.2}, df={}, p={:.4} | {}", chi2, df, p, detail),
    }
}

fn column_chi2_test(numbers: &[Vec<u8>], _n_rows: usize, row_size: usize, label: &str) -> TestResult {
    let n_draws = numbers.len();
    let pick_count = 5; // balls

    let mut col_counts = vec![0u64; row_size];
    for draw_nums in numbers {
        for &num in draw_nums {
            let col = ((num - 1) as usize) % row_size;
            col_counts[col] += 1;
        }
    }

    let total_picks = (n_draws * pick_count) as f64;
    let expected_per_col = total_picks / row_size as f64;

    // Adjust: column 0 (units digit 0) corresponds to numbers 10,20,30,40,50
    // which has n_rows numbers, same as other columns. So uniform expected.
    let observed: Vec<f64> = col_counts.iter().map(|&c| c as f64).collect();
    let expected: Vec<f64> = vec![expected_per_col; row_size];

    let chi2 = chi_squared(&observed, &expected);
    let df = row_size - 1;
    let p = chi_squared_p_value(chi2, df);

    let max_dev = observed.iter().zip(expected.iter())
        .map(|(&o, &e)| if e > 0.0 { ((o - e) / e).abs() } else { 0.0 })
        .fold(0.0f64, f64::max);

    let detail = col_counts.iter().enumerate()
        .map(|(i, &c)| format!("C{}={}", i, c))
        .collect::<Vec<_>>()
        .join(", ");

    TestResult {
        test_name: label.to_string(),
        category: "physical".to_string(),
        statistic: chi2,
        p_value: Some(p),
        effect_size: max_dev,
        verdict: verdict_from_p(p),
        detail: format!("chi2={:.2}, df={}, p={:.4} | {}", chi2, df, p, detail),
    }
}

// ════════════════════════════════════════════════════════════════
// 2. Analyse de la trappe (biais positionnel)
// ════════════════════════════════════════════════════════════════

fn trap_bias_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();
    let n_draws = draws.len();

    // Ball frequency deviation
    let mut ball_freq = vec![0u64; 50];
    for d in draws {
        for &b in &d.balls {
            ball_freq[(b - 1) as usize] += 1;
        }
    }

    let expected_ball = n_draws as f64 * 5.0 / 50.0; // = n_draws * 0.1
    let p_ball = 5.0 / 50.0;

    // Binomial test per number with Bonferroni correction
    let mut significant_balls = Vec::new();
    for (i, &count) in ball_freq.iter().enumerate() {
        let z = (count as f64 - expected_ball) / (n_draws as f64 * p_ball * (1.0 - p_ball)).sqrt();
        let p_raw = two_sided_p(z);
        let p_bonf = (p_raw * 50.0).min(1.0); // Bonferroni correction

        if p_bonf < 0.05 {
            let direction = if (count as f64) > expected_ball { "+" } else { "-" };
            significant_balls.push(format!("{}{}(p={:.3})", i + 1, direction, p_bonf));
        }
    }

    let max_dev_ball = ball_freq.iter()
        .map(|&c| ((c as f64 - expected_ball) / expected_ball).abs())
        .fold(0.0f64, f64::max);

    // Chi-squared over all 50 numbers
    let obs_balls: Vec<f64> = ball_freq.iter().map(|&c| c as f64).collect();
    let exp_balls: Vec<f64> = vec![expected_ball; 50];
    let chi2_balls = chi_squared(&obs_balls, &exp_balls);
    let p_chi2_balls = chi_squared_p_value(chi2_balls, 49);

    results.push(TestResult {
        test_name: "Biais trappe - fréquences boules".to_string(),
        category: "physical".to_string(),
        statistic: chi2_balls,
        p_value: Some(p_chi2_balls),
        effect_size: max_dev_ball,
        verdict: verdict_from_p(p_chi2_balls),
        detail: if significant_balls.is_empty() {
            format!("chi2={:.2}, p={:.4}, aucun numéro significatif (Bonferroni)", chi2_balls, p_chi2_balls)
        } else {
            format!("chi2={:.2}, p={:.4}, sig(Bonf): {}", chi2_balls, p_chi2_balls, significant_balls.join(", "))
        },
    });

    // Star frequency deviation
    let mut star_freq = [0u64; 12];
    for d in draws {
        for &s in &d.stars {
            star_freq[(s - 1) as usize] += 1;
        }
    }
    let expected_star = n_draws as f64 * 2.0 / 12.0;
    let obs_stars: Vec<f64> = star_freq.iter().map(|&c| c as f64).collect();
    let exp_stars: Vec<f64> = vec![expected_star; 12];
    let chi2_stars = chi_squared(&obs_stars, &exp_stars);
    let p_chi2_stars = chi_squared_p_value(chi2_stars, 11);

    let max_dev_star = star_freq.iter()
        .map(|&c| ((c as f64 - expected_star) / expected_star).abs())
        .fold(0.0f64, f64::max);

    results.push(TestResult {
        test_name: "Biais trappe - fréquences étoiles".to_string(),
        category: "physical".to_string(),
        statistic: chi2_stars,
        p_value: Some(p_chi2_stars),
        effect_size: max_dev_star,
        verdict: verdict_from_p(p_chi2_stars),
        detail: format!("chi2={:.2}, p={:.4}, max_dev={:.3}", chi2_stars, p_chi2_stars, max_dev_star),
    });

    // Temporal drift: compare frequency across eras
    results.push(temporal_drift_test(draws));

    results
}

fn temporal_drift_test(draws: &[Draw]) -> TestResult {
    let era_size = 100;
    let n_eras = draws.len() / era_size;

    if n_eras < 2 {
        return TestResult {
            test_name: "Drift temporel - fréquences".to_string(),
            category: "physical".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: format!("Insuffisant : {} tirages pour {} ères de {}", draws.len(), n_eras, era_size),
        };
    }

    // For each era, compute frequency of each ball number
    let mut max_drift = 0.0f64;
    let mut drifting_numbers = Vec::new();

    for num in 1..=50u8 {
        let mut era_freqs = Vec::new();
        for era in 0..n_eras {
            let start = era * era_size;
            let end = start + era_size;
            let count = draws[start..end].iter()
                .filter(|d| d.balls.contains(&num))
                .count();
            era_freqs.push(count as f64 / era_size as f64);
        }

        // Check trend: linear regression of frequency vs era index
        let x: Vec<f64> = (0..n_eras).map(|i| i as f64).collect();
        let n = n_eras as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = era_freqs.iter().sum();
        let sum_xy: f64 = x.iter().zip(era_freqs.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            continue;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let drift = slope.abs();

        if drift > max_drift {
            max_drift = drift;
        }

        // A drift > 0.001 per era (~0.1% per 100 draws) is notable
        if drift > 0.001 {
            let direction = if slope > 0.0 { "↑" } else { "↓" };
            drifting_numbers.push(format!("{}{}", num, direction));
        }
    }

    let verdict = if max_drift > 0.003 {
        ResearchVerdict::Significant
    } else if max_drift > 0.001 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    TestResult {
        test_name: "Drift temporel - fréquences".to_string(),
        category: "physical".to_string(),
        statistic: max_drift,
        p_value: None,
        effect_size: max_drift,
        verdict,
        detail: if drifting_numbers.is_empty() {
            format!("{} ères de {} tirages, max_drift={:.5}, aucun drift notable", n_eras, era_size, max_drift)
        } else {
            format!("{} ères, max_drift={:.5}, drift: {}", n_eras, max_drift, drifting_numbers.join(", "))
        },
    }
}

// ════════════════════════════════════════════════════════════════
// 3. Corrélation inter-extraction (co-occurrence)
// ════════════════════════════════════════════════════════════════

fn cooccurrence_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();
    let n = draws.len();

    // Ball pairs
    let mut pair_counts = HashMap::new();
    for d in draws {
        for i in 0..5 {
            for j in (i + 1)..5 {
                let pair = (d.balls[i].min(d.balls[j]), d.balls[i].max(d.balls[j]));
                *pair_counts.entry(pair).or_insert(0u64) += 1;
            }
        }
    }

    // Expected pair frequency under independence:
    // P(both i and j drawn) = C(48,3)/C(50,5) = (48*47*46)/(50*49*48) * ...
    // Actually: P(pair) = C(48,3)/C(50,5) = 5*4/(50*49) * C(48,3)/C(48,3) ...
    // Simpler: P(i in draw) = 5/50 = 0.1, but draws without replacement
    // P(i and j both drawn) = 5/50 * 4/49 = 20/2450 ≈ 0.008163
    let p_pair = 5.0 * 4.0 / (50.0 * 49.0);
    let expected_pair = n as f64 * p_pair;

    // Chi-squared test over all pairs
    let total_pairs = 50 * 49 / 2; // C(50,2) = 1225
    let mut chi2_pairs = 0.0;
    let mut excess_pairs = Vec::new();
    let mut deficit_pairs = Vec::new();

    for i in 1..=50u8 {
        for j in (i + 1)..=50u8 {
            let count = pair_counts.get(&(i, j)).copied().unwrap_or(0) as f64;
            chi2_pairs += (count - expected_pair).powi(2) / expected_pair;

            // Flag extreme deviations
            let z = (count - expected_pair) / (n as f64 * p_pair * (1.0 - p_pair)).sqrt();
            if z > 3.5 {
                excess_pairs.push(format!("({},{})+{:.0}", i, j, count - expected_pair));
            } else if z < -3.5 {
                deficit_pairs.push(format!("({},{}){:.0}", i, j, count - expected_pair));
            }
        }
    }

    let df_pairs = total_pairs - 1;
    let p_pairs = chi_squared_p_value(chi2_pairs, df_pairs);

    let max_pair_dev = pair_counts.values()
        .map(|&c| ((c as f64 - expected_pair) / expected_pair).abs())
        .fold(0.0f64, f64::max);

    let mut detail_parts = vec![format!("chi2={:.2}, df={}, p={:.4}", chi2_pairs, df_pairs, p_pairs)];
    if !excess_pairs.is_empty() {
        detail_parts.push(format!("excès: {}", excess_pairs.iter().take(5).cloned().collect::<Vec<_>>().join(", ")));
    }
    if !deficit_pairs.is_empty() {
        detail_parts.push(format!("déficit: {}", deficit_pairs.iter().take(5).cloned().collect::<Vec<_>>().join(", ")));
    }

    results.push(TestResult {
        test_name: "Co-occurrence paires boules".to_string(),
        category: "physical".to_string(),
        statistic: chi2_pairs,
        p_value: Some(p_pairs),
        effect_size: max_pair_dev,
        verdict: verdict_from_p(p_pairs),
        detail: detail_parts.join(" | "),
    });

    // Triplet analysis (sampling top deviating pairs)
    results.push(triplet_excess_test(draws));

    results
}

fn triplet_excess_test(draws: &[Draw]) -> TestResult {
    let n = draws.len();

    // P(triplet i,j,k drawn together) = 5*4*3/(50*49*48) ≈ 0.000510
    let p_triplet = (5.0 * 4.0 * 3.0) / (50.0 * 49.0 * 48.0);
    let expected_triplet = n as f64 * p_triplet;

    let mut triplet_counts: HashMap<(u8, u8, u8), u64> = HashMap::new();
    for d in draws {
        for i in 0..5 {
            for j in (i + 1)..5 {
                for k in (j + 1)..5 {
                    let mut trio = [d.balls[i], d.balls[j], d.balls[k]];
                    trio.sort();
                    *triplet_counts.entry((trio[0], trio[1], trio[2])).or_insert(0) += 1;
                }
            }
        }
    }

    // Find most extreme triplets
    let mut deviations: Vec<((u8, u8, u8), f64)> = triplet_counts.iter()
        .map(|(&t, &c)| {
            let z = (c as f64 - expected_triplet) / (n as f64 * p_triplet * (1.0 - p_triplet)).sqrt();
            (t, z)
        })
        .collect();
    deviations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let max_z = deviations.first().map(|(_, z)| z.abs()).unwrap_or(0.0);
    let significant_count = deviations.iter().filter(|(_, z)| z.abs() > 3.0).count();
    let total_triplets = deviations.len();

    // Expected number of |z| > 3 under null ~ total * 0.0027
    let expected_sig = total_triplets as f64 * 0.0027;
    let excess_sig = significant_count as f64 / expected_sig.max(1.0);

    let verdict = if excess_sig > 3.0 && significant_count > 5 {
        ResearchVerdict::Significant
    } else if excess_sig > 1.5 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    let top_triplets: Vec<String> = deviations.iter().take(3)
        .map(|((a, b, c), z)| format!("({},{},{}) z={:.1}", a, b, c, z))
        .collect();

    TestResult {
        test_name: "Co-occurrence triplets boules".to_string(),
        category: "physical".to_string(),
        statistic: max_z,
        p_value: None,
        effect_size: excess_sig,
        verdict,
        detail: format!(
            "{}/{} triplets |z|>3 (attendu ~{:.0}), top: {}",
            significant_count, total_triplets, expected_sig, top_triplets.join(", ")
        ),
    }
}

// ════════════════════════════════════════════════════════════════
// 4. Boules vs Étoiles : indépendance
// ════════════════════════════════════════════════════════════════

fn balls_stars_independence(draws: &[Draw]) -> TestResult {
    let n = draws.len();

    // Mutual information between ball sum and star sum
    let ball_sums: Vec<f64> = draws.iter().map(|d| d.balls.iter().map(|&b| b as f64).sum()).collect();
    let star_sums: Vec<f64> = draws.iter().map(|d| d.stars.iter().map(|&s| s as f64).sum()).collect();

    // Discretize into bins
    let n_bins_ball = 5; // quintiles of ball sum
    let n_bins_star = 3; // tertiles of star sum

    let ball_bins = discretize_into_quantiles(&ball_sums, n_bins_ball);
    let star_bins = discretize_into_quantiles(&star_sums, n_bins_star);

    // Contingency table
    let mut contingency = vec![vec![0u64; n_bins_star]; n_bins_ball];
    let mut row_totals = vec![0u64; n_bins_ball];
    let mut col_totals = vec![0u64; n_bins_star];

    for (&bb, &sb) in ball_bins.iter().zip(star_bins.iter()) {
        contingency[bb][sb] += 1;
        row_totals[bb] += 1;
        col_totals[sb] += 1;
    }

    // Chi-squared independence test
    let mut chi2 = 0.0;
    for i in 0..n_bins_ball {
        for j in 0..n_bins_star {
            let expected = (row_totals[i] as f64) * (col_totals[j] as f64) / n as f64;
            if expected > 0.0 {
                chi2 += (contingency[i][j] as f64 - expected).powi(2) / expected;
            }
        }
    }

    let df = (n_bins_ball - 1) * (n_bins_star - 1);
    let p = chi_squared_p_value(chi2, df);

    // Mutual information
    let mi = compute_mi(&ball_bins, &star_bins, n_bins_ball, n_bins_star);

    TestResult {
        test_name: "Indépendance boules/étoiles".to_string(),
        category: "physical".to_string(),
        statistic: chi2,
        p_value: Some(p),
        effect_size: mi,
        verdict: verdict_from_p(p),
        detail: format!(
            "chi2={:.2}, df={}, p={:.4}, MI={:.5} | Contingence {}x{}",
            chi2, df, p, mi, n_bins_ball, n_bins_star
        ),
    }
}

fn discretize_into_quantiles(values: &[f64], n_bins: usize) -> Vec<usize> {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let thresholds: Vec<f64> = (1..n_bins)
        .map(|i| {
            let idx = (i * sorted.len() / n_bins).min(sorted.len() - 1);
            sorted[idx]
        })
        .collect();

    values.iter().map(|&v| {
        let mut bin = 0;
        for &t in &thresholds {
            if v > t {
                bin += 1;
            }
        }
        bin.min(n_bins - 1)
    }).collect()
}

fn compute_mi(x: &[usize], y: &[usize], n_x: usize, n_y: usize) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mut joint = vec![vec![0u32; n_y]; n_x];
    let mut mx = vec![0u32; n_x];
    let mut my = vec![0u32; n_y];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        joint[xi][yi] += 1;
        mx[xi] += 1;
        my[yi] += 1;
    }

    let mut mi = 0.0;
    for i in 0..n_x {
        for j in 0..n_y {
            if joint[i][j] > 0 {
                let pxy = joint[i][j] as f64 / n;
                let px = mx[i] as f64 / n;
                let py = my[j] as f64 / n;
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }
    mi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_rack_position_no_panic() {
        let draws = make_test_draws(100);
        let results = rack_position_tests(&draws);
        assert!(results.len() >= 3); // at least row balls, col balls, row stars
        for r in &results {
            assert!(r.p_value.is_some());
        }
    }

    #[test]
    fn test_trap_bias_no_panic() {
        let draws = make_test_draws(100);
        let results = trap_bias_tests(&draws);
        assert!(results.len() >= 2); // balls freq, stars freq, drift
    }

    #[test]
    fn test_cooccurrence_no_panic() {
        let draws = make_test_draws(100);
        let results = cooccurrence_tests(&draws);
        assert!(results.len() >= 2); // pairs + triplets
    }

    #[test]
    fn test_independence_no_panic() {
        let draws = make_test_draws(100);
        let result = balls_stars_independence(&draws);
        assert!(result.p_value.is_some());
    }

    #[test]
    fn test_discretize_quantiles() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bins = discretize_into_quantiles(&values, 2);
        // First half should be bin 0, second half bin 1
        let low = bins.iter().filter(|&&b| b == 0).count();
        let high = bins.iter().filter(|&&b| b == 1).count();
        assert!(low >= 4 && high >= 4);
    }

    #[test]
    fn test_run_physical_tests() {
        let draws = make_test_draws(200);
        let results = run_physical_tests(&draws);
        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.category, "physical");
        }
    }
}
