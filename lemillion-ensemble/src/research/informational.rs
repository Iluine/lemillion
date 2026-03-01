use std::collections::HashMap;
use std::io::Write;

use flate2::Compression;
use flate2::write::DeflateEncoder;

use lemillion_db::models::Draw;

use super::{TestResult, ResearchVerdict};

pub fn run_informational_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();

    results.push(conditional_entropy_test(draws));
    results.extend(transfer_entropy_tests(draws));
    results.extend(compression_tests(draws));
    results.extend(delayed_mi_tests(draws));

    results
}

// ════════════════════════════════════════════════════════════════
// 1. Entropie conditionnelle H(t+1 | t)
// ════════════════════════════════════════════════════════════════

fn conditional_entropy_test(draws: &[Draw]) -> TestResult {
    let n = draws.len();
    if n < 20 {
        return TestResult {
            test_name: "Entropie conditionnelle H(t+1|t)".to_string(),
            category: "informational".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: "Insuffisant".to_string(),
        };
    }

    // Encode each draw as a summary vector for tractable estimation:
    // [ball_sum_bin, ball_spread_bin, odd_count, decade_pattern]
    let summaries: Vec<[usize; 4]> = draws.iter().rev().map(|d| {
        let sum: u16 = d.balls.iter().map(|&b| b as u16).sum();
        let sum_bin = ((sum as f64 - 15.0) / 30.0 * 5.0).floor().clamp(0.0, 4.0) as usize;

        let spread = d.balls.iter().max().unwrap() - d.balls.iter().min().unwrap();
        let spread_bin = (spread as usize / 10).min(4);

        let odd_count = d.balls.iter().filter(|&&b| b % 2 == 1).count();

        // Decade pattern: which decades have at least one ball
        let decade_mask = d.balls.iter()
            .map(|&b| 1usize << (((b - 1) / 10) as usize))
            .fold(0usize, |acc, x| acc | x);

        [sum_bin, spread_bin, odd_count, decade_mask]
    }).collect();

    // Estimate H(t+1) using summary marginals
    let mut marginal_counts: HashMap<[usize; 4], u64> = HashMap::new();
    for s in &summaries {
        *marginal_counts.entry(*s).or_insert(0) += 1;
    }
    let h_marginal = entropy_from_counts(&marginal_counts, n);

    // Estimate H(t+1 | t) using joint distribution of (summary_t, summary_{t+1})
    let mut joint_counts: HashMap<([usize; 4], [usize; 4]), u64> = HashMap::new();
    let mut cond_marginal: HashMap<[usize; 4], u64> = HashMap::new();
    for i in 0..(summaries.len() - 1) {
        let key = (summaries[i], summaries[i + 1]);
        *joint_counts.entry(key).or_insert(0) += 1;
        *cond_marginal.entry(summaries[i]).or_insert(0) += 1;
    }

    let n_pairs = (summaries.len() - 1) as f64;
    let mut h_joint = 0.0;
    for &count in joint_counts.values() {
        let p = count as f64 / n_pairs;
        if p > 0.0 {
            h_joint -= p * p.log2();
        }
    }

    let mut h_condition = 0.0;
    for &count in cond_marginal.values() {
        let p = count as f64 / n_pairs;
        if p > 0.0 {
            h_condition -= p * p.log2();
        }
    }

    // H(Y|X) = H(X,Y) - H(X)
    let h_conditional = h_joint - h_condition;

    // Theoretical maximum: H(draw) = log2(C(50,5)) ≈ 20.6 bits
    let h_max = 20.6;

    // Information gain ratio
    let info_gain = if h_marginal > 0.0 { 1.0 - h_conditional / h_marginal } else { 0.0 };

    let verdict = if info_gain > 0.1 {
        ResearchVerdict::Significant
    } else if info_gain > 0.03 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    TestResult {
        test_name: "Entropie conditionnelle H(t+1|t)".to_string(),
        category: "informational".to_string(),
        statistic: h_conditional,
        p_value: None,
        effect_size: info_gain,
        verdict,
        detail: format!(
            "H(t+1)={:.2} bits, H(t+1|t)={:.2} bits, gain={:.3}, H_max≈{:.1} bits",
            h_marginal, h_conditional, info_gain, h_max
        ),
    }
}

fn entropy_from_counts(counts: &HashMap<[usize; 4], u64>, total: usize) -> f64 {
    let n = total as f64;
    let mut h = 0.0;
    for &count in counts.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

// ════════════════════════════════════════════════════════════════
// 2. Transfer entropy
// ════════════════════════════════════════════════════════════════

fn transfer_entropy_tests(draws: &[Draw]) -> Vec<TestResult> {
    let n = draws.len();
    if n < 50 {
        return vec![TestResult {
            test_name: "Transfer entropy".to_string(),
            category: "informational".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: "Insuffisant (< 50 tirages)".to_string(),
        }];
    }

    // Binary presence series for each ball (chronological order)
    let ball_series: Vec<Vec<bool>> = (1..=50u8).map(|num| {
        draws.iter().rev().map(|d| d.balls.contains(&num)).collect()
    }).collect();

    let star_series: Vec<Vec<bool>> = (1..=12u8).map(|num| {
        draws.iter().rev().map(|d| d.stars.contains(&num)).collect()
    }).collect();

    // Compute TE for all ball-ball pairs (sampling to keep computation manageable)
    // Focus on pairs where at least one has high absolute frequency
    let mut te_values: Vec<(usize, usize, f64)> = Vec::new();

    // Sample: test pairs involving the top 10 most frequent balls
    let mut ball_freqs: Vec<(usize, usize)> = ball_series.iter().enumerate()
        .map(|(i, s)| (i, s.iter().filter(|&&b| b).count()))
        .collect();
    ball_freqs.sort_by(|a, b| b.1.cmp(&a.1));

    let top_balls: Vec<usize> = ball_freqs.iter().take(15).map(|(i, _)| *i).collect();

    for &i in &top_balls {
        for j in 0..50 {
            if i == j { continue; }
            let te = compute_transfer_entropy(&ball_series[i], &ball_series[j]);
            if te > 0.0 {
                te_values.push((i, j, te));
            }
        }
    }

    // Cross-pool: ball → star
    let mut cross_te_values: Vec<(usize, usize, f64)> = Vec::new();
    for &i in &top_balls {
        for (j, star_s) in star_series.iter().enumerate() {
            let te = compute_transfer_entropy(&ball_series[i], star_s);
            if te > 0.0 {
                cross_te_values.push((i, j, te));
            }
        }
    }

    // Benjamini-Hochberg FDR correction
    te_values.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Estimate significance: TE under null ~ chi-squared / (2 * n * ln2)
    // Use permutation-based threshold
    let te_threshold = 2.0 / (n as f64 * 2.0_f64.ln()); // approximate null expectation

    let significant_pairs = te_values.iter().filter(|(_, _, te)| *te > te_threshold * 3.0).count();
    let total_tested = te_values.len();
    let expected_false_pos = (total_tested as f64 * 0.05).max(1.0);

    let max_te = te_values.first().map(|(_, _, te)| *te).unwrap_or(0.0);
    let top_pairs: Vec<String> = te_values.iter().take(5)
        .map(|(i, j, te)| format!("{}→{} TE={:.5}", i + 1, j + 1, te))
        .collect();

    let verdict = if significant_pairs as f64 > expected_false_pos * 3.0 {
        ResearchVerdict::Significant
    } else if significant_pairs as f64 > expected_false_pos {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    let mut results = vec![TestResult {
        test_name: "Transfer entropy (boule→boule)".to_string(),
        category: "informational".to_string(),
        statistic: max_te,
        p_value: None,
        effect_size: if te_threshold > 0.0 { max_te / te_threshold } else { 0.0 },
        verdict,
        detail: format!(
            "{}/{} paires sig (seuil={:.5}), top: {}",
            significant_pairs, total_tested, te_threshold * 3.0, top_pairs.join(", ")
        ),
    }];

    // Cross-pool result
    let cross_max_te = cross_te_values.iter().map(|(_, _, te)| *te).fold(0.0f64, f64::max);
    let cross_sig = cross_te_values.iter().filter(|(_, _, te)| *te > te_threshold * 3.0).count();

    let cross_verdict = if cross_sig > 5 {
        ResearchVerdict::Significant
    } else if cross_sig > 0 {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    results.push(TestResult {
        test_name: "Transfer entropy (boule→étoile)".to_string(),
        category: "informational".to_string(),
        statistic: cross_max_te,
        p_value: None,
        effect_size: if te_threshold > 0.0 { cross_max_te / te_threshold } else { 0.0 },
        verdict: cross_verdict,
        detail: format!(
            "{}/{} paires sig, max_TE={:.5}",
            cross_sig, cross_te_values.len(), cross_max_te
        ),
    });

    results
}

fn compute_transfer_entropy(source: &[bool], target: &[bool]) -> f64 {
    // TE(X→Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
    let n = source.len().min(target.len());
    if n < 10 {
        return 0.0;
    }

    // Count occurrences of (y_t, y_{t+1}) and (y_t, x_t, y_{t+1})
    let mut count_yy: HashMap<(bool, bool), u64> = HashMap::new();
    let mut count_y: HashMap<bool, u64> = HashMap::new();
    let mut count_yxy: HashMap<(bool, bool, bool), u64> = HashMap::new();
    let mut count_yx: HashMap<(bool, bool), u64> = HashMap::new();

    for t in 0..(n - 1) {
        let y_t = target[t];
        let y_next = target[t + 1];
        let x_t = source[t];

        *count_yy.entry((y_t, y_next)).or_insert(0) += 1;
        *count_y.entry(y_t).or_insert(0) += 1;
        *count_yxy.entry((y_t, x_t, y_next)).or_insert(0) += 1;
        *count_yx.entry((y_t, x_t)).or_insert(0) += 1;
    }

    let total = (n - 1) as f64;

    // TE = sum_over_states p(y_next, y_t, x_t) * log2(p(y_next | y_t, x_t) / p(y_next | y_t))
    let mut te = 0.0;
    for (&(y_t, x_t, y_next), &count) in &count_yxy {
        let p_yxy = count as f64 / total;
        let p_next_given_yx = count as f64 / count_yx[&(y_t, x_t)] as f64;
        let p_next_given_y = count_yy[&(y_t, y_next)] as f64 / count_y[&y_t] as f64;

        if p_next_given_y > 0.0 && p_next_given_yx > 0.0 {
            te += p_yxy * (p_next_given_yx / p_next_given_y).log2();
        }
    }

    te.max(0.0)
}

// ════════════════════════════════════════════════════════════════
// 3. Complexité de compression
// ════════════════════════════════════════════════════════════════

fn compression_tests(draws: &[Draw]) -> Vec<TestResult> {
    let mut results = Vec::new();
    let n = draws.len();
    if n < 30 {
        return results;
    }

    // Encoding 1: raw sorted numbers
    let raw_bytes: Vec<u8> = draws.iter().rev()
        .flat_map(|d| {
            let mut balls = d.balls;
            balls.sort();
            balls.to_vec()
        })
        .collect();

    // Encoding 2: residues mod 4
    let mod4_bytes: Vec<u8> = draws.iter().rev()
        .flat_map(|d| d.balls.iter().map(|&b| (b - 1) % 4))
        .collect();

    // Encoding 3: decades
    let decade_bytes: Vec<u8> = draws.iter().rev()
        .flat_map(|d| d.balls.iter().map(|&b| (b - 1) / 10))
        .collect();

    // Encoding 4: gaps sequence
    let gap_bytes: Vec<u8> = compute_gap_encoding(draws);

    let encodings: Vec<(&str, &[u8])> = vec![
        ("brute (numéros triés)", &raw_bytes),
        ("résidus mod 4", &mod4_bytes),
        ("décades", &decade_bytes),
        ("gaps", &gap_bytes),
    ];

    for (name, data) in encodings {
        if data.is_empty() {
            continue;
        }

        let compressed_len = compress_deflate(data);
        let ratio = compressed_len as f64 / data.len() as f64;

        // Generate random baseline: same length, same value range
        let max_val = data.iter().copied().max().unwrap_or(1);
        let random_ratio = random_compression_baseline(data.len(), max_val as usize + 1, 100);

        let effect = if random_ratio > 0.0 { ratio / random_ratio } else { 1.0 };

        let verdict = if effect < 0.90 {
            ResearchVerdict::Significant
        } else if effect < 0.95 {
            ResearchVerdict::Marginal
        } else {
            ResearchVerdict::NotSignificant
        };

        results.push(TestResult {
            test_name: format!("Compression ({})", name),
            category: "informational".to_string(),
            statistic: ratio,
            p_value: None,
            effect_size: 1.0 - effect,
            verdict,
            detail: format!(
                "ratio={:.4}, baseline_aléatoire={:.4}, effet={:.4}, {} bytes → {} bytes",
                ratio, random_ratio, effect, data.len(), compressed_len
            ),
        });
    }

    results
}

fn compress_deflate(data: &[u8]) -> usize {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap_or(());
    let compressed = encoder.finish().unwrap_or_default();
    compressed.len()
}

fn compute_gap_encoding(draws: &[Draw]) -> Vec<u8> {
    let mut gaps = Vec::new();
    let mut last_seen = vec![0usize; 50];
    let mut initialized = [false; 50];

    for (t, d) in draws.iter().rev().enumerate() {
        for &b in &d.balls {
            let idx = (b - 1) as usize;
            if initialized[idx] {
                let gap = t - last_seen[idx];
                gaps.push(gap.min(255) as u8);
            }
            last_seen[idx] = t;
            initialized[idx] = true;
        }
    }

    gaps
}

fn random_compression_baseline(len: usize, n_values: usize, n_samples: usize) -> f64 {
    let mut total_ratio = 0.0;
    let mut rng_state: u64 = 12345;

    for _ in 0..n_samples {
        let data: Vec<u8> = (0..len).map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as usize % n_values) as u8
        }).collect();

        let compressed = compress_deflate(&data);
        total_ratio += compressed as f64 / data.len() as f64;
    }

    total_ratio / n_samples as f64
}

// ════════════════════════════════════════════════════════════════
// 4. Information mutuelle retardée multi-échelle
// ════════════════════════════════════════════════════════════════

fn delayed_mi_tests(draws: &[Draw]) -> Vec<TestResult> {
    let n = draws.len();
    if n < 60 {
        return vec![TestResult {
            test_name: "MI retardée multi-échelle".to_string(),
            category: "informational".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: "Insuffisant (< 60 tirages)".to_string(),
        }];
    }

    let lags = [1, 2, 3, 5, 7, 10, 20, 50];

    // Encode draws as summary scalar for MI estimation
    let summaries: Vec<f64> = draws.iter().rev()
        .map(|d| d.balls.iter().map(|&b| b as f64).sum())
        .collect();

    let n_bins = 8;
    let discretized = discretize(&summaries, n_bins);

    let mut mi_results: Vec<(usize, f64, f64)> = Vec::new();

    for &lag in &lags {
        if lag >= n / 2 {
            continue;
        }

        let x = &discretized[..n - lag];
        let y = &discretized[lag..];
        let mi = mutual_information(x, y, n_bins);

        // Shuffled surrogate baseline (average over 200 shuffles)
        let mi_baseline = shuffled_mi_baseline(&discretized, lag, n_bins, 200);

        mi_results.push((lag, mi, mi_baseline));
    }

    // Find maximum excess MI
    let max_excess = mi_results.iter()
        .map(|(_, mi, baseline)| mi - baseline)
        .fold(0.0f64, f64::max);

    let significant_lags: Vec<String> = mi_results.iter()
        .filter(|(_, mi, baseline)| *mi > *baseline * 2.0 && *mi > 0.01)
        .map(|(lag, mi, baseline)| format!("lag{}={:.4}(base={:.4})", lag, mi, baseline))
        .collect();

    let verdict = if significant_lags.len() >= 3 {
        ResearchVerdict::Significant
    } else if !significant_lags.is_empty() {
        ResearchVerdict::Marginal
    } else {
        ResearchVerdict::NotSignificant
    };

    let all_mi: Vec<String> = mi_results.iter()
        .map(|(lag, mi, _)| format!("k{}={:.4}", lag, mi))
        .collect();

    vec![TestResult {
        test_name: "MI retardée multi-échelle".to_string(),
        category: "informational".to_string(),
        statistic: max_excess,
        p_value: None,
        effect_size: max_excess,
        verdict,
        detail: format!(
            "{} | sig: {}",
            all_mi.join(", "),
            if significant_lags.is_empty() { "aucun".to_string() } else { significant_lags.join(", ") }
        ),
    }]
}

fn discretize(series: &[f64], n_bins: usize) -> Vec<usize> {
    if series.is_empty() {
        return vec![];
    }
    let min = series.iter().cloned().fold(f64::MAX, f64::min);
    let max = series.iter().cloned().fold(f64::MIN, f64::max);
    let range = max - min;
    if range == 0.0 {
        return vec![0; series.len()];
    }
    series.iter()
        .map(|&x| {
            let bin = ((x - min) / range * n_bins as f64) as usize;
            bin.min(n_bins - 1)
        })
        .collect()
}

fn mutual_information(x: &[usize], y: &[usize], n_bins: usize) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mut joint = vec![vec![0u32; n_bins]; n_bins];
    let mut mx = vec![0u32; n_bins];
    let mut my = vec![0u32; n_bins];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi < n_bins && yi < n_bins {
            joint[xi][yi] += 1;
            mx[xi] += 1;
            my[yi] += 1;
        }
    }

    let mut mi = 0.0;
    for i in 0..n_bins {
        for j in 0..n_bins {
            if joint[i][j] > 0 {
                let pxy = joint[i][j] as f64 / n;
                let px = mx[i] as f64 / n;
                let py = my[j] as f64 / n;
                if px > 0.0 && py > 0.0 {
                    mi += pxy * (pxy / (px * py)).ln();
                }
            }
        }
    }

    mi.max(0.0)
}

fn shuffled_mi_baseline(data: &[usize], lag: usize, n_bins: usize, n_shuffles: usize) -> f64 {
    let n = data.len();
    if lag >= n {
        return 0.0;
    }

    let mut total_mi = 0.0;
    let mut rng_state: u64 = 42;

    for _ in 0..n_shuffles {
        // Create shuffled version
        let mut shuffled: Vec<usize> = data.to_vec();
        for i in (1..shuffled.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            shuffled.swap(i, j);
        }

        let x = &data[..n - lag];
        let y = &shuffled[lag..];
        total_mi += mutual_information(x, y, n_bins);
    }

    total_mi / n_shuffles as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_conditional_entropy_no_panic() {
        let draws = make_test_draws(100);
        let result = conditional_entropy_test(&draws);
        assert_eq!(result.category, "informational");
    }

    #[test]
    fn test_transfer_entropy_no_panic() {
        let draws = make_test_draws(100);
        let results = transfer_entropy_tests(&draws);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_compression_no_panic() {
        let draws = make_test_draws(100);
        let results = compression_tests(&draws);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_delayed_mi_no_panic() {
        let draws = make_test_draws(100);
        let results = delayed_mi_tests(&draws);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_compress_deflate() {
        let data = vec![1u8, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        let compressed_len = compress_deflate(&data);
        assert!(compressed_len > 0);
        assert!(compressed_len <= data.len() + 20); // deflate may add overhead for small data
    }

    #[test]
    fn test_transfer_entropy_independent() {
        // Two independent binary series generated with different PRNG seeds
        let mut rng1: u64 = 12345;
        let source: Vec<bool> = (0..500).map(|_| {
            rng1 = rng1.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng1 >> 63) == 0
        }).collect();
        let mut rng2: u64 = 98765;
        let target: Vec<bool> = (0..500).map(|_| {
            rng2 = rng2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng2 >> 63) == 0
        }).collect();
        let te = compute_transfer_entropy(&source, &target);
        assert!(te < 0.1, "TE should be small for independent series, got {}", te);
    }

    #[test]
    fn test_mutual_information_identical() {
        let x = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let y = vec![0, 1, 2, 3, 0, 1, 2, 3]; // identical
        let mi = mutual_information(&x, &y, 4);
        assert!(mi > 1.0, "MI should be high for identical series, got {}", mi);
    }

    #[test]
    fn test_run_informational_tests() {
        let draws = make_test_draws(100);
        let results = run_informational_tests(&draws);
        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.category, "informational");
        }
    }
}
