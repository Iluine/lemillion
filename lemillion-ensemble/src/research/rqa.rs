use lemillion_db::models::Draw;

use super::{TestResult, ResearchVerdict, two_sided_p};

/// Recurrence Quantification Analysis (RQA)
///
/// Computes RQA metrics on the recurrence matrix from Takens embedding
/// of ball/star frequency series. Compares against shuffled surrogates
/// to assess significance.

const EMBEDDING_DIM: usize = 3;
const DELAY: usize = 1;
const EPSILON_FACTOR: f64 = 0.2;
const MIN_LINE_LENGTH: usize = 2;
const N_SURROGATES: usize = 100;

pub fn run_rqa_tests(draws: &[Draw]) -> Vec<TestResult> {
    if draws.len() < 60 {
        return vec![TestResult {
            test_name: "RQA Recurrence Rate".to_string(),
            category: "rqa".to_string(),
            statistic: 0.0,
            p_value: None,
            effect_size: 0.0,
            verdict: ResearchVerdict::NotSignificant,
            detail: format!("Insuffisant : {} tirages (minimum 60)", draws.len()),
        }];
    }

    // Build a multivariate frequency series from ball sums + star sums
    // Use ball sum as the primary series for Takens embedding
    let ball_sums: Vec<f64> = draws.iter().rev()
        .map(|d| d.balls.iter().map(|&b| b as f64).sum())
        .collect();

    // Compute RQA on the real series
    let real_metrics = compute_rqa_metrics(&ball_sums);

    // Compute RQA on shuffled surrogates for baseline
    let surrogate_metrics = compute_surrogate_baselines(&ball_sums, N_SURROGATES);

    // Build results comparing real vs surrogates
    let metric_names = [
        ("RQA Recurrence Rate (RR)", "Fraction des points de recurrence"),
        ("RQA Determinisme (DET)", "Fraction en lignes diagonales"),
        ("RQA L_max (diagonale max)", "Plus longue ligne diagonale"),
        ("RQA Entropie diag (ENTR)", "Entropie Shannon des longueurs diag"),
        ("RQA Laminarite (LAM)", "Fraction en lignes verticales"),
    ];

    let mut results = Vec::new();

    for (i, (name, desc)) in metric_names.iter().enumerate() {
        let real_val = real_metrics[i];
        let surr_vals: Vec<f64> = surrogate_metrics.iter().map(|m| m[i]).collect();

        let (surr_mean, surr_std) = mean_std(&surr_vals);

        // z-score vs surrogate distribution
        let z = if surr_std > 1e-12 {
            (real_val - surr_mean) / surr_std
        } else {
            0.0
        };
        let p = two_sided_p(z);

        // Effect size: Cohen's d
        let effect_size = if surr_std > 1e-12 {
            (real_val - surr_mean).abs() / surr_std
        } else {
            0.0
        };

        // p-value from rank in surrogates
        let rank_p = compute_rank_p_value(real_val, &surr_vals);

        // Use the more conservative p-value
        let final_p = rank_p.max(p);

        let verdict = if final_p < 0.01 {
            ResearchVerdict::Significant
        } else if final_p < 0.05 {
            ResearchVerdict::Marginal
        } else {
            ResearchVerdict::NotSignificant
        };

        results.push(TestResult {
            test_name: name.to_string(),
            category: "rqa".to_string(),
            statistic: real_val,
            p_value: Some(final_p),
            effect_size,
            verdict,
            detail: format!(
                "{}: reel={:.4}, surr_moy={:.4}, surr_std={:.4}, z={:.2}, p_rank={:.4} | {}",
                desc, real_val, surr_mean, surr_std, z, rank_p, desc
            ),
        });
    }

    results
}

/// RQA metrics: [RR, DET, L_max, ENTR, LAM]
fn compute_rqa_metrics(series: &[f64]) -> [f64; 5] {
    let embedded = takens_embedding(series, EMBEDDING_DIM, DELAY);
    if embedded.len() < 10 {
        return [0.0; 5];
    }

    let std_dev = series_std(series);
    let epsilon = EPSILON_FACTOR * std_dev;
    if epsilon < 1e-15 {
        return [0.0; 5];
    }

    let n = embedded.len();
    let recurrence = build_recurrence_matrix(&embedded, epsilon);

    // RR: Recurrence Rate
    let total_points = n * n;
    let rr = recurrence.iter().map(|row| row.iter().filter(|&&v| v).count()).sum::<usize>() as f64
        / total_points as f64;

    // Diagonal lines (exclude main diagonal)
    let diag_lines = extract_diagonal_lines(&recurrence, MIN_LINE_LENGTH);

    // DET: fraction of recurrence points in diagonal lines
    let diag_points: usize = diag_lines.iter().sum();
    let total_recurrence = recurrence.iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().filter(move |(j, v)| **v && *j != i).map(|(_, _)| 1))
        .count();
    let det = if total_recurrence > 0 {
        diag_points as f64 / total_recurrence as f64
    } else {
        0.0
    };

    // L_max: longest diagonal line
    let l_max = diag_lines.iter().copied().max().unwrap_or(0) as f64;

    // ENTR: Shannon entropy of diagonal line length distribution
    let entr = diagonal_entropy(&diag_lines);

    // Vertical lines for LAM
    let vert_lines = extract_vertical_lines(&recurrence, MIN_LINE_LENGTH);
    let vert_points: usize = vert_lines.iter().sum();
    let lam = if total_recurrence > 0 {
        vert_points as f64 / total_recurrence as f64
    } else {
        0.0
    };

    [rr, det, l_max, entr, lam]
}

/// Takens time-delay embedding.
fn takens_embedding(series: &[f64], dim: usize, delay: usize) -> Vec<Vec<f64>> {
    let n = series.len();
    let required = (dim - 1) * delay + 1;
    if n < required {
        return vec![];
    }

    let m = n - (dim - 1) * delay;
    (0..m)
        .map(|i| {
            (0..dim).map(|d| series[i + d * delay]).collect()
        })
        .collect()
}

/// Build the recurrence matrix: R[i][j] = true if ||x_i - x_j|| < epsilon.
fn build_recurrence_matrix(embedded: &[Vec<f64>], epsilon: f64) -> Vec<Vec<bool>> {
    let n = embedded.len();
    let eps_sq = epsilon * epsilon;
    let mut matrix = vec![vec![false; n]; n];

    for i in 0..n {
        for j in i..n {
            let dist_sq: f64 = embedded[i].iter().zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist_sq < eps_sq {
                matrix[i][j] = true;
                matrix[j][i] = true;
            }
        }
    }

    matrix
}

/// Extract diagonal line lengths (excluding main diagonal).
/// A diagonal line at offset k is the set of (i, i+k) where R[i][i+k] = true for consecutive i.
fn extract_diagonal_lines(matrix: &[Vec<bool>], min_length: usize) -> Vec<usize> {
    let n = matrix.len();
    let mut line_lengths = Vec::new();

    // Scan diagonals above the main diagonal (k > 0)
    for k in 1..n {
        let mut current_length = 0;
        for i in 0..(n - k) {
            if matrix[i][i + k] {
                current_length += 1;
            } else {
                if current_length >= min_length {
                    line_lengths.push(current_length);
                }
                current_length = 0;
            }
        }
        if current_length >= min_length {
            line_lengths.push(current_length);
        }
    }

    // Also scan diagonals below (by symmetry, same as above for symmetric matrix)
    // but for completeness (and since we already have a symmetric matrix),
    // the below-diagonal is the mirror, so we double-count. We only count above.

    line_lengths
}

/// Shannon entropy of the diagonal line length distribution.
fn diagonal_entropy(line_lengths: &[usize]) -> f64 {
    if line_lengths.is_empty() {
        return 0.0;
    }

    let total = line_lengths.len() as f64;

    // Count frequency of each length
    let max_len = line_lengths.iter().copied().max().unwrap_or(0);
    let mut freq = vec![0usize; max_len + 1];
    for &l in line_lengths {
        freq[l] += 1;
    }

    let mut entropy = 0.0;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / total;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Extract vertical line lengths.
/// A vertical line at column j is the set of consecutive rows i where R[i][j] = true.
fn extract_vertical_lines(matrix: &[Vec<bool>], min_length: usize) -> Vec<usize> {
    let n = matrix.len();
    let mut line_lengths = Vec::new();

    for j in 0..n {
        let mut current_length = 0;
        for i in 0..n {
            if i == j {
                // Skip main diagonal
                if current_length >= min_length {
                    line_lengths.push(current_length);
                }
                current_length = 0;
                continue;
            }
            if matrix[i][j] {
                current_length += 1;
            } else {
                if current_length >= min_length {
                    line_lengths.push(current_length);
                }
                current_length = 0;
            }
        }
        if current_length >= min_length {
            line_lengths.push(current_length);
        }
    }

    line_lengths
}

/// Compute RQA baselines from shuffled surrogates.
fn compute_surrogate_baselines(series: &[f64], n_surrogates: usize) -> Vec<[f64; 5]> {
    let mut rng_state: u64 = 42;
    let mut results = Vec::with_capacity(n_surrogates);

    for _ in 0..n_surrogates {
        // Shuffle the series
        let mut shuffled = series.to_vec();
        for i in (1..shuffled.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            shuffled.swap(i, j);
        }

        let metrics = compute_rqa_metrics(&shuffled);
        results.push(metrics);
    }

    results
}

/// Two-sided rank-based p-value: proportion of surrogates with |metric - mean| >= |real - mean|.
fn compute_rank_p_value(real_val: f64, surrogate_vals: &[f64]) -> f64 {
    if surrogate_vals.is_empty() {
        return 1.0;
    }

    let mean = surrogate_vals.iter().sum::<f64>() / surrogate_vals.len() as f64;
    let real_dev = (real_val - mean).abs();

    let more_extreme = surrogate_vals.iter()
        .filter(|&&v| (v - mean).abs() >= real_dev)
        .count();

    // Add 1 for the real observation (conservative)
    (more_extreme + 1) as f64 / (surrogate_vals.len() + 1) as f64
}

fn series_std(series: &[f64]) -> f64 {
    let n = series.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = series.iter().sum::<f64>() / n;
    let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
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
    fn test_rqa_no_panic() {
        let draws = make_test_draws(100);
        let results = run_rqa_tests(&draws);
        assert_eq!(results.len(), 5, "Expected 5 RQA metrics, got {}", results.len());
        for r in &results {
            assert_eq!(r.category, "rqa");
            assert!(r.p_value.is_some(), "Each RQA metric should have a p-value");
        }
    }

    #[test]
    fn test_rqa_insufficient_data() {
        let draws = make_test_draws(10);
        let results = run_rqa_tests(&draws);
        assert!(!results.is_empty());
        assert_eq!(results[0].verdict, ResearchVerdict::NotSignificant);
    }

    #[test]
    fn test_takens_embedding() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedded = takens_embedding(&series, 3, 1);
        assert_eq!(embedded.len(), 3); // 5 - (3-1)*1 = 3
        assert_eq!(embedded[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(embedded[1], vec![2.0, 3.0, 4.0]);
        assert_eq!(embedded[2], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_takens_embedding_delay() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let embedded = takens_embedding(&series, 2, 3);
        // n=7, dim=2, delay=3: required = 1*3+1 = 4, m = 7-3 = 4
        assert_eq!(embedded.len(), 4);
        assert_eq!(embedded[0], vec![1.0, 4.0]);
        assert_eq!(embedded[1], vec![2.0, 5.0]);
    }

    #[test]
    fn test_takens_embedding_too_short() {
        let series = vec![1.0, 2.0];
        let embedded = takens_embedding(&series, 3, 1);
        assert!(embedded.is_empty());
    }

    #[test]
    fn test_recurrence_matrix_identical() {
        let embedded = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![5.0, 5.0],
        ];
        let rm = build_recurrence_matrix(&embedded, 0.5);
        // Points 0 and 1 are identical (distance 0 < 0.5), point 2 is far
        assert!(rm[0][0]);
        assert!(rm[0][1]);
        assert!(rm[1][0]);
        assert!(!rm[0][2]);
        assert!(!rm[2][0]);
    }

    #[test]
    fn test_diagonal_lines_simple() {
        // Matrix where (0,1), (1,2), (2,3) are all recurrent -> diagonal of length 3
        let n = 5;
        let mut matrix = vec![vec![false; n]; n];
        for i in 0..n {
            matrix[i][i] = true;
        }
        // Create a diagonal: (0,1), (1,2), (2,3)
        matrix[0][1] = true; matrix[1][0] = true;
        matrix[1][2] = true; matrix[2][1] = true;
        matrix[2][3] = true; matrix[3][2] = true;

        let lines = extract_diagonal_lines(&matrix, 2);
        // Should find a line of length 3 on diagonal k=1
        assert!(lines.contains(&3), "Expected a diagonal line of length 3, got {:?}", lines);
    }

    #[test]
    fn test_vertical_lines_simple() {
        let n = 5;
        let mut matrix = vec![vec![false; n]; n];
        // Vertical line at column 3: rows 0, 1, 2 (length 3, excluding main diag)
        matrix[0][3] = true;
        matrix[1][3] = true;
        matrix[2][3] = true;

        let lines = extract_vertical_lines(&matrix, 2);
        assert!(lines.contains(&3), "Expected a vertical line of length 3, got {:?}", lines);
    }

    #[test]
    fn test_diagonal_entropy_uniform() {
        // All lines same length -> entropy = 0
        let lines = vec![3, 3, 3, 3];
        let e = diagonal_entropy(&lines);
        assert!(e.abs() < 1e-10, "Entropy should be 0 for uniform lengths, got {}", e);
    }

    #[test]
    fn test_diagonal_entropy_varied() {
        // Different lengths -> positive entropy
        let lines = vec![2, 3, 4, 5, 2, 3];
        let e = diagonal_entropy(&lines);
        assert!(e > 0.0, "Entropy should be > 0 for varied lengths, got {}", e);
    }

    #[test]
    fn test_rqa_metrics_constant_series() {
        // Constant series: all embedded points identical, RR should be 1.0
        let series = vec![127.5; 100];
        let metrics = compute_rqa_metrics(&series);
        // std_dev = 0, so epsilon = 0 -> no recurrence except exact matches
        // Actually epsilon = 0.2 * 0.0 = 0, so early return with zeros
        assert_eq!(metrics, [0.0; 5]);
    }

    #[test]
    fn test_rank_p_value() {
        // Real value much larger than all surrogates -> small p-value
        let surrogates = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        let p = compute_rank_p_value(5.0, &surrogates);
        // Only 1 (the real value itself in the conservative count) out of 6
        assert!(p < 0.3, "p should be small for extreme value, got {}", p);
    }

    #[test]
    fn test_rank_p_value_typical() {
        // Real value within range of surrogates -> large p-value
        let surrogates = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = compute_rank_p_value(3.0, &surrogates);
        assert!(p > 0.3, "p should be large for typical value, got {}", p);
    }

    #[test]
    fn test_series_std() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = series_std(&series);
        // std = sqrt(2.0)
        assert!((s - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_rqa_full_pipeline() {
        let draws = make_test_draws(150);
        let results = run_rqa_tests(&draws);
        for r in &results {
            assert!(!r.test_name.is_empty());
            assert!(!r.detail.is_empty());
            assert!(r.statistic.is_finite(), "Statistic should be finite");
            if let Some(p) = r.p_value {
                assert!(p >= 0.0 && p <= 1.0, "p-value should be in [0,1], got {}", p);
            }
        }
    }
}
