use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Signal,
    Neutral,
    Random,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Signal => write!(f, "SIGNAL"),
            Verdict::Neutral => write!(f, "NEUTRE"),
            Verdict::Random => write!(f, "ALÉATOIRE"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub test_name: String,
    pub value: f64,
    pub expected_random: f64,
    pub verdict: Verdict,
    pub detail: String,
}

/// Exécute tous les tests de non-aléatoire sur l'historique des tirages.
pub fn run_all_tests(draws: &[Draw]) -> Vec<AnalysisResult> {
    // Séries scalaires en ordre chronologique (draws[0] = le plus récent → on inverse)
    let ball_sums: Vec<f64> = draws
        .iter()
        .rev()
        .map(|d| d.balls.iter().map(|&b| b as f64).sum())
        .collect();

    // Séries binaires par numéro (chronologique)
    let ball_binary = extract_binary_sequences(draws, Pool::Balls);
    let star_binary = extract_binary_sequences(draws, Pool::Stars);

    vec![
        permutation_entropy_test(&ball_sums),
        runs_test_aggregate(&ball_binary, &star_binary),
        ami_test(&ball_sums),
        correlation_dimension_test(&ball_sums),
        lempel_ziv_test(&ball_binary, &star_binary),
    ]
}

/// Pour chaque numéro du pool, extrait une série binaire (true = présent) en ordre chronologique.
fn extract_binary_sequences(draws: &[Draw], pool: Pool) -> Vec<Vec<bool>> {
    let size = pool.size();
    (1..=size as u8)
        .map(|num| {
            draws
                .iter()
                .rev()
                .map(|d| pool.numbers_from(d).contains(&num))
                .collect()
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════
// 1. Entropie de permutation
// ════════════════════════════════════════════════════════════════

fn ordinal_pattern(window: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..window.len()).collect();
    indices.sort_by(|&a, &b| {
        window[a]
            .partial_cmp(&window[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut pattern = vec![0usize; window.len()];
    for (rank, &idx) in indices.iter().enumerate() {
        pattern[idx] = rank;
    }
    pattern
}

fn permutation_entropy_test(series: &[f64]) -> AnalysisResult {
    let d = 5usize; // dimension d'embedding
    let tau = 1usize;

    if series.len() < d * tau + 1 {
        return AnalysisResult {
            test_name: "Entropie de permutation".into(),
            value: f64::NAN,
            expected_random: 1.0,
            verdict: Verdict::Neutral,
            detail: "Série trop courte".into(),
        };
    }

    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    let n_windows = series.len() - (d - 1) * tau;

    for i in 0..n_windows {
        let window: Vec<f64> = (0..d).map(|j| series[i + j * tau]).collect();
        let pattern = ordinal_pattern(&window);
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    let total = n_windows as f64;
    let h: f64 = pattern_counts
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let factorial: f64 = (1..=d).map(|i| i as f64).product::<f64>();
    let h_max = factorial.ln();
    let h_norm = h / h_max;

    let verdict = if h_norm < 0.90 {
        Verdict::Signal
    } else if h_norm > 0.97 {
        Verdict::Random
    } else {
        Verdict::Neutral
    };

    AnalysisResult {
        test_name: "Entropie de permutation".into(),
        value: h_norm,
        expected_random: 1.0,
        verdict,
        detail: format!(
            "H_norm={:.4} (d={}, H_max={:.2}), {} patterns distincts sur {}",
            h_norm,
            d,
            h_max,
            pattern_counts.len(),
            factorial as u64
        ),
    }
}

// ════════════════════════════════════════════════════════════════
// 2. Test des runs (Wald-Wolfowitz)
// ════════════════════════════════════════════════════════════════

fn runs_test_single(seq: &[bool]) -> f64 {
    if seq.len() < 20 {
        return 0.0;
    }

    let n1 = seq.iter().filter(|&&x| x).count() as f64;
    let n2 = seq.iter().filter(|&&x| !x).count() as f64;
    let n = n1 + n2;

    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }

    let mut runs = 1u64;
    for i in 1..seq.len() {
        if seq[i] != seq[i - 1] {
            runs += 1;
        }
    }

    let r = runs as f64;
    let expected = 1.0 + 2.0 * n1 * n2 / n;
    let variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0));

    if variance <= 0.0 {
        return 0.0;
    }

    (r - expected) / variance.sqrt()
}

fn runs_test_aggregate(ball_binary: &[Vec<bool>], star_binary: &[Vec<bool>]) -> AnalysisResult {
    let mut z_scores = Vec::new();

    for seq in ball_binary.iter().chain(star_binary.iter()) {
        let z = runs_test_single(seq);
        z_scores.push(z);
    }

    let significant = z_scores.iter().filter(|&&z| z.abs() > 1.96).count();
    let total = z_scores.len();
    let fraction_sig = significant as f64 / total as f64;

    let verdict = if fraction_sig > 0.15 {
        Verdict::Signal
    } else if fraction_sig < 0.08 {
        Verdict::Random
    } else {
        Verdict::Neutral
    };

    let avg_abs_z: f64 = z_scores.iter().map(|z| z.abs()).sum::<f64>() / total as f64;

    AnalysisResult {
        test_name: "Test des runs (Wald-Wolfowitz)".into(),
        value: fraction_sig,
        expected_random: 0.05,
        verdict,
        detail: format!(
            "{}/{} numéros significatifs (p<0.05), |z| moyen={:.2}",
            significant, total, avg_abs_z
        ),
    }
}

// ════════════════════════════════════════════════════════════════
// 3. Auto-information mutuelle (AMI)
// ════════════════════════════════════════════════════════════════

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
    series
        .iter()
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
    let mut marginal_x = vec![0u32; n_bins];
    let mut marginal_y = vec![0u32; n_bins];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        joint[xi][yi] += 1;
        marginal_x[xi] += 1;
        marginal_y[yi] += 1;
    }

    let mut mi = 0.0;
    for i in 0..n_bins {
        for j in 0..n_bins {
            if joint[i][j] > 0 {
                let pxy = joint[i][j] as f64 / n;
                let px = marginal_x[i] as f64 / n;
                let py = marginal_y[j] as f64 / n;
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }

    mi
}

fn ami_test(series: &[f64]) -> AnalysisResult {
    let n_bins = 10;
    let discretized = discretize(series, n_bins);
    let max_lag = 20.min(series.len() / 5);

    let mut max_ami = 0.0f64;
    let mut max_lag_val = 0;

    for lag in 1..=max_lag {
        let x = &discretized[..discretized.len() - lag];
        let y = &discretized[lag..];
        let ami = mutual_information(x, y, n_bins);
        if ami > max_ami {
            max_ami = ami;
            max_lag_val = lag;
        }
    }

    // Biais estimé pour données aléatoires
    let bias = (n_bins * n_bins) as f64 / (2.0 * series.len() as f64);

    let verdict = if max_ami > 0.1 {
        Verdict::Signal
    } else if max_ami < bias * 3.0 {
        Verdict::Random
    } else {
        Verdict::Neutral
    };

    AnalysisResult {
        test_name: "Auto-information mutuelle".into(),
        value: max_ami,
        expected_random: bias,
        verdict,
        detail: format!(
            "Max AMI={:.4} au lag {} (biais estimé={:.4})",
            max_ami, max_lag_val, bias
        ),
    }
}

// ════════════════════════════════════════════════════════════════
// 4. Dimension de corrélation (Grassberger-Procaccia)
// ════════════════════════════════════════════════════════════════

fn embed_series(series: &[f64], dim: usize, tau: usize) -> Vec<Vec<f64>> {
    let n = series.len();
    let offset = (dim - 1) * tau;
    if offset >= n {
        return vec![];
    }
    (offset..n)
        .map(|i| (0..dim).map(|d| series[i - d * tau]).collect())
        .collect()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn correlation_integral(embedded: &[Vec<f64>], r: f64) -> f64 {
    let n = embedded.len();
    if n < 2 {
        return 0.0;
    }

    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            if euclidean_distance(&embedded[i], &embedded[j]) < r {
                count += 1;
            }
        }
    }

    2.0 * count as f64 / (n as f64 * (n as f64 - 1.0))
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

fn correlation_dimension_test(series: &[f64]) -> AnalysisResult {
    let tau = 1;
    let dims = [3, 5, 7];

    let mut d2_values = Vec::new();

    for &dim in &dims {
        let embedded = embed_series(series, dim, tau);
        if embedded.len() < 50 {
            continue;
        }

        // Sous-échantillonnage pour performance (max 400 points)
        let sub: Vec<Vec<f64>> = if embedded.len() > 400 {
            let step = embedded.len() / 400;
            embedded.into_iter().step_by(step).collect()
        } else {
            embedded
        };

        // Calcul des distances pour déterminer la plage de r
        let mut distances = Vec::new();
        let n_dist = sub.len().min(200);
        for i in 0..n_dist {
            for j in (i + 1)..n_dist {
                distances.push(euclidean_distance(&sub[i], &sub[j]));
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if distances.is_empty() {
            continue;
        }

        let r_low = distances[distances.len() / 10];
        let r_high = distances[distances.len() * 9 / 10];

        if r_low <= 0.0 || r_high <= r_low {
            continue;
        }

        let n_r = 15;
        let log_r_low = r_low.ln();
        let log_r_high = r_high.ln();
        let step = (log_r_high - log_r_low) / (n_r - 1) as f64;

        let mut log_r_vec = Vec::new();
        let mut log_c_vec = Vec::new();

        for i in 0..n_r {
            let r = (log_r_low + i as f64 * step).exp();
            let c = correlation_integral(&sub, r);
            if c > 0.0 {
                log_r_vec.push(r.ln());
                log_c_vec.push(c.ln());
            }
        }

        if log_r_vec.len() >= 5 {
            let d2 = linear_regression_slope(&log_r_vec, &log_c_vec);
            d2_values.push((dim, d2));
        }
    }

    if d2_values.is_empty() {
        return AnalysisResult {
            test_name: "Dimension de corrélation".into(),
            value: f64::NAN,
            expected_random: f64::INFINITY,
            verdict: Verdict::Neutral,
            detail: "Données insuffisantes pour estimer D2".into(),
        };
    }

    let last_d2 = d2_values.last().unwrap().1;
    let first_d2 = d2_values.first().unwrap().1;
    let last_dim = d2_values.last().unwrap().0;

    let verdict = if last_d2 < last_dim as f64 * 0.6 && (last_d2 - first_d2).abs() < 1.0 {
        Verdict::Signal // D2 sature en-dessous de la dimension d'embedding
    } else if last_d2 > last_dim as f64 * 0.8 {
        Verdict::Random // D2 croît avec la dimension d'embedding
    } else {
        Verdict::Neutral
    };

    let d2_str = d2_values
        .iter()
        .map(|(dim, d2)| format!("m={}->D2={:.2}", dim, d2))
        .collect::<Vec<_>>()
        .join(", ");

    AnalysisResult {
        test_name: "Dimension de corrélation".into(),
        value: last_d2,
        expected_random: last_dim as f64,
        verdict,
        detail: d2_str,
    }
}

// ════════════════════════════════════════════════════════════════
// 5. Complexité de Lempel-Ziv
// ════════════════════════════════════════════════════════════════

fn lempel_ziv_complexity(seq: &[bool]) -> usize {
    if seq.is_empty() {
        return 0;
    }

    let mut seen = std::collections::HashSet::new();
    let mut word = Vec::new();
    let mut complexity = 0usize;

    for &bit in seq {
        word.push(bit);
        if !seen.contains(&word) {
            seen.insert(word.clone());
            complexity += 1;
            word.clear();
        }
    }
    if !word.is_empty() {
        complexity += 1;
    }

    complexity
}

fn lempel_ziv_test(ball_binary: &[Vec<bool>], star_binary: &[Vec<bool>]) -> AnalysisResult {
    let mut ratios = Vec::new();

    for seq in ball_binary.iter().chain(star_binary.iter()) {
        if seq.len() < 30 {
            continue;
        }
        let c = lempel_ziv_complexity(seq);
        let n = seq.len() as f64;
        let c_norm = c as f64 * n.ln() / n;

        // Entropie attendue pour un Bernoulli(p) : h = -p*ln(p) - (1-p)*ln(1-p)
        let p = seq.iter().filter(|&&x| x).count() as f64 / n;
        let h = if p > 0.0 && p < 1.0 {
            -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
        } else {
            continue; // séquence constante
        };

        if h > 0.0 {
            ratios.push(c_norm / h);
        }
    }

    if ratios.is_empty() {
        return AnalysisResult {
            test_name: "Complexité de Lempel-Ziv".into(),
            value: f64::NAN,
            expected_random: 1.0,
            verdict: Verdict::Neutral,
            detail: "Données insuffisantes".into(),
        };
    }

    let avg = ratios.iter().sum::<f64>() / ratios.len() as f64;

    let verdict = if avg < 0.85 {
        Verdict::Signal
    } else if avg > 0.95 {
        Verdict::Random
    } else {
        Verdict::Neutral
    };

    let below_threshold = ratios.iter().filter(|&&c| c < 0.85).count();

    AnalysisResult {
        test_name: "Complexité de Lempel-Ziv".into(),
        value: avg,
        expected_random: 1.0,
        verdict,
        detail: format!(
            "c/h moyen={:.4}, {}/{} numéros avec c/h < 0.85",
            avg,
            below_threshold,
            ratios.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_ordinal_pattern() {
        assert_eq!(ordinal_pattern(&[3.0, 1.0, 2.0]), vec![2, 0, 1]);
        assert_eq!(ordinal_pattern(&[1.0, 2.0, 3.0]), vec![0, 1, 2]);
    }

    #[test]
    fn test_lempel_ziv_constant() {
        // LZ78 sur séquence constante : complexité O(√N), pas O(1)
        // Pour N=1000, C ≈ √(2*1000) ≈ 45
        let constant = vec![true; 1000];
        let c_const = lempel_ziv_complexity(&constant);
        // Séquence pseudo-aléatoire pour comparaison
        let pseudo: Vec<bool> = (0..1000).map(|i| (i * 7 + 13) % 11 < 5).collect();
        let c_pseudo = lempel_ziv_complexity(&pseudo);
        assert!(
            c_const < c_pseudo,
            "séquence constante devrait être moins complexe: {} vs {}",
            c_const,
            c_pseudo
        );
    }

    #[test]
    fn test_lempel_ziv_periodic() {
        let periodic: Vec<bool> = (0..200).map(|i| i % 2 == 0).collect();
        let random_like: Vec<bool> = (0..200).map(|i| (i * 7 + 13) % 11 < 5).collect();
        let c_periodic = lempel_ziv_complexity(&periodic);
        let c_random = lempel_ziv_complexity(&random_like);
        assert!(
            c_periodic < c_random,
            "séquence périodique devrait être moins complexe: {} vs {}",
            c_periodic,
            c_random
        );
    }

    #[test]
    fn test_runs_test_alternating() {
        let seq: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let z = runs_test_single(&seq);
        assert!(
            z.abs() > 1.96,
            "séquence alternante devrait être significative, z={z}"
        );
    }

    #[test]
    fn test_discretize() {
        let series = vec![0.0, 0.5, 1.0];
        let bins = discretize(&series, 4);
        assert_eq!(bins[0], 0);
        assert_eq!(bins[2], 3);
    }

    #[test]
    fn test_embed_series() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedded = embed_series(&series, 3, 1);
        assert_eq!(embedded.len(), 3);
        assert_eq!(embedded[0], vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Deux distributions indépendantes uniformes → MI ≈ 0
        let x = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let mi = mutual_information(&x, &y, 4);
        // Pas exactement 0 car dépend de la distribution conjointe exacte
        assert!(mi < 0.5, "MI devrait être faible pour des variables quasi-indépendantes, got {mi}");
    }

    #[test]
    fn test_run_all_tests_no_panic() {
        let draws = make_test_draws(100);
        let results = run_all_tests(&draws);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(!r.test_name.is_empty());
        }
    }

    #[test]
    fn test_linear_regression_slope() {
        // y = 2x → slope = 2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let slope = linear_regression_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
    }
}
