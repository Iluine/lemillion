use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// RqaPredictabilityModel — exploits RQA (Recurrence Quantification Analysis)
/// metrics to adapt prediction strategy based on system predictability.
///
/// Signal: RQA DET and LAM measure deterministic structure in the draw sequence.
/// - High DET (above threshold): the system is in a predictable regime.
///   Identify recurrent states and boost numbers that appear in recent
///   recurrent episodes via EWMA frequency weighted by DET.
/// - Low DET: the system is chaotic, return near-uniform (high smoothing).
///
/// Implements lightweight inline RQA computation (Takens embedding +
/// recurrence matrix + diagonal/vertical line analysis) to avoid coupling
/// with research::rqa.
///
/// For stars: simpler EWMA with DET-based smoothing adjustment.
pub struct RqaPredictabilityModel {
    smoothing: f64,
    min_draws: usize,
    embedding_dim: usize,
    delay: usize,
    epsilon_factor: f64,
    min_line_length: usize,
    ewma_alpha: f64,
    det_threshold: f64,
}

impl Default for RqaPredictabilityModel {
    fn default() -> Self {
        Self {
            smoothing: 0.20,
            min_draws: 30,
            embedding_dim: 3,
            delay: 1,
            epsilon_factor: 0.2,
            min_line_length: 2,
            ewma_alpha: 0.10,
            det_threshold: 0.3,
        }
    }
}

impl ForecastModel for RqaPredictabilityModel {
    fn name(&self) -> &str {
        "RqaPredictability"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        match pool {
            Pool::Balls => self.predict_balls(draws, size),
            Pool::Stars => self.predict_stars(draws, size),
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("embedding_dim".into(), self.embedding_dim as f64),
            ("delay".into(), self.delay as f64),
            ("epsilon_factor".into(), self.epsilon_factor),
            ("min_line_length".into(), self.min_line_length as f64),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("det_threshold".into(), self.det_threshold),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn calibration_stride(&self) -> usize {
        2
    }
}

impl RqaPredictabilityModel {
    fn predict_balls(&self, draws: &[Draw], size: usize) -> Vec<f64> {
        let uniform = vec![1.0 / size as f64; size];

        // Build ball sum time series in chronological order
        let ball_sums: Vec<f64> = draws
            .iter()
            .rev()
            .map(|d| d.balls.iter().map(|&b| b as f64).sum())
            .collect();

        // Compute RQA metrics on ball sums
        let [_rr, det, _l_max, _entr, lam] =
            compute_rqa_metrics(&ball_sums, self.embedding_dim, self.delay, self.epsilon_factor, self.min_line_length);

        // Compute EWMA frequency per number (chronological iteration)
        let mut freq_ewma = vec![1.0 / size as f64; size];
        for draw in draws.iter().rev() {
            for (idx, freq) in freq_ewma.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if draw.balls.contains(&num) { 1.0 } else { 0.0 };
                *freq = self.ewma_alpha * present + (1.0 - self.ewma_alpha) * *freq;
            }
        }

        // Normalize EWMA frequencies
        let freq_sum: f64 = freq_ewma.iter().sum();
        if freq_sum > 0.0 {
            for f in &mut freq_ewma {
                *f /= freq_sum;
            }
        }

        // Adapt based on DET/LAM: combine predictability score
        let predictability = (det + lam) / 2.0;

        if predictability < self.det_threshold {
            // Low predictability: near-uniform with heavy smoothing
            let heavy_smoothing = 0.6;
            let uniform_val = 1.0 / size as f64;
            let mut probs = freq_ewma;
            for p in &mut probs {
                *p = (1.0 - heavy_smoothing) * *p + heavy_smoothing * uniform_val;
            }
            floor_only(&mut probs, PROB_FLOOR_BALLS);
            return probs;
        }

        // High predictability: identify recurrent patterns
        // Find which numbers co-occur with recent recurrent states
        let recurrence_boost = self.compute_recurrence_boost(draws, &ball_sums, size);

        // Blend: weight by predictability strength
        // predictability_weight scales from 0.0 (at threshold) to ~0.5 (at DET=1.0)
        let predictability_weight = ((predictability - self.det_threshold) / (1.0 - self.det_threshold)).min(0.5);

        let mut probs = vec![0.0f64; size];
        for k in 0..size {
            probs[k] = (1.0 - predictability_weight) * freq_ewma[k]
                + predictability_weight * recurrence_boost[k];
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smoothing: less smoothing when more predictable
        let adaptive_smoothing = self.smoothing * (1.0 - 0.3 * predictability);
        let uniform_val = 1.0 / size as f64;
        for p in &mut probs {
            *p = (1.0 - adaptive_smoothing) * *p + adaptive_smoothing * uniform_val;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    /// Compute a recurrence-based boost distribution for balls.
    ///
    /// Identifies which time points are recurrent with the most recent state,
    /// then counts the ball frequencies in those recurrent draws, producing
    /// a distribution that reflects "what numbers appeared when the system
    /// was in a similar state to now".
    fn compute_recurrence_boost(
        &self,
        draws: &[Draw],
        ball_sums: &[f64],
        size: usize,
    ) -> Vec<f64> {
        let uniform_val = 1.0 / size as f64;
        let mut boost = vec![uniform_val; size];

        let embedded = takens_embedding(ball_sums, self.embedding_dim, self.delay);
        if embedded.len() < 10 {
            return boost;
        }

        let std_dev = series_std(ball_sums);
        let epsilon = self.epsilon_factor * std_dev;
        if epsilon < 1e-15 {
            return boost;
        }

        // Find recurrence neighbors of the most recent embedded state
        let last_idx = embedded.len() - 1;
        let last_state = &embedded[last_idx];

        let mut recurrence_weights = vec![0.0f64; embedded.len()];
        for (i, state) in embedded.iter().enumerate() {
            if i == last_idx {
                continue;
            }
            let dist_sq: f64 = last_state
                .iter()
                .zip(state.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist_sq < epsilon * epsilon {
                // Weight by proximity: closer recurrences get higher weight
                let dist = dist_sq.sqrt();
                recurrence_weights[i] = 1.0 - (dist / epsilon);
            }
        }

        let total_weight: f64 = recurrence_weights.iter().sum();
        if total_weight < 1e-15 {
            return boost;
        }

        // Map embedded indices back to draw indices.
        // embedded[i] corresponds to chronological draw index i,
        // which is draws[draws.len() - 1 - i] (since draws[0] = most recent).
        let n_draws = draws.len();
        let mut weighted_freq = vec![0.0f64; size];

        for (emb_idx, &w) in recurrence_weights.iter().enumerate() {
            if w < 1e-15 {
                continue;
            }
            let draw_idx = n_draws - 1 - emb_idx;
            if draw_idx >= n_draws {
                continue;
            }
            let draw = &draws[draw_idx];
            for &b in &draw.balls {
                weighted_freq[(b - 1) as usize] += w;
            }
        }

        let wf_sum: f64 = weighted_freq.iter().sum();
        if wf_sum > 0.0 {
            for (k, f) in weighted_freq.iter().enumerate() {
                boost[k] = f / wf_sum;
            }
        }

        boost
    }

    fn predict_stars(&self, draws: &[Draw], size: usize) -> Vec<f64> {
        let uniform = vec![1.0 / size as f64; size];

        // Build star sum time series in chronological order
        let star_sums: Vec<f64> = draws
            .iter()
            .rev()
            .map(|d| d.stars.iter().map(|&s| s as f64).sum())
            .collect();

        // Compute DET for smoothing adaptation
        let [_rr, det, _l_max, _entr, _lam] =
            compute_rqa_metrics(&star_sums, self.embedding_dim, self.delay, self.epsilon_factor, self.min_line_length);

        // EWMA frequency for stars
        let mut freq_ewma = vec![1.0 / size as f64; size];
        for draw in draws.iter().rev() {
            for (idx, freq) in freq_ewma.iter_mut().enumerate() {
                let num = (idx + 1) as u8;
                let present = if draw.stars.contains(&num) { 1.0 } else { 0.0 };
                *freq = self.ewma_alpha * present + (1.0 - self.ewma_alpha) * *freq;
            }
        }

        // Normalize
        let sum: f64 = freq_ewma.iter().sum();
        if sum > 0.0 {
            for f in &mut freq_ewma {
                *f /= sum;
            }
        } else {
            return uniform;
        }

        // Adaptive smoothing based on DET: more predictable -> less smoothing
        let base_star_smoothing = 0.40;
        let star_smoothing = base_star_smoothing * (1.0 - 0.25 * det);
        let uniform_val = 1.0 / size as f64;
        for p in &mut freq_ewma {
            *p = (1.0 - star_smoothing) * *p + star_smoothing * uniform_val;
        }

        floor_only(&mut freq_ewma, PROB_FLOOR_STARS);
        freq_ewma
    }
}

// ---------------------------------------------------------------------------
// Lightweight inline RQA computation
// ---------------------------------------------------------------------------

/// Compute RQA metrics: [RR, DET, L_max, ENTR, LAM].
fn compute_rqa_metrics(
    series: &[f64],
    dim: usize,
    delay: usize,
    epsilon_factor: f64,
    min_line_length: usize,
) -> [f64; 5] {
    let embedded = takens_embedding(series, dim, delay);
    if embedded.len() < 10 {
        return [0.0; 5];
    }

    let std_dev = series_std(series);
    let epsilon = epsilon_factor * std_dev;
    if epsilon < 1e-15 {
        return [0.0; 5];
    }

    let n = embedded.len();
    let recurrence = build_recurrence_matrix(&embedded, epsilon);

    // RR: Recurrence Rate
    let total_points = n * n;
    let rr = recurrence
        .iter()
        .map(|row| row.iter().filter(|&&v| v).count())
        .sum::<usize>() as f64
        / total_points as f64;

    // Diagonal lines (exclude main diagonal)
    let diag_lines = extract_diagonal_lines(&recurrence, min_line_length);

    // DET: fraction of recurrence points in diagonal lines
    let diag_points: usize = diag_lines.iter().sum();
    let total_recurrence = recurrence
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter()
                .enumerate()
                .filter(move |(j, v)| **v && *j != i)
        })
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
    let vert_lines = extract_vertical_lines(&recurrence, min_line_length);
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
        .map(|i| (0..dim).map(|d| series[i + d * delay]).collect())
        .collect()
}

/// Build the recurrence matrix: R[i][j] = true if ||x_i - x_j|| < epsilon.
fn build_recurrence_matrix(embedded: &[Vec<f64>], epsilon: f64) -> Vec<Vec<bool>> {
    let n = embedded.len();
    let eps_sq = epsilon * epsilon;
    let mut matrix = vec![vec![false; n]; n];

    for i in 0..n {
        for j in i..n {
            let dist_sq: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
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
fn extract_diagonal_lines(matrix: &[Vec<bool>], min_length: usize) -> Vec<usize> {
    let n = matrix.len();
    let mut line_lengths = Vec::new();

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

    line_lengths
}

/// Shannon entropy of the diagonal line length distribution.
fn diagonal_entropy(line_lengths: &[usize]) -> f64 {
    if line_lengths.is_empty() {
        return 0.0;
    }

    let total = line_lengths.len() as f64;
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

/// Extract vertical line lengths (excluding main diagonal).
fn extract_vertical_lines(matrix: &[Vec<bool>], min_length: usize) -> Vec<usize> {
    let n = matrix.len();
    let mut line_lengths = Vec::new();

    for j in 0..n {
        let mut current_length = 0;
        for i in 0..n {
            if i == j {
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

/// Standard deviation of a series (population std).
fn series_std(series: &[f64]) -> f64 {
    let n = series.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = series.iter().sum::<f64>() / n;
    let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_rqa_predict_balls_valid() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_rqa_predict_stars_valid() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_rqa_predict_few_draws() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Too few draws should return uniform");
        }
    }

    #[test]
    fn test_rqa_predict_empty_draws() {
        let model = RqaPredictabilityModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Empty draws should return uniform");
        }
    }

    #[test]
    fn test_rqa_predict_no_negative() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            for &p in &dist {
                assert!(p >= 0.0, "Negative probability: {} for {:?}", p, pool);
            }
        }
    }

    #[test]
    fn test_rqa_predict_deterministic() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_rqa_predict_stars_deterministic() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Stars);
        let dist2 = model.predict(&draws, Pool::Stars);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_rqa_predict_sparse_strategy() {
        let model = RqaPredictabilityModel::default();
        assert!(matches!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        ));
    }

    #[test]
    fn test_rqa_predict_calibration_stride() {
        let model = RqaPredictabilityModel::default();
        assert_eq!(model.calibration_stride(), 2);
    }

    #[test]
    fn test_rqa_predict_large_draws() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_rqa_predict_balls_not_uniform() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        let all_uniform = dist.iter().all(|&p| (p - expected).abs() < 1e-6);
        assert!(
            !all_uniform || draws.len() < model.min_draws,
            "Should have some signal"
        );
    }

    #[test]
    fn test_rqa_predict_stars_not_uniform() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        let all_uniform = dist.iter().all(|&p| (p - expected).abs() < 1e-6);
        assert!(
            !all_uniform || draws.len() < model.min_draws,
            "Should have some signal"
        );
    }

    // --- Inline RQA helper tests ---

    #[test]
    fn test_takens_embedding_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedded = takens_embedding(&series, 3, 1);
        assert_eq!(embedded.len(), 3);
        assert_eq!(embedded[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(embedded[1], vec![2.0, 3.0, 4.0]);
        assert_eq!(embedded[2], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_takens_embedding_with_delay() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let embedded = takens_embedding(&series, 2, 3);
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
    fn test_recurrence_matrix_identical_points() {
        let embedded = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![5.0, 5.0]];
        let rm = build_recurrence_matrix(&embedded, 0.5);
        assert!(rm[0][0]);
        assert!(rm[0][1]);
        assert!(rm[1][0]);
        assert!(!rm[0][2]);
        assert!(!rm[2][0]);
    }

    #[test]
    fn test_diagonal_lines_extraction() {
        let n = 5;
        let mut matrix = vec![vec![false; n]; n];
        for i in 0..n {
            matrix[i][i] = true;
        }
        // Diagonal at k=1: (0,1), (1,2), (2,3)
        matrix[0][1] = true;
        matrix[1][0] = true;
        matrix[1][2] = true;
        matrix[2][1] = true;
        matrix[2][3] = true;
        matrix[3][2] = true;

        let lines = extract_diagonal_lines(&matrix, 2);
        assert!(
            lines.contains(&3),
            "Expected a diagonal line of length 3, got {:?}",
            lines
        );
    }

    #[test]
    fn test_vertical_lines_extraction() {
        let n = 5;
        let mut matrix = vec![vec![false; n]; n];
        matrix[0][3] = true;
        matrix[1][3] = true;
        matrix[2][3] = true;

        let lines = extract_vertical_lines(&matrix, 2);
        assert!(
            lines.contains(&3),
            "Expected a vertical line of length 3, got {:?}",
            lines
        );
    }

    #[test]
    fn test_diagonal_entropy_uniform_lengths() {
        let lines = vec![3, 3, 3, 3];
        let e = diagonal_entropy(&lines);
        assert!(e.abs() < 1e-10, "Entropy should be 0 for uniform lengths");
    }

    #[test]
    fn test_diagonal_entropy_varied_lengths() {
        let lines = vec![2, 3, 4, 5, 2, 3];
        let e = diagonal_entropy(&lines);
        assert!(e > 0.0, "Entropy should be positive for varied lengths");
    }

    #[test]
    fn test_diagonal_entropy_empty() {
        let lines: Vec<usize> = vec![];
        let e = diagonal_entropy(&lines);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn test_series_std_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = series_std(&series);
        assert!((s - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_series_std_constant() {
        let series = vec![5.0; 50];
        let s = series_std(&series);
        assert!(s.abs() < 1e-15, "Constant series should have std=0");
    }

    #[test]
    fn test_series_std_short() {
        let series = vec![1.0];
        let s = series_std(&series);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_rqa_metrics_constant_series() {
        let series = vec![127.5; 100];
        let metrics = compute_rqa_metrics(&series, 3, 1, 0.2, 2);
        assert_eq!(metrics, [0.0; 5], "Constant series should yield all-zero metrics");
    }

    #[test]
    fn test_rqa_metrics_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let metrics = compute_rqa_metrics(&series, 3, 1, 0.2, 2);
        // Only 1 embedded point, below threshold of 10
        assert_eq!(metrics, [0.0; 5]);
    }

    #[test]
    fn test_rqa_metrics_periodic_series() {
        // Periodic series should have high DET
        let mut series = Vec::with_capacity(200);
        for i in 0..200 {
            series.push((i % 5) as f64 * 10.0 + 100.0);
        }
        let metrics = compute_rqa_metrics(&series, 3, 1, 0.2, 2);
        let rr = metrics[0];
        let det = metrics[1];
        // Periodic data should show recurrence
        assert!(rr > 0.0, "Periodic series should have positive RR: {}", rr);
        assert!(det >= 0.0, "DET should be non-negative: {}", det);
    }

    #[test]
    fn test_compute_recurrence_boost_valid() {
        let model = RqaPredictabilityModel::default();
        let draws = make_test_draws(100);
        let ball_sums: Vec<f64> = draws
            .iter()
            .rev()
            .map(|d| d.balls.iter().map(|&b| b as f64).sum())
            .collect();
        let boost = model.compute_recurrence_boost(&draws, &ball_sums, 50);
        assert_eq!(boost.len(), 50);
        let sum: f64 = boost.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Recurrence boost should sum to 1.0, got {}",
            sum
        );
        for &b in &boost {
            assert!(b >= 0.0, "Negative boost: {}", b);
        }
    }

    #[test]
    fn test_params_complete() {
        let model = RqaPredictabilityModel::default();
        let params = model.params();
        assert_eq!(params.len(), 8);
        assert!(params.contains_key("smoothing"));
        assert!(params.contains_key("min_draws"));
        assert!(params.contains_key("embedding_dim"));
        assert!(params.contains_key("delay"));
        assert!(params.contains_key("epsilon_factor"));
        assert!(params.contains_key("min_line_length"));
        assert!(params.contains_key("ewma_alpha"));
        assert!(params.contains_key("det_threshold"));
    }

    #[test]
    fn test_name() {
        let model = RqaPredictabilityModel::default();
        assert_eq!(model.name(), "RqaPredictability");
    }
}
