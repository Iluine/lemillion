use serde::{Deserialize, Serialize};
use lemillion_db::models::{Draw, Pool};
use crate::ensemble::meta::RegimeFeatures;
use crate::models::ForecastModel;

/// Poids appris par stacking per-number ridge regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingWeights {
    /// Poids par numéro par modèle: [pool_size][n_models]
    pub model_weights: Vec<Vec<f64>>,
    /// Poids contextuels par numéro: [pool_size][n_ctx_features]
    pub context_weights: Vec<Vec<f64>>,
    /// Biais par numéro: [pool_size]
    pub bias: Vec<f64>,
    pub model_names: Vec<String>,
    pub pool_size: usize,
}

/// Point de données pour le stacking.
pub struct StackingDataPoint {
    /// Distributions de chaque modèle: [n_models][pool_size]
    pub model_distributions: Vec<Vec<f64>>,
    /// Numéros réellement tirés
    pub actual_numbers: Vec<u8>,
    /// Features de contexte
    pub context: RegimeFeatures,
}

/// Collecte les données de stacking par walk-forward.
/// Parcourt ~200 points test avec stride, pour chaque point collecte
/// les distributions de tous les modèles + le tirage réel.
pub fn collect_stacking_data(
    models: &[Box<dyn ForecastModel>],
    draws: &[Draw],
    pool: Pool,
    window: usize,
) -> Vec<StackingDataPoint> {
    let max_t = draws.len().saturating_sub(window + 1);
    if max_t == 0 {
        return vec![];
    }

    let target_points = 200;
    let stride = (max_t / target_points).max(1);

    let test_indices: Vec<usize> = (0..max_t).step_by(stride).collect();

    test_indices
        .iter()
        .filter_map(|&t| {
            let train_end = (t + 1 + window).min(draws.len());
            let train_data = &draws[t + 1..train_end];
            if train_data.len() < 10 {
                return None;
            }

            let model_distributions: Vec<Vec<f64>> = models
                .iter()
                .map(|model| model.predict(train_data, pool))
                .collect();

            let test_draw = &draws[t];
            let actual_numbers = pool.numbers_from(test_draw).to_vec();
            let context = RegimeFeatures::from_draws(&draws[t + 1..]);

            Some(StackingDataPoint {
                model_distributions,
                actual_numbers,
                context,
            })
        })
        .collect()
}

/// Entraîne les poids de stacking par ridge regression per-number.
///
/// Pour chaque numéro k: features = [P_m1(k), ..., P_mN(k), ctx_1..4],
/// target = 1.0 si k tiré, 0.0 sinon.
pub fn train_stacking(
    data: &[StackingDataPoint],
    pool: Pool,
    lambda: f64,
) -> Option<StackingWeights> {
    if data.len() < 20 {
        return None;
    }

    let pool_size = pool.size();
    let n_models = data[0].model_distributions.len();
    let n_ctx = data[0].context.as_vec().len();
    let n_features = n_models + n_ctx; // model probs + context features
    let n_samples = data.len();

    let model_names: Vec<String> = (0..n_models).map(|i| format!("model_{}", i)).collect();

    let mut all_model_weights = Vec::with_capacity(pool_size);
    let mut all_context_weights = Vec::with_capacity(pool_size);
    let mut all_bias = Vec::with_capacity(pool_size);

    for k in 0..pool_size {
        // Build X matrix and y vector for number k
        let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        let mut y: Vec<f64> = Vec::with_capacity(n_samples);

        for dp in data {
            let mut row = Vec::with_capacity(n_features);
            // Model predictions for number k
            for dist in &dp.model_distributions {
                row.push(dist[k]);
            }
            // Context features
            let ctx = dp.context.as_vec();
            row.extend_from_slice(&ctx);

            x_rows.push(row);

            let number = (k + 1) as u8;
            let label = if dp.actual_numbers.contains(&number) { 1.0 } else { 0.0 };
            y.push(label);
        }

        // Elastic net (LASSO + small L2) for automatic model elimination
        let (intercept, beta) = elastic_net_cd(&x_rows, &y, n_features, lambda, lambda * 0.1, 200);

        let mut mw = Vec::with_capacity(n_models);
        for b in &beta[..n_models] {
            mw.push(*b);
        }
        let cw: Vec<f64> = beta[n_models..].to_vec();

        all_model_weights.push(mw);
        all_context_weights.push(cw);
        all_bias.push(intercept);
    }

    Some(StackingWeights {
        model_weights: all_model_weights,
        context_weights: all_context_weights,
        bias: all_bias,
        model_names,
        pool_size,
    })
}

/// Prédit avec les poids de stacking.
/// Retourne une distribution normalisée sur le pool.
pub fn predict_stacked(
    weights: &StackingWeights,
    model_distributions: &[Vec<f64>],
    context: &RegimeFeatures,
) -> Vec<f64> {
    let ctx = context.as_vec();
    let mut probs = Vec::with_capacity(weights.pool_size);

    for k in 0..weights.pool_size {
        let mut val = weights.bias[k];
        for (m, w) in weights.model_weights[k].iter().enumerate() {
            if m < model_distributions.len() {
                val += w * model_distributions[m][k];
            }
        }
        let n_cw = weights.context_weights[k].len().min(ctx.len());
        for i in 0..n_cw {
            val += weights.context_weights[k][i] * ctx[i];
        }
        // Clamp to positive
        probs.push(val.max(1e-10));
    }

    // Normalize
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    probs
}

/// Ridge regression N-dimensionnelle avec centrage (kept for fallback/tests).
/// Retourne (intercept, coefficients).
#[allow(dead_code)]
fn ridge_regression_nd(
    x_rows: &[Vec<f64>],
    y: &[f64],
    n_features: usize,
    lambda: f64,
) -> (f64, Vec<f64>) {
    let n = x_rows.len();
    if n == 0 || n_features == 0 {
        return (0.0, vec![0.0; n_features]);
    }

    // Center y
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    // Center x
    let mut mean_x = vec![0.0f64; n_features];
    for row in x_rows {
        for (j, &v) in row.iter().enumerate() {
            mean_x[j] += v;
        }
    }
    for m in &mut mean_x {
        *m /= n as f64;
    }

    // X^T X + lambda*I
    let mut xtx = vec![vec![0.0f64; n_features]; n_features];
    let mut xty = vec![0.0f64; n_features];

    for (row, &target) in x_rows.iter().zip(y.iter()) {
        let yc = target - mean_y;
        for j in 0..n_features {
            let xj = row[j] - mean_x[j];
            xty[j] += xj * yc;
            for k in 0..n_features {
                let xk = row[k] - mean_x[k];
                xtx[j][k] += xj * xk;
            }
        }
    }

    // Add regularization
    for j in 0..n_features {
        xtx[j][j] += lambda * n as f64;
    }

    // Solve by Gauss-Jordan
    let beta = solve_nxn(&mut xtx, &mut xty, n_features);

    // Intercept = mean_y - beta . mean_x
    let mut intercept = mean_y;
    for j in 0..n_features {
        intercept -= beta[j] * mean_x[j];
    }

    (intercept, beta)
}

/// Soft thresholding operator for LASSO.
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda { x - lambda }
    else if x < -lambda { x + lambda }
    else { 0.0 }
}

/// Elastic net coordinate descent: L1 (lambda1) + L2 (lambda2) regularization.
/// L1 drives irrelevant model weights to exactly zero (automatic elimination).
/// L2 provides numerical stability for correlated features.
fn elastic_net_cd(
    x_rows: &[Vec<f64>],
    y: &[f64],
    n_features: usize,
    lambda1: f64,
    lambda2: f64,
    max_iter: usize,
) -> (f64, Vec<f64>) {
    let n = x_rows.len();
    if n == 0 || n_features == 0 {
        return (0.0, vec![0.0; n_features]);
    }

    // Center y
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - mean_y).collect();

    // Center x and compute means
    let mut mean_x = vec![0.0f64; n_features];
    for row in x_rows {
        for (j, &v) in row.iter().enumerate() {
            mean_x[j] += v;
        }
    }
    for m in &mut mean_x {
        *m /= n as f64;
    }

    // Precompute X^T X diagonal (for coordinate descent denominator)
    let mut xx_diag = vec![0.0f64; n_features];
    for row in x_rows {
        for j in 0..n_features {
            let xc = row[j] - mean_x[j];
            xx_diag[j] += xc * xc;
        }
    }

    let mut beta = vec![0.0f64; n_features];
    let mut residuals = y_centered.clone();

    let l1 = lambda1 * n as f64;
    let l2 = lambda2 * n as f64;

    for _iter in 0..max_iter {
        let mut max_change = 0.0f64;

        for j in 0..n_features {
            let denom = xx_diag[j] + l2;
            if denom < 1e-15 { continue; }

            let old_beta = beta[j];

            // Compute rho = X_j^T × (residuals + old_beta × X_j)
            let mut rho = 0.0f64;
            for (i, row) in x_rows.iter().enumerate() {
                let xc = row[j] - mean_x[j];
                rho += xc * (residuals[i] + old_beta * xc);
            }

            // Soft thresholding with elastic net
            let new_beta = soft_threshold(rho, l1) / denom;

            // Update residuals
            let delta = new_beta - old_beta;
            if delta.abs() > 1e-15 {
                for (i, row) in x_rows.iter().enumerate() {
                    residuals[i] -= delta * (row[j] - mean_x[j]);
                }
            }

            beta[j] = new_beta;
            max_change = max_change.max(delta.abs());
        }

        if max_change < 1e-7 { break; }
    }

    // Intercept = mean_y - beta · mean_x
    let mut intercept = mean_y;
    for j in 0..n_features {
        intercept -= beta[j] * mean_x[j];
    }

    (intercept, beta)
}

/// Gauss-Jordan elimination with partial pivoting for NxN system.
#[allow(dead_code)]
fn solve_nxn(a: &mut [Vec<f64>], b: &mut [f64], n: usize) -> Vec<f64> {
    // Forward elimination
    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        if a[col][col].abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for col in (0..n).rev() {
        if a[col][col].abs() < 1e-15 {
            continue;
        }
        x[col] = b[col];
        for k in (col + 1)..n {
            x[col] -= a[col][k] * x[k];
        }
        x[col] /= a[col][col];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_predict_stacked_normalizes() {
        let weights = StackingWeights {
            model_weights: vec![vec![0.5, 0.5]; 50],
            context_weights: vec![vec![0.1; 12]; 50],
            bias: vec![0.02; 50],
            model_names: vec!["A".into(), "B".into()],
            pool_size: 50,
        };
        let dists = vec![
            vec![1.0 / 50.0; 50],
            vec![1.0 / 50.0; 50],
        ];
        let ctx = RegimeFeatures {
            sum_norm: 0.5, spread_norm: 0.5, mod4_cosine: 0.5,
            recent_entropy: 0.5, day_of_week: 0.5, gap_compression: 1.0,
            hurst_exponent: 0.5, perm_entropy_ratio: 1.0, runs_ratio: 1.0,
            ami_ratio: 1.0, lz_ratio: 1.0, signal_fraction: 0.0,
        };
        let result = predict_stacked(&weights, &dists, &ctx);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Stacked prediction should sum to 1, got {}", sum);
    }

    #[test]
    fn test_predict_stacked_backward_compat() {
        // Old 7-element context weights should still work (min of ctx/cw lengths used)
        let weights = StackingWeights {
            model_weights: vec![vec![0.5, 0.5]; 50],
            context_weights: vec![vec![0.1; 7]; 50],
            bias: vec![0.02; 50],
            model_names: vec!["A".into(), "B".into()],
            pool_size: 50,
        };
        let dists = vec![
            vec![1.0 / 50.0; 50],
            vec![1.0 / 50.0; 50],
        ];
        let ctx = RegimeFeatures {
            sum_norm: 0.5, spread_norm: 0.5, mod4_cosine: 0.5,
            recent_entropy: 0.5, day_of_week: 0.5, gap_compression: 1.0,
            hurst_exponent: 0.5, perm_entropy_ratio: 1.0, runs_ratio: 1.0,
            ami_ratio: 1.0, lz_ratio: 1.0, signal_fraction: 0.0,
        };
        let result = predict_stacked(&weights, &dists, &ctx);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Should still normalize with old 7D weights, got {}", sum);
    }

    #[test]
    fn test_collect_stacking_data_returns_data() {
        let draws = make_test_draws(100);
        let models = crate::models::all_models();
        let data = collect_stacking_data(&models, &draws, Pool::Balls, 30);
        assert!(!data.is_empty(), "Should collect stacking data");
        assert_eq!(data[0].model_distributions.len(), models.len());
        assert_eq!(data[0].model_distributions[0].len(), 50);
    }

    #[test]
    fn test_train_stacking_with_enough_data() {
        let draws = make_test_draws(100);
        let models = crate::models::all_models();
        let data = collect_stacking_data(&models, &draws, Pool::Balls, 30);
        let result = train_stacking(&data, Pool::Balls, 0.01);
        assert!(result.is_some(), "Should train stacking with sufficient data");
        let w = result.unwrap();
        assert_eq!(w.pool_size, 50);
        assert_eq!(w.model_weights.len(), 50);
    }

    #[test]
    fn test_train_stacking_too_few_data() {
        let data: Vec<StackingDataPoint> = vec![];
        let result = train_stacking(&data, Pool::Balls, 0.01);
        assert!(result.is_none());
    }

    #[test]
    fn test_ridge_regression_nd_constant() {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 / 30.0, 0.5]).collect();
        let y = vec![1.0; 30];
        let (intercept, coeff) = ridge_regression_nd(&x, &y, 2, 1.0);
        assert!((intercept - 1.0).abs() < 0.5, "Intercept ~1.0, got {}", intercept);
        for &c in &coeff {
            assert!(c.abs() < 2.0, "Coeffs should be small, got {}", c);
        }
    }
}
