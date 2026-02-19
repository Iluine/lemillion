use std::collections::HashMap;
use ndarray::{Array1, Array2};
use lemillion_db::models::{Draw, Pool};
use super::ForecastModel;
use crate::features;

pub struct LogisticModel {
    learning_rate: f64,
    lambda: f64,
    epochs: usize,
    window: usize,
}

impl LogisticModel {
    pub fn new(learning_rate: f64, lambda: f64, epochs: usize, window: usize) -> Self {
        Self { learning_rate, lambda, epochs, window }
    }
}

impl ForecastModel for LogisticModel {
    fn name(&self) -> &str {
        "Logistic"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        if draws.len() < 3 {
            return vec![1.0 / size as f64; size];
        }

        let n_features = features::FEATURE_NAMES.len();
        let effective_window = self.window.min(draws.len().saturating_sub(1));
        if effective_window < 2 {
            return vec![1.0 / size as f64; size];
        }

        // Collecter les données d'entraînement (fenêtre glissante)
        let mut all_features = Vec::new();
        let mut all_labels = Vec::new();

        for t in 1..effective_window {
            let rows = features::extract_features_for_draw(draws, pool, t);
            for row in rows {
                all_features.push(row.features);
                all_labels.push(row.label);
            }
        }

        if all_features.is_empty() {
            return vec![1.0 / size as f64; size];
        }

        let n_samples = all_features.len();

        // Construire la matrice X et le vecteur y
        let mut x = Array2::<f64>::zeros((n_samples, n_features));
        let y = Array1::from_vec(all_labels);

        for (i, feats) in all_features.iter().enumerate() {
            for (j, &val) in feats.iter().enumerate() {
                x[[i, j]] = val;
            }
        }

        // Normaliser les features (z-score)
        let means = x.mean_axis(ndarray::Axis(0)).unwrap();
        let stds: Array1<f64> = x
            .axis_iter(ndarray::Axis(0))
            .fold(Array1::zeros(n_features), |acc, row| {
                let diff = &row - &means;
                acc + &diff.mapv(|v| v * v)
            }) / n_samples as f64;
        let stds = stds.mapv(|v| v.sqrt().max(1e-10));

        let mut x_norm = x.clone();
        for mut row in x_norm.rows_mut() {
            for j in 0..n_features {
                row[j] = (row[j] - means[j]) / stds[j];
            }
        }

        // SGD avec L2 régularisation
        let mut weights = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0f64;

        for _ in 0..self.epochs {
            let logits = x_norm.dot(&weights) + bias;
            let preds = logits.mapv(sigmoid);
            let errors = &preds - &y;

            // Gradient
            let grad_w = x_norm.t().dot(&errors) / n_samples as f64
                + &weights * (self.lambda * 2.0);
            let grad_b = errors.sum() / n_samples as f64;

            weights = weights - &grad_w * self.learning_rate;
            bias -= grad_b * self.learning_rate;
        }

        // Prédire pour le tirage le plus récent (index 0)
        let current_rows = features::extract_features_for_draw(draws, pool, 0);
        let mut scores: Vec<f64> = current_rows
            .iter()
            .map(|row| {
                let normalized: Vec<f64> = row.features
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| (v - means[j]) / stds[j])
                    .collect();
                let feat = Array1::from_vec(normalized);
                sigmoid(feat.dot(&weights) + bias)
            })
            .collect();

        // Normaliser en distribution (softmax-like via normalisation directe)
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in &mut scores {
                *s /= total;
            }
        } else {
            scores = vec![1.0 / size as f64; size];
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("learning_rate".to_string(), self.learning_rate),
            ("lambda".to_string(), self.lambda),
            ("epochs".to_string(), self.epochs as f64),
            ("window".to_string(), self.window as f64),
        ])
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_logistic_balls_sums_to_one() {
        let model = LogisticModel::new(0.01, 0.001, 50, 50);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_logistic_stars_sums_to_one() {
        let model = LogisticModel::new(0.01, 0.001, 50, 50);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_logistic_no_negative() {
        let model = LogisticModel::new(0.01, 0.001, 50, 50);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_logistic_empty_draws() {
        let model = LogisticModel::new(0.01, 0.001, 50, 50);
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
    }
}
