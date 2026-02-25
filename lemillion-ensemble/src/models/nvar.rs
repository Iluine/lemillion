use std::collections::HashMap;

use ndarray::{Array1, Array2};

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// NVAR (Nonlinear Vector Autoregression) — "Next Generation Reservoir Computing".
///
/// Basé sur Gauthier et al., Nature Communications 2021.
/// Remplace le réservoir aléatoire de l'ESN par un espace de features polynomiales
/// de variables retardées (delay embedding + cross-products quadratiques).
///
/// Déterministe, interprétable, seulement 3 hyperparamètres.
pub struct NvarModel {
    delay: usize,
    poly_degree: usize,
    ridge_lambda: f64,
    smoothing: f64,
}

impl NvarModel {
    pub fn new(delay: usize, poly_degree: usize, ridge_lambda: f64, smoothing: f64) -> Self {
        Self {
            delay,
            poly_degree,
            ridge_lambda,
            smoothing,
        }
    }
}

impl Default for NvarModel {
    fn default() -> Self {
        Self {
            delay: 5,
            poly_degree: 2,
            ridge_lambda: 1e-4,
            smoothing: 0.6,
        }
    }
}

/// Encode un tirage en vecteur de statistiques résumées (5 features).
/// - somme normalisée des numéros
/// - écart (max - min) normalisé
/// - ratio de numéros impairs
/// - centroïde normalisé
/// - variance normalisée
fn encode_draw(numbers: &[u8], pool_size: usize) -> Vec<f64> {
    let n = numbers.len() as f64;
    let max_val = pool_size as f64;

    let sum: f64 = numbers.iter().map(|&x| x as f64).sum();
    let sum_norm = sum / (n * max_val);

    let min_v = numbers.iter().copied().min().unwrap_or(1) as f64;
    let max_v = numbers.iter().copied().max().unwrap_or(1) as f64;
    let spread_norm = (max_v - min_v) / max_val;

    let odd_ratio = numbers.iter().filter(|&&x| x % 2 == 1).count() as f64 / n;

    let centroid = sum / n / max_val;

    let mean = sum / n;
    let variance = numbers.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let var_norm = variance / (max_val * max_val);

    vec![sum_norm, spread_norm, odd_ratio, centroid, var_norm]
}

/// Construit les features polynomiales à partir du delay embedding.
/// Pour degree=2 : [x_t, x_{t-1}, ..., x_{t-d+1}, x_i*x_j pour tout i<=j]
fn build_polynomial_features(delay_vectors: &[Vec<f64>], degree: usize) -> Vec<f64> {
    // Concaténer tous les delay vectors
    let mut linear: Vec<f64> = Vec::new();
    for v in delay_vectors {
        linear.extend_from_slice(v);
    }

    if degree < 2 {
        return linear;
    }

    // Ajouter les cross-products quadratiques
    let n = linear.len();
    let mut features = linear.clone();
    for i in 0..n {
        for j in i..n {
            features.push(linear[i] * linear[j]);
        }
    }

    features
}

impl ForecastModel for NvarModel {
    fn name(&self) -> &str {
        "NVAR"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Besoin d'au moins delay + 2 tirages
        let min_draws = self.delay + 2;
        if draws.len() < min_draws {
            return uniform;
        }

        // Ordre chronologique (draws[0] = plus récent → inverser)
        let chronological: Vec<&Draw> = draws.iter().rev().collect();
        let n = chronological.len();

        // Encoder chaque tirage
        let encoded: Vec<Vec<f64>> = chronological
            .iter()
            .map(|d| encode_draw(pool.numbers_from(d), size))
            .collect();

        // Construire les paires (features, target) via delay embedding
        let start = self.delay;
        let n_samples = n - start - 1; // -1 car on a besoin du target au step suivant
        if n_samples < 2 {
            return uniform;
        }

        // Construire les features polynomiales pour chaque pas de temps
        let mut all_features: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for t in start..n - 1 {
            let delay_vecs: Vec<Vec<f64>> = (0..self.delay)
                .map(|d| encoded[t - d].clone())
                .collect();
            let features = build_polynomial_features(&delay_vecs, self.poly_degree);
            all_features.push(features);
        }

        let n_features = all_features[0].len();

        // Construire les targets : pour chaque numéro, binaire présence/absence
        // Target : vecteur binaire de taille `size` pour le tirage à t+1
        let mut targets = Array2::<f64>::zeros((size, n_samples));
        for (s, t) in (start..n - 1).enumerate() {
            for &num in pool.numbers_from(chronological[t + 1]) {
                let idx = (num - 1) as usize;
                if idx < size {
                    targets[[idx, s]] = 1.0;
                }
            }
        }

        // Construire la matrice H [n_features, n_samples]
        let mut h = Array2::<f64>::zeros((n_features, n_samples));
        for (s, feat) in all_features.iter().enumerate() {
            for (f, &val) in feat.iter().enumerate() {
                h[[f, s]] = val;
            }
        }

        // Ridge regression : W_out = targets * H^T * (H * H^T + lambda * I)^{-1}
        let w_out = match lemillion_esn::linalg::ridge_regression(&h, &targets, self.ridge_lambda) {
            Ok(w) => w,
            Err(_) => return uniform,
        };

        // Prédire pour le dernier pas de temps
        let last_delay_vecs: Vec<Vec<f64>> = (0..self.delay)
            .map(|d| encoded[n - 1 - d].clone())
            .collect();
        let last_features = build_polynomial_features(&last_delay_vecs, self.poly_degree);

        let input = Array1::from_vec(last_features);
        let prediction = w_out.dot(&input);

        // Convertir en probabilités positives
        let mut raw_probs: Vec<f64> = prediction.iter().map(|&x| x.max(1e-10)).collect();

        // Lisser avec l'uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut raw_probs {
            *p = self.smoothing * *p + (1.0 - self.smoothing) * uniform_val;
        }

        // Normaliser
        let sum: f64 = raw_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut raw_probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        raw_probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("delay".into(), self.delay as f64),
            ("poly_degree".into(), self.poly_degree as f64),
            ("ridge_lambda".into(), self.ridge_lambda),
            ("smoothing".into(), self.smoothing),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_nvar_balls_sums_to_one() {
        let model = NvarModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_nvar_stars_sums_to_one() {
        let model = NvarModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_nvar_no_negative() {
        let model = NvarModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_nvar_empty_draws() {
        let model = NvarModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nvar_few_draws() {
        let model = NvarModel::default();
        let draws = make_test_draws(3);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_encode_draw_basic() {
        let numbers = [1u8, 10, 20, 30, 50];
        let encoded = encode_draw(&numbers, 50);
        assert_eq!(encoded.len(), 5);
        // Toutes les valeurs devraient être dans [0, 1]
        for &v in &encoded {
            assert!(v >= 0.0 && v <= 1.0, "Encoded value out of range: {v}");
        }
    }

    #[test]
    fn test_polynomial_features_degree1() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let features = build_polynomial_features(&vecs, 1);
        assert_eq!(features, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_polynomial_features_degree2() {
        let vecs = vec![vec![1.0, 2.0]];
        let features = build_polynomial_features(&vecs, 2);
        // linear: [1, 2] + quadratic: [1*1, 1*2, 2*2] = [1, 2, 1, 2, 4]
        assert_eq!(features, vec![1.0, 2.0, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_nvar_deterministic() {
        let model = NvarModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "NVAR should be deterministic");
        }
    }
}
