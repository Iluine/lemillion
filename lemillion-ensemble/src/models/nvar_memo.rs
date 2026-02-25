use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// NVAR-Memo — Mémorisation via Random Fourier Features.
///
/// Crée un espace de features surcomplèt (dim > N échantillons)
/// via Random Fourier Features (Rahimi & Recht, 2007) approximant
/// un noyau RBF infini-dimensionnel.
///
/// Avec lambda → 0, interpole parfaitement toutes les données d'entraînement.
/// La question : est-ce que cette mémorisation extrapole ?
pub struct NvarMemoModel {
    n_features: usize,
    bandwidth: f64,
    ridge_lambda: f64,
    delay: usize,
    smoothing: f64,
    seed: u64,
}

impl NvarMemoModel {
    pub fn new(
        n_features: usize,
        bandwidth: f64,
        ridge_lambda: f64,
        delay: usize,
        smoothing: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_features,
            bandwidth,
            ridge_lambda,
            delay,
            smoothing,
            seed,
        }
    }
}

impl Default for NvarMemoModel {
    fn default() -> Self {
        Self {
            n_features: 200,
            bandwidth: 1.0,
            ridge_lambda: 1e-6,
            delay: 3,
            smoothing: 0.5,
            seed: 42,
        }
    }
}

/// Encode un tirage en vecteur compact de statistiques.
fn encode_draw(numbers: &[u8], pool_size: usize) -> Vec<f64> {
    let n = numbers.len() as f64;
    let max_val = pool_size as f64;

    let sum: f64 = numbers.iter().map(|&x| x as f64).sum();
    let mean = sum / n;

    let variance = numbers.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;

    let min_v = numbers.iter().copied().min().unwrap_or(1) as f64;
    let max_v = numbers.iter().copied().max().unwrap_or(1) as f64;

    let odd_count = numbers.iter().filter(|&&x| x % 2 == 1).count() as f64;

    vec![
        sum / (n * max_val),
        (max_v - min_v) / max_val,
        odd_count / n,
        mean / max_val,
        variance.sqrt() / max_val,
    ]
}

/// Construit le delay embedding : concatène les encodages des `delay` derniers tirages.
fn build_delay_vector(encoded: &[Vec<f64>], t: usize, delay: usize) -> Vec<f64> {
    let mut vec = Vec::new();
    for d in 0..delay {
        if t >= d {
            vec.extend_from_slice(&encoded[t - d]);
        } else {
            vec.extend(std::iter::repeat(0.0).take(encoded[0].len()));
        }
    }
    vec
}

/// Applique la transformation Random Fourier Features.
/// Φ(x) = sqrt(2/D) * cos(Wx + b)
/// où W ~ N(0, 1/bandwidth²), b ~ Uniform(0, 2π)
struct RffTransform {
    w: Array2<f64>,
    b: Array1<f64>,
    scale: f64,
}

impl RffTransform {
    fn new(input_dim: usize, n_features: usize, bandwidth: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let sigma = 1.0 / bandwidth;

        let w = Array2::from_shape_fn((n_features, input_dim), |_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.random::<f64>();
            let u2: f64 = rng.random::<f64>();
            let u1_safe: f64 = u1.max(1e-15);
            sigma * (-2.0f64 * u1_safe.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });

        let b = Array1::from_shape_fn(n_features, |_| {
            rng.random::<f64>() * 2.0 * std::f64::consts::PI
        });

        let scale = (2.0 / n_features as f64).sqrt();

        Self { w, b, scale }
    }

    fn transform(&self, x: &Array1<f64>) -> Array1<f64> {
        let z = self.w.dot(x) + &self.b;
        z.mapv(|v| self.scale * v.cos())
    }
}

impl ForecastModel for NvarMemoModel {
    fn name(&self) -> &str {
        "NVAR-Memo"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        let min_draws = self.delay + 3;
        if draws.len() < min_draws {
            return uniform;
        }

        // Ordre chronologique
        let chronological: Vec<&Draw> = draws.iter().rev().collect();
        let n = chronological.len();

        // Encoder chaque tirage
        let encoded: Vec<Vec<f64>> = chronological
            .iter()
            .map(|d| encode_draw(pool.numbers_from(d), size))
            .collect();

        let input_dim = encoded[0].len() * self.delay;

        // Construire les delay vectors
        let start = self.delay;
        let n_samples = n - start - 1;
        if n_samples < 2 {
            return uniform;
        }

        // Initialiser la transformation RFF
        let rff = RffTransform::new(input_dim, self.n_features, self.bandwidth, self.seed);

        // Construire la matrice de features transformées H [n_features, n_samples]
        let mut h = Array2::<f64>::zeros((self.n_features, n_samples));
        for (s, t) in (start..n - 1).enumerate() {
            let dv = build_delay_vector(&encoded, t, self.delay);
            let input = Array1::from_vec(dv);
            let phi = rff.transform(&input);
            for (f, &val) in phi.iter().enumerate() {
                h[[f, s]] = val;
            }
        }

        // Construire les targets
        let mut targets = Array2::<f64>::zeros((size, n_samples));
        for (s, t) in (start..n - 1).enumerate() {
            for &num in pool.numbers_from(chronological[t + 1]) {
                let idx = (num - 1) as usize;
                if idx < size {
                    targets[[idx, s]] = 1.0;
                }
            }
        }

        // Ridge regression
        let w_out = match lemillion_esn::linalg::ridge_regression(&h, &targets, self.ridge_lambda) {
            Ok(w) => w,
            Err(_) => return uniform,
        };

        // Prédire pour le dernier pas de temps
        let last_dv = build_delay_vector(&encoded, n - 1, self.delay);
        let last_input = Array1::from_vec(last_dv);
        let last_phi = rff.transform(&last_input);
        let prediction = w_out.dot(&last_phi);

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
            ("n_features".into(), self.n_features as f64),
            ("bandwidth".into(), self.bandwidth),
            ("ridge_lambda".into(), self.ridge_lambda),
            ("delay".into(), self.delay as f64),
            ("smoothing".into(), self.smoothing),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_nvar_memo_balls_sums_to_one() {
        let model = NvarMemoModel::default();
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
    fn test_nvar_memo_stars_sums_to_one() {
        let model = NvarMemoModel::default();
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
    fn test_nvar_memo_no_negative() {
        let model = NvarMemoModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_nvar_memo_empty_draws() {
        let model = NvarMemoModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nvar_memo_few_draws() {
        let model = NvarMemoModel::default();
        let draws = make_test_draws(3);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rff_deterministic() {
        let rff = RffTransform::new(10, 100, 1.0, 42);
        let x = Array1::from_vec(vec![0.1; 10]);
        let phi1 = rff.transform(&x);
        let phi2 = rff.transform(&x);
        for (a, b) in phi1.iter().zip(phi2.iter()) {
            assert!((a - b).abs() < 1e-15, "RFF should be deterministic for same input");
        }
    }

    #[test]
    fn test_rff_correct_dimension() {
        let rff = RffTransform::new(10, 200, 1.0, 42);
        let x = Array1::from_vec(vec![0.5; 10]);
        let phi = rff.transform(&x);
        assert_eq!(phi.len(), 200);
    }

    #[test]
    fn test_nvar_memo_deterministic() {
        let model = NvarMemoModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-12, "NVAR-Memo should be deterministic with same seed");
        }
    }
}
