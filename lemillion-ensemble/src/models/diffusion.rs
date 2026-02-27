use std::collections::HashMap;

use ndarray::{Array1, Array2};

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// Diffusion (Denoising Autoencoder multi-échelle) — apprentissage de distribution jointe.
///
/// Au lieu du DDPM complet (nécessite backprop), utilise des autoencodeurs de
/// débruitage linéaires entraînés par ridge regression à plusieurs niveaux de bruit.
///
/// Algorithme :
/// 1. Encoder chaque tirage en vecteur binaire + contexte (5 derniers tirages)
/// 2. Pour chaque niveau de bruit σ_t : bruiter + ridge regression pour débruiter
/// 3. Prédiction itérative : partir de la moyenne + petit bruit, débruiter en 10 pas
pub struct DiffusionModel {
    n_noise_levels: usize,
    sigma_min: f64,
    sigma_max: f64,
    ridge_lambda: f64,
    context_draws: usize,
    n_denoise_steps: usize,
    smoothing: f64,
    seed: u64,
}

impl DiffusionModel {
    pub fn new(
        n_noise_levels: usize,
        sigma_min: f64,
        sigma_max: f64,
        ridge_lambda: f64,
        context_draws: usize,
        n_denoise_steps: usize,
        smoothing: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_noise_levels,
            sigma_min,
            sigma_max,
            ridge_lambda,
            context_draws,
            n_denoise_steps,
            smoothing,
            seed,
        }
    }
}

impl Default for DiffusionModel {
    fn default() -> Self {
        Self {
            n_noise_levels: 5,
            sigma_min: 0.1,
            sigma_max: 2.0,
            ridge_lambda: 1e-3,
            context_draws: 5,
            n_denoise_steps: 10,
            smoothing: 0.5,
            seed: 42,
        }
    }
}

/// Simple PRNG (xorshift64) pour bruit reproductible.
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Gaussian approximé via Box-Muller (utilise 2 uniformes)
    fn next_gaussian(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-15);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Encode un tirage en vecteur binaire de taille pool_size.
fn encode_binary(draw: &Draw, pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let mut binary = vec![0.0; size];
    for &n in pool.numbers_from(draw) {
        let idx = (n - 1) as usize;
        if idx < size {
            binary[idx] = 1.0;
        }
    }
    binary
}

/// Construit le vecteur de contexte : concaténation des `context_draws` tirages précédents.
fn build_context(draws: &[Draw], pool: Pool, context_draws: usize) -> Vec<f64> {
    let size = pool.size();
    let mut ctx = Vec::with_capacity(size * context_draws);
    for i in 0..context_draws {
        if i < draws.len() {
            ctx.extend_from_slice(&encode_binary(&draws[i], pool));
        } else {
            ctx.extend(std::iter::repeat_n(0.0, size));
        }
    }
    ctx
}

/// Génère les niveaux de bruit en progression géométrique.
fn noise_schedule(n_levels: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    if n_levels <= 1 {
        return vec![sigma_min];
    }
    let ratio = (sigma_max / sigma_min).powf(1.0 / (n_levels - 1) as f64);
    (0..n_levels)
        .map(|i| sigma_min * ratio.powi(i as i32))
        .collect()
}

impl ForecastModel for DiffusionModel {
    fn name(&self) -> &str {
        "Diffusion"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        let min_required = self.context_draws + 5;
        if draws.len() < min_required {
            return uniform;
        }

        // Ordre chronologique
        let chronological: Vec<&Draw> = draws.iter().rev().collect();
        let n = chronological.len();

        let sigmas = noise_schedule(self.n_noise_levels, self.sigma_min, self.sigma_max);

        // Dimension de l'input = pool_size (noisy x) + pool_size * context_draws (contexte)
        let context_dim = size * self.context_draws;
        let input_dim = size + context_dim;

        // Construire les données d'entraînement pour chaque niveau de bruit
        // Pour chaque tirage t (avec contexte disponible), on crée :
        //   x_clean = binary encoding du tirage t
        //   context = binary encoding des context_draws tirages avant t
        //   x_noisy = x_clean + sigma * noise

        let n_train = n.saturating_sub(self.context_draws + 1);
        if n_train < 3 {
            return uniform;
        }

        // Limiter pour la performance
        let max_train = 150;
        let stride = if n_train > max_train {
            n_train / max_train
        } else {
            1
        };
        let train_indices: Vec<usize> = (self.context_draws..self.context_draws + n_train)
            .step_by(stride)
            .collect();
        let n_actual = train_indices.len();

        if n_actual < 3 {
            return uniform;
        }

        // Entraîner un débruiteur (matrice W) par niveau de bruit
        let mut denoisers: Vec<Array2<f64>> = Vec::with_capacity(sigmas.len());

        for &sigma in &sigmas {
            let mut rng = Rng64::new(self.seed.wrapping_add((sigma * 1000.0) as u64));

            // H = [input_dim, n_actual] — inputs bruités + contexte
            // Y = [size, n_actual] — targets propres
            let mut h = Array2::<f64>::zeros((input_dim, n_actual));
            let mut y = Array2::<f64>::zeros((size, n_actual));

            for (s, &t) in train_indices.iter().enumerate() {
                // x_clean
                let x_clean = encode_binary(chronological[t], pool);

                // contexte : tirages t-1, t-2, ..., t-context_draws
                let mut ctx = Vec::with_capacity(context_dim);
                for c in 1..=self.context_draws {
                    if t >= c {
                        ctx.extend_from_slice(&encode_binary(chronological[t - c], pool));
                    } else {
                        ctx.extend(std::iter::repeat_n(0.0, size));
                    }
                }

                // Bruiter x_clean
                for i in 0..size {
                    let noise = rng.next_gaussian() * sigma;
                    h[[i, s]] = x_clean[i] + noise;
                }
                // Ajouter le contexte
                for (i, &c) in ctx.iter().enumerate() {
                    h[[size + i, s]] = c;
                }

                // Target
                for (i, &v) in x_clean.iter().enumerate() {
                    y[[i, s]] = v;
                }
            }

            // Ridge regression : W = Y @ H^T @ (H @ H^T + λI)^{-1}
            match lemillion_esn::linalg::ridge_regression(&h, &y, self.ridge_lambda) {
                Ok(w) => denoisers.push(w),
                Err(_) => return uniform,
            }
        }

        if denoisers.len() != sigmas.len() {
            return uniform;
        }

        // Prédiction itérative :
        // Partir de x = moyenne des tirages récents + petit bruit
        let mut rng = Rng64::new(self.seed.wrapping_add(999));

        // Moyenne des derniers tirages comme point de départ
        let recent_count = self.context_draws.min(n);
        let mut x_mean = vec![0.0f64; size];
        for i in 0..recent_count {
            let enc = encode_binary(chronological[n - 1 - i], pool);
            for (k, &v) in enc.iter().enumerate() {
                x_mean[k] += v;
            }
        }
        for v in &mut x_mean {
            *v /= recent_count as f64;
        }

        // Ajouter du bruit initial (au niveau sigma_max)
        let mut x_current: Vec<f64> = x_mean
            .iter()
            .map(|&m| m + rng.next_gaussian() * self.sigma_max * 0.5)
            .collect();

        // Contexte pour la prédiction : les context_draws tirages les plus récents
        let pred_ctx = build_context(draws, pool, self.context_draws);

        // Débruitage itératif du plus bruité au moins bruité
        for step in 0..self.n_denoise_steps {
            // Quel débruiteur utiliser ? Interpoler entre les niveaux
            let progress = step as f64 / self.n_denoise_steps as f64;
            let denoiser_idx =
                ((1.0 - progress) * (denoisers.len() - 1) as f64).round() as usize;
            let denoiser_idx = denoiser_idx.min(denoisers.len() - 1);

            // Construire l'input : [x_current, context]
            let mut input_vec = Vec::with_capacity(input_dim);
            input_vec.extend_from_slice(&x_current);
            input_vec.extend_from_slice(&pred_ctx);

            let input = Array1::from_vec(input_vec);
            let output = denoisers[denoiser_idx].dot(&input);

            // Clamp [0, 1] et mettre à jour
            x_current = output.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
        }

        // Convertir en probabilités
        let mut dist = x_current;

        // Smooth + normalize
        let uniform_val = 1.0 / size as f64;
        for p in &mut dist {
            *p = (1.0 - self.smoothing) * (*p).max(1e-10) + self.smoothing * uniform_val;
        }
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for p in &mut dist {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_noise_levels".into(), self.n_noise_levels as f64),
            ("sigma_min".into(), self.sigma_min),
            ("sigma_max".into(), self.sigma_max),
            ("ridge_lambda".into(), self.ridge_lambda),
            ("context_draws".into(), self.context_draws as f64),
            ("n_denoise_steps".into(), self.n_denoise_steps as f64),
            ("smoothing".into(), self.smoothing),
            ("seed".into(), self.seed as f64),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_diffusion_balls_sums_to_one() {
        let model = DiffusionModel::default();
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
    fn test_diffusion_stars_sums_to_one() {
        let model = DiffusionModel::default();
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
    fn test_diffusion_no_negative() {
        let model = DiffusionModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_diffusion_empty_draws() {
        let model = DiffusionModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_diffusion_few_draws() {
        let model = DiffusionModel::default();
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_diffusion_deterministic() {
        let model = DiffusionModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-12, "Diffusion should be deterministic");
        }
    }

    #[test]
    fn test_noise_schedule_geometric() {
        let sigmas = noise_schedule(5, 0.1, 2.0);
        assert_eq!(sigmas.len(), 5);
        assert!((sigmas[0] - 0.1).abs() < 1e-10);
        assert!((sigmas[4] - 2.0).abs() < 1e-10);
        // Monotonically increasing
        for i in 1..sigmas.len() {
            assert!(sigmas[i] > sigmas[i - 1]);
        }
    }

    #[test]
    fn test_encode_binary() {
        let draw = Draw {
            draw_id: "001".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-01".to_string(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
        };
        let enc = encode_binary(&draw, Pool::Balls);
        assert_eq!(enc.len(), 50);
        assert_eq!(enc[0], 1.0); // ball 1
        assert_eq!(enc[4], 1.0); // ball 5
        assert_eq!(enc[5], 0.0); // ball 6 absent
        let sum: f64 = enc.iter().sum();
        assert_eq!(sum, 5.0);
    }

    #[test]
    fn test_build_context_size() {
        let draws = make_test_draws(10);
        let ctx = build_context(&draws, Pool::Balls, 5);
        assert_eq!(ctx.len(), 50 * 5);
    }

    #[test]
    fn test_noise_schedule_single() {
        let sigmas = noise_schedule(1, 0.5, 2.0);
        assert_eq!(sigmas.len(), 1);
        assert!((sigmas[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_large_draws() {
        let model = DiffusionModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
