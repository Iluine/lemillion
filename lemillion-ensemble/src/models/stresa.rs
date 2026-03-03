use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

// ─────────────────────────────────────────────────────────────────────────────
// Paramètres structurels de la machine Stresa (9 params boules, 12 étoiles)
// ─────────────────────────────────────────────────────────────────────────────

/// Paramètres structurels pour les boules (7 params).
///
/// Score(k) = row_bias[(k-1)/10] * persistence_factor(decade)
/// blade_bias supprimé (redondant avec mod4_transition_predict_v2).
/// persistence basée sur decade-overlap (orthogonal à mod4).
#[derive(Debug, Clone)]
struct StresaStructuralParams {
    /// Biais de rangée dans le rack (5 rangées = décades 1-10, ..., 41-50).
    row_bias: [f64; 5],

    /// Corrélation inter-tirages (0 = indépendant, >0 = persistance).
    persistence: f64,

    /// Température (0 → déterministe, ∞ → uniforme).
    temperature: f64,
}

impl StresaStructuralParams {
    /// Prior uniforme : tous les facteurs à 1.0.
    fn uniform() -> Self {
        Self {
            row_bias: [1.0; 5],
            persistence: 0.05,
            temperature: 1.0,
        }
    }

    /// Score brut pour la boule k (1-indexed).
    fn ball_score(&self, ball: u8, prev_draw: Option<&[u8; 5]>) -> f64 {
        let k = (ball - 1) as usize;
        let row = k / 10;

        let mut score = self.row_bias[row];

        // Persistence basée sur decade (row-overlap), orthogonal à mod4
        if let Some(prev) = prev_draw {
            let row_overlap = prev.iter()
                .filter(|&&b| ((b - 1) as usize / 10) == row)
                .count();
            score *= 1.0 + self.persistence * row_overlap as f64 / 5.0;
        }

        // Température
        if self.temperature != 1.0 && self.temperature > 0.0 {
            score = score.powf(1.0 / self.temperature);
        }

        score.max(1e-15)
    }

    /// Distribution complète sur les 50 boules, normalisée.
    fn ball_distribution(&self, prev_draw: Option<&[u8; 5]>) -> Vec<f64> {
        let mut probs: Vec<f64> = (1..=50)
            .map(|b| self.ball_score(b, prev_draw))
            .collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
        probs
    }

    /// Log-vraisemblance des boules d'un tirage observé.
    fn log_likelihood_balls(&self, draw: &Draw, prev_draw: Option<&Draw>) -> f64 {
        let dist = self.ball_distribution(prev_draw.map(|d| &d.balls));
        draw.balls.iter()
            .map(|&b| dist[(b - 1) as usize].max(1e-15).ln())
            .sum()
    }

    /// Gradient analytique de la LL par rapport aux paramètres structurels.
    ///
    /// LL = Σ_{b observé} ln(p_b), p_b = s_b / Z, Z = Σ_k s_k
    ///    = Σ_{b observé} ln(s_b) - 5 * ln(Z)
    ///
    /// ∂LL/∂θ = Σ_{b observé} (∂s_b/∂θ)/s_b - 5 * Σ_k p_k * (∂s_k/∂θ)/s_k
    ///
    /// Avec s_k = raw_k^(1/T), on a (∂s_k/∂θ)/s_k = (1/T) * (∂raw_k/∂θ)/raw_k
    fn analytical_gradient_balls(&self, draw: &Draw, prev_draw: Option<&Draw>) -> StresaGradient {
        let prev_balls = prev_draw.map(|d| &d.balls);
        let scores: Vec<f64> = (1..=50u8)
            .map(|b| self.ball_score(b, prev_balls))
            .collect();
        let total_score: f64 = scores.iter().sum();
        let probs: Vec<f64> = scores.iter().map(|&s| s / total_score).collect();

        let pick_count = 5.0;
        let t_eff = if self.temperature != 1.0 && self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };

        let mut grad = StresaGradient::zero();

        for k in 0..50usize {
            let ball = (k + 1) as u8;
            let row = k / 10;
            let p_k = probs[k];
            let is_observed = draw.balls.contains(&ball);

            // Facteur commun : (1(obs) - 5*p_k)
            let common = if is_observed { 1.0 } else { 0.0 } - pick_count * p_k;

            // (∂s_k/∂row[row])/s_k = 1/(T_eff * row[row])
            grad.row[row] += common / (t_eff * self.row_bias[row]);

            // persistence (decade-based)
            if let Some(prev) = prev_balls {
                let overlap = prev.iter()
                    .filter(|&&b| ((b - 1) as usize / 10) == row)
                    .count() as f64;
                let pers_factor = 1.0 + self.persistence * overlap / 5.0;
                grad.persistence += common * (overlap / 5.0) / (t_eff * pers_factor);
            }

            // temperature : (∂s_k/∂T)/s_k = (-1/T²) * ln(raw_k)
            if self.temperature != 1.0 && self.temperature > 0.0 {
                let base = self.row_bias[row];
                let pers_factor = if let Some(prev) = prev_balls {
                    let ov = prev.iter()
                        .filter(|&&b| ((b - 1) as usize / 10) == row)
                        .count() as f64;
                    1.0 + self.persistence * ov / 5.0
                } else {
                    1.0
                };
                let raw = (base * pers_factor).max(1e-15);
                grad.temperature += common * (-1.0 / (self.temperature * self.temperature)) * raw.ln();
            }
        }

        grad
    }

    /// Ajoute du bruit gaussien (jittering pour SMC).
    fn jitter(&mut self, noise: f64, rng_state: &mut u64) {
        for v in &mut self.row_bias {
            *v = (*v + noise * pseudo_normal(rng_state)).max(0.5);
        }
        self.persistence = (self.persistence + noise * 0.1 * pseudo_normal(rng_state)).clamp(0.0, 0.5);
        self.temperature = (self.temperature + noise * 0.1 * pseudo_normal(rng_state)).clamp(0.3, 3.0);
    }
}

/// Gradient structurel pour les boules.
#[derive(Debug, Clone)]
struct StresaGradient {
    row: [f64; 5],
    persistence: f64,
    temperature: f64,
}

impl StresaGradient {
    fn zero() -> Self {
        Self {
            row: [0.0; 5],
            persistence: 0.0,
            temperature: 0.0,
        }
    }
}

/// Paramètres pour les étoiles (12 affinités — ratio raisonnable).
#[derive(Debug, Clone)]
struct StarParams {
    affinity: [f64; 12],
    temperature: f64,
}

#[allow(dead_code)]
impl StarParams {
    fn uniform() -> Self {
        Self {
            affinity: [1.0; 12],
            temperature: 1.0,
        }
    }

    fn star_probability(&self, star: u8) -> f64 {
        let k = (star - 1) as usize;
        let score = self.affinity[k];
        if self.temperature != 1.0 && self.temperature > 0.0 {
            score.powf(1.0 / self.temperature).max(1e-15)
        } else {
            score.max(1e-15)
        }
    }

    fn star_distribution(&self) -> Vec<f64> {
        let mut probs: Vec<f64> = (1..=12)
            .map(|s| self.star_probability(s))
            .collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
        probs
    }

    fn log_likelihood_stars(&self, draw: &Draw) -> f64 {
        let dist = self.star_distribution();
        draw.stars.iter()
            .map(|&s| dist[(s - 1) as usize].max(1e-15).ln())
            .sum()
    }

    fn jitter(&mut self, noise: f64, rng_state: &mut u64) {
        for v in &mut self.affinity {
            *v = (*v + noise * 0.5 * pseudo_normal(rng_state)).max(0.5);
        }
        self.temperature = (self.temperature + noise * 0.1 * pseudo_normal(rng_state)).clamp(0.3, 3.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG léger (xorshift64) — pas de dépendance externe
// ─────────────────────────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn pseudo_uniform(state: &mut u64) -> f64 {
    (xorshift64(state) as f64) / (u64::MAX as f64)
}

/// Box-Muller pour générer un pseudo-normal.
fn pseudo_normal(state: &mut u64) -> f64 {
    let u1 = pseudo_uniform(state).max(1e-15);
    let u2 = pseudo_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Modèle 1 : StresaSGD (Maximum a Posteriori par gradient ascent)
// ─────────────────────────────────────────────────────────────────────────────

/// StresaSGD — simulateur physique Bayésien avec optimisation SGD.
///
/// v3 : 7 paramètres structurels (row_bias[5], persistence, temperature).
/// blade_bias supprimé (redondant avec mod4_transition_v2).
/// Gradients analytiques. Multi-epoch training.
pub struct StresaSgdModel {
    learning_rate: f64,
    regularization: f64,
    smoothing: f64,
    star_smoothing: f64,
    n_epochs: usize,
    mod4_weight: f64,
    ewma_weight: f64,
    spatial_sigma: f64,
}

impl StresaSgdModel {
    pub fn new(learning_rate: f64, regularization: f64, smoothing: f64, mod4_weight: f64,
               ewma_weight: f64, spatial_sigma: f64) -> Self {
        Self { learning_rate, regularization, smoothing, star_smoothing: 0.18,
               n_epochs: 5, mod4_weight, ewma_weight, spatial_sigma }
    }
}

impl Default for StresaSgdModel {
    fn default() -> Self {
        Self {
            learning_rate: 0.02,
            regularization: 0.01,
            smoothing: 0.30,
            star_smoothing: 0.18,
            n_epochs: 10,
            mod4_weight: 0.85,
            ewma_weight: 0.0,
            spatial_sigma: 0.0,
        }
    }
}

/// Clip le gradient pour éviter la divergence.
fn clip_grad(grad: f64, max_grad: f64) -> f64 {
    grad.clamp(-max_grad, max_grad)
}

/// Adam optimizer state for structural parameters.
struct AdamState {
    m_row: [f64; 5],
    v_row: [f64; 5],
    m_persistence: f64,
    v_persistence: f64,
    m_temperature: f64,
    v_temperature: f64,
    t: usize,
}

impl AdamState {
    fn new() -> Self {
        Self {
            m_row: [0.0; 5],
            v_row: [0.0; 5],
            m_persistence: 0.0,
            v_persistence: 0.0,
            m_temperature: 0.0,
            v_temperature: 0.0,
            t: 0,
        }
    }
}

/// Adam optimizer step for structural ball parameters.
fn adam_step_analytical(
    params: &mut StresaStructuralParams,
    adam: &mut AdamState,
    draw: &Draw,
    prev_draw: Option<&Draw>,
    lr: f64,
    reg: f64,
) {
    let max_grad: f64 = 10.0;
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let epsilon: f64 = 1e-8;

    let grad = params.analytical_gradient_balls(draw, prev_draw);
    adam.t += 1;
    let t = adam.t as f64;
    let bc1 = 1.0 - beta1.powf(t);
    let bc2 = 1.0 - beta2.powf(t);

    // row_bias
    for i in 0..5 {
        let g = clip_grad(grad.row[i], max_grad);
        adam.m_row[i] = beta1 * adam.m_row[i] + (1.0 - beta1) * g;
        adam.v_row[i] = beta2 * adam.v_row[i] + (1.0 - beta2) * g * g;
        let m_hat = adam.m_row[i] / bc1;
        let v_hat = adam.v_row[i] / bc2;
        params.row_bias[i] = (params.row_bias[i]
            + lr * m_hat / (v_hat.sqrt() + epsilon)
            - lr * reg * (params.row_bias[i] - 1.0))
            .clamp(0.5, 3.0);
    }

    // persistence
    {
        let g = clip_grad(grad.persistence, max_grad);
        adam.m_persistence = beta1 * adam.m_persistence + (1.0 - beta1) * g;
        adam.v_persistence = beta2 * adam.v_persistence + (1.0 - beta2) * g * g;
        let m_hat = adam.m_persistence / bc1;
        let v_hat = adam.v_persistence / bc2;
        params.persistence = (params.persistence
            + lr * m_hat / (v_hat.sqrt() + epsilon)
            - lr * reg * params.persistence)
            .clamp(0.0, 0.5);
    }

    // temperature
    {
        let g = clip_grad(grad.temperature, max_grad);
        adam.m_temperature = beta1 * adam.m_temperature + (1.0 - beta1) * g;
        adam.v_temperature = beta2 * adam.v_temperature + (1.0 - beta2) * g * g;
        let m_hat = adam.m_temperature / bc1;
        let v_hat = adam.v_temperature / bc2;
        params.temperature = (params.temperature
            + lr * m_hat / (v_hat.sqrt() + epsilon)
            - lr * reg * (params.temperature - 1.0))
            .clamp(0.3, 3.0);
    }
}

/// Met à jour les paramètres structurels des boules par gradient analytique.
#[allow(dead_code)]
fn sgd_step_analytical(
    params: &mut StresaStructuralParams,
    draw: &Draw,
    prev_draw: Option<&Draw>,
    lr: f64,
    reg: f64,
) {
    let max_grad = 10.0;
    let grad = params.analytical_gradient_balls(draw, prev_draw);

    // row_bias
    for i in 0..5 {
        let g = clip_grad(grad.row[i], max_grad);
        params.row_bias[i] = (params.row_bias[i] + lr * g - lr * reg * (params.row_bias[i] - 1.0)).clamp(0.5, 3.0);
    }

    // persistence
    {
        let g = clip_grad(grad.persistence, max_grad);
        params.persistence = (params.persistence + lr * g - lr * reg * params.persistence).clamp(0.0, 0.5);
    }

    // temperature
    {
        let g = clip_grad(grad.temperature, max_grad);
        params.temperature = (params.temperature + lr * g - lr * reg * (params.temperature - 1.0)).clamp(0.3, 3.0);
    }
}

impl ForecastModel for StresaSgdModel {
    fn name(&self) -> &str {
        "StresaSGD"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 10 {
            return uniform;
        }

        let n = draws.len();
        let base_lr = self.learning_rate;

        match pool {
            Pool::Balls => {
                // 1. SGD structural (Adam optimizer)
                let mut params = StresaStructuralParams::uniform();
                let mut adam = AdamState::new();

                for epoch in 0..self.n_epochs {
                    let epoch_lr = base_lr / (1.0 + epoch as f64 * 0.5);
                    for t in (0..n).rev() {
                        let prev = if t + 1 < n { Some(&draws[t + 1]) } else { None };
                        let lr = epoch_lr / (1.0 + (n - 1 - t) as f64 * 0.001);
                        adam_step_analytical(&mut params, &mut adam, &draws[t], prev,
                                            lr, self.regularization);
                    }
                }

                let structural = params.ball_distribution(Some(&draws[0].balls));

                // 2. Mod4 transition (second-order)
                let mod4 = mod4_transition_predict_v2(draws, pool);

                // 3. EWMA frequency
                let ewma = ewma_ball_predict(draws, 0.3);

                // 4. Blend 3 channels
                let w_struct = (1.0 - self.mod4_weight - self.ewma_weight).max(0.0);
                let mut dist: Vec<f64> = (0..size)
                    .map(|i| w_struct * structural[i] + self.mod4_weight * mod4[i] + self.ewma_weight * ewma[i])
                    .collect();

                // 5. Spatial smoothing
                spatial_smooth(&mut dist, self.spatial_sigma);

                // 6. Uniform smoothing
                let uniform_val = 1.0 / size as f64;
                for p in &mut dist {
                    *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
                }
                normalize(&mut dist);
                dist
            }
            Pool::Stars => {
                if draws.len() < 20 {
                    return uniform;
                }
                // EWMA rapide (alpha=0.08) + mod4 transition étoiles (30/70 blend)
                let ewma = ewma_star_predict_alpha(draws, 0.08, self.star_smoothing);
                let mod4 = mod4_star_transition(draws);
                let mod4_w = 0.30;
                let mut dist: Vec<f64> = (0..size)
                    .map(|i| (1.0 - mod4_w) * ewma[i] + mod4_w * mod4[i])
                    .collect();
                normalize(&mut dist);
                dist
            }
        }
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("learning_rate".into(), self.learning_rate),
            ("regularization".into(), self.regularization),
            ("smoothing".into(), self.smoothing),
            ("star_smoothing".into(), self.star_smoothing),
            ("n_epochs".into(), self.n_epochs as f64),
            ("mod4_weight".into(), self.mod4_weight),
            ("ewma_weight".into(), self.ewma_weight),
            ("spatial_sigma".into(), self.spatial_sigma),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Modèle 2 : StresaSMC (Sequential Monte Carlo / Filtre particulaire)
// ─────────────────────────────────────────────────────────────────────────────

/// StresaSMC — simulateur physique Bayésien avec filtre particulaire.
///
/// v3 : particules sur StresaStructuralParams (7D) au lieu de 9D.
/// blade_bias supprimé (redondant avec mod4_transition_v2).
/// 500 particules en 7D = couverture correcte.
/// Jitter adaptatif décroissant avec sqrt(n_observations).
pub struct StresaSmcModel {
    n_particles: usize,
    jitter_noise: f64,
    smoothing: f64,
    star_smoothing: f64,
    seed: u64,
    mod4_weight: f64,
    ewma_weight: f64,
    spatial_sigma: f64,
    warmup: usize,
}

impl StresaSmcModel {
    pub fn new(n_particles: usize, jitter_noise: f64, smoothing: f64, seed: u64) -> Self {
        Self { n_particles, jitter_noise, smoothing, star_smoothing: 0.18,
               seed, mod4_weight: 0.70, ewma_weight: 0.0, spatial_sigma: 0.0, warmup: 0 }
    }
}

impl Default for StresaSmcModel {
    fn default() -> Self {
        Self {
            n_particles: 1000,
            jitter_noise: 0.02,
            smoothing: 0.20,
            star_smoothing: 0.18,
            seed: 42,
            mod4_weight: 0.85,
            ewma_weight: 0.0,
            spatial_sigma: 0.0,
            warmup: 30,
        }
    }
}

/// Particule du filtre SMC : paramètres structurels boules + étoiles.
#[derive(Debug, Clone)]
struct SmcParticle {
    balls: StresaStructuralParams,
    stars: StarParams,
}

impl SmcParticle {
    fn new_perturbed(sigma: f64, rng: &mut u64) -> Self {
        let mut balls = StresaStructuralParams::uniform();
        balls.jitter(sigma, rng);
        let mut stars = StarParams::uniform();
        stars.jitter(sigma, rng);
        Self { balls, stars }
    }

    fn jitter(&mut self, noise: f64, rng: &mut u64) {
        self.balls.jitter(noise, rng);
        self.stars.jitter(noise, rng);
    }
}

struct ParticleFilter {
    particles: Vec<SmcParticle>,
    weights: Vec<f64>,
    rng_state: u64,
    observations_seen: usize,
    base_jitter: f64,
    warmup: usize,
}

impl ParticleFilter {
    fn new(n_particles: usize, seed: u64, jitter_noise: f64, warmup: usize) -> Self {
        let mut rng_state = seed.wrapping_add(1);
        if rng_state == 0 { rng_state = 1; }

        // Prior d'init plus large (sigma=0.1) pour meilleure exploration en 9D
        let particles: Vec<SmcParticle> = (0..n_particles)
            .map(|_| SmcParticle::new_perturbed(0.1, &mut rng_state))
            .collect();

        let uniform_w = 1.0 / n_particles as f64;
        let weights = vec![uniform_w; n_particles];

        Self {
            particles,
            weights,
            rng_state,
            observations_seen: 0,
            base_jitter: jitter_noise,
            warmup,
        }
    }

    /// Jitter adaptatif : décroît avec sqrt(n_observations).
    fn adaptive_jitter(&self) -> f64 {
        self.base_jitter / (1.0 + (self.observations_seen as f64).sqrt() * 0.1)
    }

    /// Observe un tirage : repondère et resample si nécessaire.
    /// Uses tempered likelihood during warmup to avoid premature particle collapse.
    fn observe_balls(&mut self, draw: &Draw, prev_draw: Option<&Draw>) {
        let n = self.particles.len();

        // Tempered likelihood: ramp-up over first warmup observations
        let beta = if self.warmup > 0 && self.observations_seen < self.warmup {
            (self.observations_seen as f64 + 1.0) / self.warmup as f64
        } else {
            1.0
        };

        for i in 0..n {
            let ll = self.particles[i].balls.log_likelihood_balls(draw, prev_draw);
            let ll_clamped = ll.clamp(-50.0, 0.0);
            self.weights[i] *= (beta * ll_clamped).exp();
        }

        self.normalize_and_resample();
        self.observations_seen += 1;
    }

    fn observe_stars(&mut self, draw: &Draw) {
        let n = self.particles.len();

        for i in 0..n {
            let ll = self.particles[i].stars.log_likelihood_stars(draw);
            let ll_clamped = ll.clamp(-50.0, 0.0);
            self.weights[i] *= ll_clamped.exp();
        }

        self.normalize_and_resample();
    }

    fn normalize_and_resample(&mut self) {
        let n = self.particles.len();
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        } else {
            let uw = 1.0 / n as f64;
            for w in &mut self.weights {
                *w = uw;
            }
            return;
        }

        let ess = 1.0 / self.weights.iter().map(|&w| w * w).sum::<f64>();
        if ess < n as f64 / 2.0 {
            self.systematic_resample();
        }
    }

    /// Systematic resampling (O(N), faible variance).
    fn systematic_resample(&mut self) {
        let n = self.particles.len();
        let u = pseudo_uniform(&mut self.rng_state) / n as f64;

        let mut cumsum = vec![0.0f64; n];
        cumsum[0] = self.weights[0];
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + self.weights[i];
        }

        let mut new_particles = Vec::with_capacity(n);
        let mut j = 0;
        for i in 0..n {
            let threshold = u + i as f64 / n as f64;
            while j < n - 1 && cumsum[j] < threshold {
                j += 1;
            }
            new_particles.push(self.particles[j].clone());
        }

        // Jittering adaptatif pour diversité
        let noise = self.adaptive_jitter();
        for p in &mut new_particles {
            p.jitter(noise, &mut self.rng_state);
        }

        self.particles = new_particles;
        let uw = 1.0 / n as f64;
        self.weights = vec![uw; n];
    }

    /// Prédiction boules = moyenne pondérée sur les particules.
    fn predict_balls(&self, prev_draw: &Draw) -> Vec<f64> {
        let mut combined = vec![0.0f64; 50];
        for (i, particle) in self.particles.iter().enumerate() {
            let dist = particle.balls.ball_distribution(Some(&prev_draw.balls));
            for (j, &p) in dist.iter().enumerate() {
                combined[j] += self.weights[i] * p;
            }
        }
        let sum: f64 = combined.iter().sum();
        if sum > 0.0 {
            for p in &mut combined {
                *p /= sum;
            }
        }
        combined
    }

    fn predict_stars(&self) -> Vec<f64> {
        let mut combined = vec![0.0f64; 12];
        for (i, particle) in self.particles.iter().enumerate() {
            let dist = particle.stars.star_distribution();
            for (j, &p) in dist.iter().enumerate() {
                combined[j] += self.weights[i] * p;
            }
        }
        let sum: f64 = combined.iter().sum();
        if sum > 0.0 {
            for p in &mut combined {
                *p /= sum;
            }
        }
        combined
    }
}

impl ForecastModel for StresaSmcModel {
    fn name(&self) -> &str {
        "StresaSMC"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 10 {
            return uniform;
        }

        // Stars: filtre particulaire dédié (200 particules pour 12 numéros)
        if pool == Pool::Stars {
            if draws.len() < 20 {
                return uniform;
            }
            let n_star_particles = 200;
            let mut pf = ParticleFilter::new(n_star_particles, self.seed.wrapping_add(7), self.jitter_noise, self.warmup);
            let n = draws.len();
            for t in (0..n).rev() {
                pf.observe_stars(&draws[t]);
            }
            let mut star_dist = pf.predict_stars();
            // Blend avec uniforme
            let uniform_val = 1.0 / size as f64;
            for p in &mut star_dist {
                *p = (1.0 - self.star_smoothing) * *p + self.star_smoothing * uniform_val;
            }
            normalize(&mut star_dist);
            return star_dist;
        }

        // Balls path: particle filter + Mod4 + EWMA blend
        let mut pf = ParticleFilter::new(self.n_particles, self.seed, self.jitter_noise, self.warmup);

        let n = draws.len();
        for t in (0..n).rev() {
            let prev = if t + 1 < n { Some(&draws[t + 1]) } else { None };
            pf.observe_balls(&draws[t], prev);
        }

        let structural = pf.predict_balls(&draws[0]);
        let mod4 = mod4_transition_predict_v2(draws, Pool::Balls);
        let ewma = ewma_ball_predict(draws, 0.3);

        let w_struct = (1.0 - self.mod4_weight - self.ewma_weight).max(0.0);
        let mut dist: Vec<f64> = (0..size)
            .map(|i| w_struct * structural[i] + self.mod4_weight * mod4[i] + self.ewma_weight * ewma[i])
            .collect();

        // Spatial smoothing
        spatial_smooth(&mut dist, self.spatial_sigma);

        // Uniform smoothing
        let uniform_val = 1.0 / size as f64;
        for p in &mut dist {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }
        normalize(&mut dist);

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_particles".into(), self.n_particles as f64),
            ("jitter_noise".into(), self.jitter_noise),
            ("smoothing".into(), self.smoothing),
            ("seed".into(), self.seed as f64),
            ("mod4_weight".into(), self.mod4_weight),
            ("star_smoothing".into(), self.star_smoothing),
            ("ewma_weight".into(), self.ewma_weight),
            ("spatial_sigma".into(), self.spatial_sigma),
            ("warmup".into(), self.warmup as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Modèle 3 : StresaChaos (Simulateur Dynamique Chaotique)
// ─────────────────────────────────────────────────────────────────────────────

/// Métriques de Recurrence Quantification Analysis (RQA).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RqaMetrics {
    recurrence_rate: f64,
    determinism: f64,
    laminarity: f64,
    mean_diagonal: f64,
    max_diagonal: f64,
    entropy_diagonal: f64,
}

/// StresaChaos — simulateur dynamique chaotique de la machine Stresa.
///
/// Deux canaux de prédiction avec poids fixes :
/// 1. Nadaraya-Watson en espace des phases (bandwidth adaptatif via k_pilot)
/// 2. Transition mod-4 (balls uniquement — signal prouvé par Mod4Trans)
///
/// Stars: EWMA fréquentiel (même approche que Physics, alpha=0.05).
/// Embedding fixe tau=1, dim=2 pour éviter la malédiction de la dimensionnalité.
pub struct StresaChaosModel {
    k_pilot: usize,
    mod4_weight: f64,
    smoothing: f64,
    star_smoothing: f64,
    ewma_weight: f64,
    spatial_sigma: f64,
    temporal_decay: f64,
}

impl StresaChaosModel {
    pub fn new(
        k_pilot: usize,
        mod4_weight: f64,
        smoothing: f64,
        ewma_weight: f64,
        spatial_sigma: f64,
        temporal_decay: f64,
    ) -> Self {
        Self { k_pilot, mod4_weight, smoothing, star_smoothing: 0.18, ewma_weight, spatial_sigma, temporal_decay }
    }
}

impl Default for StresaChaosModel {
    fn default() -> Self {
        Self {
            k_pilot: 20,
            mod4_weight: 0.90,
            smoothing: 0.25,
            star_smoothing: 0.18,
            ewma_weight: 0.0,
            spatial_sigma: 0.0,
            temporal_decay: 0.0,
        }
    }
}

// ── Chaos utility functions ──────────────────────────────────────────────────

/// Shannon entropy of a frequency vector (unnormalized counts OK).
fn shannon_entropy(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Encode a balls draw as a 3D state vector.
///
/// 3 features physiquement pertinentes :
/// - mod4_momentum : cosine inter-tirages du vecteur mod-4 (LE signal prouvé)
/// - sum_norm : somme normalisée (corrèle avec la physique machine)
/// - row_entropy : entropie des décades (diversité spatiale)
fn encode_state_balls(draw: &Draw, prev: Option<&Draw>) -> Vec<f64> {
    let balls = &draw.balls;
    let sum: f64 = balls.iter().map(|&b| b as f64).sum();

    // Mod-4 momentum (cosine change from previous)
    let mut mod4 = [0.0f64; 4];
    for &b in balls {
        mod4[((b - 1) % 4) as usize] += 1.0;
    }
    let mod4_momentum = if let Some(prev) = prev {
        let mut prev_mod4 = [0.0f64; 4];
        for &b in &prev.balls {
            prev_mod4[((b - 1) % 4) as usize] += 1.0;
        }
        cosine_similarity(&mod4, &prev_mod4)
    } else {
        0.5
    };

    // Row distribution entropy (5 rows = decades)
    let mut rows = [0.0f64; 5];
    for &b in balls {
        rows[((b - 1) / 10) as usize] += 1.0;
    }
    let row_entropy = shannon_entropy(&rows);

    vec![
        mod4_momentum,            // [0] blade correlation with prev
        sum / 250.0,              // [1] sum normalized
        row_entropy / 1.609,      // [2] row spread (normalized by ln(5))
    ]
}

/// Encode a stars draw as a 2D state vector.
///
/// 2 features : sum_norm et spread_norm.
fn encode_state_stars(draw: &Draw, _prev: Option<&Draw>) -> Vec<f64> {
    let stars = &draw.stars;
    let sum = (stars[0] as f64 + stars[1] as f64) / 24.0;
    let spread = (stars[1] - stars[0]) as f64 / 11.0;
    vec![sum, spread]
}

/// Encode all draws into state vectors (chronological order, oldest first).
fn encode_all_states(draws: &[Draw], pool: Pool) -> Vec<Vec<f64>> {
    let n = draws.len();
    // draws[0] = most recent, we process oldest-first
    let mut states = Vec::with_capacity(n);
    for t in (0..n).rev() {
        let prev = if t + 1 < n { Some(&draws[t + 1]) } else { None };
        let state = match pool {
            Pool::Balls => encode_state_balls(&draws[t], prev),
            Pool::Stars => encode_state_stars(&draws[t], prev),
        };
        states.push(state);
    }
    states
}

/// Auto Mutual Information — find optimal tau (first minimum of I(x_t, x_{t+tau})).
#[allow(dead_code)]
fn optimal_tau(series: &[Vec<f64>], max_tau: usize, n_bins: usize) -> usize {
    if series.len() < max_tau + 2 || series.is_empty() {
        return 1;
    }

    let feat_dim = series[0].len();
    let mut prev_mi = f64::MAX;

    for tau in 1..=max_tau {
        let mut total_mi = 0.0;

        for d in 0..feat_dim {
            // Extract the d-th feature series
            let x: Vec<f64> = series.iter().map(|s| s[d]).collect();
            let n = x.len() - tau;
            if n < 10 {
                return tau.max(1);
            }

            // Find min/max for binning
            let (mut x_min, mut x_max) = (f64::MAX, f64::NEG_INFINITY);
            for i in 0..n {
                x_min = x_min.min(x[i]).min(x[i + tau]);
                x_max = x_max.max(x[i]).max(x[i + tau]);
            }
            let range = (x_max - x_min).max(1e-15);

            // Build joint histogram
            let mut joint = vec![vec![0.0f64; n_bins]; n_bins];
            for i in 0..n {
                let bi = (((x[i] - x_min) / range * (n_bins as f64 - 1e-10)) as usize).min(n_bins - 1);
                let bj = (((x[i + tau] - x_min) / range * (n_bins as f64 - 1e-10)) as usize).min(n_bins - 1);
                joint[bi][bj] += 1.0;
            }

            // Marginals
            let mut px = vec![0.0f64; n_bins];
            let mut py = vec![0.0f64; n_bins];
            for i in 0..n_bins {
                for j in 0..n_bins {
                    px[i] += joint[i][j];
                    py[j] += joint[i][j];
                }
            }

            // MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
            let n_f = n as f64;
            let mut mi = 0.0;
            for i in 0..n_bins {
                for j in 0..n_bins {
                    if joint[i][j] > 0.0 && px[i] > 0.0 && py[j] > 0.0 {
                        let pxy = joint[i][j] / n_f;
                        let px_val = px[i] / n_f;
                        let py_val = py[j] / n_f;
                        mi += pxy * (pxy / (px_val * py_val)).ln();
                    }
                }
            }
            total_mi += mi;
        }

        let avg_mi = total_mi / feat_dim as f64;

        // First local minimum
        if avg_mi > prev_mi {
            return (tau - 1).max(1);
        }
        prev_mi = avg_mi;
    }

    max_tau.max(1)
}

/// False Nearest Neighbors — find optimal embedding dimension.
#[allow(dead_code)]
fn optimal_dim(series: &[Vec<f64>], tau: usize, max_dim: usize, threshold: f64) -> usize {
    if series.is_empty() {
        return 3;
    }

    let feat_dim = series[0].len();

    for dim in 2..=max_dim {
        let offset = dim * tau;
        if offset >= series.len() {
            return (dim - 1).max(2);
        }

        // Build embedding at dim and dim+1
        let n_pts = series.len() - offset;
        if n_pts < 10 {
            return (dim - 1).max(2);
        }

        let mut embedded_d: Vec<Vec<f64>> = Vec::with_capacity(n_pts);
        let mut embedded_d1: Vec<Vec<f64>> = Vec::with_capacity(n_pts);
        let offset_d1 = (dim + 1) * tau;
        let n_pts_d1 = if offset_d1 < series.len() { series.len() - offset_d1 } else { 0 };

        for t in 0..n_pts {
            let mut v = Vec::with_capacity(dim * feat_dim);
            for dd in 0..dim {
                let idx = offset - dd * tau + t;
                if idx < series.len() {
                    v.extend_from_slice(&series[idx]);
                }
            }
            embedded_d.push(v);
        }

        if n_pts_d1 < 10 {
            return dim;
        }

        for t in 0..n_pts_d1 {
            let mut v = Vec::with_capacity((dim + 1) * feat_dim);
            for dd in 0..=dim {
                let idx = offset_d1 - dd * tau + t;
                if idx < series.len() {
                    v.extend_from_slice(&series[idx]);
                }
            }
            embedded_d1.push(v);
        }

        // Count false nearest neighbors
        let mut fnn_count = 0;
        let mut total_count = 0;
        let sample_size = n_pts_d1.min(100); // sample for speed

        for i in 0..sample_size {
            // Find nearest neighbor in dim-d embedding
            let mut best_dist = f64::MAX;
            let mut best_j = 0;
            for j in 0..n_pts {
                if i == j { continue; }
                let d = euclidean_dist(&embedded_d[i], &embedded_d[j]);
                if d < best_dist {
                    best_dist = d;
                    best_j = j;
                }
            }

            if best_dist < 1e-15 { continue; }

            // Check if neighbor is still close in dim+1
            if i < embedded_d1.len() && best_j < embedded_d1.len() {
                let d1 = euclidean_dist(&embedded_d1[i], &embedded_d1[best_j]);
                let ratio = ((d1 * d1 - best_dist * best_dist).max(0.0)).sqrt() / best_dist;
                if ratio > 10.0 {
                    fnn_count += 1;
                }
                total_count += 1;
            }
        }

        if total_count > 0 {
            let fnn_frac = fnn_count as f64 / total_count as f64;
            if fnn_frac < threshold {
                return dim;
            }
        }
    }

    max_dim.min(6)
}

/// Euclidean distance between two vectors.
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Takens embedding with given tau and dim.
fn chaos_takens_embed(states: &[Vec<f64>], tau: usize, dim: usize) -> Vec<Vec<f64>> {
    let feat_dim = if states.is_empty() { 0 } else { states[0].len() };
    let offset = (dim - 1) * tau;
    if offset >= states.len() {
        return vec![];
    }

    let mut embedded = Vec::with_capacity(states.len() - offset);
    for t in offset..states.len() {
        let mut v = Vec::with_capacity(dim * feat_dim);
        for d in 0..dim {
            v.extend_from_slice(&states[t - d * tau]);
        }
        embedded.push(v);
    }
    embedded
}

/// Local Lyapunov exponent at a given index (kept as diagnostic utility).
#[allow(dead_code)]
fn local_lyapunov(embedded: &[Vec<f64>], index: usize, delta_t: usize) -> f64 {
    if embedded.len() < 3 || index == 0 {
        return 0.0;
    }

    let query = &embedded[index];

    // Find nearest neighbor (excluding index itself and last delta_t points)
    let search_end = if index + delta_t < embedded.len() {
        embedded.len() - delta_t
    } else {
        return 0.0;
    };

    let mut best_dist = f64::MAX;
    let mut best_idx = 0;
    for i in 0..search_end {
        if i == index { continue; }
        if i + delta_t >= embedded.len() { continue; }
        let d = euclidean_dist(query, &embedded[i]);
        if d > 1e-15 && d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }

    if best_dist >= f64::MAX - 1.0 {
        return 0.0;
    }

    // Check divergence after delta_t steps
    let idx_future = index + delta_t;
    let nn_future = best_idx + delta_t;
    if idx_future >= embedded.len() || nn_future >= embedded.len() {
        return 0.0;
    }

    let future_dist = euclidean_dist(&embedded[idx_future], &embedded[nn_future]);
    if future_dist < 1e-15 {
        return -5.0; // convergence
    }

    (future_dist / best_dist).ln() / delta_t as f64
}

/// Recurrence Quantification Analysis.
#[allow(dead_code)]
fn recurrence_analysis(embedded: &[Vec<f64>], epsilon: f64) -> RqaMetrics {
    let n = embedded.len();
    if n < 5 {
        return RqaMetrics {
            recurrence_rate: 0.0, determinism: 0.0, laminarity: 0.0,
            mean_diagonal: 0.0, max_diagonal: 0.0, entropy_diagonal: 0.0,
        };
    }

    // Cap size for O(n^2) analysis
    let max_n = n.min(300);
    let embedded = &embedded[n - max_n..];
    let n = embedded.len();

    // Build recurrence matrix (sparse — only count)
    let mut recurrence_count = 0usize;
    let total_pairs = n * n;

    // Diagonal line lengths (l_min = 2)
    let mut diag_lengths: Vec<usize> = Vec::new();
    let mut vert_lengths: Vec<usize> = Vec::new();

    // Scan diagonals (upper triangle only, then double)
    for offset in 1..n {
        let mut diag_len = 0usize;
        for i in 0..n - offset {
            let j = i + offset;
            let d = euclidean_dist(&embedded[i], &embedded[j]);
            if d < epsilon {
                recurrence_count += 2; // symmetric
                diag_len += 1;
            } else {
                if diag_len >= 2 {
                    diag_lengths.push(diag_len);
                }
                diag_len = 0;
            }
        }
        if diag_len >= 2 {
            diag_lengths.push(diag_len);
        }
    }

    // Vertical lines: for each column j, scan consecutive recurrent points
    for j in 0..n {
        let mut vert_len = 0usize;
        for i in 0..n {
            if i == j { continue; }
            let d = euclidean_dist(&embedded[i], &embedded[j]);
            if d < epsilon {
                vert_len += 1;
            } else {
                if vert_len >= 2 {
                    vert_lengths.push(vert_len);
                }
                vert_len = 0;
            }
        }
        if vert_len >= 2 {
            vert_lengths.push(vert_len);
        }
    }

    let rr = recurrence_count as f64 / total_pairs as f64;

    let diag_points: usize = diag_lengths.iter().sum();
    let det = if recurrence_count > 0 {
        diag_points as f64 / (recurrence_count as f64 / 2.0)
    } else {
        0.0
    };

    let vert_points: usize = vert_lengths.iter().sum();
    let lam = if recurrence_count > 0 {
        vert_points as f64 / recurrence_count as f64
    } else {
        0.0
    };

    let mean_diag = if diag_lengths.is_empty() {
        0.0
    } else {
        diag_lengths.iter().sum::<usize>() as f64 / diag_lengths.len() as f64
    };

    let max_diag = diag_lengths.iter().copied().max().unwrap_or(0) as f64;

    let ent_diag = if !diag_lengths.is_empty() {
        let counts: Vec<f64> = diag_lengths.iter().map(|&l| l as f64).collect();
        shannon_entropy(&counts)
    } else {
        0.0
    };

    RqaMetrics {
        recurrence_rate: rr.clamp(0.0, 1.0),
        determinism: det.clamp(0.0, 1.0),
        laminarity: lam.clamp(0.0, 1.0),
        mean_diagonal: mean_diag,
        max_diagonal: max_diag,
        entropy_diagonal: ent_diag,
    }
}

/// Detect unstable periodic orbits (UPO) — quasi-periodicities (kept as utility).
#[allow(dead_code)]
fn detect_upo(embedded: &[Vec<f64>], max_period: usize) -> Vec<(usize, f64)> {
    let n = embedded.len();
    if n < max_period + 5 {
        return vec![];
    }

    let mut scores: Vec<(usize, f64)> = Vec::new();
    let max_p = max_period.min(n / 2);

    for p in 2..=max_p {
        let mut total_sim = 0.0;
        let mut count = 0;
        for t in p..n {
            let d = euclidean_dist(&embedded[t], &embedded[t - p]);
            // Similarity = exp(-d²)
            total_sim += (-d * d).exp();
            count += 1;
        }
        if count > 0 {
            let mean_sim = total_sim / count as f64;
            scores.push((p, mean_sim));
        }
    }

    // Sort by score descending, keep top peaks
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(10);
    scores
}

/// Mod-4 transition prediction (reused from Mod4Trans logic).
fn mod4_transition_predict(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 10 {
        return uniform;
    }

    let numbers_list: Vec<Vec<u8>> = draws.iter()
        .map(|d| pool.numbers_from(d).to_vec())
        .collect();

    // Transition matrix T[i][j] with Laplace smoothing
    let mut transition = [[1.0f64; 4]; 4];
    for t in 0..draws.len() - 1 {
        let mut current_mod4 = [0.0f64; 4];
        let mut next_mod4 = [0.0f64; 4];
        for &n in &numbers_list[t] {
            current_mod4[((n - 1) % 4) as usize] += 1.0;
        }
        for &n in &numbers_list[t + 1] {
            next_mod4[((n - 1) % 4) as usize] += 1.0;
        }
        for (i, &c_i) in current_mod4.iter().enumerate() {
            if c_i > 0.0 {
                for (j, &n_j) in next_mod4.iter().enumerate() {
                    transition[i][j] += c_i * n_j;
                }
            }
        }
    }

    // Normalize rows
    for row in &mut transition {
        let total: f64 = row.iter().sum();
        if total > 0.0 {
            for v in row.iter_mut() { *v /= total; }
        }
    }

    // Predict from last draw
    let mut current_mod4 = [0.0f64; 4];
    for &n in &numbers_list[0] {
        current_mod4[((n - 1) % 4) as usize] += 1.0;
    }
    let total_current: f64 = current_mod4.iter().sum();

    let mut p_next = [0.0f64; 4];
    if total_current > 0.0 {
        for (i, &c) in current_mod4.iter().enumerate() {
            let w = c / total_current;
            for (j, p) in p_next.iter_mut().enumerate() {
                *p += w * transition[i][j];
            }
        }
    } else {
        p_next = [0.25; 4];
    }

    // Intra-class redistribution by historical frequency
    let mut freq = vec![1.0f64; size]; // Laplace +1
    for nums in &numbers_list {
        for &n in nums {
            let idx = (n - 1) as usize;
            if idx < size {
                freq[idx] += 1.0;
            }
        }
    }

    let mut class_freq_sum = [0.0f64; 4];
    for (k, &f) in freq.iter().enumerate() {
        class_freq_sum[k % 4] += f;
    }

    let mut prob = vec![0.0f64; size];
    for (k, p) in prob.iter_mut().enumerate() {
        let r = k % 4;
        if class_freq_sum[r] > 0.0 {
            *p = p_next[r] * freq[k] / class_freq_sum[r];
        }
    }

    // Normalize
    let total: f64 = prob.iter().sum();
    if total > 0.0 {
        for p in &mut prob { *p /= total; }
    } else {
        return uniform;
    }

    prob
}

/// Second-order mod-4 transition prediction.
/// Conditions on the 2 last draws instead of 1 for higher-order Markov dependency.
/// Falls back to first-order if fewer than 30 draws available.
fn mod4_transition_predict_v2(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let uniform = vec![1.0 / size as f64; size];

    // Fallback to first-order if insufficient data
    if draws.len() < 30 {
        return mod4_transition_predict(draws, pool);
    }

    let numbers_list: Vec<Vec<u8>> = draws.iter()
        .map(|d| pool.numbers_from(d).to_vec())
        .collect();

    // 3D transition table: T[prev_blade][curr_blade][next_blade]
    // With Laplace smoothing (+1)
    let mut transition = [[[1.0f64; 4]; 4]; 4];

    for t in 0..draws.len() - 2 {
        // t = current (most recent), t+1 = previous, t+2 = before that
        let mut curr_mod4 = [0.0f64; 4];
        let mut prev_mod4 = [0.0f64; 4];
        let mut next_mod4 = [0.0f64; 4];

        for &n in &numbers_list[t] {
            next_mod4[((n - 1) % 4) as usize] += 1.0;
        }
        for &n in &numbers_list[t + 1] {
            curr_mod4[((n - 1) % 4) as usize] += 1.0;
        }
        for &n in &numbers_list[t + 2] {
            prev_mod4[((n - 1) % 4) as usize] += 1.0;
        }

        // Weight by counts in each class
        for (i, &p_i) in prev_mod4.iter().enumerate() {
            if p_i > 0.0 {
                for (j, &c_j) in curr_mod4.iter().enumerate() {
                    if c_j > 0.0 {
                        for (k, &n_k) in next_mod4.iter().enumerate() {
                            transition[i][j][k] += p_i * c_j * n_k;
                        }
                    }
                }
            }
        }
    }

    // Normalize: for each (i,j), normalize over k
    for i in 0..4 {
        for j in 0..4 {
            let total: f64 = transition[i][j].iter().sum();
            if total > 0.0 {
                for k in 0..4 {
                    transition[i][j][k] /= total;
                }
            }
        }
    }

    // Predict: condition on draws[1] (prev) and draws[0] (current)
    let mut prev_mod4 = [0.0f64; 4];
    let mut curr_mod4 = [0.0f64; 4];
    for &n in &numbers_list[1] {
        prev_mod4[((n - 1) % 4) as usize] += 1.0;
    }
    for &n in &numbers_list[0] {
        curr_mod4[((n - 1) % 4) as usize] += 1.0;
    }

    let total_prev: f64 = prev_mod4.iter().sum();
    let total_curr: f64 = curr_mod4.iter().sum();

    let mut p_next = [0.0f64; 4];
    if total_prev > 0.0 && total_curr > 0.0 {
        for (i, &p_i) in prev_mod4.iter().enumerate() {
            for (j, &c_j) in curr_mod4.iter().enumerate() {
                let w = (p_i / total_prev) * (c_j / total_curr);
                for (k, &t_k) in transition[i][j].iter().enumerate() {
                    p_next[k] += w * t_k;
                }
            }
        }
    } else {
        p_next = [0.25; 4];
    }

    // Intra-class redistribution by historical frequency (same as v1)
    let mut freq = vec![1.0f64; size];
    for nums in &numbers_list {
        for &n in nums {
            let idx = (n - 1) as usize;
            if idx < size { freq[idx] += 1.0; }
        }
    }

    let mut class_freq_sum = [0.0f64; 4];
    for (k, &f) in freq.iter().enumerate() {
        class_freq_sum[k % 4] += f;
    }

    let mut prob = vec![0.0f64; size];
    for (k, p) in prob.iter_mut().enumerate() {
        let r = k % 4;
        if class_freq_sum[r] > 0.0 {
            *p = p_next[r] * freq[k] / class_freq_sum[r];
        }
    }

    let total: f64 = prob.iter().sum();
    if total > 0.0 {
        for p in &mut prob { *p /= total; }
    } else {
        return uniform;
    }
    prob
}

/// Mod4 transition prediction v3 — temporal weighting improvements over v2.
///
/// Two improvements:
/// 1. EWMA intra-class redistribution: frequency weights decay exponentially
///    (alpha_freq controls recency bias, ~0.05 = effective memory of ~20 draws)
/// 2. Temporal decay on transition table: recent transitions weighted more
///    (transition_decay controls decay rate, ~0.003 = ~333 draws effective memory)
///
/// Falls back to v2 for < 30 draws.
///
/// NOTE: Benchmarked in Phase 4 — both C (EWMA freq) and D (transition decay)
/// regress significantly vs v2 flat frequency. Kept for reference/testing only.
#[allow(dead_code)]
fn mod4_transition_predict_v3(
    draws: &[Draw],
    pool: Pool,
    alpha_freq: f64,
    transition_decay: f64,
) -> Vec<f64> {
    let size = pool.size();
    let uniform = vec![1.0 / size as f64; size];

    // Fallback to v2 (which itself falls back to v1) if insufficient data
    if draws.len() < 30 {
        return mod4_transition_predict_v2(draws, pool);
    }

    let numbers_list: Vec<Vec<u8>> = draws.iter()
        .map(|d| pool.numbers_from(d).to_vec())
        .collect();

    // 3D transition table: T[prev_blade][curr_blade][next_blade]
    // With Laplace smoothing (+1)
    let mut transition = [[[1.0f64; 4]; 4]; 4];

    for t in 0..draws.len() - 2 {
        // t = current (most recent), t+1 = previous, t+2 = before that
        let mut curr_mod4 = [0.0f64; 4];
        let mut prev_mod4 = [0.0f64; 4];
        let mut next_mod4 = [0.0f64; 4];

        for &n in &numbers_list[t] {
            next_mod4[((n - 1) % 4) as usize] += 1.0;
        }
        for &n in &numbers_list[t + 1] {
            curr_mod4[((n - 1) % 4) as usize] += 1.0;
        }
        for &n in &numbers_list[t + 2] {
            prev_mod4[((n - 1) % 4) as usize] += 1.0;
        }

        // Temporal weighting: t=0 is most recent → weight=1
        let weight = if transition_decay > 0.0 {
            (-transition_decay * t as f64).exp()
        } else {
            1.0
        };

        // Weight by counts in each class
        for (i, &p_i) in prev_mod4.iter().enumerate() {
            if p_i > 0.0 {
                for (j, &c_j) in curr_mod4.iter().enumerate() {
                    if c_j > 0.0 {
                        for (k, &n_k) in next_mod4.iter().enumerate() {
                            transition[i][j][k] += weight * p_i * c_j * n_k;
                        }
                    }
                }
            }
        }
    }

    // Normalize: for each (i,j), normalize over k
    for i in 0..4 {
        for j in 0..4 {
            let total: f64 = transition[i][j].iter().sum();
            if total > 0.0 {
                for k in 0..4 {
                    transition[i][j][k] /= total;
                }
            }
        }
    }

    // Predict: condition on draws[1] (prev) and draws[0] (current)
    let mut prev_mod4 = [0.0f64; 4];
    let mut curr_mod4 = [0.0f64; 4];
    for &n in &numbers_list[1] {
        prev_mod4[((n - 1) % 4) as usize] += 1.0;
    }
    for &n in &numbers_list[0] {
        curr_mod4[((n - 1) % 4) as usize] += 1.0;
    }

    let total_prev: f64 = prev_mod4.iter().sum();
    let total_curr: f64 = curr_mod4.iter().sum();

    let mut p_next = [0.0f64; 4];
    if total_prev > 0.0 && total_curr > 0.0 {
        for (i, &p_i) in prev_mod4.iter().enumerate() {
            for (j, &c_j) in curr_mod4.iter().enumerate() {
                let w = (p_i / total_prev) * (c_j / total_curr);
                for (k, &t_k) in transition[i][j].iter().enumerate() {
                    p_next[k] += w * t_k;
                }
            }
        }
    } else {
        p_next = [0.25; 4];
    }

    // Intra-class redistribution with EWMA frequency (improvement C)
    let mut freq = vec![1.0 / size as f64; size]; // uniform prior
    for t in (0..draws.len()).rev() {  // oldest first
        let mut current = vec![0.0; size];
        for &n in &numbers_list[t] {
            let idx = (n - 1) as usize;
            if idx < size { current[idx] = 1.0; }
        }
        for i in 0..size {
            freq[i] = (1.0 - alpha_freq) * freq[i] + alpha_freq * current[i];
        }
    }
    // Normalize freq
    normalize(&mut freq);

    let mut class_freq_sum = [0.0f64; 4];
    for (k, &f) in freq.iter().enumerate() {
        class_freq_sum[k % 4] += f;
    }

    let mut prob = vec![0.0f64; size];
    for (k, p) in prob.iter_mut().enumerate() {
        let r = k % 4;
        if class_freq_sum[r] > 0.0 {
            *p = p_next[r] * freq[k] / class_freq_sum[r];
        }
    }

    let total: f64 = prob.iter().sum();
    if total > 0.0 {
        for p in &mut prob { *p /= total; }
    } else {
        return uniform;
    }
    prob
}

/// UPO-based prediction (kept as utility, no longer used in predict).
#[allow(dead_code)]
fn upo_predict(
    upo_scores: &[(usize, f64)],
    draws: &[Draw],
    pool: Pool,
    threshold: f64,
) -> Vec<f64> {
    let size = pool.size();
    let uniform_val = 1.0 / size as f64;
    let mut dist = vec![uniform_val; size];

    if upo_scores.is_empty() || draws.is_empty() {
        return dist;
    }

    // Use the strongest UPO if above threshold
    let (best_period, best_score) = upo_scores[0];
    if best_score < threshold || best_period >= draws.len() {
        return dist;
    }

    // The draw `best_period` draws ago should predict the next one
    let reference_draw = &draws[best_period];
    let upo_weight = (best_score - threshold).min(0.5);

    // Mix: UPO successor distribution + uniform
    let mut upo_dist = vec![0.0f64; size];
    for &num in pool.numbers_from(reference_draw) {
        let idx = (num - 1) as usize;
        if idx < size {
            upo_dist[idx] += 1.0;
        }
    }
    let upo_sum: f64 = upo_dist.iter().sum();
    if upo_sum > 0.0 {
        for p in &mut upo_dist { *p /= upo_sum; }
    }

    for (i, p) in dist.iter_mut().enumerate() {
        *p = upo_weight * upo_dist[i] + (1.0 - upo_weight) * uniform_val;
    }

    // Normalize
    let total: f64 = dist.iter().sum();
    if total > 0.0 {
        for p in &mut dist { *p /= total; }
    }

    dist
}

/// Sigmoid function (kept as utility).
#[allow(dead_code)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Normalize a distribution to sum to 1.
fn normalize(dist: &mut [f64]) {
    let sum: f64 = dist.iter().sum();
    if sum > 0.0 {
        for p in dist.iter_mut() {
            *p /= sum;
        }
    }
}

/// EWMA star prediction avec alpha configurable.
fn ewma_star_predict_alpha(draws: &[Draw], alpha: f64, smoothing: f64) -> Vec<f64> {
    let size = 12;
    let mut freq = vec![1.0 / size as f64; size];

    for draw in draws.iter().rev() {
        let mut current = vec![0.0; size];
        for &s in &draw.stars {
            current[(s - 1) as usize] = 1.0;
        }
        for i in 0..size {
            freq[i] = (1.0 - alpha) * freq[i] + alpha * current[i];
        }
    }

    normalize(&mut freq);
    let uniform_val = 1.0 / size as f64;
    for f in &mut freq {
        *f = (1.0 - smoothing) * *f + smoothing * uniform_val;
    }
    normalize(&mut freq);
    freq
}

/// Matrice de transition mod-4 pour étoiles.
/// (star-1) % 4 → 4 classes, matrice de transition 4×4, redistribution intra-classe.
fn mod4_star_transition(draws: &[Draw]) -> Vec<f64> {
    let size = 12;
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 10 {
        return uniform;
    }

    let n_classes = 4;
    let laplace = 0.5;
    let mut transition = vec![vec![laplace; n_classes]; n_classes];
    let mut from_totals = vec![laplace * n_classes as f64; n_classes];

    for i in 0..draws.len() - 1 {
        for &s_from in &draws[i + 1].stars {
            let class_from = ((s_from - 1) % 4) as usize;
            for &s_to in &draws[i].stars {
                let class_to = ((s_to - 1) % 4) as usize;
                transition[class_from][class_to] += 1.0;
                from_totals[class_from] += 1.0;
            }
        }
    }

    let mut class_probs = vec![0.0f64; n_classes];
    for &s in &draws[0].stars {
        let class_from = ((s - 1) % 4) as usize;
        if from_totals[class_from] > 0.0 {
            for c in 0..n_classes {
                class_probs[c] += transition[class_from][c] / from_totals[class_from];
            }
        }
    }

    let class_sum: f64 = class_probs.iter().sum();
    if class_sum > 0.0 {
        for p in &mut class_probs {
            *p /= class_sum;
        }
    }

    let mut dist = vec![0.0f64; size];
    for star in 1..=12u8 {
        let idx = (star - 1) as usize;
        let class = ((star - 1) % 4) as usize;
        let members = (1..=12u8).filter(|&s| ((s - 1) % 4) as usize == class).count();
        dist[idx] = class_probs[class] / members as f64;
    }

    dist
}

/// EWMA star prediction — same approach as Physics model (alpha=0.03).
fn ewma_star_predict(draws: &[Draw], smoothing: f64) -> Vec<f64> {
    let size = 12;
    let alpha = 0.04;
    let mut freq = vec![1.0 / size as f64; size];

    // Du plus ancien au plus récent
    for draw in draws.iter().rev() {
        let mut current = vec![0.0; size];
        for &s in &draw.stars {
            current[(s - 1) as usize] = 1.0;
        }
        for i in 0..size {
            freq[i] = (1.0 - alpha) * freq[i] + alpha * current[i];
        }
    }

    // Normaliser + blend avec uniform
    normalize(&mut freq);
    let uniform_val = 1.0 / size as f64;
    for f in &mut freq {
        *f = (1.0 - smoothing) * *f + smoothing * uniform_val;
    }
    normalize(&mut freq);
    freq
}

/// EWMA ball prediction — same approach as Physics model (alpha=0.05).
fn ewma_ball_predict(draws: &[Draw], smoothing: f64) -> Vec<f64> {
    let size = 50;
    let alpha = 0.05;
    let mut freq = vec![1.0 / size as f64; size];

    // Du plus ancien au plus récent
    for draw in draws.iter().rev() {
        let mut current = vec![0.0; size];
        for &b in &draw.balls {
            current[(b - 1) as usize] = 1.0;
        }
        for i in 0..size {
            freq[i] = (1.0 - alpha) * freq[i] + alpha * current[i];
        }
    }

    normalize(&mut freq);
    let uniform_val = 1.0 / size as f64;
    for f in &mut freq {
        *f = (1.0 - smoothing) * *f + smoothing * uniform_val;
    }
    normalize(&mut freq);
    freq
}

/// Gaussian spatial smoothing (technique from Physics model).
/// Encodes mechanical proximity: nearby ball numbers behave similarly.
fn spatial_smooth(dist: &mut [f64], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let size = dist.len();
    let mut smoothed = vec![0.0; size];
    for i in 0..size {
        let mut wsum = 0.0;
        let mut vsum = 0.0;
        for j in 0..size {
            let d = (i as f64 - j as f64).abs();
            let w = (-d * d / (2.0 * sigma * sigma)).exp();
            wsum += w;
            vsum += w * dist[j];
        }
        if wsum > 0.0 {
            smoothed[i] = vsum / wsum;
        }
    }
    dist.copy_from_slice(&smoothed);
}

impl ForecastModel for StresaChaosModel {
    fn name(&self) -> &str {
        "StresaChaos"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 20 {
            return uniform;
        }

        // Stars: Nadaraya-Watson en espace des phases
        if pool == Pool::Stars {
            // Encode étoiles en 2D : [sum_norm_stars, spread_norm_stars]
            let star_states = encode_all_states(draws, Pool::Stars);
            let tau = 1;
            let dim = 2;
            let embedded = chaos_takens_embed(&star_states, tau, dim);
            if embedded.len() >= self.k_pilot + 2 {
                let offset = (dim - 1) * tau;
                let pred_nw = nadaraya_watson_predict(
                    &embedded, draws, Pool::Stars,
                    self.k_pilot.min(10), offset, self.temporal_decay,
                );
                // Blend avec uniforme
                let uniform_val = 1.0 / size as f64;
                let mut star_dist = pred_nw;
                for p in &mut star_dist {
                    *p = (1.0 - self.star_smoothing) * *p + self.star_smoothing * uniform_val;
                }
                normalize(&mut star_dist);
                return star_dist;
            }
            // Fallback: EWMA
            return ewma_star_predict(draws, self.star_smoothing);
        }

        // Balls path: NW + Mod4 + EWMA fusion
        // 1. Encode all draws as state vectors (chronological, oldest first)
        let states = encode_all_states(draws, pool);

        // 2. Fixed embedding parameters (no auto-optimization)
        let tau = 1;
        let dim = 2;

        // 3. Build Takens embedding
        let embedded = chaos_takens_embed(&states, tau, dim);
        if embedded.len() < self.k_pilot + 2 {
            return uniform;
        }

        // 4. NW prediction with temporal weighting
        let offset = (dim - 1) * tau;
        let pred_nw = nadaraya_watson_predict(
            &embedded, draws, pool,
            self.k_pilot, offset, self.temporal_decay,
        );

        // 5. Mod4 + EWMA
        let pred_mod4 = mod4_transition_predict_v2(draws, pool);
        let pred_ewma = ewma_ball_predict(draws, 0.3);

        let w_nw = (1.0 - self.mod4_weight - self.ewma_weight).max(0.0);
        let mut dist: Vec<f64> = (0..size)
            .map(|i| w_nw * pred_nw[i] + self.mod4_weight * pred_mod4[i] + self.ewma_weight * pred_ewma[i])
            .collect();

        // 6. Spatial smoothing
        spatial_smooth(&mut dist, self.spatial_sigma);

        // 7. Uniform smoothing
        let uniform_val = 1.0 / size as f64;
        for p in &mut dist {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }
        normalize(&mut dist);

        // Safety: ensure no negative or NaN
        for p in &mut dist {
            if p.is_nan() || *p < 0.0 {
                *p = uniform_val;
            }
        }
        normalize(&mut dist);

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("k_pilot".into(), self.k_pilot as f64),
            ("mod4_weight".into(), self.mod4_weight),
            ("smoothing".into(), self.smoothing),
            ("star_smoothing".into(), self.star_smoothing),
            ("ewma_weight".into(), self.ewma_weight),
            ("spatial_sigma".into(), self.spatial_sigma),
            ("temporal_decay".into(), self.temporal_decay),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }
}

/// Nadaraya-Watson prediction with adaptive bandwidth and temporal weighting.
///
/// Uses ALL embedded points weighted by a Gaussian kernel with locally adaptive
/// bandwidth (set to 1.5× distance to k_pilot-th neighbor). Avoids the truncation
/// noise of KNN while keeping the same offset-aware index mapping.
/// temporal_decay > 0 gives more weight to recent neighbors (drift compensation).
fn nadaraya_watson_predict(
    embedded: &[Vec<f64>],
    draws: &[Draw],
    pool: Pool,
    k_pilot: usize,
    offset: usize,
    temporal_decay: f64,
) -> Vec<f64> {
    let size = pool.size();
    let uniform = vec![1.0 / size as f64; size];

    if embedded.len() < k_pilot + 2 {
        return uniform;
    }

    let query = embedded.last().unwrap();
    let search_end = embedded.len() - 1;
    let n = draws.len();

    // 1. Compute all distances to query
    let mut distances: Vec<(usize, f64)> = (0..search_end)
        .map(|i| (i, euclidean_dist(&embedded[i], query)))
        .collect();

    // 2. Find k_pilot-th distance for adaptive bandwidth
    //    (partial sort is O(N) on average vs O(N log N) for full sort)
    let k_idx = k_pilot.min(distances.len() - 1);
    distances.select_nth_unstable_by(k_idx, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let sigma = distances[k_idx].1 * 2.0;

    if sigma < 1e-15 {
        return uniform;
    }

    // 3. Weight ALL points with Gaussian kernel
    let sigma_sq_2 = 2.0 * sigma * sigma;
    let mut weights = vec![0.0f64; size];
    let mut total_weight = 0.0f64;

    for &(embed_idx, dist) in &distances {
        // Successor mapping (same as knn_predict_with_offset)
        let chrono_successor = offset + embed_idx + 1;
        if chrono_successor >= n {
            continue;
        }
        let draw_idx = n - 1 - chrono_successor;

        let w_kernel = (-dist * dist / sigma_sq_2).exp();
        if w_kernel < 1e-15 {
            continue;
        }

        // Temporal weighting: recent neighbors matter more
        let w = if temporal_decay > 0.0 && n > 1 {
            let age = draw_idx as f64 / n as f64; // 0 = newest, ~1 = oldest
            w_kernel * (-temporal_decay * age).exp()
        } else {
            w_kernel
        };

        total_weight += w;

        let successor_draw = &draws[draw_idx];
        for &num in pool.numbers_from(successor_draw) {
            let idx = (num - 1) as usize;
            if idx < size {
                weights[idx] += w;
            }
        }
    }

    if total_weight <= 0.0 {
        return uniform;
    }

    normalize(&mut weights);
    weights
}

/// KNN prediction with proper offset-aware index mapping (superseded by Nadaraya-Watson).
#[allow(dead_code)]
fn knn_predict_with_offset(
    embedded: &[Vec<f64>],
    draws: &[Draw],
    pool: Pool,
    k: usize,
    sigma: f64,
    offset: usize,
) -> Vec<f64> {
    let size = pool.size();
    let uniform = vec![1.0 / size as f64; size];

    if embedded.len() < k + 2 {
        return uniform;
    }

    let query = embedded.last().unwrap();
    let search_end = embedded.len() - 1;
    let mut distances: Vec<(usize, f64)> = (0..search_end)
        .map(|i| (i, euclidean_dist(&embedded[i], query)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);

    if distances.is_empty() {
        return uniform;
    }

    let mut weights = vec![0.0f64; size];
    let mut total_weight = 0.0f64;
    let n = draws.len();

    for &(embed_idx, dist) in &distances {
        // embedded[embed_idx] corresponds to chronological index (offset + embed_idx)
        // Its successor is chronological index (offset + embed_idx + 1)
        // In draws[] (reverse-chrono): draws[n - 1 - (offset + embed_idx + 1)]
        let chrono_successor = offset + embed_idx + 1;
        if chrono_successor >= n {
            continue;
        }
        let draw_idx = n - 1 - chrono_successor;

        let successor_draw = &draws[draw_idx];
        let w = (-dist * dist / (2.0 * sigma * sigma)).exp();
        total_weight += w;

        for &num in pool.numbers_from(successor_draw) {
            let idx = (num - 1) as usize;
            if idx < size {
                weights[idx] += w;
            }
        }
    }

    if total_weight <= 0.0 {
        return uniform;
    }

    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return uniform;
    }
    for w in &mut weights {
        *w /= sum;
    }

    weights
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_stresa_sgd_balls_sums_to_one() {
        let model = StresaSgdModel::default();
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
    fn test_stresa_sgd_stars_sums_to_one() {
        let model = StresaSgdModel::default();
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
    fn test_stresa_sgd_no_negative() {
        let model = StresaSgdModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_stresa_sgd_empty_draws() {
        let model = StresaSgdModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stresa_sgd_few_draws() {
        let model = StresaSgdModel::default();
        let draws = make_test_draws(5);
        // Balls: uniform for <10 draws
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
        // Stars: uniform for <20 draws (EWMA threshold)
        let draws_15 = make_test_draws(15);
        let dist_stars = model.predict(&draws_15, Pool::Stars);
        let expected_stars = 1.0 / 12.0;
        for &p in &dist_stars {
            assert!((p - expected_stars).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stresa_smc_balls_sums_to_one() {
        let model = StresaSmcModel::default();
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
    fn test_stresa_smc_stars_sums_to_one() {
        let model = StresaSmcModel::default();
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
    fn test_stresa_smc_no_negative() {
        let model = StresaSmcModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_stresa_smc_empty_draws() {
        let model = StresaSmcModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stresa_smc_few_draws() {
        let model = StresaSmcModel::default();
        let draws = make_test_draws(5);
        // Balls: uniform for <10 draws
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
        // Stars: uniform for <20 draws (EWMA threshold)
        let draws_15 = make_test_draws(15);
        let dist_stars = model.predict(&draws_15, Pool::Stars);
        let expected_stars = 1.0 / 12.0;
        for &p in &dist_stars {
            assert!((p - expected_stars).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stresa_params_convergence() {
        // Vérifie que les paramètres ne divergent pas
        let mut params = StresaStructuralParams::uniform();
        let draws = make_test_draws(50);
        for t in (0..draws.len()).rev() {
            let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
            sgd_step_analytical(&mut params, &draws[t], prev, 0.02, 0.01);
        }
        // Tous les paramètres doivent rester dans des bornes raisonnables
        for &r in &params.row_bias {
            assert!(r >= 0.5 && r < 10.0, "row_bias diverged: {}", r);
        }
    }

    #[test]
    fn test_machine_params_ball_distribution_sums_to_one() {
        let params = StresaStructuralParams::uniform();
        let dist = params.ball_distribution(None);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Ball distribution sum = {}", sum);
    }

    #[test]
    fn test_star_params_distribution_sums_to_one() {
        let params = StarParams::uniform();
        let dist = params.star_distribution();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Star distribution sum = {}", sum);
    }

    #[test]
    fn test_particle_filter_resampling() {
        let mut pf = ParticleFilter::new(100, 42, 0.02, 0);
        // Mettre tout le poids sur la première particule
        pf.weights = vec![0.0; 100];
        pf.weights[0] = 1.0;
        pf.systematic_resample();
        // Après resampling, les poids doivent être uniformes
        let expected_w = 1.0 / 100.0;
        for &w in &pf.weights {
            assert!((w - expected_w).abs() < 1e-10);
        }
    }

    #[test]
    fn test_stresa_sgd_large_draws() {
        let model = StresaSgdModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_stresa_smc_large_draws() {
        let model = StresaSmcModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tests v2 : gradient analytique et convergence multi-epoch
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_analytical_gradient_matches_finite_diff() {
        let draws = make_test_draws(30);
        let params = StresaStructuralParams {
            row_bias: [1.0, 1.1, 0.95, 1.02, 0.98],
            persistence: 0.08,
            temperature: 1.2,
        };

        let draw = &draws[5];
        let prev = Some(&draws[6]);

        // Gradient analytique
        let grad = params.analytical_gradient_balls(draw, prev);

        // Gradient par différences finies
        let eps = 1e-6;

        // Vérifier row_bias
        for i in 0..5 {
            let mut p_plus = params.clone();
            p_plus.row_bias[i] += eps;
            let mut p_minus = params.clone();
            p_minus.row_bias[i] -= eps;
            let fd = (p_plus.log_likelihood_balls(draw, prev) - p_minus.log_likelihood_balls(draw, prev)) / (2.0 * eps);
            assert!(
                (grad.row[i] - fd).abs() < 1e-4,
                "row[{}]: analytical={:.6}, finite_diff={:.6}, diff={:.2e}",
                i, grad.row[i], fd, (grad.row[i] - fd).abs()
            );
        }

        // Vérifier persistence
        {
            let mut p_plus = params.clone();
            p_plus.persistence += eps;
            let mut p_minus = params.clone();
            p_minus.persistence -= eps;
            let fd = (p_plus.log_likelihood_balls(draw, prev) - p_minus.log_likelihood_balls(draw, prev)) / (2.0 * eps);
            assert!(
                (grad.persistence - fd).abs() < 1e-4,
                "persistence: analytical={:.6}, finite_diff={:.6}, diff={:.2e}",
                grad.persistence, fd, (grad.persistence - fd).abs()
            );
        }

        // Vérifier temperature
        {
            let mut p_plus = params.clone();
            p_plus.temperature += eps;
            let mut p_minus = params.clone();
            p_minus.temperature -= eps;
            let fd = (p_plus.log_likelihood_balls(draw, prev) - p_minus.log_likelihood_balls(draw, prev)) / (2.0 * eps);
            assert!(
                (grad.temperature - fd).abs() < 1e-4,
                "temperature: analytical={:.6}, finite_diff={:.6}, diff={:.2e}",
                grad.temperature, fd, (grad.temperature - fd).abs()
            );
        }
    }

    #[test]
    fn test_multi_epoch_convergence() {
        let draws = make_test_draws(50);
        let mut params = StresaStructuralParams::uniform();

        // Calculer LL initiale sur tous les draws
        let initial_ll: f64 = (0..draws.len()).rev().map(|t| {
            let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
            params.log_likelihood_balls(&draws[t], prev)
        }).sum();

        // Entraîner 3 epochs
        for epoch in 0..3 {
            let lr = 0.02 / (1.0 + epoch as f64 * 0.5);
            for t in (0..draws.len()).rev() {
                let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
                sgd_step_analytical(&mut params, &draws[t], prev, lr, 0.01);
            }
        }

        // LL finale doit être >= initiale (la régularisation peut empêcher
        // une amélioration stricte, donc on autorise une petite marge)
        let final_ll: f64 = (0..draws.len()).rev().map(|t| {
            let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
            params.log_likelihood_balls(&draws[t], prev)
        }).sum();

        assert!(
            final_ll >= initial_ll - 1.0,
            "Multi-epoch should improve or maintain LL: initial={:.4}, final={:.4}",
            initial_ll, final_ll
        );
    }

    #[test]
    fn test_analytical_gradient_at_uniform() {
        // Au prior uniforme, le gradient devrait être non-nul si les données
        // ne sont pas parfaitement uniformes
        let draws = make_test_draws(20);
        let params = StresaStructuralParams::uniform();
        let draw = &draws[0];
        let prev = Some(&draws[1]);

        let grad = params.analytical_gradient_balls(draw, prev);

        // Le gradient ne devrait pas être exactement zéro
        let grad_norm = grad.row.iter().map(|g| g * g).sum::<f64>()
            + grad.persistence * grad.persistence
            + grad.temperature * grad.temperature;

        assert!(grad_norm > 0.0, "Gradient should be non-zero at uniform prior");
    }

    #[test]
    fn test_structural_params_7_dimensions() {
        // Vérifier qu'on a bien 7 paramètres (5+2) pour les boules
        let params = StresaStructuralParams::uniform();
        assert_eq!(params.row_bias.len(), 5);
        // persistence + temperature = 2 scalaires
        // Total = 5 + 2 = 7
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tests StresaChaos
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_chaos_balls_sums_to_one() {
        let model = StresaChaosModel::default();
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
    fn test_chaos_stars_sums_to_one() {
        let model = StresaChaosModel::default();
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
    fn test_chaos_no_negative() {
        let model = StresaChaosModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_chaos_empty_draws() {
        let model = StresaChaosModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chaos_few_draws() {
        let model = StresaChaosModel::default();
        let draws = make_test_draws(15);
        // Balls: uniform (< 20 draws)
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
        // Stars: also uniform (< 20 draws)
        let dist_s = model.predict(&draws, Pool::Stars);
        let expected_s = 1.0 / 12.0;
        for &p in &dist_s {
            assert!((p - expected_s).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chaos_large_draws() {
        let model = StresaChaosModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_optimal_tau_sinusoid() {
        // For a sinusoidal signal, tau should be around period/4
        let period = 12;
        let n = 100;
        let series: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![
                (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin(),
                (2.0 * std::f64::consts::PI * i as f64 / period as f64).cos(),
            ])
            .collect();
        let tau = optimal_tau(&series, 8, 10);
        // Expected: tau ~ period/4 = 3, accept 1-5
        assert!(tau >= 1 && tau <= 5, "tau={} for period={}", tau, period);
    }

    #[test]
    fn test_optimal_dim_simple() {
        // For a 2D signal embedded, FNN should stabilize at dim ~2-3
        let n = 200;
        let series: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos()]
            })
            .collect();
        let dim = optimal_dim(&series, 1, 8, 0.01);
        // Should be small (2-4) for a simple 2D attractor
        assert!(dim >= 2 && dim <= 6, "dim={}", dim);
    }

    #[test]
    fn test_lyapunov_periodic() {
        // For a periodic series, Lyapunov exponent should be near 0 or negative
        let n = 100;
        let series: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(i as f64 * 0.5).sin(), (i as f64 * 0.5).cos()])
            .collect();
        let embedded = chaos_takens_embed(&series, 1, 3);
        if embedded.len() > 5 {
            let lambda = local_lyapunov(&embedded, embedded.len() - 2, 1);
            assert!(lambda < 2.0, "Lambda={} should be moderate for periodic signal", lambda);
        }
    }

    #[test]
    fn test_lyapunov_random() {
        // For random data, Lyapunov exponent should tend positive
        let mut rng = 12345u64;
        let n = 100;
        let series: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![pseudo_uniform(&mut rng), pseudo_uniform(&mut rng)])
            .collect();
        let embedded = chaos_takens_embed(&series, 1, 3);
        if embedded.len() > 5 {
            let lambda = local_lyapunov(&embedded, embedded.len() / 2, 1);
            // Random should generally show positive or zero lambda
            assert!(lambda > -5.0, "Lambda={} should not be strongly negative for random data", lambda);
        }
    }

    #[test]
    fn test_rqa_deterministic() {
        // For a deterministic periodic series, DET should be relatively high
        let n = 50;
        let series: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(i as f64 * 0.3).sin(), (i as f64 * 0.3).cos()])
            .collect();
        let embedded = chaos_takens_embed(&series, 1, 3);
        let rqa = recurrence_analysis(&embedded, 0.5);
        assert!(rqa.recurrence_rate >= 0.0);
        assert!(rqa.determinism >= 0.0);
    }

    #[test]
    fn test_upo_detection() {
        // Create a signal with a known period of 5
        let n = 100;
        let period = 5;
        let series: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let phase = (i % period) as f64 / period as f64;
                vec![phase, (phase * 2.0 * std::f64::consts::PI).sin()]
            })
            .collect();
        let embedded = chaos_takens_embed(&series, 1, 2);
        let upo = detect_upo(&embedded, 20);
        // Should detect period 5 or multiples thereof
        assert!(!upo.is_empty(), "Should detect at least one UPO");
        let detected_periods: Vec<usize> = upo.iter().map(|&(p, _)| p).collect();
        let has_period_5 = detected_periods.iter().any(|&p| p == period || p == period * 2);
        assert!(has_period_5, "Should detect period {} in {:?}", period, detected_periods);
    }

    #[test]
    fn test_encode_state_balls_dimensions() {
        let draw = Draw {
            draw_id: "001".into(),
            day: "MARDI".into(),
            date: "2024-01-01".into(),
            balls: [5, 15, 25, 35, 45],
            stars: [3, 7],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
        };
        let state = encode_state_balls(&draw, None);
        assert_eq!(state.len(), 3, "Balls state should be 3D");
        for &v in &state {
            assert!(v.is_finite(), "State value should be finite: {}", v);
        }
    }

    #[test]
    fn test_encode_state_stars_dimensions() {
        let draw = Draw {
            draw_id: "001".into(),
            day: "MARDI".into(),
            date: "2024-01-01".into(),
            balls: [5, 15, 25, 35, 45],
            stars: [3, 7],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
        };
        let state = encode_state_stars(&draw, None);
        assert_eq!(state.len(), 2, "Stars state should be 2D");
        for &v in &state {
            assert!(v.is_finite(), "State value should be finite: {}", v);
        }
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution should have max entropy
        let uniform = [1.0, 1.0, 1.0, 1.0];
        let h = shannon_entropy(&uniform);
        assert!((h - 4.0f64.ln()).abs() < 1e-10);

        // Degenerate distribution should have zero entropy
        let degenerate = [1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&degenerate);
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_ewma_star_predict_basic() {
        let draws = make_test_draws(50);
        let dist = ewma_star_predict(&draws, 0.65);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "EWMA star sum = {}", sum);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_nadaraya_watson_normalization() {
        let draws = make_test_draws(60);
        let states = encode_all_states(&draws, Pool::Balls);
        let embedded = chaos_takens_embed(&states, 1, 2);
        let offset = 1; // (dim-1) * tau = 1
        let dist = nadaraya_watson_predict(&embedded, &draws, Pool::Balls, 20, offset, 0.0);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "NW sum = {}", sum);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tests Phase 2 : ewma_ball, spatial_smooth, Adam
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_ewma_ball_predict_basic() {
        let draws = make_test_draws(50);
        let dist = ewma_ball_predict(&draws, 0.3);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "EWMA ball sum = {}", sum);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_spatial_smooth_no_negative_and_smooths() {
        // Start with a normalized distribution
        let mut dist = vec![1.0 / 50.0; 50];
        // Create a spike
        dist[25] = 0.10;
        normalize(&mut dist);
        let peak_before = dist[25];
        spatial_smooth(&mut dist, 2.0);
        // No negative values
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability after spatial smooth: {}", p);
        }
        // Peak should be reduced (smoothed out)
        assert!(dist[25] < peak_before, "Spatial smooth should reduce peaks");
        // All values should be finite
        for &p in &dist {
            assert!(p.is_finite(), "Non-finite value after spatial smooth: {}", p);
        }
    }

    #[test]
    fn test_adam_state_initialization() {
        let adam = AdamState::new();
        assert_eq!(adam.t, 0);
        for &m in &adam.m_row { assert_eq!(m, 0.0); }
        for &v in &adam.v_row { assert_eq!(v, 0.0); }
        assert_eq!(adam.m_persistence, 0.0);
        assert_eq!(adam.v_persistence, 0.0);
        assert_eq!(adam.m_temperature, 0.0);
        assert_eq!(adam.v_temperature, 0.0);
    }

    #[test]
    fn test_adam_convergence() {
        let draws = make_test_draws(50);
        let mut params = StresaStructuralParams::uniform();
        let mut adam = AdamState::new();

        let initial_ll: f64 = (0..draws.len()).rev().map(|t| {
            let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
            params.log_likelihood_balls(&draws[t], prev)
        }).sum();

        for epoch in 0..3 {
            let lr = 0.02 / (1.0 + epoch as f64 * 0.5);
            for t in (0..draws.len()).rev() {
                let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
                adam_step_analytical(&mut params, &mut adam, &draws[t], prev, lr, 0.01);
            }
        }

        let final_ll: f64 = (0..draws.len()).rev().map(|t| {
            let prev = if t + 1 < draws.len() { Some(&draws[t + 1]) } else { None };
            params.log_likelihood_balls(&draws[t], prev)
        }).sum();

        assert!(
            final_ll >= initial_ll - 1.0,
            "Adam should improve or maintain LL: initial={:.4}, final={:.4}",
            initial_ll, final_ll
        );
    }

    #[test]
    fn test_nadaraya_watson_temporal_decay() {
        let draws = make_test_draws(60);
        let states = encode_all_states(&draws, Pool::Balls);
        let embedded = chaos_takens_embed(&states, 1, 2);
        let offset = 1;
        // With temporal decay
        let dist = nadaraya_watson_predict(&embedded, &draws, Pool::Balls, 20, offset, 1.0);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "NW temporal sum = {}", sum);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_particle_filter_warmup() {
        let mut pf = ParticleFilter::new(100, 42, 0.02, 10);
        assert_eq!(pf.warmup, 10);
        assert_eq!(pf.observations_seen, 0);
        // After warmup, observations_seen should track
        let draw = make_test_draws(2);
        pf.observe_balls(&draw[0], Some(&draw[1]));
        assert_eq!(pf.observations_seen, 1);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tests Phase 3 : mod4_transition_v2
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mod4_transition_v2_basic() {
        let draws = make_test_draws(50);

        // Balls: sum=1, no negative, len=50
        let dist_balls = mod4_transition_predict_v2(&draws, Pool::Balls);
        assert_eq!(dist_balls.len(), 50);
        let sum: f64 = dist_balls.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "v2 balls sum = {}", sum);
        for &p in &dist_balls {
            assert!(p >= 0.0, "Negative probability in v2 balls: {}", p);
        }

        // Stars: sum=1, no negative, len=12
        let dist_stars = mod4_transition_predict_v2(&draws, Pool::Stars);
        assert_eq!(dist_stars.len(), 12);
        let sum_s: f64 = dist_stars.iter().sum();
        assert!((sum_s - 1.0).abs() < 1e-9, "v2 stars sum = {}", sum_s);
        for &p in &dist_stars {
            assert!(p >= 0.0, "Negative probability in v2 stars: {}", p);
        }
    }

    #[test]
    fn test_mod4_transition_v2_fallback_small() {
        // < 30 draws: should fallback to v1
        let draws = make_test_draws(20);
        let dist_v1 = mod4_transition_predict(&draws, Pool::Balls);
        let dist_v2 = mod4_transition_predict_v2(&draws, Pool::Balls);
        // v2 should be identical to v1 for small datasets
        for (i, (&a, &b)) in dist_v1.iter().zip(dist_v2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-15, "v1/v2 differ at {}: {} vs {}", i, a, b);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tests Phase 4 : mod4_transition_v3
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mod4_transition_v3_basic() {
        let draws = make_test_draws(50);

        // Balls: sum=1, no negative, len=50
        let dist_balls = mod4_transition_predict_v3(&draws, Pool::Balls, 0.05, 0.003);
        assert_eq!(dist_balls.len(), 50);
        let sum: f64 = dist_balls.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "v3 balls sum = {}", sum);
        for &p in &dist_balls {
            assert!(p >= 0.0, "Negative probability in v3 balls: {}", p);
        }

        // Stars: sum=1, no negative, len=12
        let dist_stars = mod4_transition_predict_v3(&draws, Pool::Stars, 0.05, 0.003);
        assert_eq!(dist_stars.len(), 12);
        let sum_s: f64 = dist_stars.iter().sum();
        assert!((sum_s - 1.0).abs() < 1e-9, "v3 stars sum = {}", sum_s);
        for &p in &dist_stars {
            assert!(p >= 0.0, "Negative probability in v3 stars: {}", p);
        }
    }

    #[test]
    fn test_mod4_transition_v3_ewma_vs_flat() {
        // v3 with EWMA freq should differ from v2 for sufficient data
        let draws = make_test_draws(80);
        let dist_v2 = mod4_transition_predict_v2(&draws, Pool::Balls);
        let dist_v3 = mod4_transition_predict_v3(&draws, Pool::Balls, 0.05, 0.0);

        // Should be different (EWMA freq vs flat freq)
        let max_diff: f64 = dist_v2.iter().zip(dist_v3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_diff > 1e-10, "v3 EWMA should differ from v2 flat: max_diff={}", max_diff);
    }

    #[test]
    fn test_mod4_transition_v3_decay() {
        // v3 with transition_decay should differ from v2
        let draws = make_test_draws(80);
        let dist_v2 = mod4_transition_predict_v2(&draws, Pool::Balls);
        let dist_v3 = mod4_transition_predict_v3(&draws, Pool::Balls, 1.0, 0.003);

        // alpha_freq=1.0 makes EWMA == last draw only, so freq effect is large
        // But transition_decay also differs
        let max_diff: f64 = dist_v2.iter().zip(dist_v3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_diff > 1e-10, "v3 with decay should differ from v2: max_diff={}", max_diff);
    }

    #[test]
    fn test_mod4_transition_v3_fallback_small() {
        // < 30 draws: v3 should fallback to v2 which falls back to v1
        let draws = make_test_draws(20);
        let dist_v2 = mod4_transition_predict_v2(&draws, Pool::Balls);
        let dist_v3 = mod4_transition_predict_v3(&draws, Pool::Balls, 0.05, 0.003);
        for (i, (&a, &b)) in dist_v2.iter().zip(dist_v3.iter()).enumerate() {
            assert!((a - b).abs() < 1e-15, "v2/v3 differ at {} for small data: {} vs {}", i, a, b);
        }
    }
}
