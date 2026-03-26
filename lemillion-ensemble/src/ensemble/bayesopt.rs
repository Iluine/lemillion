/// v19 H8: Bayesian Optimization for Ensemble Hyperparameters
///
/// Uses a Gaussian Process surrogate with Matérn 5/2 kernel and Expected Improvement
/// acquisition function to optimize ensemble hyperparameters.
///
/// Parameters optimized: T_balls, T_stars, coherence weights, online_alpha, hedge_eta, etc.

use std::collections::HashMap;

/// A single point in the hyperparameter space with its observed objective value.
#[derive(Debug, Clone)]
pub struct Observation {
    pub params: Vec<f64>,
    pub value: f64,
}

/// Gaussian Process with Matérn 5/2 kernel for surrogate modeling.
pub struct GaussianProcess {
    observations: Vec<Observation>,
    length_scales: Vec<f64>,
    signal_variance: f64,
    noise_variance: f64,
    /// Cached inverse kernel matrix (updated on each observation)
    k_inv_y: Vec<f64>,
    k_inv: Vec<Vec<f64>>,
}

impl GaussianProcess {
    pub fn new(n_dims: usize) -> Self {
        Self {
            observations: Vec::new(),
            length_scales: vec![1.0; n_dims],
            signal_variance: 1.0,
            noise_variance: 0.01,
            k_inv_y: Vec::new(),
            k_inv: Vec::new(),
        }
    }

    /// Matérn 5/2 kernel: k(r) = σ² (1 + √5·r + 5/3·r²) exp(-√5·r)
    fn matern52(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let r_sq: f64 = x1.iter().zip(x2.iter()).zip(self.length_scales.iter())
            .map(|((a, b), l)| ((a - b) / l).powi(2))
            .sum();
        let r = r_sq.sqrt();
        let sqrt5_r = 5.0_f64.sqrt() * r;
        self.signal_variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r_sq) * (-sqrt5_r).exp()
    }

    /// Add an observation and update the GP posterior.
    pub fn observe(&mut self, params: Vec<f64>, value: f64) {
        self.observations.push(Observation { params, value });
        self.update_posterior();
    }

    /// Update cached posterior (K⁻¹y and K⁻¹).
    fn update_posterior(&mut self) {
        let n = self.observations.len();
        if n == 0 { return; }

        // Build kernel matrix K + σ²_noise·I
        let mut k = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let kij = self.matern52(&self.observations[i].params, &self.observations[j].params);
                k[i][j] = kij;
                k[j][i] = kij;
            }
            k[i][i] += self.noise_variance;
        }

        // Cholesky decomposition L·L^T = K (simple implementation)
        let mut l = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0f64;
                for k_idx in 0..j {
                    sum += l[i][k_idx] * l[j][k_idx];
                }
                if i == j {
                    let val = k[i][i] - sum;
                    l[i][j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
                } else {
                    l[i][j] = if l[j][j].abs() > 1e-15 { (k[i][j] - sum) / l[j][j] } else { 0.0 };
                }
            }
        }

        // Solve L·z = y, then L^T · α = z (forward/backward substitution)
        let y: Vec<f64> = self.observations.iter().map(|o| o.value).collect();
        let mut z = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0f64;
            for j in 0..i { sum += l[i][j] * z[j]; }
            z[i] = if l[i][i].abs() > 1e-15 { (y[i] - sum) / l[i][i] } else { 0.0 };
        }
        let mut alpha = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            for j in (i + 1)..n { sum += l[j][i] * alpha[j]; }
            alpha[i] = if l[i][i].abs() > 1e-15 { (z[i] - sum) / l[i][i] } else { 0.0 };
        }

        self.k_inv_y = alpha;

        // Approximate K⁻¹ (for variance computation, use diagonal approximation)
        self.k_inv = k; // store K for variance computation
    }

    /// Predict mean and variance at a new point.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let n = self.observations.len();
        if n == 0 {
            return (0.0, self.signal_variance);
        }

        // k_star = [k(x, x_i) for each observation]
        let k_star: Vec<f64> = self.observations.iter()
            .map(|o| self.matern52(x, &o.params))
            .collect();

        // Mean = k_star^T · K⁻¹ · y
        let mean: f64 = k_star.iter().zip(self.k_inv_y.iter()).map(|(a, b)| a * b).sum();

        // Variance = k(x,x) - k_star^T · K⁻¹ · k_star (approximate)
        let k_xx = self.matern52(x, x);
        // Simple variance estimate using training set spread
        let var = (k_xx - k_star.iter().map(|k| k * k / (self.signal_variance + self.noise_variance)).sum::<f64>())
            .max(1e-6);

        (mean, var)
    }
}

/// Expected Improvement acquisition function.
/// EI(x) = (μ - f_best) · Φ(z) + σ · φ(z), where z = (μ - f_best) / σ
pub fn expected_improvement(mean: f64, variance: f64, f_best: f64) -> f64 {
    let sigma = variance.sqrt();
    if sigma < 1e-10 {
        return if mean > f_best { mean - f_best } else { 0.0 };
    }
    let z = (mean - f_best) / sigma;
    // Standard normal CDF and PDF approximations
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf(z / 2.0_f64.sqrt()));
    (mean - f_best) * big_phi + sigma * phi
}

/// Gauss error function approximation (Abramowitz & Stegun)
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

/// Hyperparameter bounds for optimization.
#[derive(Debug, Clone)]
pub struct ParamBounds {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

/// Run Bayesian optimization to find optimal hyperparameters.
/// Returns the best parameter set found after `n_iterations` evaluations.
pub fn optimize(
    bounds: &[ParamBounds],
    objective: &mut dyn FnMut(&[f64]) -> f64,
    n_iterations: usize,
    n_initial: usize,
    seed: u64,
) -> (Vec<f64>, f64) {
    let n_dims = bounds.len();
    let mut gp = GaussianProcess::new(n_dims);
    let mut rng_state = seed;

    // Simple LCG random number generator
    let mut rand_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // Initial random sampling (Latin Hypercube-like)
    for _ in 0..n_initial {
        let params: Vec<f64> = bounds.iter()
            .map(|b| b.min + rand_f64() * (b.max - b.min))
            .collect();
        let value = objective(&params);
        gp.observe(params, value);
    }

    // Bayesian optimization loop
    for _ in n_initial..n_iterations {
        let f_best = gp.observations.iter().map(|o| o.value).fold(f64::NEG_INFINITY, f64::max);

        // Find next point by maximizing EI over random candidates
        let n_candidates = 1000;
        let mut best_ei = f64::NEG_INFINITY;
        let mut best_params = vec![0.0f64; n_dims];

        for _ in 0..n_candidates {
            let params: Vec<f64> = bounds.iter()
                .map(|b| b.min + rand_f64() * (b.max - b.min))
                .collect();
            let (mean, var) = gp.predict(&params);
            let ei = expected_improvement(mean, var, f_best);
            if ei > best_ei {
                best_ei = ei;
                best_params = params;
            }
        }

        let value = objective(&best_params);
        gp.observe(best_params, value);
    }

    // Return best observed parameters
    let best = gp.observations.iter()
        .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    (best.params.clone(), best.value)
}

/// Default parameter bounds for ensemble hyperparameters.
pub fn default_ensemble_bounds() -> Vec<ParamBounds> {
    vec![
        ParamBounds { name: "t_balls".into(), min: 0.05, max: 0.50 },
        ParamBounds { name: "t_stars".into(), min: 0.15, max: 0.50 },
        ParamBounds { name: "coherence_weight".into(), min: 5.0, max: 60.0 },
        ParamBounds { name: "star_coherence_weight".into(), min: 5.0, max: 30.0 },
        ParamBounds { name: "hedge_eta".into(), min: 0.01, max: 0.30 },
        ParamBounds { name: "stability_sigma".into(), min: 0.10, max: 1.0 },
        ParamBounds { name: "redundancy_center".into(), min: 0.30, max: 0.70 },
    ]
}

/// Parameter bounds matching HyperParams fields (7 params).
pub fn hyperparams_bounds() -> Vec<ParamBounds> {
    vec![
        ParamBounds { name: "t_balls".into(), min: 0.05, max: 1.0 },
        ParamBounds { name: "t_stars".into(), min: 0.10, max: 0.50 },
        ParamBounds { name: "coherence_weight".into(), min: 5.0, max: 60.0 },
        ParamBounds { name: "star_coherence_weight".into(), min: 3.0, max: 40.0 },
        ParamBounds { name: "hedge_eta".into(), min: 0.01, max: 0.50 },
        ParamBounds { name: "joint_blend".into(), min: 0.0, max: 0.8 },
        ParamBounds { name: "k_balls".into(), min: 12.0, max: 35.0 },
    ]
}

/// Compute pair co-occurrence frequencies from draws.
fn compute_pair_freqs(draws: &[lemillion_db::models::Draw]) -> HashMap<(u8, u8), f64> {
    let mut counts: HashMap<(u8, u8), usize> = HashMap::new();
    let n = draws.len().min(500); // recent history
    for draw in &draws[..n] {
        for i in 0..5 {
            for j in (i + 1)..5 {
                let a = draw.balls[i].min(draw.balls[j]);
                let b = draw.balls[i].max(draw.balls[j]);
                *counts.entry((a, b)).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().map(|(k, v)| (k, v as f64 / n as f64)).collect()
}

/// Compute triplet co-occurrence frequencies from draws.
fn compute_triplet_freqs(draws: &[lemillion_db::models::Draw]) -> HashMap<(u8, u8, u8), f64> {
    let mut counts: HashMap<(u8, u8, u8), usize> = HashMap::new();
    let n = draws.len().min(500);
    for draw in &draws[..n] {
        for i in 0..5 {
            for j in (i + 1)..5 {
                for k in (j + 1)..5 {
                    let mut triple = [draw.balls[i], draw.balls[j], draw.balls[k]];
                    triple.sort();
                    *counts.entry((triple[0], triple[1], triple[2])).or_insert(0) += 1;
                }
            }
        }
    }
    counts.into_iter().map(|(k, v)| (k, v as f64 / n as f64)).collect()
}

/// Compute ball sum statistics from draws.
fn compute_sum_stats(draws: &[lemillion_db::models::Draw]) -> (f64, f64) {
    let sums: Vec<f64> = draws.iter().take(500)
        .map(|d| d.balls.iter().map(|&b| b as f64).sum::<f64>())
        .collect();
    if sums.is_empty() { return (127.5, 30.0); }
    let mean = sums.iter().sum::<f64>() / sums.len() as f64;
    let var = sums.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sums.len() as f64;
    (mean, var.sqrt().max(1.0))
}

/// Compute ball spread (max-min) statistics from draws.
fn compute_spread_stats(draws: &[lemillion_db::models::Draw]) -> (f64, f64) {
    let spreads: Vec<f64> = draws.iter().take(500)
        .map(|d| (d.balls.iter().max().unwrap() - d.balls.iter().min().unwrap()) as f64)
        .collect();
    if spreads.is_empty() { return (35.0, 8.0); }
    let mean = spreads.iter().sum::<f64>() / spreads.len() as f64;
    let var = spreads.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / spreads.len() as f64;
    (mean, var.sqrt().max(1.0))
}

/// Pre-computed data for one backtest draw point — everything that does NOT
/// depend on HyperParams (model distributions and hedge losses).
pub struct DrawCache {
    /// Per-model ball distributions [n_models][50]
    ball_dists: Vec<Vec<f64>>,
    /// Per-model star distributions [n_models][12]
    star_dists: Vec<Vec<f64>>,
    /// Per-model per-recent-draw ball losses for Hedge replay [n_recent][n_models]
    hedge_ball_losses: Vec<Vec<f64>>,
    /// Per-model per-recent-draw star losses for Hedge replay [n_recent][n_models]
    hedge_star_losses: Vec<Vec<f64>>,
    /// Base calibration weights
    base_ball_w: Vec<f64>,
    base_star_w: Vec<f64>,
    /// Which models are stars-only
    is_stars_only: Vec<bool>,
    /// Pair frequency scores for coherence [C(50,2) pairs]
    pair_freqs: HashMap<(u8, u8), f64>,
    /// Triplet frequency scores for coherence [C(50,3) triplets]
    triplet_freqs: HashMap<(u8, u8, u8), f64>,
    /// Historical sum/spread stats
    sum_mean: f64,
    sum_std: f64,
    spread_mean: f64,
    spread_std: f64,
    /// Star pair probabilities from StarPairModel (66 pairs)
    star_pair_probs: Option<[f64; 66]>,
}

/// Pre-computed cache for all backtest draw points.
pub struct BacktestCache {
    pub draw_caches: Vec<DrawCache>,
    pub suggestions_count: usize,
}

impl BacktestCache {
    /// Build the cache — this is the expensive step, done ONCE.
    /// Pre-computes model predictions and hedge losses for all backtest points.
    pub fn build(
        draws: &[lemillion_db::models::Draw],
        weights: &crate::ensemble::calibration::EnsembleWeights,
        last_n: usize,
        suggestions_count: usize,
    ) -> Self {
        use lemillion_db::models::Pool;
        use rayon::prelude::*;
        use crate::models::all_models;

        let last = last_n.min(draws.len().saturating_sub(1));
        let mut draw_caches = Vec::with_capacity(last);

        for i in 0..last {
            let training_draws = &draws[i + 1..];
            if training_draws.len() < 100 {
                continue;
            }

            let models = all_models();
            let n_models = models.len();

            // Base weights from calibration
            let base_ball_w: Vec<f64> = models.iter()
                .map(|m| weights.ball_weights.iter()
                    .find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                .collect();
            let base_star_w: Vec<f64> = models.iter()
                .map(|m| weights.star_weights.iter()
                    .find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                .collect();
            let is_stars_only: Vec<bool> = models.iter()
                .map(|m| m.is_stars_only()).collect();

            // Pre-compute per-model distributions on training_draws (parallel)
            let all_ball_dists: Vec<Vec<f64>> = models.par_iter()
                .map(|m| m.predict(training_draws, Pool::Balls))
                .collect();
            let all_star_dists: Vec<Vec<f64>> = models.par_iter()
                .map(|m| m.predict(training_draws, Pool::Stars))
                .collect();

            // Pre-compute hedge losses: for each recent draw, compute per-model losses
            let n_recent = 20.min(training_draws.len().saturating_sub(1));
            let floor_threshold = 0.02;
            let mut hedge_ball_losses: Vec<Vec<f64>> = Vec::with_capacity(n_recent);
            let mut hedge_star_losses: Vec<Vec<f64>> = Vec::with_capacity(n_recent);

            for t in (0..n_recent).rev() {
                let test_draw = &training_draws[t];
                let sub_training = &training_draws[t + 1..];
                if sub_training.len() < 10 {
                    hedge_ball_losses.push(vec![0.0; n_models]);
                    hedge_star_losses.push(vec![0.0; n_models]);
                    continue;
                }

                let ball_losses: Vec<f64> = models.iter().enumerate().map(|(m, model)| {
                    if base_ball_w[m] <= floor_threshold { return 0.0; }
                    let ball_ll_cap = 2.0 * (1.0_f64 / 50.0).ln();
                    let dist = model.predict(sub_training, Pool::Balls);
                    test_draw.balls.iter()
                        .map(|&b| -dist[(b - 1) as usize].max(1e-15).ln().max(ball_ll_cap))
                        .sum()
                }).collect();

                let star_losses: Vec<f64> = models.iter().enumerate().map(|(m, model)| {
                    if base_star_w[m] <= floor_threshold { return 0.0; }
                    let star_ll_cap = 2.0 * (1.0_f64 / 12.0).ln();
                    let star_uniform = 1.0 / 12.0_f64;
                    let dist = model.predict(sub_training, Pool::Stars);
                    test_draw.stars.iter()
                        .map(|&s| {
                            let p = dist[(s - 1) as usize].max(1e-15);
                            let ll = p.ln().max(star_ll_cap);
                            let loss = -ll;
                            if p < star_uniform { loss * 0.5 } else { loss }
                        })
                        .sum()
                }).collect();

                hedge_ball_losses.push(ball_losses);
                hedge_star_losses.push(star_losses);
            }

            // v21: Pre-compute coherence data for proxy fidelity
            let pair_freqs = compute_pair_freqs(training_draws);
            let triplet_freqs = compute_triplet_freqs(training_draws);
            let (sum_mean, sum_std) = compute_sum_stats(training_draws);
            let (spread_mean, spread_std) = compute_spread_stats(training_draws);

            // Star pair probs from StarPairModel if available
            let star_pair_probs = {
                let sp_model = crate::models::star_pair::StarPairModel::default();
                sp_model.predict_pair_distribution(training_draws)
            };

            draw_caches.push(DrawCache {
                ball_dists: all_ball_dists,
                star_dists: all_star_dists,
                hedge_ball_losses,
                hedge_star_losses,
                base_ball_w,
                base_star_w,
                is_stars_only,
                pair_freqs,
                triplet_freqs,
                sum_mean,
                sum_std,
                spread_mean,
                spread_std,
                star_pair_probs,
            });
        }

        BacktestCache { draw_caches, suggestions_count }
    }
}

/// Fast objective using pre-computed cache.
/// Uses analytical top-N grid scoring instead of full jackpot enumeration.
/// Parallelized across backtest points with rayon.
pub fn fast_backtest_objective(
    cache: &BacktestCache,
    params: &crate::sampler::HyperParams,
    n_grids: usize,
) -> f64 {
    use rayon::prelude::*;
    use crate::sampler::apply_temperature;

    const TOTAL_COMBINATIONS: f64 = 139_838_160.0;

    let results: Vec<f64> = cache.draw_caches.par_iter().filter_map(|dc| {
        // 1. Replay hedge weight update with this eta (fast — pure arithmetic)
        let (ball_w, star_w) = replay_hedge_weights(
            &dc.base_ball_w, &dc.base_star_w,
            &dc.hedge_ball_losses, &dc.hedge_star_losses,
            params.hedge_eta,
        );

        // 2. Log-linear pool with hedged weights (fast — no model.predict())
        let ball_dist = log_linear_pool(&dc.ball_dists, &ball_w, &dc.is_stars_only, 50, true);
        let star_dist = log_linear_pool(&dc.star_dists, &star_w, &dc.is_stars_only, 12, false);

        // 3. Apply temperature
        let bd = apply_temperature(&ball_dist, params.t_balls);
        let sd = apply_temperature(&star_dist, params.t_stars);

        // 4. Analytical top-N grid scoring v2 (fast — no enumeration, with coherence)
        let p52 = analytical_top_n_grids(
            &bd, &sd, n_grids,
            params.k_balls,
            params.coherence_weight,
            &dc.pair_freqs, &dc.triplet_freqs,
            dc.sum_mean, dc.sum_std, dc.spread_mean, dc.spread_std,
            &dc.star_pair_probs,
        );
        Some(p52 / TOTAL_COMBINATIONS)
    }).collect();

    if results.is_empty() { return 0.0; }
    results.iter().sum::<f64>() / results.len() as f64
}

/// Analytical scoring v2: includes coherence and star pair scoring.
/// Uses K from HyperParams instead of hardcoded 15.
/// Finds the top-N ball 5-subsets × top star pair, with diversity (min 2 different balls).
fn analytical_top_n_grids(
    ball_probs: &[f64],
    star_probs: &[f64],
    n_grids: usize,
    k_balls: usize,
    coherence_weight: f64,
    pair_freqs: &HashMap<(u8, u8), f64>,
    _triplet_freqs: &HashMap<(u8, u8, u8), f64>,
    sum_mean: f64,
    sum_std: f64,
    spread_mean: f64,
    spread_std: f64,
    star_pair_probs: &Option<[f64; 66]>,
) -> f64 {
    // Sort balls by probability descending
    let mut ball_idx: Vec<usize> = (0..ball_probs.len()).collect();
    ball_idx.sort_by(|&a, &b| ball_probs[b].partial_cmp(&ball_probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Sort star pairs by joint probability (marginal product)
    let mut star_pairs: Vec<([u8; 2], f64)> = Vec::with_capacity(66);
    for s1 in 0usize..11 {
        for s2 in (s1 + 1)..12usize {
            star_pairs.push(([s1 as u8 + 1, s2 as u8 + 1], star_probs[s1] * star_probs[s2]));
        }
    }
    star_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Generate top ball 5-subsets from top-K balls (K from HyperParams)
    let k = k_balls.min(ball_idx.len());
    let top = &ball_idx[..k];

    // Expected pair frequency (baseline for pair scoring)
    let expected_pair = 10.0 / (50.0 * 49.0 / 2.0);

    let mut ball_combos: Vec<([u8; 5], f64)> = Vec::new();
    for i0 in 0..k {
        for i1 in (i0 + 1)..k {
            for i2 in (i1 + 1)..k {
                for i3 in (i2 + 1)..k {
                    for i4 in (i3 + 1)..k {
                        let mut balls = [
                            top[i0] as u8 + 1,
                            top[i1] as u8 + 1,
                            top[i2] as u8 + 1,
                            top[i3] as u8 + 1,
                            top[i4] as u8 + 1,
                        ];
                        balls.sort();
                        let ball_score: f64 = [i0, i1, i2, i3, i4].iter()
                            .map(|&i| ball_probs[top[i]])
                            .product();

                        // Coherence: sum score
                        let ball_sum: f64 = balls.iter().map(|&b| b as f64).sum();
                        let z_sum = (ball_sum - sum_mean) / sum_std;
                        let sum_score = (-0.5 * z_sum * z_sum).exp();

                        // Coherence: spread score
                        let ball_spread = (balls[4] - balls[0]) as f64;
                        let z_spread = (ball_spread - spread_mean) / spread_std;
                        let spread_score = (-0.5 * z_spread * z_spread).exp();

                        // Coherence: pair score
                        let mut pair_total = 0.0;
                        let mut pair_count = 0;
                        for pi in 0..5 {
                            for pj in (pi + 1)..5 {
                                let a = balls[pi].min(balls[pj]);
                                let b = balls[pi].max(balls[pj]);
                                if let Some(&freq) = pair_freqs.get(&(a, b)) {
                                    pair_total += freq;
                                }
                                pair_count += 1;
                            }
                        }
                        let avg_pair = if pair_count > 0 { pair_total / pair_count as f64 } else { expected_pair };
                        let pair_score = (1.0 + (avg_pair / expected_pair).max(0.01).ln().max(0.0)).min(3.0) / 3.0;

                        // Combined coherence (same weights as CoherenceScorer)
                        let coherence = sum_score * 0.35 + spread_score * 0.25 + pair_score * 0.25 + 0.15;
                        let sort_score = ball_score.max(1e-300).ln() + coherence.max(1e-300).ln() * (coherence_weight / 30.0);

                        ball_combos.push((balls, sort_score));
                    }
                }
            }
        }
    }
    ball_combos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Best star score: use star_pair_probs if available, else marginal product
    let best_star_score = if let Some(spp) = star_pair_probs {
        spp.iter().cloned().fold(0.0_f64, f64::max)
    } else {
        star_pairs[0].1
    };

    let mut selected: Vec<([u8; 5], f64)> = Vec::with_capacity(n_grids);
    for &(balls, sort_score) in &ball_combos {
        if selected.len() >= n_grids { break; }
        // Diversity check: min 2 different balls from all selected
        let diverse = selected.iter().all(|(sel_balls, _)| {
            let common = balls.iter().filter(|b| sel_balls.contains(b)).count();
            common <= 3 // at most 3 common = at least 2 different
        });
        if diverse {
            selected.push((balls, sort_score));
        }
    }

    // If not enough diverse combos found, fill with remaining
    if selected.len() < n_grids {
        for &(balls, sort_score) in &ball_combos {
            if selected.len() >= n_grids { break; }
            if !selected.iter().any(|(s, _)| *s == balls) {
                selected.push((balls, sort_score));
            }
        }
    }

    // Compute combined P(5+2) = 1 - Π(1 - P_i)
    // Recover actual ball probability product for each selected grid
    let uniform_ball: f64 = (1.0_f64 / 50.0).powi(5);
    let uniform_star: f64 = (1.0_f64 / 12.0).powi(2);

    let grid_probs: Vec<f64> = selected.iter()
        .map(|&(ref balls, _sort_score)| {
            // Recompute actual ball probability product for the grid
            let actual_ball_score: f64 = balls.iter()
                .map(|&b| ball_probs[(b - 1) as usize])
                .product();
            (actual_ball_score / uniform_ball) * (best_star_score / uniform_star)
                / (139_838_160.0_f64) // normalize to actual probability
        })
        .collect();

    // Return improvement factor numerator (will be divided by TOTAL_COMBINATIONS in caller)
    // Actually return raw P(5+2) × TOTAL_COMBINATIONS
    let combined_p = 1.0 - grid_probs.iter().map(|&p| 1.0 - p).product::<f64>();

    // The caller divides by TOTAL_COMBINATIONS, so return combined_p × TOTAL_COMBINATIONS
    combined_p * 139_838_160.0
}

/// Replay Hedge weight update from pre-computed losses.
/// Mirrors compute_hedge_weights() logic but uses cached losses instead of model.predict().
fn replay_hedge_weights(
    base_ball_w: &[f64],
    base_star_w: &[f64],
    ball_losses: &[Vec<f64>],   // [n_recent][n_models]
    star_losses: &[Vec<f64>],   // [n_recent][n_models]
    eta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_models = base_ball_w.len();
    let min_weight = 0.005;
    let floor_threshold = 0.02;

    let mut ball_w = base_ball_w.to_vec();
    let mut star_w = base_star_w.to_vec();

    // Replay the hedge loop (losses already computed)
    for t in 0..ball_losses.len() {
        for m in 0..n_models {
            if ball_w[m] > floor_threshold && ball_losses[t][m] > 0.0 {
                let natural_eta = eta * ball_w[m].max(1e-6);
                ball_w[m] *= (-natural_eta * ball_losses[t][m]).exp();
            }
            if star_w[m] > floor_threshold && star_losses[t][m] > 0.0 {
                let natural_eta = eta * star_w[m].max(1e-6);
                star_w[m] *= (-natural_eta * star_losses[t][m]).exp();
            }
        }
        // Normalize after each draw
        let bs: f64 = ball_w.iter().sum();
        if bs > 0.0 { for w in &mut ball_w { *w /= bs; } }
        let ss: f64 = star_w.iter().sum();
        if ss > 0.0 { for w in &mut star_w { *w /= ss; } }
    }

    // Tsallis q=0.5 mirror descent
    let tsallis_exp = 1.0 / 1.5; // 1/(2-0.5)
    for w in ball_w.iter_mut() { *w = w.powf(tsallis_exp); }
    let bs: f64 = ball_w.iter().sum();
    if bs > 0.0 { for w in &mut ball_w { *w /= bs; } }
    for w in star_w.iter_mut() { *w = w.powf(tsallis_exp); }
    let ss: f64 = star_w.iter().sum();
    if ss > 0.0 { for w in &mut star_w { *w /= ss; } }

    // Floor for models with meaningful base weight
    for (i, w) in ball_w.iter_mut().enumerate() {
        if base_ball_w[i] > floor_threshold { *w = w.max(min_weight); }
    }
    let bs: f64 = ball_w.iter().sum();
    if bs > 0.0 { for w in &mut ball_w { *w /= bs; } }
    for (i, w) in star_w.iter_mut().enumerate() {
        if base_star_w[i] > floor_threshold { *w = w.max(min_weight); }
    }
    let ss: f64 = star_w.iter().sum();
    if ss > 0.0 { for w in &mut star_w { *w /= ss; } }

    (ball_w, star_w)
}

/// Log-linear pool: P(x) ∝ Π P_i(x)^w_i.
/// Mirrors EnsembleCombiner::predict() logic but uses pre-computed distributions.
fn log_linear_pool(
    model_dists: &[Vec<f64>],
    weights: &[f64],
    is_stars_only: &[bool],
    size: usize,
    is_balls: bool,
) -> Vec<f64> {
    // Zero out ball weights for star-only models
    let effective_w: Vec<f64> = if is_balls {
        let mut w = weights.to_vec();
        for (i, &so) in is_stars_only.iter().enumerate() {
            if so { w[i] = 0.0; }
        }
        let total: f64 = w.iter().sum();
        if total > 0.0 { w.iter_mut().for_each(|v| *v /= total); }
        w
    } else {
        weights.to_vec()
    };

    let max_w = effective_w.iter().cloned().fold(0.0f64, f64::max);
    let uniform_p = 1.0 / size as f64;
    let mut log_combined = vec![0.0f64; size];

    for (i, dist) in model_dists.iter().enumerate() {
        let w = effective_w[i];
        if w < 1e-15 { continue; }

        let model_h: f64 = dist.iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum();
        let h_ratio = model_h / (size as f64).ln();
        let w_ratio = if max_w > 1e-15 { w / max_w } else { 0.5 };
        let max_positive_log_ratio = 4.0 + 6.0 * (1.0 - h_ratio) * w_ratio;

        for j in 0..size {
            let raw_log = dist[j].max(1e-15).ln();
            let log_ratio = raw_log - uniform_p.ln();
            if log_ratio > max_positive_log_ratio {
                log_combined[j] += w * (uniform_p.ln() + max_positive_log_ratio);
            } else {
                log_combined[j] += w * raw_log;
            }
        }
    }

    // Exponentiate with max-subtraction
    let max_lc = log_combined.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut combined: Vec<f64> = log_combined.iter().map(|&lc| (lc - max_lc).exp()).collect();
    let total: f64 = combined.iter().sum();
    if total > 0.0 {
        for p in &mut combined { *p /= total; }
    }

    // Star floor
    if !is_balls {
        let floor = 1.0 / (size as f64 * 5.0);
        let mut floored = false;
        for p in combined.iter_mut() {
            if *p < floor { *p = floor; floored = true; }
        }
        if floored {
            let total: f64 = combined.iter().sum();
            if total > 0.0 { for p in &mut combined { *p /= total; } }
        }
    }

    combined
}

/// Legacy backtest objective (non-cached, for reference/testing).
pub fn backtest_objective(
    draws: &[lemillion_db::models::Draw],
    weights: &crate::ensemble::calibration::EnsembleWeights,
    params: &crate::sampler::HyperParams,
    n_grids: usize,
    last_n: usize,
    suggestions_count: usize,
) -> f64 {
    let cache = BacktestCache::build(draws, weights, last_n, suggestions_count);
    fast_backtest_objective(&cache, params, n_grids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_prior() {
        let gp = GaussianProcess::new(2);
        let (mean, var) = gp.predict(&[0.5, 0.5]);
        assert!((mean - 0.0).abs() < 1e-10, "Prior mean should be 0");
        assert!(var > 0.0, "Prior variance should be positive");
    }

    #[test]
    fn test_gp_observe_and_predict() {
        let mut gp = GaussianProcess::new(1);
        gp.observe(vec![0.0], 1.0);
        gp.observe(vec![1.0], 2.0);
        let (mean, _var) = gp.predict(&[0.5]);
        // Mean should be approximately between 1 and 2
        assert!(mean > 0.5 && mean < 2.5, "Mean at 0.5 should be ~1.5, got {}", mean);
    }

    #[test]
    fn test_expected_improvement() {
        let ei = expected_improvement(2.0, 1.0, 1.5);
        assert!(ei > 0.0, "EI should be positive when mean > f_best");

        let ei_low = expected_improvement(0.5, 0.01, 1.5);
        assert!(ei_low < ei, "EI should be lower when mean << f_best");
    }

    #[test]
    fn test_optimize_quadratic() {
        // Minimize -f(x) = -(x-0.5)² → maximum at x=0.5
        let bounds = vec![ParamBounds { name: "x".into(), min: 0.0, max: 1.0 }];
        let mut objective = |x: &[f64]| -> f64 {
            -(x[0] - 0.5).powi(2)
        };
        let (best_params, best_value) = optimize(&bounds, &mut objective, 30, 10, 42);
        assert!(best_value > -0.1, "Should find near-optimal, got {}", best_value);
        assert!((best_params[0] - 0.5).abs() < 0.3, "Should be near 0.5, got {}", best_params[0]);
    }

    #[test]
    fn test_erf_bounds() {
        assert!((erf(0.0)).abs() < 1e-8, "erf(0) ≈ 0, got {}", erf(0.0));
        assert!((erf(3.0) - 1.0).abs() < 0.01, "erf(3) ≈ 1");
        assert!((erf(-3.0) + 1.0).abs() < 0.01, "erf(-3) ≈ -1");
    }
}
