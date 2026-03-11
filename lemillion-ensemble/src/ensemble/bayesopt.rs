/// v19 H8: Bayesian Optimization for Ensemble Hyperparameters
///
/// Uses a Gaussian Process surrogate with Matérn 5/2 kernel and Expected Improvement
/// acquisition function to optimize ensemble hyperparameters.
///
/// Parameters optimized: T_balls, T_stars, coherence weights, online_alpha, hedge_eta, etc.

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
    objective: &dyn Fn(&[f64]) -> f64,
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
        let objective = |x: &[f64]| -> f64 {
            -(x[0] - 0.5).powi(2)
        };
        let (best_params, best_value) = optimize(&bounds, &objective, 30, 10, 42);
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
