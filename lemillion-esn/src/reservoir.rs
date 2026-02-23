use ndarray::{Array1, Array2};
use rand::{Rng, RngExt};
use rand::distr::Uniform;

pub struct Reservoir {
    pub w_in: Array2<f64>,
    pub w_res: Array2<f64>,
    pub state: Array1<f64>,
    pub leaking_rate: f64,
    pub noise_amplitude: f64,
}

impl Reservoir {
    pub fn new(
        input_dim: usize,
        reservoir_size: usize,
        spectral_radius: f64,
        sparsity: f64,
        input_scaling: f64,
        leaking_rate: f64,
        noise_amplitude: f64,
        rng: &mut impl Rng,
    ) -> Self {
        // W_in: dense, Uniform[-input_scaling, +input_scaling]
        let dist_in = Uniform::new(-input_scaling, input_scaling).unwrap();
        let w_in = Array2::from_shape_fn((reservoir_size, input_dim), |_| rng.sample(dist_in));

        // W_res: sparse (sparsity fraction of zeros), non-zeros Uniform[-1, 1]
        let dist_res = Uniform::new(-1.0, 1.0).unwrap();
        let mut w_res =
            Array2::from_shape_fn((reservoir_size, reservoir_size), |_| {
                if rng.random::<f64>() < sparsity {
                    0.0
                } else {
                    rng.sample(dist_res)
                }
            });

        // Scale W_res by spectral_radius / estimated_rho
        let rho = power_iteration(&w_res, rng);
        if rho > 1e-10 {
            w_res.mapv_inplace(|x| x * spectral_radius / rho);
        }

        let state = Array1::zeros(reservoir_size);

        Reservoir {
            w_in,
            w_res,
            state,
            leaking_rate,
            noise_amplitude,
        }
    }

    /// Single reservoir step: h(t) = (1-a)*h(t-1) + a*tanh(W_in*x + W_res*h(t-1)) + noise
    pub fn step(&mut self, input: &Array1<f64>, rng: &mut impl Rng) {
        let pre = self.w_in.dot(input) + self.w_res.dot(&self.state);
        let activated = pre.mapv(|x| x.tanh());
        let a = self.leaking_rate;

        self.state = &self.state * (1.0 - a) + &activated * a;

        // Add noise
        if self.noise_amplitude > 0.0 {
            let noise_dist = Uniform::new(-self.noise_amplitude, self.noise_amplitude).unwrap();
            for v in self.state.iter_mut() {
                *v += rng.sample(noise_dist);
            }
        }
    }

    /// Run the reservoir over a sequence of inputs, returning all states.
    pub fn run_sequence(
        &mut self,
        inputs: &[Array1<f64>],
        rng: &mut impl Rng,
    ) -> Vec<Array1<f64>> {
        inputs.iter().map(|x| {
            self.step(x, rng);
            self.state.clone()
        }).collect()
    }

    /// Reset the reservoir state to zeros.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }
}

/// Estimate the spectral radius of a matrix via power iteration.
pub fn power_iteration(w: &Array2<f64>, rng: &mut impl Rng) -> f64 {
    let n = w.nrows();
    if n == 0 {
        return 0.0;
    }

    // Random unit vector
    let dist = Uniform::new(-1.0, 1.0).unwrap();
    let mut v: Array1<f64> = Array1::from_shape_fn(n, |_| rng.sample(dist));
    let norm = v.dot(&v).sqrt();
    if norm < 1e-15 {
        return 0.0;
    }
    v /= norm;

    let mut lambda = 0.0;

    for _ in 0..1000 {
        let w_v: Array1<f64> = w.dot(&v);
        let new_lambda = w_v.dot(&w_v).sqrt();
        if new_lambda < 1e-15 {
            return 0.0;
        }
        v = &w_v / new_lambda;

        if (new_lambda - lambda).abs() < 1e-10 {
            return new_lambda;
        }
        lambda = new_lambda;
    }

    lambda
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_matrix_shapes() {
        let mut rng = StdRng::seed_from_u64(42);
        let r = Reservoir::new(62, 100, 0.95, 0.9, 0.1, 0.3, 1e-4, &mut rng);
        assert_eq!(r.w_in.shape(), &[100, 62]);
        assert_eq!(r.w_res.shape(), &[100, 100]);
        assert_eq!(r.state.len(), 100);
    }

    #[test]
    fn test_spectral_radius_within_tolerance() {
        let mut rng = StdRng::seed_from_u64(42);
        let target_rho = 0.95;
        let r = Reservoir::new(10, 50, target_rho, 0.8, 0.1, 0.3, 0.0, &mut rng);
        let rho = power_iteration(&r.w_res, &mut rng);
        assert!(
            (rho - target_rho).abs() / target_rho < 0.05,
            "rho={rho}, target={target_rho}, error={}%",
            ((rho - target_rho).abs() / target_rho) * 100.0
        );
    }

    #[test]
    fn test_sparsity() {
        let mut rng = StdRng::seed_from_u64(42);
        let sparsity = 0.9;
        let r = Reservoir::new(10, 200, 0.95, sparsity, 0.1, 0.3, 0.0, &mut rng);
        let total = (200 * 200) as f64;
        let zeros = r.w_res.iter().filter(|&&x| x == 0.0).count() as f64;
        let actual_sparsity = zeros / total;
        // Should be close to target sparsity (before rescaling, zeros stay zero)
        assert!(
            (actual_sparsity - sparsity).abs() < 0.05,
            "actual_sparsity={actual_sparsity}, target={sparsity}"
        );
    }

    #[test]
    fn test_deterministic_seed() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let r1 = Reservoir::new(10, 50, 0.95, 0.9, 0.1, 0.3, 0.0, &mut rng1);
        let mut rng2 = StdRng::seed_from_u64(42);
        let r2 = Reservoir::new(10, 50, 0.95, 0.9, 0.1, 0.3, 0.0, &mut rng2);

        assert_eq!(r1.w_in, r2.w_in);
        assert_eq!(r1.w_res, r2.w_res);
    }

    #[test]
    fn test_leaking_rate_zero_no_update() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut r = Reservoir::new(10, 20, 0.95, 0.5, 0.1, 0.0, 0.0, &mut rng);
        let input = Array1::ones(10);
        r.step(&input, &mut rng);
        // With alpha=0 and noise=0: h(t) = h(t-1) = 0
        for &v in r.state.iter() {
            assert!(v.abs() < 1e-10, "state should remain zero with alpha=0");
        }
    }

    #[test]
    fn test_leaking_rate_one_full_update() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut r = Reservoir::new(10, 20, 0.95, 0.5, 0.1, 1.0, 0.0, &mut rng);
        let input = Array1::ones(10);
        r.step(&input, &mut rng);
        // With alpha=1 and noise=0: h(t) = tanh(W_in*x + W_res*0) = tanh(W_in*x)
        let expected = r.w_in.dot(&input).mapv(|x| x.tanh());
        for i in 0..20 {
            assert!(
                (r.state[i] - expected[i]).abs() < 1e-10,
                "state[{i}]={}, expected={}",
                r.state[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_run_sequence_length() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut r = Reservoir::new(10, 20, 0.95, 0.9, 0.1, 0.3, 0.0, &mut rng);
        let inputs: Vec<Array1<f64>> = (0..5).map(|_| Array1::ones(10)).collect();
        let states = r.run_sequence(&inputs, &mut rng);
        assert_eq!(states.len(), 5);
        for s in &states {
            assert_eq!(s.len(), 20);
        }
    }

    #[test]
    fn test_reset_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut r = Reservoir::new(10, 20, 0.95, 0.9, 0.1, 0.3, 0.0, &mut rng);
        let input = Array1::ones(10);
        r.step(&input, &mut rng);
        assert!(r.state.iter().any(|&v| v.abs() > 1e-10));
        r.reset_state();
        for &v in r.state.iter() {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn test_power_iteration_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let eye = Array2::eye(5);
        let rho = power_iteration(&eye, &mut rng);
        assert!(
            (rho - 1.0).abs() < 0.01,
            "rho of identity should be 1.0, got {rho}"
        );
    }

    #[test]
    fn test_power_iteration_scaled() {
        let mut rng = StdRng::seed_from_u64(42);
        let scaled = Array2::eye(5) * 3.0;
        let rho = power_iteration(&scaled, &mut rng);
        assert!(
            (rho - 3.0).abs() < 0.01,
            "rho of 3*I should be 3.0, got {rho}"
        );
    }
}
