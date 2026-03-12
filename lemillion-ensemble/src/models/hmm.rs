use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// HMM — Hidden Markov Model with K=4 hidden states.
///
/// Corresponds to the 4 Stresa blade positions. Each hidden state has
/// a multinomial emission distribution over numbers. The model learns
/// transition and emission probabilities via Baum-Welch (EM), then
/// uses the forward algorithm to compute P(state | observations) at
/// the last timestep.
///
/// Prediction: P(num) = sum_k P(state=k | history) * P(num | state=k)
pub struct HmmModel {
    n_states: usize,
    max_iter: usize,
    smoothing: f64,
    min_draws: usize,
}

impl Default for HmmModel {
    fn default() -> Self {
        Self {
            n_states: 8, // 8 barres extérieures Stresa (ajusté par pool dans predict)
            max_iter: 20,
            smoothing: 0.12,
            min_draws: 30,
        }
    }
}

impl HmmModel {
    /// Encode a draw as a vector of binary observations for the given pool.
    /// Returns a vector of indices (0-based) of numbers present in the draw.
    fn encode_draw(draw: &Draw, pool: Pool) -> Vec<usize> {
        pool.numbers_from(draw)
            .iter()
            .map(|&n| (n - 1) as usize)
            .collect()
    }

    /// Run Baum-Welch to learn HMM parameters, then forward algorithm
    /// to get state posterior at the last timestep.
    fn run(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let k = self.n_states;
        let t_len = draws.len();

        // Convert draws to observations (chronological: oldest first)
        let observations: Vec<Vec<usize>> = draws.iter().rev().map(|d| Self::encode_draw(d, pool)).collect();

        // Initialize parameters
        // Transition matrix: slight preference for staying in same state
        let mut trans = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    trans[i][j] = 0.7;
                } else {
                    trans[i][j] = 0.3 / (k - 1) as f64;
                }
            }
        }

        // Emission probabilities: P(num | state) initialized with slight mod-4 bias
        let mut emission = vec![vec![0.0f64; n]; k];
        for s in 0..k {
            let total_weight: f64 = (0..n)
                .map(|i| if i % k == s { 1.5 } else { 1.0 })
                .sum();
            for i in 0..n {
                let w = if i % k == s { 1.5 } else { 1.0 };
                emission[s][i] = w / total_weight;
            }
        }

        // Initial state distribution (uniform)
        let mut pi = vec![1.0 / k as f64; k];

        // Baum-Welch iterations
        let pick = pool.pick_count();
        let laplace_alpha = 1.0; // Laplace smoothing parameter

        // Pre-allocate matrices for Baum-Welch (reused across iterations)
        let mut alpha = vec![vec![0.0f64; k]; t_len];
        let mut beta = vec![vec![0.0f64; k]; t_len];
        let mut gamma = vec![vec![0.0f64; k]; t_len];
        let mut xi_sum = vec![vec![0.0f64; k]; k];
        let mut xi_t = vec![vec![0.0f64; k]; k];

        // Pre-compute emission probabilities per observation per state (reused in forward/backward/xi)
        let mut emit_cache = vec![vec![0.0f64; k]; t_len];

        let old_trans_buf = vec![vec![0.0f64; k]; k];
        let old_emission_buf = vec![vec![0.0f64; n]; k];
        let mut old_trans = old_trans_buf;
        let mut old_emission = old_emission_buf;

        for _iter in 0..self.max_iter {
            // Pre-compute emission probabilities for this iteration
            for t in 0..t_len {
                for s in 0..k {
                    emit_cache[t][s] = Self::emission_prob(&emission[s], &observations[t], pick);
                }
            }

            // E-step: forward-backward
            // Forward pass: alpha[t][s] = P(o_1..o_t, state_t = s)
            for s in 0..k {
                alpha[0][s] = pi[s] * emit_cache[0][s];
            }
            Self::normalize_vec(&mut alpha[0]);

            for t in 1..t_len {
                for j in 0..k {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += alpha[t - 1][i] * trans[i][j];
                    }
                    alpha[t][j] = sum * emit_cache[t][j];
                }
                Self::normalize_vec(&mut alpha[t]);
            }

            // Backward pass: beta[t][s] = P(o_{t+1}..o_T | state_t = s)
            for s in 0..k {
                beta[t_len - 1][s] = 1.0;
            }

            for t in (0..t_len - 1).rev() {
                for i in 0..k {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += trans[i][j] * emit_cache[t + 1][j] * beta[t + 1][j];
                    }
                    beta[t][i] = sum;
                }
                Self::normalize_vec(&mut beta[t]);
            }

            // Compute gamma[t][s] = P(state_t = s | all observations)
            for t in 0..t_len {
                for s in 0..k {
                    gamma[t][s] = alpha[t][s] * beta[t][s];
                }
                Self::normalize_vec(&mut gamma[t]);
            }

            // Compute xi sums directly for M-step
            for i in 0..k {
                for j in 0..k {
                    xi_sum[i][j] = 0.0;
                }
            }
            for t in 0..t_len - 1 {
                for i in 0..k {
                    for j in 0..k {
                        xi_t[i][j] = alpha[t][i] * trans[i][j] * emit_cache[t + 1][j] * beta[t + 1][j];
                    }
                }
                // Normalize xi_t
                let xi_total: f64 = xi_t.iter().flat_map(|row| row.iter()).sum();
                if xi_total > 1e-300 {
                    for i in 0..k {
                        for j in 0..k {
                            xi_t[i][j] /= xi_total;
                            xi_sum[i][j] += xi_t[i][j];
                        }
                    }
                }
            }

            // M-step: update parameters
            // Save old values for convergence check
            for i in 0..k {
                old_trans[i][..k].copy_from_slice(&trans[i][..k]);
                old_emission[i][..n].copy_from_slice(&emission[i][..n]);
            }

            // Update pi
            for s in 0..k {
                pi[s] = gamma[0][s];
            }

            // Update transition matrix with Laplace smoothing
            for i in 0..k {
                let row_sum: f64 = xi_sum[i].iter().sum::<f64>() + laplace_alpha * k as f64;
                for j in 0..k {
                    trans[i][j] = (xi_sum[i][j] + laplace_alpha) / row_sum;
                }
            }

            // Update emission probabilities with Laplace smoothing
            for s in 0..k {
                let mut new_emit = vec![laplace_alpha; n];
                for t in 0..t_len {
                    for &obs_idx in &observations[t] {
                        if obs_idx < n {
                            new_emit[obs_idx] += gamma[t][s];
                        }
                    }
                }
                let emit_sum: f64 = new_emit.iter().sum();
                if emit_sum > 0.0 {
                    for e in &mut new_emit {
                        *e /= emit_sum;
                    }
                }
                emission[s] = new_emit;
            }

            // Check convergence
            let mut max_diff = 0.0f64;
            for i in 0..k {
                for j in 0..k {
                    max_diff = max_diff.max((trans[i][j] - old_trans[i][j]).abs());
                }
                for j in 0..n {
                    max_diff = max_diff.max((emission[i][j] - old_emission[i][j]).abs());
                }
            }
            if max_diff < 1e-6 {
                break;
            }
        }

        // Final forward pass to get state posterior at last timestep
        let mut alpha_final = vec![vec![0.0f64; k]; t_len];
        for s in 0..k {
            alpha_final[0][s] = pi[s] * Self::emission_prob(&emission[s], &observations[0], pick);
        }
        Self::normalize_vec(&mut alpha_final[0]);

        for t in 1..t_len {
            for j in 0..k {
                let mut sum = 0.0;
                for i in 0..k {
                    sum += alpha_final[t - 1][i] * trans[i][j];
                }
                alpha_final[t][j] = sum * Self::emission_prob(&emission[j], &observations[t], pick);
            }
            Self::normalize_vec(&mut alpha_final[t]);
        }

        // State posterior at last timestep
        let state_posterior = &alpha_final[t_len - 1];

        // Predicted next state distribution: P(next_state) = sum_s P(s|data) * P(next_state | s)
        let mut next_state = vec![0.0f64; k];
        for s in 0..k {
            for ns in 0..k {
                next_state[ns] += state_posterior[s] * trans[s][ns];
            }
        }

        // Predict: P(num) = sum_k P(next_state=k) * P(num | state=k)
        let mut probs = vec![0.0f64; n];
        for num in 0..n {
            for s in 0..k {
                probs[num] += next_state[s] * emission[s][num];
            }
        }

        probs
    }

    /// Compute emission probability for a set of observed numbers given state emission probs.
    /// P(obs | state) = product of emission[obs_i] (multinomial-like, ignoring normalization constant)
    fn emission_prob(emission: &[f64], obs: &[usize], _pick: usize) -> f64 {
        let mut p = 1.0f64;
        for &idx in obs {
            if idx < emission.len() {
                p *= emission[idx].max(1e-300);
            }
        }
        p.max(1e-300)
    }

    /// Normalize a vector to sum to 1.
    fn normalize_vec(v: &mut [f64]) {
        let sum: f64 = v.iter().sum();
        if sum > 1e-300 {
            for x in v.iter_mut() {
                *x /= sum;
            }
        }
    }
}

impl ForecastModel for HmmModel {
    fn name(&self) -> &str {
        "HMM"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        if draws.len() < self.min_draws {
            return vec![1.0 / n as f64; n];
        }

        // Pool-aware: 8 states for balls (Stresa bars), 4 for stars (Pâquerette blades)
        let effective_states = match pool {
            Pool::Balls => self.n_states,
            Pool::Stars => 4,
        };
        let model_eff = Self {
            n_states: effective_states,
            max_iter: self.max_iter,
            smoothing: self.smoothing,
            min_draws: self.min_draws,
        };
        let mut probs = model_eff.run(draws, pool);

        // Smooth towards uniform
        let uniform = 1.0 / n as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
        }

        let floor = if pool == Pool::Balls {
            PROB_FLOOR_BALLS
        } else {
            PROB_FLOOR_STARS
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_states".into(), self.n_states as f64),
            ("max_iter".into(), self.max_iter as f64),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn calibration_stride(&self) -> usize {
        2 // Baum-Welch is expensive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_hmm_valid_distribution() {
        let draws = make_test_draws(100);
        let model = HmmModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_hmm_stars() {
        let draws = make_test_draws(100);
        let model = HmmModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_hmm_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = HmmModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hmm_no_negative() {
        let draws = make_test_draws(100);
        let model = HmmModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_hmm_deterministic() {
        let draws = make_test_draws(50);
        let model = HmmModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-10, "HMM should be deterministic");
        }
    }

    #[test]
    fn test_hmm_empty_draws() {
        let model = HmmModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hmm_emission_prob() {
        let emission = vec![0.1, 0.2, 0.3, 0.4];
        let obs = vec![0, 2];
        let p = HmmModel::emission_prob(&emission, &obs, 2);
        assert!((p - 0.1 * 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_hmm_normalize_vec() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        HmmModel::normalize_vec(&mut v);
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((v[0] - 0.1).abs() < 1e-10);
    }
}
