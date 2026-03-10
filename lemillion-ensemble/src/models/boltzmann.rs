use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Boltzmann MRF — Markov Random Field for ball interactions.
///
/// Models P(b1,...,b5) proportional to exp(-E) where:
///   E = sum_i h_i + sum_{i<j} J_{ij}
///
/// - h_i = bias field (from frequency deviations)
/// - J_{ij} = coupling (from pair co-occurrence excess vs expected)
///
/// Parameters learned from observed co-occurrences vs expected under independence.
/// Approximate marginalization via mean-field:
///   P_k proportional to exp(-h_k - sum_j J_{kj} * m_j)
/// where m_j = mean activation frequency.
///
/// The last draw's context adjusts mean-field expectations: numbers present
/// in the last draw get a boosted m_j (recency effect).
pub struct BoltzmannModel {
    coupling_strength: f64,
    smoothing: f64,
    min_draws: usize,
}

impl Default for BoltzmannModel {
    fn default() -> Self {
        Self {
            coupling_strength: 0.3,
            smoothing: 0.20,
            min_draws: 50,
        }
    }
}

impl ForecastModel for BoltzmannModel {
    fn name(&self) -> &str {
        "Boltzmann"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let n = pool.size();
        let k = pool.pick_count();
        let uniform = vec![1.0 / n as f64; n];

        if draws.len() < self.min_draws {
            return uniform;
        }

        let n_draws = draws.len() as f64;

        // 1. Compute frequency of each number
        let mut freq = vec![0.0f64; n];
        for d in draws {
            for &num in pool.numbers_from(d) {
                let idx = (num - 1) as usize;
                if idx < n {
                    freq[idx] += 1.0;
                }
            }
        }
        for f in &mut freq {
            *f /= n_draws;
        }

        // Expected frequency under uniform: k / n
        let p_expected = k as f64 / n as f64;

        // 2. Compute bias field h_i = log(freq_i / p_expected)
        //    Negative h_i = number appears more often = lower energy = more likely
        let mut h = vec![0.0f64; n];
        for i in 0..n {
            // Use log-ratio as bias; negative means over-represented (lower energy)
            h[i] = -(freq[i].max(1e-10) / p_expected).ln();
        }

        // 3. Compute pair co-occurrence counts (flat matrix for cache locality)
        let mut pair_counts = vec![0u32; n * n];
        for d in draws {
            let nums: Vec<usize> = pool
                .numbers_from(d)
                .iter()
                .map(|&x| (x - 1) as usize)
                .collect();
            for a in 0..nums.len() {
                for b in (a + 1)..nums.len() {
                    pair_counts[nums[a] * n + nums[b]] += 1;
                    pair_counts[nums[b] * n + nums[a]] += 1;
                }
            }
        }

        // Expected pair count under independence:
        // P(i and j both drawn) = C(n-2, k-2) / C(n, k)
        //                       = k*(k-1) / (n*(n-1))
        let p_pair = (k as f64 * (k as f64 - 1.0)) / (n as f64 * (n as f64 - 1.0));
        let expected_pair_count = n_draws * p_pair;

        // 4. Compute coupling J_{ij} from pair excess (flat matrix for cache locality)
        //    J_{ij} = -coupling_strength * (observed - expected) / expected
        //    Negative J means pair appears more often = attractive = lower energy
        let mut j_coupling = vec![0.0f64; n * n];
        if expected_pair_count > 1.0 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let obs = pair_counts[i * n + j] as f64;
                    let excess = (obs - expected_pair_count) / expected_pair_count.sqrt();
                    // Clamp to avoid extreme couplings
                    let j_val = -self.coupling_strength * excess.clamp(-5.0, 5.0) * 0.1;
                    j_coupling[i * n + j] = j_val;
                    j_coupling[j * n + i] = j_val;
                }
            }
        }

        // 5. Mean-field approximation
        //    m_j = base frequency, adjusted by last draw context
        let mut m = freq.clone();

        // Boost m_j for numbers in the last draw (recency context)
        if !draws.is_empty() {
            let last_nums: Vec<usize> = pool
                .numbers_from(&draws[0])
                .iter()
                .map(|&x| (x - 1) as usize)
                .collect();
            for &idx in &last_nums {
                if idx < n {
                    // Boost mean-field for recent numbers (persistence effect)
                    m[idx] = (m[idx] * 1.5).min(1.0);
                }
            }
        }

        // 6. Compute mean-field energy for each number
        //    E_k = h_k + sum_j J_{kj} * m_j
        //    P_k proportional to exp(-E_k)
        let mut energies = vec![0.0f64; n];
        for i in 0..n {
            let mut coupling_sum = 0.0;
            let row_offset = i * n;
            for j in 0..n {
                if j != i {
                    coupling_sum += j_coupling[row_offset + j] * m[j];
                }
            }
            energies[i] = h[i] + coupling_sum;
        }

        // Convert to probabilities via softmax
        // P proportional to exp(-E), so shift by min to avoid overflow
        let min_e = energies
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let mut probs: Vec<f64> = energies.iter().map(|&e| (-(e - min_e)).exp()).collect();

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smooth towards uniform
        let uniform_val = 1.0 / n as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
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
            ("coupling_strength".into(), self.coupling_strength),
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_boltzmann_valid_distribution() {
        let draws = make_test_draws(100);
        let model = BoltzmannModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_boltzmann_stars() {
        let draws = make_test_draws(100);
        let model = BoltzmannModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_boltzmann_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = BoltzmannModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_boltzmann_no_negative() {
        let draws = make_test_draws(100);
        let model = BoltzmannModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_boltzmann_deterministic() {
        let draws = make_test_draws(100);
        let model = BoltzmannModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Boltzmann should be deterministic");
        }
    }

    #[test]
    fn test_boltzmann_empty_draws() {
        let model = BoltzmannModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_boltzmann_large_draws() {
        let draws = make_test_draws(200);
        let model = BoltzmannModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }
}
