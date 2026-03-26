use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::RngExt;

use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Particle representing a hypothetical state of the Stresa machine.
///
/// State: [row_bias[5], persistence, noise_level] = 7 dimensions.
/// Row bias models the 5 decade rows (1-10, 11-20, ..., 41-50).
struct Particle {
    row_bias: [f64; 5],
    persistence: f64,
    noise: f64,
    weight: f64,
}

impl Particle {
    fn from_prior(rng: &mut SmallRng) -> Self {
        let mut row_bias = [0.0f64; 5];
        for b in &mut row_bias {
            *b = (rng.random::<f64>() - 0.5) * 0.2;
        }
        Self {
            row_bias,
            persistence: rng.random::<f64>() * 0.5,
            noise: 0.8 + rng.random::<f64>() * 0.8, // [0.8, 1.6]
            weight: 1.0,
        }
    }

    /// Compute ball distribution given current particle state.
    fn ball_distribution(&self, prev_balls: Option<&[u8; 5]>) -> Vec<f64> {
        let mut logits = vec![0.0f64; 50];
        for k in 0..50 {
            let row = k / 10;
            logits[k] += self.row_bias[row];
            if let Some(prev) = prev_balls {
                if prev.contains(&((k + 1) as u8)) {
                    logits[k] += self.persistence * 0.3;
                }
            }
        }
        // softmax with temperature = self.noise
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = logits.iter()
            .map(|&l| ((l - max_l) / self.noise).exp())
            .collect();
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            for p in &mut probs {
                *p /= total;
            }
        }
        probs
    }

    /// Transition: jitter + mean-reversion towards prior.
    fn transition(&mut self, process_noise: f64, mean_reversion: f64, rng: &mut SmallRng) {
        for b in &mut self.row_bias {
            *b += (rng.random::<f64>() - 0.5) * process_noise * 2.0;
            *b *= 1.0 - mean_reversion; // mean-revert towards 0
        }
        self.persistence += (rng.random::<f64>() - 0.5) * process_noise;
        self.persistence = self.persistence.clamp(0.0, 1.0);
        self.noise += (rng.random::<f64>() - 0.5) * process_noise * 0.5;
        self.noise = self.noise.clamp(0.5, 2.0);
        self.weight = 1.0;
    }
}

/// Systematic resampling for particle filter.
fn systematic_resample(particles: &[Particle], rng: &mut SmallRng) -> Vec<Particle> {
    let n = particles.len();
    let u: f64 = rng.random::<f64>() / n as f64;

    let mut new_particles = Vec::with_capacity(n);
    let mut cumsum = 0.0f64;
    let mut j = 0;

    for i in 0..n {
        let threshold = u + i as f64 / n as f64;
        while cumsum + particles[j].weight < threshold && j + 1 < n {
            cumsum += particles[j].weight;
            j += 1;
        }
        new_particles.push(Particle {
            row_bias: particles[j].row_bias,
            persistence: particles[j].persistence,
            noise: particles[j].noise,
            weight: 1.0 / n as f64,
        });
    }

    new_particles
}

/// Simple EWMA fallback for stars.
fn star_ewma(draws: &[Draw], smoothing: f64) -> Vec<f64> {
    let size = 12;
    let uniform = 1.0 / size as f64;
    if draws.len() < 10 {
        return vec![uniform; size];
    }

    let alpha = 0.05;
    let mut freq = vec![0.0f64; size];
    let mut total = 0.0f64;
    for (t, draw) in draws.iter().enumerate() {
        let w = (-alpha * t as f64).exp();
        for &s in &draw.stars {
            freq[(s - 1) as usize] += w;
        }
        total += w;
    }
    if total > 0.0 {
        for f in &mut freq { *f /= total; }
    }
    for f in &mut freq {
        *f = *f * (1.0 - smoothing) + uniform * smoothing;
    }
    floor_only(&mut freq, PROB_FLOOR_STARS);
    freq
}

/// ParticleStresaModel — Bayesian particle filter for the Stresa machine state.
///
/// Maintains a distribution over the hidden state of the Stresa machine
/// (row biases, persistence, noise level) using Sequential Monte Carlo.
/// Replaces point-estimate SGD with a full posterior.
pub struct ParticleStresaModel {
    n_particles: usize,
    process_noise: f64,
    mean_reversion: f64,
    smoothing: f64,
    min_draws: usize,
    seed: u64,
}

impl Default for ParticleStresaModel {
    fn default() -> Self {
        Self {
            n_particles: 200,
            process_noise: 0.015,
            mean_reversion: 0.02,
            smoothing: 0.25,
            min_draws: 30,
            seed: 42,
        }
    }
}

impl ForecastModel for ParticleStresaModel {
    fn name(&self) -> &str {
        "ParticleStresa"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        if pool == Pool::Stars {
            return star_ewma(draws, self.smoothing);
        }

        let size = pool.size(); // 50
        let uniform = 1.0 / size as f64;
        if draws.len() < self.min_draws {
            return vec![uniform; size];
        }

        // 1. Init particles from prior
        let mut rng = SmallRng::seed_from_u64(self.seed);
        let mut particles: Vec<Particle> = (0..self.n_particles)
            .map(|_| Particle::from_prior(&mut rng))
            .collect();

        // 2. Walk-forward: oldest → newest (draws is newest-first)
        let n = draws.len();
        for t in (0..n).rev() {
            let draw = &draws[t];
            let prev = if t + 1 < n { Some(&draws[t + 1]) } else { None };
            let prev_balls = prev.map(|d| &d.balls);

            // 2a. Observation update: P(draw | particle)
            for p in &mut particles {
                let dist = p.ball_distribution(prev_balls);
                let ll: f64 = draw.balls.iter()
                    .map(|&b| dist[(b - 1) as usize].max(1e-15).ln())
                    .sum();
                p.weight *= ll.exp();
            }

            // 2b. Normaliser
            let total_w: f64 = particles.iter().map(|p| p.weight).sum();
            if total_w > 0.0 {
                for p in &mut particles {
                    p.weight /= total_w;
                }
            } else {
                // Degenerate: reset
                let uw = 1.0 / self.n_particles as f64;
                for p in &mut particles {
                    p.weight = uw;
                }
            }

            // 2c. ESS check + systematic resampling
            let ess = 1.0 / particles.iter().map(|p| p.weight.powi(2)).sum::<f64>();
            if ess < self.n_particles as f64 / 2.0 {
                particles = systematic_resample(&particles, &mut rng);
            }

            // 2d. Transition: jitter + mean-reversion
            for p in &mut particles {
                p.transition(self.process_noise, self.mean_reversion, &mut rng);
            }
        }

        // 3. Prédiction: moyenne pondérée des distributions
        let prev_balls = Some(&draws[0].balls);
        let mut probs = vec![0.0f64; size];
        for p in &particles {
            let dist = p.ball_distribution(prev_balls);
            for (i, &prob) in dist.iter().enumerate() {
                probs[i] += p.weight * prob;
            }
        }

        // 4. Smooth + normalize
        for p in &mut probs {
            *p = *p * (1.0 - self.smoothing) + uniform * self.smoothing;
        }
        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("n_particles".to_string(), self.n_particles as f64);
        m.insert("process_noise".to_string(), self.process_noise);
        m.insert("mean_reversion".to_string(), self.mean_reversion);
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("seed".to_string(), self.seed as f64);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 4 }
    }

    fn calibration_stride(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_particle_stresa_valid_distribution() {
        let model = ParticleStresaModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
        assert!(dist.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_particle_stresa_insufficient_data() {
        let model = ParticleStresaModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9);
        }
    }

    #[test]
    fn test_particle_stresa_deterministic_seed() {
        let model = ParticleStresaModel::default();
        let draws = make_test_draws(60);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15, "Same seed should be deterministic");
        }
    }

    #[test]
    fn test_particle_stresa_detects_row_bias() {
        // Draws heavily biased towards decade 1-10
        let draws: Vec<Draw> = (0..80).map(|i| {
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [1, 3, 5, 7, 9], // all in first decade
                stars: [1, 2],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
        prize_tiers: None,
            }
        }).collect();

        let model = ParticleStresaModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // First decade should have higher average probability
        let decade1_avg: f64 = dist[0..10].iter().sum::<f64>() / 10.0;
        let decade5_avg: f64 = dist[40..50].iter().sum::<f64>() / 10.0;
        assert!(decade1_avg > decade5_avg,
            "Decade 1 should be favored: {} vs {}", decade1_avg, decade5_avg);
    }

    #[test]
    fn test_particle_stresa_stars_valid() {
        let model = ParticleStresaModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_particle_stresa_sampling_strategy() {
        let model = ParticleStresaModel::default();
        assert!(matches!(model.sampling_strategy(), SamplingStrategy::Sparse { .. }));
        assert_eq!(model.calibration_stride(), 2);
    }
}
