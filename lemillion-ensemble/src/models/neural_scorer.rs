use serde::{Deserialize, Serialize};
use lemillion_db::models::Draw;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::prelude::Distribution;

/// Poids du réseau de neurones 3 couches (72→32→16→1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralScorerWeights {
    pub w1: Vec<Vec<f64>>,  // [32][72]
    pub b1: Vec<f64>,       // [32]
    pub w2: Vec<Vec<f64>>,  // [16][32]
    pub b2: Vec<f64>,       // [16]
    pub w3: Vec<f64>,       // [16]
    pub b3: f64,
}

/// Scorer neuronal combinatoire pour les grilles EuroMillions.
pub struct NeuralScorer {
    pub weights: NeuralScorerWeights,
}

const INPUT_DIM: usize = 72;
const HIDDEN1: usize = 32;
const HIDDEN2: usize = 16;

/// Encode une grille (5 boules + 2 étoiles) en vecteur 72-dim.
/// 50-dim one-hot balls + 12-dim one-hot stars + 10 features engineered.
pub fn encode(balls: &[u8; 5], stars: &[u8; 2]) -> Vec<f64> {
    let mut input = vec![0.0f64; INPUT_DIM];

    // One-hot balls (dim 0..49)
    for &b in balls {
        input[(b - 1) as usize] = 1.0;
    }

    // One-hot stars (dim 50..61)
    for &s in stars {
        input[49 + s as usize] = 1.0;
    }

    // Engineered features (dim 62..71)
    let mut sorted_balls = *balls;
    sorted_balls.sort();

    // sum_norm
    let sum: f64 = balls.iter().map(|&b| b as f64).sum();
    input[62] = (sum - 15.0) / (240.0 - 15.0);

    // spread_norm
    let spread = (sorted_balls[4] - sorted_balls[0]) as f64;
    input[63] = spread / 49.0;

    // odd_ratio
    let odd_count = balls.iter().filter(|&&b| b % 2 == 1).count() as f64;
    input[64] = odd_count / 5.0;

    // max_gap_norm
    let mut max_gap = 0u8;
    for w in sorted_balls.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    input[65] = max_gap as f64 / 49.0;

    // mod4 distribution [0..3] → 2 features (first 2 classes normalized)
    let mut mod4 = [0.0f64; 4];
    for &b in balls {
        mod4[((b - 1) % 4) as usize] += 1.0;
    }
    input[66] = mod4[0] / 5.0;
    input[67] = mod4[1] / 5.0;

    // decade_spread : number of distinct decades
    let mut decades = [false; 5];
    for &b in balls {
        decades[((b - 1) / 10) as usize] = true;
    }
    input[68] = decades.iter().filter(|&&d| d).count() as f64 / 5.0;

    // star_sum_norm
    let star_sum: f64 = stars.iter().map(|&s| s as f64).sum();
    input[69] = (star_sum - 3.0) / (23.0 - 3.0);

    // consecutive_ratio : fraction of consecutive pairs
    let consec = sorted_balls.windows(2).filter(|w| w[1] == w[0] + 1).count() as f64;
    input[70] = consec / 4.0;

    // mean_gap_norm
    let gap_sum: f64 = sorted_balls.windows(2).map(|w| (w[1] - w[0]) as f64).sum();
    input[71] = (gap_sum / 4.0) / 49.0;

    input
}

impl NeuralScorer {
    /// Forward pass : retourne un score [0, 1].
    pub fn score(&self, balls: &[u8; 5], stars: &[u8; 2]) -> f64 {
        let input = encode(balls, stars);
        let w = &self.weights;

        // Layer 1: Linear(72→32) + ReLU
        let mut h1 = vec![0.0f64; HIDDEN1];
        for i in 0..HIDDEN1 {
            let mut sum = w.b1[i];
            for j in 0..INPUT_DIM {
                sum += w.w1[i][j] * input[j];
            }
            h1[i] = sum.max(0.0); // ReLU
        }

        // Layer 2: Linear(32→16) + ReLU
        let mut h2 = vec![0.0f64; HIDDEN2];
        for i in 0..HIDDEN2 {
            let mut sum = w.b2[i];
            for j in 0..HIDDEN1 {
                sum += w.w2[i][j] * h1[j];
            }
            h2[i] = sum.max(0.0); // ReLU
        }

        // Output: Linear(16→1) + Sigmoid
        let mut out = w.b3;
        for j in 0..HIDDEN2 {
            out += w.w3[j] * h2[j];
        }

        sigmoid(out)
    }

    /// Entraîne le scorer par walk-forward.
    /// Positifs = tirages réels, négatifs = grilles aléatoires.
    pub fn train(draws: &[Draw], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = xavier_init(&mut rng);

        let lr = 0.005;
        let l2 = 1e-4;
        let n_epochs = 15;
        let n_negatives = 50;

        // Prepare training data
        let n_train = draws.len().saturating_sub(10);
        if n_train < 20 {
            return Self { weights };
        }

        for _epoch in 0..n_epochs {
            for t in 0..n_train {
                let draw = &draws[t];
                let balls = &draw.balls;
                let stars = &draw.stars;

                // Positive example
                let input_pos = encode(balls, stars);
                backprop(&mut weights, &input_pos, 1.0, lr, l2);

                // Negative examples: random grids
                for _ in 0..n_negatives {
                    let (neg_balls, neg_stars) = random_grid(&mut rng);
                    let input_neg = encode(&neg_balls, &neg_stars);
                    backprop(&mut weights, &input_neg, 0.0, lr, l2);
                }
            }
        }

        Self { weights }
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string(&self.weights)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let weights: NeuralScorerWeights = serde_json::from_str(&json)?;
        Ok(Self { weights })
    }
}

fn rng_uniform(rng: &mut StdRng) -> f64 {
    let dist = rand::distr::StandardUniform;
    let val: f64 = dist.sample(rng);
    val
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Xavier initialization for weights.
fn xavier_init(rng: &mut StdRng) -> NeuralScorerWeights {
    let scale1 = (2.0 / (INPUT_DIM + HIDDEN1) as f64).sqrt();
    let w1: Vec<Vec<f64>> = (0..HIDDEN1)
        .map(|_| (0..INPUT_DIM).map(|_| (rng_uniform(rng) * 2.0 - 1.0) * scale1).collect())
        .collect();
    let b1 = vec![0.0; HIDDEN1];

    let scale2 = (2.0 / (HIDDEN1 + HIDDEN2) as f64).sqrt();
    let w2: Vec<Vec<f64>> = (0..HIDDEN2)
        .map(|_| (0..HIDDEN1).map(|_| (rng_uniform(rng) * 2.0 - 1.0) * scale2).collect())
        .collect();
    let b2 = vec![0.0; HIDDEN2];

    let scale3 = (2.0 / (HIDDEN2 + 1) as f64).sqrt();
    let w3: Vec<f64> = (0..HIDDEN2).map(|_| (rng_uniform(rng) * 2.0 - 1.0) * scale3).collect();
    let b3 = 0.0;

    NeuralScorerWeights { w1, b1, w2, b2, w3, b3 }
}

/// Backpropagation pour une seule donnée (BCE loss, SGD).
fn backprop(w: &mut NeuralScorerWeights, input: &[f64], target: f64, lr: f64, l2: f64) {
    // Forward pass with intermediate values
    let mut z1 = vec![0.0f64; HIDDEN1];
    let mut h1 = vec![0.0f64; HIDDEN1];
    for i in 0..HIDDEN1 {
        let mut sum = w.b1[i];
        for j in 0..INPUT_DIM {
            sum += w.w1[i][j] * input[j];
        }
        z1[i] = sum;
        h1[i] = sum.max(0.0);
    }

    let mut z2 = vec![0.0f64; HIDDEN2];
    let mut h2 = vec![0.0f64; HIDDEN2];
    for i in 0..HIDDEN2 {
        let mut sum = w.b2[i];
        for j in 0..HIDDEN1 {
            sum += w.w2[i][j] * h1[j];
        }
        z2[i] = sum;
        h2[i] = sum.max(0.0);
    }

    let mut z3 = w.b3;
    for j in 0..HIDDEN2 {
        z3 += w.w3[j] * h2[j];
    }
    let output = sigmoid(z3);

    // BCE gradient at output
    let d_output = output - target; // dL/dz3

    // Gradients for layer 3
    for j in 0..HIDDEN2 {
        w.w3[j] -= lr * (d_output * h2[j] + l2 * w.w3[j]);
    }
    w.b3 -= lr * d_output;

    // Gradients for layer 2
    let mut d_h2 = vec![0.0f64; HIDDEN2];
    for j in 0..HIDDEN2 {
        d_h2[j] = d_output * w.w3[j];
    }
    // ReLU derivative
    let mut d_z2 = vec![0.0f64; HIDDEN2];
    for i in 0..HIDDEN2 {
        d_z2[i] = if z2[i] > 0.0 { d_h2[i] } else { 0.0 };
    }

    for i in 0..HIDDEN2 {
        for j in 0..HIDDEN1 {
            w.w2[i][j] -= lr * (d_z2[i] * h1[j] + l2 * w.w2[i][j]);
        }
        w.b2[i] -= lr * d_z2[i];
    }

    // Gradients for layer 1
    let mut d_h1 = vec![0.0f64; HIDDEN1];
    for j in 0..HIDDEN1 {
        for i in 0..HIDDEN2 {
            d_h1[j] += d_z2[i] * w.w2[i][j];
        }
    }
    let mut d_z1 = vec![0.0f64; HIDDEN1];
    for i in 0..HIDDEN1 {
        d_z1[i] = if z1[i] > 0.0 { d_h1[i] } else { 0.0 };
    }

    for i in 0..HIDDEN1 {
        for j in 0..INPUT_DIM {
            w.w1[i][j] -= lr * (d_z1[i] * input[j] + l2 * w.w1[i][j]);
        }
        w.b1[i] -= lr * d_z1[i];
    }
}

/// Génère une grille aléatoire (5 boules + 2 étoiles).
fn random_grid(rng: &mut StdRng) -> ([u8; 5], [u8; 2]) {
    use rand::distr::Uniform;

    let ball_dist = Uniform::new(1u8, 51).unwrap();
    let star_dist = Uniform::new(1u8, 13).unwrap();

    let mut balls = [0u8; 5];
    let mut i = 0;
    while i < 5 {
        let b = ball_dist.sample(rng);
        if !balls[..i].contains(&b) {
            balls[i] = b;
            i += 1;
        }
    }
    balls.sort();

    let mut stars = [0u8; 2];
    stars[0] = star_dist.sample(rng);
    loop {
        let s = star_dist.sample(rng);
        if s != stars[0] {
            stars[1] = s;
            break;
        }
    }
    stars.sort();

    (balls, stars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_encode_dimensions() {
        let balls = [1, 10, 20, 30, 50];
        let stars = [3, 7];
        let encoded = encode(&balls, &stars);
        assert_eq!(encoded.len(), INPUT_DIM);
    }

    #[test]
    fn test_encode_one_hot_correct() {
        let balls = [1, 10, 20, 30, 50];
        let stars = [3, 7];
        let encoded = encode(&balls, &stars);
        // Ball 1 → index 0
        assert_eq!(encoded[0], 1.0);
        // Ball 10 → index 9
        assert_eq!(encoded[9], 1.0);
        // Ball 50 → index 49
        assert_eq!(encoded[49], 1.0);
        // Star 3 → index 52
        assert_eq!(encoded[52], 1.0);
        // Non-selected ball should be 0
        assert_eq!(encoded[1], 0.0);
    }

    #[test]
    fn test_scorer_output_range() {
        let draws = make_test_draws(50);
        let scorer = NeuralScorer::train(&draws, 42);
        let score = scorer.score(&[1, 10, 20, 30, 50], &[3, 7]);
        assert!(score >= 0.0 && score <= 1.0, "Score should be in [0,1], got {}", score);
    }

    #[test]
    fn test_scorer_save_load() {
        let draws = make_test_draws(50);
        let scorer = NeuralScorer::train(&draws, 42);

        let path = std::path::PathBuf::from("/tmp/test_neural_scorer.json");
        scorer.save(&path).unwrap();
        let loaded = NeuralScorer::load(&path).unwrap();

        let s1 = scorer.score(&[5, 15, 25, 35, 45], &[1, 12]);
        let s2 = loaded.score(&[5, 15, 25, 35, 45], &[1, 12]);
        assert!((s1 - s2).abs() < 1e-10, "Loaded scorer should produce same results");

        let _ = std::fs::remove_file(&path);
    }
}
