use serde::{Deserialize, Serialize};
use lemillion_db::models::Draw;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::prelude::Distribution;

/// Poids du réseau de neurones 5 couches (92→64→32→16→8→1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralScorerWeights {
    pub w1: Vec<Vec<f64>>,  // [64][92]
    pub b1: Vec<f64>,       // [64]
    pub w2: Vec<Vec<f64>>,  // [32][64]
    pub b2: Vec<f64>,       // [32]
    pub w3: Vec<Vec<f64>>,  // [16][32]
    pub b3: Vec<f64>,       // [16]
    pub w4: Vec<Vec<f64>>,  // [8][16]
    pub b4: Vec<f64>,       // [8]
    pub w5: Vec<f64>,       // [8]
    pub b5: f64,
}

/// Ensemble de scorers neuronaux (5 seeds) pour réduire la variance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralScorerEnsembleWeights {
    pub members: Vec<NeuralScorerWeights>,
}

/// Scorer neuronal combinatoire pour les grilles EuroMillions.
pub struct NeuralScorer {
    pub weights: Vec<NeuralScorerWeights>,
}

const INPUT_DIM: usize = 92;
const H1: usize = 64;
const H2: usize = 32;
const H3: usize = 16;
const H4: usize = 8;

/// Encode une grille (5 boules + 2 étoiles) en vecteur 92-dim.
/// 50-dim one-hot balls + 12-dim one-hot stars + 30 features engineered.
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

    // Engineered features (dim 62..91)
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

    // mod4 distribution [0..3] → 2 features
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

    // consecutive_ratio
    let consec = sorted_balls.windows(2).filter(|w| w[1] == w[0] + 1).count() as f64;
    input[70] = consec / 4.0;

    // mean_gap_norm
    let gap_sum: f64 = sorted_balls.windows(2).map(|w| (w[1] - w[0]) as f64).sum();
    input[71] = (gap_sum / 4.0) / 49.0;

    // ── New features (v2) ──

    // ball_sum_zscore : écart à la moyenne historique (~127.5)
    input[72] = (sum - 127.5) / 30.0;

    // spread_percentile (rough mapping)
    input[73] = (spread - 20.0) / 20.0;

    // mod4_cosine with typical (1,1,1,2) profile
    let typical = [1.25f64, 1.25, 1.25, 1.25];
    let dot: f64 = mod4.iter().zip(typical.iter()).map(|(a, b)| a * b).sum();
    let mag_a: f64 = mod4.iter().map(|a| a * a).sum::<f64>().sqrt();
    let mag_b: f64 = typical.iter().map(|b| b * b).sum::<f64>().sqrt();
    input[74] = if mag_a > 0.0 && mag_b > 0.0 { dot / (mag_a * mag_b) } else { 0.0 };

    // star_spread_norm
    let mut sorted_stars = *stars;
    sorted_stars.sort();
    input[75] = (sorted_stars[1] - sorted_stars[0]) as f64 / 11.0;

    // low_half_ratio : fraction of balls in 1-25
    let low_count = balls.iter().filter(|&&b| b <= 25).count() as f64;
    input[76] = low_count / 5.0;

    // mod4 remaining classes
    input[77] = mod4[2] / 5.0;
    input[78] = mod4[3] / 5.0;

    // decade counts (5 features)
    let mut decade_counts = [0.0f64; 5];
    for &b in balls {
        decade_counts[((b - 1) / 10) as usize] += 1.0;
    }
    for i in 0..5 {
        input[79 + i] = decade_counts[i] / 5.0;
    }

    // decade_entropy
    let total_dec = 5.0f64;
    let mut dec_entropy = 0.0f64;
    for &c in &decade_counts {
        if c > 0.0 {
            let p = c / total_dec;
            dec_entropy -= p * p.ln();
        }
    }
    input[84] = dec_entropy / 5.0f64.ln(); // normalized

    // pair_sum features: mean of all C(5,2)=10 ball pairs
    let mut pair_sum = 0.0f64;
    let mut n_pairs = 0;
    for i in 0..5 {
        for j in (i + 1)..5 {
            pair_sum += (balls[i] as f64 + balls[j] as f64) / 100.0;
            n_pairs += 1;
        }
    }
    input[85] = pair_sum / n_pairs as f64;

    // min_ball_norm and max_ball_norm
    input[86] = sorted_balls[0] as f64 / 50.0;
    input[87] = sorted_balls[4] as f64 / 50.0;

    // star_high_low : both stars in same half?
    input[88] = if (sorted_stars[0] <= 6) == (sorted_stars[1] <= 6) { 1.0 } else { 0.0 };

    // ball_variance_norm
    let mean = sum / 5.0;
    let var: f64 = balls.iter().map(|&b| (b as f64 - mean).powi(2)).sum::<f64>() / 5.0;
    input[89] = (var.sqrt()) / 20.0;

    // star_mod4_match : stars same mod4 class?
    input[90] = if (stars[0] - 1) % 4 == (stars[1] - 1) % 4 { 1.0 } else { 0.0 };

    // consecutive_count_norm (actual number of consecutive pairs, not ratio)
    input[91] = consec / 4.0;

    input
}

impl NeuralScorer {
    /// Forward pass : retourne un score [0, 1] (moyenne de l'ensemble).
    pub fn score(&self, balls: &[u8; 5], stars: &[u8; 2]) -> f64 {
        let input = encode(balls, stars);
        let sum: f64 = self.weights.iter().map(|w| forward_pass(w, &input)).sum();
        sum / self.weights.len() as f64
    }

    /// Entraîne un ensemble de 5 scorers avec hard negatives.
    pub fn train(draws: &[Draw], seed: u64) -> Self {
        let seeds = [seed, seed + 1, seed + 2, seed + 3, seed + 4];
        let weights: Vec<NeuralScorerWeights> = seeds.iter().map(|&s| {
            train_single(draws, s)
        }).collect();
        Self { weights }
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let ensemble = NeuralScorerEnsembleWeights { members: self.weights.clone() };
        let json = serde_json::to_string(&ensemble)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        // Try ensemble format first, fall back to single
        if let Ok(ensemble) = serde_json::from_str::<NeuralScorerEnsembleWeights>(&json) {
            return Ok(Self { weights: ensemble.members });
        }
        // Legacy single-weight format — not compatible with v2 architecture, retrain
        Err(anyhow::anyhow!("Incompatible neural scorer format, retrain needed"))
    }
}

fn forward_pass(w: &NeuralScorerWeights, input: &[f64]) -> f64 {
    // Layer 1: Linear(92→64) + ReLU
    let mut h1 = vec![0.0f64; H1];
    for i in 0..H1 {
        let mut sum = w.b1[i];
        for j in 0..INPUT_DIM {
            sum += w.w1[i][j] * input[j];
        }
        h1[i] = sum.max(0.0);
    }

    // Layer 2: Linear(64→32) + ReLU
    let mut h2 = vec![0.0f64; H2];
    for i in 0..H2 {
        let mut sum = w.b2[i];
        for j in 0..H1 {
            sum += w.w2[i][j] * h1[j];
        }
        h2[i] = sum.max(0.0);
    }

    // Layer 3: Linear(32→16) + ReLU
    let mut h3 = vec![0.0f64; H3];
    for i in 0..H3 {
        let mut sum = w.b3[i];
        for j in 0..H2 {
            sum += w.w3[i][j] * h2[j];
        }
        h3[i] = sum.max(0.0);
    }

    // Layer 4: Linear(16→8) + ReLU
    let mut h4 = vec![0.0f64; H4];
    for i in 0..H4 {
        let mut sum = w.b4[i];
        for j in 0..H3 {
            sum += w.w4[i][j] * h3[j];
        }
        h4[i] = sum.max(0.0);
    }

    // Output: Linear(8→1) + Sigmoid
    let mut out = w.b5;
    for j in 0..H4 {
        out += w.w5[j] * h4[j];
    }

    sigmoid(out)
}

fn rng_uniform(rng: &mut StdRng) -> f64 {
    let dist = rand::distr::StandardUniform;
    let val: f64 = dist.sample(rng);
    val
}

fn sigmoid(x: f64) -> f64 {
    let clamped = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Xavier initialization for 5-layer network.
fn xavier_init(rng: &mut StdRng) -> NeuralScorerWeights {
    let init_layer = |rows: usize, cols: usize, rng: &mut StdRng| -> (Vec<Vec<f64>>, Vec<f64>) {
        let scale = (2.0 / (rows + cols) as f64).sqrt();
        let w: Vec<Vec<f64>> = (0..rows)
            .map(|_| (0..cols).map(|_| (rng_uniform(rng) * 2.0 - 1.0) * scale).collect())
            .collect();
        let b = vec![0.0; rows];
        (w, b)
    };

    let (w1, b1) = init_layer(H1, INPUT_DIM, rng);
    let (w2, b2) = init_layer(H2, H1, rng);
    let (w3, b3) = init_layer(H3, H2, rng);
    let (w4, b4) = init_layer(H4, H3, rng);

    let scale5 = (2.0 / (H4 + 1) as f64).sqrt();
    let w5: Vec<f64> = (0..H4).map(|_| (rng_uniform(rng) * 2.0 - 1.0) * scale5).collect();
    let b5 = 0.0;

    NeuralScorerWeights { w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 }
}

/// Entraîne un seul scorer avec hard negatives.
fn train_single(draws: &[Draw], seed: u64) -> NeuralScorerWeights {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = xavier_init(&mut rng);

    let lr = 0.003;
    let l2 = 1e-4;
    let n_epochs = 20;
    let n_negatives = 40;

    let n_train = draws.len().saturating_sub(10);
    if n_train < 20 {
        return weights;
    }

    for _epoch in 0..n_epochs {
        for t in 0..n_train {
            let draw = &draws[t];
            let balls = &draw.balls;
            let stars = &draw.stars;

            // Positive example
            let input_pos = encode(balls, stars);
            backprop(&mut weights, &input_pos, 1.0, lr, l2);

            // Negative examples: 50% random, 25% near-miss, 25% high-score random
            let n_random = n_negatives / 2;
            let n_near_miss = n_negatives / 4;
            let n_high_score = n_negatives - n_random - n_near_miss;

            // Random negatives
            for _ in 0..n_random {
                let (neg_balls, neg_stars) = random_grid(&mut rng);
                let input_neg = encode(&neg_balls, &neg_stars);
                backprop(&mut weights, &input_neg, 0.0, lr, l2);
            }

            // Near-miss negatives: change 1 ball or 1 star from real draw
            for _ in 0..n_near_miss {
                let (neg_balls, neg_stars) = near_miss_grid(balls, stars, &mut rng);
                let input_neg = encode(&neg_balls, &neg_stars);
                backprop(&mut weights, &input_neg, 0.0, lr, l2);
            }

            // High-score random: balls from popular range, not actual draw
            for _ in 0..n_high_score {
                let (neg_balls, neg_stars) = high_score_random_grid(&mut rng);
                let input_neg = encode(&neg_balls, &neg_stars);
                backprop(&mut weights, &input_neg, 0.0, lr, l2);
            }
        }
    }

    weights
}

/// Near-miss: take a real draw and change 1 ball or 1 star.
fn near_miss_grid(balls: &[u8; 5], stars: &[u8; 2], rng: &mut StdRng) -> ([u8; 5], [u8; 2]) {
    use rand::distr::Uniform;
    let mut new_balls = *balls;
    let mut new_stars = *stars;

    let change_ball: bool = rng_uniform(rng) < 0.7; // 70% change ball, 30% change star
    if change_ball {
        let idx = (rng_uniform(rng) * 5.0) as usize % 5;
        let ball_dist = Uniform::new(1u8, 51).unwrap();
        loop {
            let b = ball_dist.sample(rng);
            if !new_balls.contains(&b) {
                new_balls[idx] = b;
                break;
            }
        }
        new_balls.sort();
    } else {
        let idx = (rng_uniform(rng) * 2.0) as usize % 2;
        let star_dist = Uniform::new(1u8, 13).unwrap();
        loop {
            let s = star_dist.sample(rng);
            if !new_stars.contains(&s) {
                new_stars[idx] = s;
                break;
            }
        }
        new_stars.sort();
    }

    (new_balls, new_stars)
}

/// High-score random: balls from common range (5-45), structurally plausible.
fn high_score_random_grid(rng: &mut StdRng) -> ([u8; 5], [u8; 2]) {
    use rand::distr::Uniform;

    // Bias toward middle range balls (more "plausible looking")
    let ball_dist = Uniform::new(3u8, 48).unwrap();
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

/// Backpropagation for 5-layer network (BCE loss, SGD).
fn backprop(w: &mut NeuralScorerWeights, input: &[f64], target: f64, lr: f64, l2: f64) {
    // Forward pass with intermediate values
    let mut z1 = vec![0.0f64; H1];
    let mut h1 = vec![0.0f64; H1];
    for i in 0..H1 {
        let mut s = w.b1[i];
        for j in 0..INPUT_DIM {
            s += w.w1[i][j] * input[j];
        }
        z1[i] = s;
        h1[i] = s.max(0.0);
    }

    let mut z2 = vec![0.0f64; H2];
    let mut h2 = vec![0.0f64; H2];
    for i in 0..H2 {
        let mut s = w.b2[i];
        for j in 0..H1 {
            s += w.w2[i][j] * h1[j];
        }
        z2[i] = s;
        h2[i] = s.max(0.0);
    }

    let mut z3 = vec![0.0f64; H3];
    let mut h3 = vec![0.0f64; H3];
    for i in 0..H3 {
        let mut s = w.b3[i];
        for j in 0..H2 {
            s += w.w3[i][j] * h2[j];
        }
        z3[i] = s;
        h3[i] = s.max(0.0);
    }

    let mut z4 = vec![0.0f64; H4];
    let mut h4 = vec![0.0f64; H4];
    for i in 0..H4 {
        let mut s = w.b4[i];
        for j in 0..H3 {
            s += w.w4[i][j] * h3[j];
        }
        z4[i] = s;
        h4[i] = s.max(0.0);
    }

    let mut z5 = w.b5;
    for j in 0..H4 {
        z5 += w.w5[j] * h4[j];
    }
    let output = sigmoid(z5);

    // BCE gradient at output
    let d_out = output - target;

    // Layer 5 gradients
    for j in 0..H4 {
        w.w5[j] -= lr * (d_out * h4[j] + l2 * w.w5[j]);
    }
    w.b5 -= lr * d_out;

    // Layer 4 gradients
    let mut d_h4 = vec![0.0f64; H4];
    for j in 0..H4 {
        d_h4[j] = d_out * w.w5[j];
    }
    let d_z4: Vec<f64> = (0..H4).map(|i| if z4[i] > 0.0 { d_h4[i] } else { 0.0 }).collect();

    for i in 0..H4 {
        for j in 0..H3 {
            w.w4[i][j] -= lr * (d_z4[i] * h3[j] + l2 * w.w4[i][j]);
        }
        w.b4[i] -= lr * d_z4[i];
    }

    // Layer 3 gradients
    let mut d_h3 = vec![0.0f64; H3];
    for j in 0..H3 {
        for i in 0..H4 {
            d_h3[j] += d_z4[i] * w.w4[i][j];
        }
    }
    let d_z3: Vec<f64> = (0..H3).map(|i| if z3[i] > 0.0 { d_h3[i] } else { 0.0 }).collect();

    for i in 0..H3 {
        for j in 0..H2 {
            w.w3[i][j] -= lr * (d_z3[i] * h2[j] + l2 * w.w3[i][j]);
        }
        w.b3[i] -= lr * d_z3[i];
    }

    // Layer 2 gradients
    let mut d_h2 = vec![0.0f64; H2];
    for j in 0..H2 {
        for i in 0..H3 {
            d_h2[j] += d_z3[i] * w.w3[i][j];
        }
    }
    let d_z2: Vec<f64> = (0..H2).map(|i| if z2[i] > 0.0 { d_h2[i] } else { 0.0 }).collect();

    for i in 0..H2 {
        for j in 0..H1 {
            w.w2[i][j] -= lr * (d_z2[i] * h1[j] + l2 * w.w2[i][j]);
        }
        w.b2[i] -= lr * d_z2[i];
    }

    // Layer 1 gradients
    let mut d_h1 = vec![0.0f64; H1];
    for j in 0..H1 {
        for i in 0..H2 {
            d_h1[j] += d_z2[i] * w.w2[i][j];
        }
    }
    let d_z1: Vec<f64> = (0..H1).map(|i| if z1[i] > 0.0 { d_h1[i] } else { 0.0 }).collect();

    for i in 0..H1 {
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
        assert_eq!(encoded[0], 1.0);
        assert_eq!(encoded[9], 1.0);
        assert_eq!(encoded[49], 1.0);
        assert_eq!(encoded[52], 1.0);
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
    fn test_scorer_ensemble_save_load() {
        let draws = make_test_draws(50);
        let scorer = NeuralScorer::train(&draws, 42);
        assert_eq!(scorer.weights.len(), 5);

        let path = std::path::PathBuf::from("/tmp/test_neural_scorer_v2.json");
        scorer.save(&path).unwrap();
        let loaded = NeuralScorer::load(&path).unwrap();

        let s1 = scorer.score(&[5, 15, 25, 35, 45], &[1, 12]);
        let s2 = loaded.score(&[5, 15, 25, 35, 45], &[1, 12]);
        assert!((s1 - s2).abs() < 1e-10, "Loaded scorer should produce same results");

        let _ = std::fs::remove_file(&path);
    }
}
