use std::collections::HashMap;

use ndarray::{Array1, Array2};

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// Transformer (Reservoir Transformer) — self-attention à poids fixes.
///
/// Comme un ESN mais avec attention au lieu de récurrence. Les matrices
/// Q, K, V sont random et gelées (frozen), seul le readout est entraîné
/// par ridge regression. Capture les dépendances long-range sans backprop.
///
/// Architecture :
/// - Encoding 5D (sum, spread, odd_ratio, centroid, variance)
/// - Projection en d_model=32 via matrice random W_embed
/// - Positional encoding sinusoïdal
/// - 2 couches d'attention (4 têtes, d_k=8, masque causal)
/// - Ridge regression pour le readout
pub struct TransformerModel {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    context_len: usize,
    ridge_lambda: f64,
    smoothing: f64,
    seed: u64,
}

impl TransformerModel {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        context_len: usize,
        ridge_lambda: f64,
        smoothing: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_layers,
            d_model,
            n_heads,
            context_len,
            ridge_lambda,
            smoothing,
            seed,
        }
    }
}

impl Default for TransformerModel {
    fn default() -> Self {
        Self {
            n_layers: 2,
            d_model: 32,
            n_heads: 4,
            context_len: 50,
            ridge_lambda: 1e-3,
            smoothing: 0.5,
            seed: 42,
        }
    }
}

/// Simple PRNG (xorshift64) pour générer des matrices random reproductibles.
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1), // Éviter état 0
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

    /// Retourne un f64 dans [-1, 1]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    /// Génère une matrice random [rows x cols] avec valeurs dans [-scale, scale]
    fn random_matrix(&mut self, rows: usize, cols: usize, scale: f64) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                m[[r, c]] = self.next_f64() * scale;
            }
        }
        m
    }
}

/// Encode un tirage en vecteur 5D (même encoding que NVAR).
fn encode_draw(numbers: &[u8], pool_size: usize) -> [f64; 5] {
    let n = numbers.len() as f64;
    let max_val = pool_size as f64;

    let sum: f64 = numbers.iter().map(|&x| x as f64).sum();
    let sum_norm = sum / (n * max_val);

    let min_v = numbers.iter().copied().min().unwrap_or(1) as f64;
    let max_v = numbers.iter().copied().max().unwrap_or(1) as f64;
    let spread_norm = (max_v - min_v) / max_val;

    let odd_ratio = numbers.iter().filter(|&&x| x % 2 == 1).count() as f64 / n;

    let centroid = sum / n / max_val;

    let mean = sum / n;
    let variance = numbers.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let var_norm = variance / (max_val * max_val);

    [sum_norm, spread_norm, odd_ratio, centroid, var_norm]
}

const INPUT_DIM: usize = 5;

/// Softmax sur un vecteur, en place.
fn softmax_row(row: &mut [f64]) {
    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Layer normalization (mean=0, std=1) sur un vecteur.
fn layer_norm(x: &mut [f64]) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-8).sqrt();
    for v in x.iter_mut() {
        *v = (*v - mean) / std;
    }
}

/// Matrices de poids frozen pour une couche d'attention multi-têtes.
struct AttentionLayer {
    w_q: Array2<f64>, // [d_model, d_model]
    w_k: Array2<f64>, // [d_model, d_model]
    w_v: Array2<f64>, // [d_model, d_model]
}

impl AttentionLayer {
    fn new(rng: &mut Rng64, d_model: usize) -> Self {
        let scale = (1.0 / d_model as f64).sqrt();
        Self {
            w_q: rng.random_matrix(d_model, d_model, scale),
            w_k: rng.random_matrix(d_model, d_model, scale),
            w_v: rng.random_matrix(d_model, d_model, scale),
        }
    }

    /// Applique multi-head causal self-attention.
    /// x: [seq_len, d_model] → output: [seq_len, d_model]
    fn forward(&self, x: &Array2<f64>, n_heads: usize) -> Array2<f64> {
        let seq_len = x.nrows();
        let d_model = x.ncols();
        let d_k = d_model / n_heads;

        // Q, K, V = X @ W_Q/K/V — [seq_len, d_model]
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let mut output = Array2::<f64>::zeros((seq_len, d_model));

        // Multi-head attention
        for h in 0..n_heads {
            let start = h * d_k;
            let end = start + d_k;

            // Extraire les slices pour cette tête
            let q_h = q.slice(ndarray::s![.., start..end]);
            let k_h = k.slice(ndarray::s![.., start..end]);
            let v_h = v.slice(ndarray::s![.., start..end]);

            // Attention scores: Q @ K^T / sqrt(d_k)
            let scale = (d_k as f64).sqrt();
            let mut attn = q_h.dot(&k_h.t()) / scale; // [seq_len, seq_len]

            // Masque causal : -inf pour les positions futures
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    attn[[i, j]] = f64::NEG_INFINITY;
                }
            }

            // Softmax par ligne
            for i in 0..seq_len {
                let mut row: Vec<f64> = (0..seq_len).map(|j| attn[[i, j]]).collect();
                softmax_row(&mut row);
                for j in 0..seq_len {
                    attn[[i, j]] = row[j];
                }
            }

            // Weighted sum: attn @ V_h
            let head_out = attn.dot(&v_h); // [seq_len, d_k]

            // Accumuler dans output
            for i in 0..seq_len {
                for j in 0..d_k {
                    output[[i, start + j]] += head_out[[i, j]];
                }
            }
        }

        output
    }
}

impl ForecastModel for TransformerModel {
    fn name(&self) -> &str {
        "Transformer"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        let min_draws = self.context_len + 2;
        if draws.len() < min_draws {
            return uniform;
        }

        let mut rng = Rng64::new(self.seed);

        // Ordre chronologique
        let chronological: Vec<&Draw> = draws.iter().rev().collect();
        let n = chronological.len();

        // Encoder tous les tirages
        let encoded: Vec<[f64; 5]> = chronological
            .iter()
            .map(|d| encode_draw(pool.numbers_from(d), size))
            .collect();

        // Matrice d'embedding : [INPUT_DIM, d_model]
        let w_embed = rng.random_matrix(INPUT_DIM, self.d_model, (1.0 / INPUT_DIM as f64).sqrt());

        // Créer les couches d'attention
        let layers: Vec<AttentionLayer> = (0..self.n_layers)
            .map(|_| AttentionLayer::new(&mut rng, self.d_model))
            .collect();

        // Nombre de séquences d'entraînement
        let n_sequences = n.saturating_sub(self.context_len + 1);
        if n_sequences < 2 {
            return uniform;
        }

        // Limiter pour la performance
        let max_sequences = 100;
        let stride = if n_sequences > max_sequences {
            n_sequences / max_sequences
        } else {
            1
        };
        let actual_sequences: Vec<usize> = (0..n_sequences).step_by(stride).collect();
        let n_actual = actual_sequences.len();

        // Pour chaque séquence, forward pass → collecter h_last
        let mut h_matrix = Array2::<f64>::zeros((self.d_model, n_actual));
        let mut targets = Array2::<f64>::zeros((size, n_actual));

        for (s_idx, &seq_start) in actual_sequences.iter().enumerate() {
            let seq_end = seq_start + self.context_len;

            // Embedding : [context_len, d_model]
            let mut x = Array2::<f64>::zeros((self.context_len, self.d_model));
            for (t, enc) in encoded[seq_start..seq_end].iter().enumerate() {
                let input = Array1::from_vec(enc.to_vec());
                let embedded = input.dot(&w_embed);
                for d in 0..self.d_model {
                    x[[t, d]] = embedded[d];
                }
            }

            // Positional encoding sinusoïdal
            for t in 0..self.context_len {
                for d in 0..self.d_model {
                    let angle = t as f64 / (10000.0f64).powf(2.0 * (d / 2) as f64 / self.d_model as f64);
                    if d % 2 == 0 {
                        x[[t, d]] += angle.sin();
                    } else {
                        x[[t, d]] += angle.cos();
                    }
                }
            }

            // Forward pass à travers les couches d'attention
            for layer in &layers {
                let attn_out = layer.forward(&x, self.n_heads);
                // Résidu + layer norm
                for t in 0..self.context_len {
                    let mut row: Vec<f64> = (0..self.d_model)
                        .map(|d| x[[t, d]] + attn_out[[t, d]])
                        .collect();
                    layer_norm(&mut row);
                    for d in 0..self.d_model {
                        x[[t, d]] = row[d];
                    }
                }
            }

            // h_last = dernière position
            for d in 0..self.d_model {
                h_matrix[[d, s_idx]] = x[[self.context_len - 1, d]];
            }

            // Target : numéros du tirage suivant la séquence
            let target_idx = seq_end;
            if target_idx < n {
                for &num in pool.numbers_from(chronological[target_idx]) {
                    let idx = (num - 1) as usize;
                    if idx < size {
                        targets[[idx, s_idx]] = 1.0;
                    }
                }
            }
        }

        // Ridge regression : W_out = targets @ H^T @ (H @ H^T + λI)^{-1}
        let w_out =
            match lemillion_esn::linalg::ridge_regression(&h_matrix, &targets, self.ridge_lambda) {
                Ok(w) => w,
                Err(_) => return uniform,
            };

        // Prédire pour la dernière séquence
        let last_start = n.saturating_sub(self.context_len);
        let mut x = Array2::<f64>::zeros((self.context_len, self.d_model));
        let ctx_len = n - last_start;
        for (t, enc) in encoded[last_start..n].iter().enumerate() {
            let input = Array1::from_vec(enc.to_vec());
            let embedded = input.dot(&w_embed);
            for d in 0..self.d_model {
                x[[t, d]] = embedded[d];
            }
        }

        // Positional encoding
        for t in 0..ctx_len {
            for d in 0..self.d_model {
                let angle = t as f64 / (10000.0f64).powf(2.0 * (d / 2) as f64 / self.d_model as f64);
                if d % 2 == 0 {
                    x[[t, d]] += angle.sin();
                } else {
                    x[[t, d]] += angle.cos();
                }
            }
        }

        // Forward pass
        for layer in &layers {
            let attn_out = layer.forward(&x, self.n_heads);
            for t in 0..ctx_len {
                let mut row: Vec<f64> = (0..self.d_model)
                    .map(|d| x[[t, d]] + attn_out[[t, d]])
                    .collect();
                layer_norm(&mut row);
                for d in 0..self.d_model {
                    x[[t, d]] = row[d];
                }
            }
        }

        let h_last = Array1::from_vec((0..self.d_model).map(|d| x[[ctx_len - 1, d]]).collect());
        let prediction = w_out.dot(&h_last);

        // Convertir en probabilités
        let mut raw_probs: Vec<f64> = prediction.iter().map(|&x| x.max(1e-10)).collect();

        // Smooth + normalize
        let uniform_val = 1.0 / size as f64;
        for p in &mut raw_probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }
        let sum: f64 = raw_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut raw_probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        raw_probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_layers".into(), self.n_layers as f64),
            ("d_model".into(), self.d_model as f64),
            ("n_heads".into(), self.n_heads as f64),
            ("context_len".into(), self.context_len as f64),
            ("ridge_lambda".into(), self.ridge_lambda),
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
    fn test_transformer_balls_sums_to_one() {
        let model = TransformerModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_transformer_stars_sums_to_one() {
        let model = TransformerModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_transformer_no_negative() {
        let model = TransformerModel::default();
        let draws = make_test_draws(80);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_transformer_empty_draws() {
        let model = TransformerModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transformer_few_draws() {
        let model = TransformerModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transformer_deterministic() {
        let model = TransformerModel::default();
        let draws = make_test_draws(80);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-12, "Transformer should be deterministic");
        }
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng64::new(42);
        let mut rng2 = Rng64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_softmax_basic() {
        let mut row = vec![1.0, 2.0, 3.0];
        softmax_row(&mut row);
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(row[2] > row[1] && row[1] > row[0]);
    }

    #[test]
    fn test_layer_norm_basic() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        layer_norm(&mut x);
        let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
        assert!(mean.abs() < 1e-10, "Mean should be ~0, got {mean}");
    }

    #[test]
    fn test_encode_draw_5d() {
        let numbers = [1u8, 10, 20, 30, 50];
        let enc = encode_draw(&numbers, 50);
        assert_eq!(enc.len(), 5);
        for &v in &enc {
            assert!(v >= 0.0 && v <= 1.0, "Value out of range: {v}");
        }
    }
}
