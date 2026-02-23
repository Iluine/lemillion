use std::ops::Range;
use std::time::Instant;

use anyhow::{bail, Result};
use ndarray::{s, Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

use lemillion_db::models::{Draw, Pool};

use crate::config::{EsnConfig, EsnResult};
use crate::encoding::{encode_draw, encode_target_onehot};
use crate::linalg::ridge_regression;
use crate::metrics;
use crate::reservoir::Reservoir;

pub struct TrainedEsn {
    pub config: EsnConfig,
    pub reservoir: Reservoir,
    pub w_out_balls: Array2<f64>,
    pub w_out_stars: Array2<f64>,
}

pub struct DataSplit {
    pub train: Range<usize>,
    pub val: Range<usize>,
    pub test: Range<usize>,
}

impl DataSplit {
    /// Create a static train/val/test split.
    /// Draws are assumed in chronological order (oldest first).
    /// ~80% train, ~8% val, ~12% test (tail = most recent).
    pub fn new(n: usize) -> Result<Self> {
        if n < 10 {
            bail!("Need at least 10 draws, got {n}");
        }
        let train_end = (n as f64 * 0.80) as usize;
        let val_end = (n as f64 * 0.88) as usize;
        if train_end < 3 || val_end <= train_end || val_end >= n {
            bail!("Split too small for {n} draws");
        }
        Ok(DataSplit {
            train: 0..train_end,
            val: train_end..val_end,
            test: val_end..n,
        })
    }
}

/// Softmax with numerical stability (subtract max).
pub fn softmax(logits: &Array1<f64>) -> Vec<f64> {
    let mut out = vec![0.0; logits.len()];
    softmax_into(logits, &mut out);
    out
}

/// Softmax in-place into a pre-allocated buffer.
pub fn softmax_into(logits: &Array1<f64>, out: &mut [f64]) {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (o, &x) in out.iter_mut().zip(logits.iter()) {
        *o = (x - max).exp();
        sum += *o;
    }
    if sum < 1e-30 {
        let n = out.len();
        let uniform = 1.0 / n as f64;
        for o in out.iter_mut() {
            *o = uniform;
        }
        return;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// Train the ESN and evaluate on val/test splits.
/// Draws are in DB order (newest-first); this function reverses them internally.
pub fn train_and_evaluate(draws: &[Draw], config: &EsnConfig) -> Result<(TrainedEsn, EsnResult)> {
    let start = Instant::now();

    // Reverse to chronological order (oldest first)
    let chrono: Vec<&Draw> = draws.iter().rev().collect();
    let n = chrono.len();

    let split = DataSplit::new(n)?;

    if config.washout >= split.train.end.saturating_sub(1) {
        bail!(
            "Washout {} too large for training set of {} draws",
            config.washout,
            split.train.len()
        );
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let input_dim = config.encoding.input_dim();
    let state_dim = config.reservoir_size + input_dim;

    // Encode all inputs and targets
    let inputs: Vec<Array1<f64>> = chrono.iter().map(|d| encode_draw(d, config.encoding)).collect();
    let ball_targets: Vec<Array1<f64>> = chrono
        .iter()
        .map(|d| encode_target_onehot(d, Pool::Balls))
        .collect();
    let star_targets: Vec<Array1<f64>> = chrono
        .iter()
        .map(|d| encode_target_onehot(d, Pool::Stars))
        .collect();

    // Create reservoir and run on training inputs
    let mut reservoir = Reservoir::new(
        input_dim,
        config.reservoir_size,
        config.spectral_radius,
        config.sparsity,
        config.input_scaling,
        config.leaking_rate,
        config.noise_amplitude,
        &mut rng,
    );

    // Run reservoir over all training inputs, collecting states
    let train_states = reservoir.run_sequence(&inputs[split.train.clone()], &mut rng);

    // Build H and Y matrices for valid training pairs
    // input[t] predicts draw[t+1], so valid t: washout..train_end-1
    let valid_start = config.washout;
    let valid_end = split.train.end - 1;
    if valid_start >= valid_end {
        bail!(
            "No valid training pairs: washout={}, train_end={}",
            config.washout,
            split.train.end
        );
    }
    let t_train = valid_end - valid_start;

    // H: [state_dim, T] - each column = concat(state, input)
    let mut h_mat = Array2::zeros((state_dim, t_train));
    let mut y_balls = Array2::zeros((Pool::Balls.size(), t_train));
    let mut y_stars = Array2::zeros((Pool::Stars.size(), t_train));

    for (col, t) in (valid_start..valid_end).enumerate() {
        let state = &train_states[t];
        let input = &inputs[t];
        h_mat.slice_mut(s![..config.reservoir_size, col]).assign(state);
        h_mat.slice_mut(s![config.reservoir_size.., col]).assign(input);
        let target_idx = split.train.start + t + 1;
        y_balls
            .column_mut(col)
            .assign(&ball_targets[target_idx]);
        y_stars
            .column_mut(col)
            .assign(&star_targets[target_idx]);
    }

    // Ridge regression
    let w_out_balls = ridge_regression(&h_mat, &y_balls, config.ridge_lambda)?;
    let w_out_stars = ridge_regression(&h_mat, &y_stars, config.ridge_lambda)?;

    // Evaluate on validation set
    // Continue running the reservoir (state carries over from training)
    let val_states = reservoir.run_sequence(&inputs[split.val.clone()], &mut rng);

    let (val_ball_hit, val_star_hit, val_ball_topk, val_star_topk) =
        evaluate_split(&val_states, &inputs, &chrono, &split.val, &w_out_balls, &w_out_stars, config);

    // Evaluate on test set
    let test_states = reservoir.run_sequence(&inputs[split.test.clone()], &mut rng);

    let (test_ball_hit, test_star_hit, test_ball_topk, test_star_topk) =
        evaluate_split(&test_states, &inputs, &chrono, &split.test, &w_out_balls, &w_out_stars, config);

    // Lyapunov exponent
    let lyapunov = metrics::lyapunov_exponent(&mut reservoir, &inputs[..split.train.end.min(inputs.len())], config.seed);

    let train_time_ms = start.elapsed().as_millis() as u64;

    let result = EsnResult {
        config: config.clone(),
        val_ball_hit_rate: val_ball_hit,
        val_star_hit_rate: val_star_hit,
        val_ball_topk: val_ball_topk,
        val_star_topk: val_star_topk,
        test_ball_hit_rate: test_ball_hit,
        test_star_hit_rate: test_star_hit,
        test_ball_topk: test_ball_topk,
        test_star_topk: test_star_topk,
        lyapunov_exponent: lyapunov,
        train_time_ms,
    };

    let trained = TrainedEsn {
        config: config.clone(),
        reservoir,
        w_out_balls,
        w_out_stars,
    };

    Ok((trained, result))
}

/// Evaluate a split (val or test). Returns (ball_hit_rate, star_hit_rate, ball_topk, star_topk).
fn evaluate_split(
    states: &[Array1<f64>],
    inputs: &[Array1<f64>],
    chrono: &[&Draw],
    split: &Range<usize>,
    w_out_balls: &Array2<f64>,
    w_out_stars: &Array2<f64>,
    config: &EsnConfig,
) -> (f64, f64, f64, f64) {
    let input_dim = config.encoding.input_dim();
    let rs = config.reservoir_size;
    // For each t in the split (except last), predict draw[t+1]
    let mut ball_preds = Vec::new();
    let mut star_preds = Vec::new();
    let mut ball_actuals = Vec::new();
    let mut star_actuals = Vec::new();

    let mut extended = Array1::zeros(rs + input_dim);
    let mut ball_buf = vec![0.0; Pool::Balls.size()];
    let mut star_buf = vec![0.0; Pool::Stars.size()];

    for (i, t) in split.clone().enumerate() {
        if t + 1 >= chrono.len() {
            break;
        }
        // Build extended state [state; input]
        extended.slice_mut(s![..rs]).assign(&states[i]);
        extended.slice_mut(s![rs..]).assign(&inputs[t]);

        // Predict with in-place softmax
        let ball_logits = w_out_balls.dot(&extended);
        let star_logits = w_out_stars.dot(&extended);
        softmax_into(&ball_logits, &mut ball_buf);
        softmax_into(&star_logits, &mut star_buf);

        ball_preds.push(ball_buf.clone());
        star_preds.push(star_buf.clone());

        let actual = chrono[t + 1];
        ball_actuals.push(actual.balls);
        star_actuals.push(actual.stars);
    }

    if ball_preds.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let ball_hit = metrics::hit_rate(&ball_preds, &ball_actuals, Pool::Balls, Pool::Balls.pick_count());
    let star_hit = metrics::hit_rate(&star_preds, &star_actuals, Pool::Stars, Pool::Stars.pick_count());
    let ball_topk = metrics::hit_rate(&ball_preds, &ball_actuals, Pool::Balls, 10);
    let star_topk = metrics::hit_rate(&star_preds, &star_actuals, Pool::Stars, 6);

    (ball_hit, star_hit, ball_topk, star_topk)
}

/// Given a trained ESN and draws (newest-first from DB), predict next draw probabilities.
/// Returns (ball_probs[50], star_probs[12]).
pub fn predict_next(esn: &mut TrainedEsn, draws: &[Draw]) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(esn.config.seed);
    let input_dim = esn.config.encoding.input_dim();

    // Reset and run through all draws in chronological order
    esn.reservoir.reset_state();
    let chrono: Vec<&Draw> = draws.iter().rev().collect();
    let inputs: Vec<Array1<f64>> = chrono
        .iter()
        .map(|d| encode_draw(d, esn.config.encoding))
        .collect();

    let states = esn.reservoir.run_sequence(&inputs, &mut rng);

    // Use the last state to predict
    if let Some(last_state) = states.last() {
        let last_input = inputs.last().unwrap();
        let mut extended = Array1::zeros(esn.config.reservoir_size + input_dim);
        extended
            .slice_mut(s![..esn.config.reservoir_size])
            .assign(last_state);
        extended
            .slice_mut(s![esn.config.reservoir_size..])
            .assign(last_input);

        let ball_logits = esn.w_out_balls.dot(&extended);
        let star_logits = esn.w_out_stars.dot(&extended);

        (softmax(&ball_logits), softmax(&star_logits))
    } else {
        // Uniform fallback
        (
            vec![1.0 / 50.0; 50],
            vec![1.0 / 12.0; 12],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_draws(n: usize) -> Vec<Draw> {
        (0..n)
            .map(|i| {
                let base = (i % 10) as u8;
                Draw {
                    draw_id: format!("{:03}", i),
                    day: if i % 2 == 0 {
                        "MARDI".to_string()
                    } else {
                        "VENDREDI".to_string()
                    },
                    date: format!("2024-01-{:02}", (i % 28) + 1),
                    balls: [
                        (base * 5 + 1).min(50).max(1),
                        (base * 5 + 2).min(50).max(1),
                        (base * 5 + 3).min(50).max(1),
                        (base * 5 + 4).min(50).max(1),
                        (base * 5 + 5).min(50).max(1),
                    ],
                    stars: [(base % 12 + 1), ((base + 1) % 12 + 1)],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                }
            })
            .collect()
    }

    #[test]
    fn test_data_split_covers_all() {
        let split = DataSplit::new(100).unwrap();
        assert_eq!(split.train.start, 0);
        assert_eq!(split.test.end, 100);
        assert_eq!(split.train.end, split.val.start);
        assert_eq!(split.val.end, split.test.start);
    }

    #[test]
    fn test_data_split_too_small() {
        assert!(DataSplit::new(5).is_err());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum={sum}");
    }

    #[test]
    fn test_softmax_no_negatives() {
        let logits = Array1::from_vec(vec![-100.0, 0.0, 100.0]);
        let probs = softmax(&logits);
        for &p in &probs {
            assert!(p >= 0.0, "negative prob: {p}");
        }
    }

    #[test]
    fn test_softmax_overflow_stability() {
        let logits = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "overflow sum={sum}");
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_train_small() {
        let draws = make_test_draws(100);
        let config = EsnConfig {
            reservoir_size: 20,
            spectral_radius: 0.9,
            sparsity: 0.8,
            leaking_rate: 0.3,
            ridge_lambda: 1e-2,
            input_scaling: 0.1,
            encoding: crate::config::Encoding::OneHot,
            washout: 5,
            noise_amplitude: 0.0,
            seed: 42,
        };
        let (_esn, result) = train_and_evaluate(&draws, &config).unwrap();
        // Probabilities should be valid
        assert!(result.val_ball_hit_rate >= 0.0);
        assert!(result.val_star_hit_rate >= 0.0);
        assert!(result.test_ball_hit_rate >= 0.0);
        assert!(result.train_time_ms > 0 || result.train_time_ms == 0); // just runs
    }

    #[test]
    fn test_predict_next_sums_to_one() {
        let draws = make_test_draws(100);
        let config = EsnConfig {
            reservoir_size: 20,
            spectral_radius: 0.9,
            sparsity: 0.8,
            leaking_rate: 0.3,
            ridge_lambda: 1e-2,
            input_scaling: 0.1,
            encoding: crate::config::Encoding::OneHot,
            washout: 5,
            noise_amplitude: 0.0,
            seed: 42,
        };
        let (mut esn, _) = train_and_evaluate(&draws, &config).unwrap();
        let (ball_probs, star_probs) = predict_next(&mut esn, &draws);

        assert_eq!(ball_probs.len(), 50);
        assert_eq!(star_probs.len(), 12);

        let ball_sum: f64 = ball_probs.iter().sum();
        let star_sum: f64 = star_probs.iter().sum();
        assert!((ball_sum - 1.0).abs() < 1e-6, "ball_sum={ball_sum}");
        assert!((star_sum - 1.0).abs() < 1e-6, "star_sum={star_sum}");

        assert!(ball_probs.iter().all(|&p| p >= 0.0));
        assert!(star_probs.iter().all(|&p| p >= 0.0));
    }
}
