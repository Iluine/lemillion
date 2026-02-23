use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{SeedableRng, RngExt};
use rand::distr::Uniform;

use lemillion_db::models::Pool;

use crate::reservoir::Reservoir;

/// Compute hit rate: average fraction of actual numbers in the top-K predicted.
/// predictions[i] is a probability distribution (50 for balls, 12 for stars).
/// actuals[i] is the actual drawn numbers ([u8; 5] for balls, [u8; 2] for stars).
pub fn hit_rate<const N: usize>(
    predictions: &[Vec<f64>],
    actuals: &[[u8; N]],
    pool: Pool,
    top_k: usize,
) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    let pick_count = pool.pick_count();
    let mut total_hits = 0.0;

    for (pred, actual) in predictions.iter().zip(actuals.iter()) {
        // Get top-K indices by probability
        let mut indices: Vec<usize> = (0..pred.len()).collect();
        indices.sort_by(|&a, &b| pred[b].partial_cmp(&pred[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_set: Vec<u8> = indices.iter().take(top_k).map(|&i| (i + 1) as u8).collect();

        let hits = actual.iter().filter(|&&n| top_set.contains(&n)).count();
        total_hits += hits as f64 / pick_count as f64;
    }

    total_hits / predictions.len() as f64
}

/// Random baseline: expected hit rate for uniform random selection.
/// E[hits] = pick_count * top_k / pool_size
pub fn random_baseline(pool: Pool, top_k: usize) -> f64 {
    let pick = pool.pick_count() as f64;
    let size = pool.size() as f64;
    let k = top_k as f64;
    (pick * k / size) / pick // = top_k / pool_size
}

/// Calibration bins: group predictions by predicted probability, compare to actual frequency.
/// Returns Vec of (bin_center, avg_predicted, actual_freq, count).
pub fn calibration_bins<const N: usize>(
    predictions: &[Vec<f64>],
    actuals: &[[u8; N]],
    _pool: Pool,
    n_bins: usize,
) -> Vec<(f64, f64, f64, usize)> {
    if n_bins == 0 || predictions.is_empty() {
        return Vec::new();
    }

    let bin_width = 1.0 / n_bins as f64;
    let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); n_bins]; // (sum_pred, sum_actual, count)

    for (pred, actual) in predictions.iter().zip(actuals.iter()) {
        let actual_set: Vec<u8> = actual.to_vec();
        for (i, &p) in pred.iter().enumerate() {
            let number = (i + 1) as u8;
            let bin_idx = ((p / bin_width) as usize).min(n_bins - 1);
            let is_hit = if actual_set.contains(&number) { 1.0 } else { 0.0 };
            bins[bin_idx].0 += p;
            bins[bin_idx].1 += is_hit;
            bins[bin_idx].2 += 1;
        }
    }

    bins.iter()
        .enumerate()
        .filter(|(_, b)| b.2 > 0)
        .map(|(i, b)| {
            let center = (i as f64 + 0.5) * bin_width;
            let avg_pred = b.0 / b.2 as f64;
            let actual_freq = b.1 / b.2 as f64;
            (center, avg_pred, actual_freq, b.2)
        })
        .collect()
}

/// Estimate the Lyapunov exponent using twin trajectory method.
/// Positive = chaotic, negative = stable (echo state property).
pub fn lyapunov_exponent(
    reservoir: &mut Reservoir,
    inputs: &[Array1<f64>],
    seed: u64,
) -> f64 {
    let epsilon = 1e-8;
    let n_probes = 10;
    let traj_len = 50;

    if inputs.len() < traj_len + n_probes {
        return 0.0;
    }

    let mut rng = StdRng::seed_from_u64(seed + 9999);

    // Run reservoir to a mid-point to get a representative state
    reservoir.reset_state();
    let warmup = inputs.len() / 4;
    for input in &inputs[..warmup] {
        reservoir.step(input, &mut rng);
    }

    let n = reservoir.state.len();
    let mut total_lambda = 0.0;
    let mut count = 0;

    let probe_stride = (inputs.len() - warmup - traj_len).max(1) / n_probes.max(1);

    // Pre-allocate buffers
    let mut states_orig: Vec<Array1<f64>> = (0..traj_len).map(|_| Array1::zeros(n)).collect();
    let mut delta = Array1::zeros(n);
    let mut original_state = Array1::zeros(n);

    for probe in 0..n_probes {
        let start_idx = warmup + probe * probe_stride;
        if start_idx + traj_len >= inputs.len() {
            break;
        }

        // Save original state
        original_state.assign(&reservoir.state);

        // Perturb state
        let dist = Uniform::new(-1.0, 1.0).unwrap();
        let perturbation: Array1<f64> = Array1::from_shape_fn(n, |_| rng.sample(dist));
        let pert_norm = perturbation.dot(&perturbation).sqrt();
        if pert_norm < 1e-15 {
            continue;
        }
        let pert_scaled = &perturbation * (epsilon / pert_norm);

        // Run original trajectory
        reservoir.state.assign(&original_state);
        let mut rng_orig = StdRng::seed_from_u64(seed + probe as u64);
        for t in 0..traj_len {
            reservoir.step(&inputs[start_idx + t], &mut rng_orig);
            states_orig[t].assign(&reservoir.state);
        }

        // Run perturbed trajectory
        reservoir.state.assign(&original_state);
        reservoir.state += &pert_scaled;
        let mut rng_pert = StdRng::seed_from_u64(seed + probe as u64);
        let mut lambda_sum = 0.0;
        let mut lambda_count = 0;
        for t in 0..traj_len {
            reservoir.step(&inputs[start_idx + t], &mut rng_pert);
            delta.assign(&reservoir.state);
            delta -= &states_orig[t];
            let delta_norm = delta.dot(&delta).sqrt();
            if delta_norm > 1e-30 {
                lambda_sum += (delta_norm / epsilon).ln();
                lambda_count += 1;
            }
        }

        if lambda_count > 0 {
            total_lambda += lambda_sum / lambda_count as f64;
            count += 1;
        }

        // Restore state
        reservoir.state.assign(&original_state);
    }

    if count > 0 {
        total_lambda / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_rate_perfect() {
        // Prediction puts all mass on the actual numbers
        let mut pred = vec![0.0; 50];
        pred[0] = 0.5; // number 1
        pred[4] = 0.3; // number 5
        pred[9] = 0.1; // number 10
        pred[19] = 0.05; // number 20
        pred[49] = 0.05; // number 50

        let actual: [u8; 5] = [1, 5, 10, 20, 50];
        let rate = hit_rate(&[pred], &[actual], Pool::Balls, 5);
        assert!((rate - 1.0).abs() < 1e-10, "perfect hit_rate should be 1.0, got {rate}");
    }

    #[test]
    fn test_hit_rate_zero() {
        // Prediction is completely wrong
        let mut pred = vec![0.0; 50];
        pred[5] = 0.3;
        pred[6] = 0.3;
        pred[7] = 0.2;
        pred[8] = 0.1;
        pred[9] = 0.1;

        let actual: [u8; 5] = [1, 2, 3, 4, 5];
        let rate = hit_rate(&[pred], &[actual], Pool::Balls, 5);
        assert!((rate - 0.0).abs() < 1e-10, "should be 0.0, got {rate}");
    }

    #[test]
    fn test_random_baseline_balls() {
        let baseline = random_baseline(Pool::Balls, 5);
        // 5 * 5 / 50 / 5 = 5/50 = 0.1
        assert!((baseline - 0.1).abs() < 1e-10, "baseline={baseline}");
    }

    #[test]
    fn test_random_baseline_stars() {
        let baseline = random_baseline(Pool::Stars, 2);
        // 2 * 2 / 12 / 2 = 2/12 = 1/6
        assert!((baseline - 1.0 / 6.0).abs() < 1e-10, "baseline={baseline}");
    }

    #[test]
    fn test_calibration_bins_basic() {
        let pred = vec![0.02; 50]; // uniform
        let actual: [u8; 5] = [1, 2, 3, 4, 5];
        let bins = calibration_bins(&[pred], &[actual], Pool::Balls, 10);
        assert!(!bins.is_empty());
        // All predictions fall in the same bin (~0.02)
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].3, 50); // 50 numbers
    }
}
