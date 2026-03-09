use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// ContextKnnModel — k-NN basé sur le contexte du dernier tirage.
///
/// Pour chaque tirage historique t, calcule un vecteur contexte 4D depuis draws[t+1].
/// Trouve les k plus proches voisins du contexte courant (draws[0]),
/// puis pondère les numéros par 1/distance.
pub struct ContextKnnModel {
    k: usize,
    smoothing: f64,
    min_draws: usize,
}

impl Default for ContextKnnModel {
    fn default() -> Self {
        Self {
            k: 15,
            smoothing: 0.30,
            min_draws: 30,
        }
    }
}

/// Compute 4D context features from a draw (and optionally the previous draw for mod4 cosine).
fn context_features(draw: &Draw, prev_draw: Option<&Draw>) -> [f64; 4] {
    let ball_sum: f64 = draw.balls.iter().map(|&b| b as f64).sum();
    let sum_norm = (ball_sum - 15.0) / 235.0;

    let spread = *draw.balls.iter().max().unwrap() as f64 - *draw.balls.iter().min().unwrap() as f64;
    let spread_norm = spread / 49.0;

    let odd_count = draw.balls.iter().filter(|&&b| b % 2 == 1).count() as f64 / 5.0;

    // Normalize mod4_cosine from [-1, 1] to [0, 1] for uniform feature scaling
    let mod4_cosine = if let Some(prev) = prev_draw {
        (compute_mod4_cosine(draw, prev) + 1.0) / 2.0
    } else {
        0.5
    };

    [sum_norm, spread_norm, odd_count, mod4_cosine]
}

fn compute_mod4_cosine(a: &Draw, b: &Draw) -> f64 {
    let mut profile_a = [0.0f64; 4];
    let mut profile_b = [0.0f64; 4];

    for &ball in &a.balls {
        profile_a[((ball - 1) % 4) as usize] += 1.0;
    }
    for &ball in &b.balls {
        profile_b[((ball - 1) % 4) as usize] += 1.0;
    }

    let dot: f64 = profile_a.iter().zip(profile_b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = profile_a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = profile_b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.5
    }
}

fn euclidean_distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    // Weighted distance: mod4_cosine (index 3) gets 2x weight
    // reflecting its physical importance (Stresa machine symmetry)
    const WEIGHTS: [f64; 4] = [1.0, 1.0, 1.0, 2.0];
    a.iter().zip(b.iter()).zip(WEIGHTS.iter())
        .map(|((x, y), &w)| w * (x - y).powi(2))
        .sum::<f64>().sqrt()
}

impl ForecastModel for ContextKnnModel {
    fn name(&self) -> &str {
        "ContextKNN"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Current context from draws[0] (and draws[1] for mod4_cosine)
        let current_ctx = context_features(
            &draws[0],
            if draws.len() > 1 { Some(&draws[1]) } else { None },
        );

        // Build historical contexts and their associated "next draw" numbers
        // For t in 1..draws.len()-1: context from draws[t+1] (and draws[t+2]),
        // target = draws[t] (the draw that followed this context)
        let n_neighbors = draws.len().saturating_sub(2);
        let mut neighbors: Vec<(f64, usize)> = Vec::with_capacity(n_neighbors);

        for t in 1..draws.len().saturating_sub(1) {
            let ctx = context_features(
                &draws[t + 1],
                if t + 2 < draws.len() { Some(&draws[t + 2]) } else { None },
            );
            let dist = euclidean_distance(&current_ctx, &ctx);
            neighbors.push((dist, t));
        }

        if neighbors.is_empty() {
            return uniform;
        }

        // Partial sort (quickselect) for top-k nearest instead of full sort
        let k = self.k.min(neighbors.len());
        neighbors.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let nearest = &neighbors[..k];

        // Weighted vote: 1/(distance + epsilon)
        let epsilon = 1e-6;
        let mut scores = vec![0.0f64; size];
        let mut total_weight = 0.0f64;

        for &(dist, target_idx) in nearest {
            let weight = 1.0 / (dist + epsilon);
            total_weight += weight;

            let numbers = pool.numbers_from(&draws[target_idx]);
            for &n in numbers {
                let idx = (n - 1) as usize;
                if idx < size {
                    scores[idx] += weight;
                }
            }
        }

        // Normalize
        if total_weight > 0.0 {
            let pick_count = pool.pick_count() as f64;
            for s in &mut scores {
                *s /= total_weight * pick_count;
            }
        }

        // Smooth with uniform
        let uniform_val = 1.0 / size as f64;
        for s in &mut scores {
            *s = (1.0 - self.smoothing) * *s + self.smoothing * uniform_val;
        }

        // Renormalize
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("k".into(), self.k as f64),
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
    fn test_context_knn_balls_valid() {
        let model = ContextKnnModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_context_knn_stars_valid() {
        let model = ContextKnnModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_context_knn_few_draws_uniform() {
        let model = ContextKnnModel::default();
        let draws = make_test_draws(10);
        let dist_b = model.predict(&draws, Pool::Balls);
        let dist_s = model.predict(&draws, Pool::Stars);
        for &p in &dist_b {
            assert!((p - 1.0 / 50.0).abs() < 1e-10);
        }
        for &p in &dist_s {
            assert!((p - 1.0 / 12.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_context_knn_no_negative() {
        let model = ContextKnnModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_context_knn_deterministic() {
        let model = ContextKnnModel::default();
        let draws = make_test_draws(60);
        let d1 = model.predict(&draws, Pool::Balls);
        let d2 = model.predict(&draws, Pool::Balls);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_context_features_range() {
        let draws = make_test_draws(5);
        let ctx = context_features(&draws[0], Some(&draws[1]));
        assert!(ctx[0] >= -0.1 && ctx[0] <= 1.1, "sum_norm out of range: {}", ctx[0]);
        assert!(ctx[1] >= 0.0 && ctx[1] <= 1.0, "spread_norm out of range: {}", ctx[1]);
        assert!(ctx[2] >= 0.0 && ctx[2] <= 1.0, "odd_count out of range: {}", ctx[2]);
        assert!(ctx[3] >= 0.0 && ctx[3] <= 1.0, "mod4_cosine out of range: {}", ctx[3]);
    }
}
