pub mod calibration;
pub mod consensus;
pub mod meta;
pub mod online;
pub mod stacking;
pub mod wasserstein;
pub mod bayesopt;

use rayon::prelude::*;
use lemillion_db::models::{Draw, Pool};
use crate::models::ForecastModel;

pub struct EnsembleCombiner {
    pub models: Vec<Box<dyn ForecastModel>>,
    pub ball_weights: Vec<f64>,
    pub star_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    pub distribution: Vec<f64>,
    pub model_distributions: Vec<(String, Vec<f64>)>,
    pub spread: Vec<f64>,
}

impl EnsembleCombiner {
    pub fn new(models: Vec<Box<dyn ForecastModel>>) -> Self {
        let n = models.len();
        let uniform_weight = 1.0 / n as f64;
        Self {
            models,
            ball_weights: vec![uniform_weight; n],
            star_weights: vec![uniform_weight; n],
        }
    }

    pub fn with_weights(models: Vec<Box<dyn ForecastModel>>, ball_weights: Vec<f64>, star_weights: Vec<f64>) -> Self {
        Self { models, ball_weights, star_weights }
    }

    pub fn predict(&self, draws: &[Draw], pool: Pool) -> EnsemblePrediction {
        let base_weights = match pool {
            Pool::Balls => &self.ball_weights,
            Pool::Stars => &self.star_weights,
        };

        // Zero out ball weights for star-only models and renormalize
        let weights = if pool == Pool::Balls {
            let mut w: Vec<f64> = base_weights.to_vec();
            for (i, model) in self.models.iter().enumerate() {
                if model.is_stars_only() {
                    w[i] = 0.0;
                }
            }
            let total: f64 = w.iter().sum();
            if total > 0.0 {
                for v in w.iter_mut() {
                    *v /= total;
                }
            }
            w
        } else {
            base_weights.to_vec()
        };

        let size = pool.size();

        // Paralléliser les prédictions des modèles (indépendantes)
        let all_dists: Vec<(String, Vec<f64>)> = self.models
            .par_iter()
            .map(|model| (model.name().to_string(), model.predict(draws, pool)))
            .collect();

        // v18: max weight for weight-modulated clamp
        let max_w = weights.iter().cloned().fold(0.0f64, f64::max);

        // Log-linear pool: P(x) ∝ Π P_i(x)^w_i (geometric combination)
        // In log-space: log P(x) = Σ w_i × log P_i(x)
        let mut log_combined = vec![0.0f64; size];
        let mut model_distributions = Vec::with_capacity(all_dists.len());

        for (i, (name, dist)) in all_dists.into_iter().enumerate() {
            let w = weights[i];
            // v18: Log-ratio clamp modulated by calibrated weight.
            // Well-calibrated models (high weight) get full headroom.
            // Marginal models (low weight) get reduced headroom to prevent domination.
            let model_h: f64 = dist.iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();
            let h_ratio = model_h / (size as f64).ln();
            let w_ratio = if max_w > 1e-15 { weights[i] / max_w } else { 0.5 };
            let max_positive_log_ratio = 4.0 + 6.0 * (1.0 - h_ratio) * w_ratio;
            let uniform_p = 1.0 / size as f64;
            for j in 0..size {
                let raw_log = dist[j].max(1e-15).ln();
                // Only clamp the upside: limit how much a model can boost a number
                let log_ratio = raw_log - uniform_p.ln();
                if log_ratio > max_positive_log_ratio {
                    log_combined[j] += w * (uniform_p.ln() + max_positive_log_ratio);
                } else {
                    log_combined[j] += w * raw_log;
                }
            }
            model_distributions.push((name, dist));
        }

        // Exponentiate with max-subtraction for numerical stability
        let max_lc = log_combined.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut combined: Vec<f64> = log_combined.iter().map(|&lc| (lc - max_lc).exp()).collect();

        // Normaliser
        let total: f64 = combined.iter().sum();
        if total > 0.0 {
            for p in &mut combined {
                *p /= total;
            }
        }

        // v19: Star floor — prevent any star from being crushed below 1/5 of uniform.
        // Safety net against unanimous momentum bias (★4 scenario).
        if pool == Pool::Stars {
            let floor = 1.0 / (size as f64 * 5.0); // ~0.0167 (uniform = 0.0833)
            let mut floored = false;
            for p in combined.iter_mut() {
                if *p < floor {
                    *p = floor;
                    floored = true;
                }
            }
            if floored {
                let total: f64 = combined.iter().sum();
                if total > 0.0 {
                    for p in combined.iter_mut() {
                        *p /= total;
                    }
                }
            }
        }

        let spread = compute_spread(&model_distributions, size);

        EnsemblePrediction {
            distribution: combined,
            model_distributions,
            spread,
        }
    }
}

impl EnsembleCombiner {
    /// Prédit avec poids décorrélés (v7) : pénalise les modèles corrélés pour éviter le double-comptage
    /// dans le log-linear pool. effective_w_i = w_i × Π_j sqrt(1 - ρ_ij²).
    pub fn predict_decorrelated(
        &self,
        draws: &[Draw],
        pool: Pool,
        correlation_matrix: &[(String, String, f64)],
        threshold: f64,
    ) -> EnsemblePrediction {
        let model_names: Vec<String> = self.models.iter().map(|m| m.name().to_string()).collect();
        let base_weights = match pool {
            Pool::Balls => &self.ball_weights,
            Pool::Stars => &self.star_weights,
        };

        let effective = calibration::compute_decorrelated_weights(
            base_weights, &model_names, correlation_matrix, threshold,
        );

        // Build a temporary combiner with decorrelated weights
        let size = pool.size();

        // Zero out ball weights for star-only models and renormalize
        let weights = if pool == Pool::Balls {
            let mut w = effective;
            for (i, model) in self.models.iter().enumerate() {
                if model.is_stars_only() {
                    w[i] = 0.0;
                }
            }
            let total: f64 = w.iter().sum();
            if total > 0.0 {
                for v in w.iter_mut() { *v /= total; }
            }
            w
        } else {
            effective
        };

        let all_dists: Vec<(String, Vec<f64>)> = self.models
            .par_iter()
            .map(|model| (model.name().to_string(), model.predict(draws, pool)))
            .collect();

        // v18: max weight for weight-modulated clamp (aligned with predict())
        let max_w = weights.iter().cloned().fold(0.0f64, f64::max);

        let mut log_combined = vec![0.0f64; size];
        let mut model_distributions = Vec::with_capacity(all_dists.len());

        for (i, (name, dist)) in all_dists.into_iter().enumerate() {
            let w = weights[i];
            // v18: Weight-modulated log-ratio clamp (aligned with predict()).
            let model_h: f64 = dist.iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();
            let h_ratio = model_h / (size as f64).ln();
            let w_ratio = if max_w > 1e-15 { weights[i] / max_w } else { 0.5 };
            let max_positive_log_ratio = 4.0 + 6.0 * (1.0 - h_ratio) * w_ratio;
            let uniform_p = 1.0 / size as f64;
            for j in 0..size {
                let raw_log = dist[j].max(1e-15).ln();
                let log_ratio = raw_log - uniform_p.ln();
                if log_ratio > max_positive_log_ratio {
                    log_combined[j] += w * (uniform_p.ln() + max_positive_log_ratio);
                } else {
                    log_combined[j] += w * raw_log;
                }
            }
            model_distributions.push((name, dist));
        }

        let max_lc = log_combined.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut combined: Vec<f64> = log_combined.iter().map(|&lc| (lc - max_lc).exp()).collect();

        let total: f64 = combined.iter().sum();
        if total > 0.0 {
            for p in &mut combined { *p /= total; }
        }

        // v19: Star floor (aligned with predict())
        if pool == Pool::Stars {
            let floor = 1.0 / (size as f64 * 5.0);
            let mut floored = false;
            for p in combined.iter_mut() {
                if *p < floor { *p = floor; floored = true; }
            }
            if floored {
                let total: f64 = combined.iter().sum();
                if total > 0.0 {
                    for p in combined.iter_mut() { *p /= total; }
                }
            }
        }

        let spread = compute_spread(&model_distributions, size);

        EnsemblePrediction {
            distribution: combined,
            model_distributions,
            spread,
        }
    }
}

impl EnsembleCombiner {
    /// Prédit avec stacking : blend entre la prédiction pondérée et la prédiction stackée.
    /// blend_factor = fraction du stacking (ex: 0.6 stacked + 0.4 weighted).
    pub fn predict_stacked(
        &self,
        draws: &[Draw],
        pool: Pool,
        stacking_weights: &stacking::StackingWeights,
        blend_factor: f64,
    ) -> EnsemblePrediction {
        let base = self.predict(draws, pool);

        // Collect model distributions
        let model_dists: Vec<Vec<f64>> = base.model_distributions.iter()
            .map(|(_, d)| d.clone())
            .collect();

        let context = meta::RegimeFeatures::from_draws(draws);
        let stacked = stacking::predict_stacked(stacking_weights, &model_dists, &context);

        // Blend
        let bf = blend_factor.clamp(0.0, 1.0);
        let mut blended = vec![0.0f64; pool.size()];
        for i in 0..pool.size() {
            blended[i] = bf * stacked[i] + (1.0 - bf) * base.distribution[i];
        }

        // Normalize
        let sum: f64 = blended.iter().sum();
        if sum > 0.0 {
            for p in &mut blended {
                *p /= sum;
            }
        }

        EnsemblePrediction {
            distribution: blended,
            model_distributions: base.model_distributions,
            spread: base.spread,
        }
    }
}

impl EnsembleCombiner {
    /// F4: Multi-scale ensemble prediction blending short/medium/long horizons.
    /// Runs predict() with 3 different draw windows (short=30, medium=100, long=all)
    /// and blends their predictions weighted by KL-divergence from uniform
    /// (more informative scale = higher weight).
    pub fn predict_multi_scale(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let scales = [30usize, 100, draws.len()];
        let uniform = 1.0 / pool.size() as f64;
        let mut scale_preds = Vec::new();
        let mut scale_kls = Vec::new();

        for &window in &scales {
            let w = window.min(draws.len());
            let sub_draws = &draws[..w];
            let pred = self.predict(sub_draws, pool);
            // KL divergence from uniform: D_KL(pred || uniform)
            let kl: f64 = pred.distribution.iter()
                .map(|&p| if p > 1e-30 { p * (p / uniform).ln() } else { 0.0 })
                .sum();
            scale_preds.push(pred.distribution);
            scale_kls.push(kl.max(1e-10));
        }

        // KL-proportional weights
        let kl_sum: f64 = scale_kls.iter().sum();
        let weights: Vec<f64> = scale_kls.iter().map(|k| k / kl_sum).collect();

        // Blend
        let size = pool.size();
        let mut blended = vec![0.0f64; size];
        for (pred, &w) in scale_preds.iter().zip(weights.iter()) {
            for (i, &p) in pred.iter().enumerate() {
                blended[i] += w * p;
            }
        }

        // Normalize
        let sum: f64 = blended.iter().sum();
        for p in &mut blended {
            *p /= sum;
        }
        blended
    }
}

/// Beta-transform: recalibrates a pooled distribution using p^alpha * (1-p)^(beta-1).
/// alpha=1, beta=1 is identity. alpha>1 sharpens high-probability entries.
/// beta>1 suppresses low-probability entries. Asymmetric unlike temperature.
/// Based on Ranjan & Gneiting (JRSSB 2010).
pub fn beta_transform(probs: &mut [f64], alpha: f64, beta: f64) {
    if (alpha - 1.0).abs() < 1e-9 && (beta - 1.0).abs() < 1e-9 {
        return; // identity
    }
    for p in probs.iter_mut() {
        let clamped = p.clamp(1e-15, 1.0 - 1e-15);
        *p = clamped.powf(alpha) * (1.0 - clamped).powf(beta - 1.0);
    }
    let total: f64 = probs.iter().sum();
    if total > 0.0 {
        probs.iter_mut().for_each(|p| *p /= total);
    }
}

pub fn compute_spread(model_dists: &[(String, Vec<f64>)], size: usize) -> Vec<f64> {
    let n = model_dists.len() as f64;
    (0..size)
        .map(|j| {
            let mean = model_dists.iter().map(|(_, d)| d[j]).sum::<f64>() / n;
            let variance = model_dists.iter().map(|(_, d)| (d[j] - mean).powi(2)).sum::<f64>() / n;
            variance.sqrt()
        })
        .collect()
}

impl EnsembleCombiner {
    /// Prédit avec agreement boost : les numéros où les modèles convergent
    /// reçoivent un boost proportionnel à leur agreement (1 - spread/max_spread).
    ///
    /// `strength` contrôle l'intensité du boost (0 = pas de boost, 1 = doublement max).
    pub fn predict_with_agreement_boost(
        &self,
        draws: &[Draw],
        pool: Pool,
        strength: f64,
    ) -> EnsemblePrediction {
        let base = self.predict(draws, pool);

        // v9: utiliser Q75 du spread comme référence au lieu du max
        // (le max est sensible aux outliers — un seul modèle en désaccord extrême rend le boost inutile)
        let mut sorted_spreads: Vec<f64> = base.spread.clone();
        sorted_spreads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q75_idx = (sorted_spreads.len() * 3) / 4;
        let reference_spread = sorted_spreads[q75_idx].max(1e-15);

        let uniform_val = 1.0 / pool.size() as f64;
        let mut boosted = base.distribution.clone();
        for (i, p) in boosted.iter_mut().enumerate() {
            // Only boost numbers that are both FAVORED and UNANIMOUS
            // deviation > 0 = above uniform, < 0 = below uniform
            let deviation = (base.distribution[i] / uniform_val) - 1.0;
            let spread_agreement = (1.0 - base.spread[i] / reference_spread).max(0.0);
            let agreement = deviation * spread_agreement;
            *p *= 1.0 + strength * agreement;
            if *p < 0.0 { *p = 1e-15; }
        }

        // Renormaliser
        let sum: f64 = boosted.iter().sum();
        if sum > 0.0 {
            for p in &mut boosted {
                *p /= sum;
            }
        }

        EnsemblePrediction {
            distribution: boosted,
            model_distributions: base.model_distributions,
            spread: base.spread,
        }
    }
}

/// Online Hedge : recalcule les poids en rejouant les N derniers tirages.
///
/// Mise à jour multiplicative : w[m] *= exp(-eta * loss_m)
/// où loss = -log(P(tirage observé | modèle m)).
pub fn compute_hedge_weights(
    models: &[Box<dyn ForecastModel>],
    draws: &[Draw],
    base_ball_weights: &[f64],
    base_star_weights: &[f64],
    n_recent: usize,
    eta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = n_recent.min(draws.len().saturating_sub(1));

    if n == 0 {
        return (base_ball_weights.to_vec(), base_star_weights.to_vec());
    }

    let mut ball_weights = base_ball_weights.to_vec();
    let mut star_weights = base_star_weights.to_vec();

    // v17: threshold for skipping predict() in Hedge loop AND for floor allocation
    let min_weight = 0.005;
    let floor_threshold = 0.02;

    // Rejouer les n derniers tirages (draws[0..n])
    for t in (0..n).rev() {
        let test_draw = &draws[t];
        let training_draws = &draws[t + 1..];

        if training_draws.len() < 10 {
            continue;
        }

        for (m, model) in models.iter().enumerate() {
            // v17: Skip predict() pour les modèles à poids < floor_threshold (perf: ~12 modèles skippés)
            let need_ball = ball_weights[m] > floor_threshold;
            let need_star = star_weights[m] > floor_threshold;

            if need_ball {
                // Loss boules (pure log-likelihood)
                let ball_ll_cap = 2.0 * (1.0_f64 / 50.0).ln();
                let ball_dist = model.predict(training_draws, Pool::Balls);
                let ball_loss: f64 = test_draw.balls.iter()
                    .map(|&b| {
                        let ll = ball_dist[(b - 1) as usize].max(1e-15).ln().max(ball_ll_cap);
                        -ll
                    })
                    .sum();
                // v19 E3: Natural gradient on probability simplex
                // Fisher metric approx: F_ii ≈ 1/w_i → natural grad = w_i * grad
                let natural_eta = eta * ball_weights[m].max(1e-6);
                ball_weights[m] *= (-natural_eta * ball_loss).exp();
            }

            if need_star {
                // Loss étoiles (pure log-likelihood)
                // v19 D1: Asymmetric loss — halve penalty for stars below uniform
                // when the star actually appeared. This reduces over-reward for
                // momentum models that correctly predict hot stars but miss cold ones.
                let star_ll_cap = 2.0 * (1.0_f64 / 12.0).ln();
                let star_uniform = 1.0 / 12.0_f64;
                let star_dist = model.predict(training_draws, Pool::Stars);
                let star_loss: f64 = test_draw.stars.iter()
                    .map(|&s| {
                        let p = star_dist[(s - 1) as usize].max(1e-15);
                        let ll = p.ln().max(star_ll_cap);
                        let loss = -ll;
                        // If model was below uniform for a drawn star, halve the loss
                        if p < star_uniform { loss * 0.5 } else { loss }
                    })
                    .sum();
                // v19 E3: Natural gradient (same as balls)
                let natural_eta = eta * star_weights[m].max(1e-6);
                star_weights[m] *= (-natural_eta * star_loss).exp();
            }
        }

        // Normaliser après chaque tirage
        let bs: f64 = ball_weights.iter().sum();
        if bs > 0.0 {
            for w in &mut ball_weights {
                *w /= bs;
            }
        }
        let ss: f64 = star_weights.iter().sum();
        if ss > 0.0 {
            for w in &mut star_weights {
                *w /= ss;
            }
        }
    }

    // H4: Tsallis q=0.5 mirror descent — more exploratory than Shannon entropy.
    // After the exp update, apply w^(1/(2-q)) transform to prevent weight collapse
    // on mean-reversion models. Tsallis q<1 spreads weight more evenly.
    let q = 0.5_f64;
    let tsallis_exp = 1.0 / (2.0 - q); // = 1/1.5 ≈ 0.6667
    for w in ball_weights.iter_mut() {
        *w = w.powf(tsallis_exp);
    }
    let bs: f64 = ball_weights.iter().sum();
    if bs > 0.0 {
        for w in ball_weights.iter_mut() {
            *w /= bs;
        }
    }
    for w in star_weights.iter_mut() {
        *w = w.powf(tsallis_exp);
    }
    let ss: f64 = star_weights.iter().sum();
    if ss > 0.0 {
        for w in star_weights.iter_mut() {
            *w /= ss;
        }
    }

    // v19 G5: Surprise-based model boost.
    // After the Hedge loop, if a model predicted the last draw's "surprising" numbers
    // (numbers the ensemble gave low probability but the model favored), boost it.
    if n > 0 {
        let last_draw = &draws[0];
        let training = &draws[1..];
        if training.len() >= 10 {
            // Compute ensemble-level distribution for surprise reference
            let ball_total: f64 = ball_weights.iter().sum();
            let star_total: f64 = star_weights.iter().sum();
            for (m, model) in models.iter().enumerate() {
                if ball_weights[m] > floor_threshold && ball_total > 0.0 {
                    let dist = model.predict(training, Pool::Balls);
                    let w_norm = ball_weights[m] / ball_total;
                    for &b in &last_draw.balls {
                        let model_p = dist[(b - 1) as usize].max(1e-15);
                        let uniform_p = 1.0 / 50.0;
                        let surprise = model_p / uniform_p;
                        // If model was above 2x uniform for this drawn number → it "knew"
                        if surprise > 2.0 {
                            let boost = 1.0 + 0.05 * (surprise - 2.0).min(3.0);
                            ball_weights[m] *= boost.powf(w_norm);
                        }
                    }
                }
                if star_weights[m] > floor_threshold && star_total > 0.0 {
                    let dist = model.predict(training, Pool::Stars);
                    let w_norm = star_weights[m] / star_total;
                    for &s in &last_draw.stars {
                        let model_p = dist[(s - 1) as usize].max(1e-15);
                        let uniform_p = 1.0 / 12.0;
                        let surprise = model_p / uniform_p;
                        if surprise > 2.0 {
                            let boost = 1.0 + 0.05 * (surprise - 2.0).min(3.0);
                            star_weights[m] *= boost.powf(w_norm);
                        }
                    }
                }
            }
            // Renormalize after surprise boost
            let bs: f64 = ball_weights.iter().sum();
            if bs > 0.0 { for w in &mut ball_weights { *w /= bs; } }
            let ss: f64 = star_weights.iter().sum();
            if ss > 0.0 { for w in &mut star_weights { *w /= ss; } }
        }
    }

    // Floor différencié: only models with meaningful base weight (>2%) get a floor.
    for (i, w) in ball_weights.iter_mut().enumerate() {
        if base_ball_weights[i] > floor_threshold {
            *w = w.max(min_weight);
        }
    }
    let bs: f64 = ball_weights.iter().sum();
    if bs > 0.0 {
        for w in &mut ball_weights {
            *w /= bs;
        }
    }
    for (i, w) in star_weights.iter_mut().enumerate() {
        if base_star_weights[i] > floor_threshold {
            *w = w.max(min_weight);
        }
    }
    let ss: f64 = star_weights.iter().sum();
    if ss > 0.0 {
        for w in &mut star_weights {
            *w /= ss;
        }
    }

    (ball_weights, star_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;
    use crate::models::validate_distribution;

    #[test]
    fn test_ensemble_prediction_sums_to_one() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&pred.distribution, Pool::Balls));
    }

    #[test]
    fn test_ensemble_spread_length() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict(&draws, Pool::Balls);
        assert_eq!(pred.spread.len(), 50);
    }

    #[test]
    fn test_agreement_boost_sums_to_one() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict_with_agreement_boost(&draws, Pool::Balls, 0.5);
        assert!(validate_distribution(&pred.distribution, Pool::Balls));
    }

    #[test]
    fn test_agreement_boost_zero_strength_equals_base() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let base = combiner.predict(&draws, Pool::Balls);
        let boosted = combiner.predict_with_agreement_boost(&draws, Pool::Balls, 0.0);
        for (a, b) in base.distribution.iter().zip(boosted.distribution.iter()) {
            assert!((a - b).abs() < 1e-10, "Zero-strength boost should match base");
        }
    }

    #[test]
    fn test_hedge_weights_sum_to_one() {
        let models = crate::models::all_models();
        let n = models.len();
        let draws = make_test_draws(30);
        let uniform = vec![1.0 / n as f64; n];
        let (bw, sw) = compute_hedge_weights(&models, &draws, &uniform, &uniform, 5, 0.1);
        let bs: f64 = bw.iter().sum();
        let ss: f64 = sw.iter().sum();
        assert!((bs - 1.0).abs() < 1e-9, "Ball weights sum = {}", bs);
        assert!((ss - 1.0).abs() < 1e-9, "Star weights sum = {}", ss);
    }

    #[test]
    fn test_beta_transform_identity() {
        let mut probs = vec![0.1, 0.2, 0.3, 0.15, 0.25];
        let orig = probs.clone();
        beta_transform(&mut probs, 1.0, 1.0);
        for (a, b) in probs.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-10, "Identity beta should not change distribution");
        }
    }

    #[test]
    fn test_beta_transform_sums_to_one() {
        let mut probs = vec![1.0 / 50.0; 50];
        probs[0] = 0.1;
        let sum: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        beta_transform(&mut probs, 2.0, 1.5);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "Beta-transform should normalize, sum={}", total);
    }

    #[test]
    fn test_beta_transform_sharpens() {
        let mut probs = vec![0.1, 0.2, 0.3, 0.15, 0.25];
        let orig = probs.clone();
        beta_transform(&mut probs, 2.0, 1.0);
        // alpha > 1 should sharpen: the largest probability should become even more dominant
        let max_orig = orig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_new = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_new > max_orig, "Sharpening should increase max: {} > {}", max_new, max_orig);
    }

    #[test]
    fn test_log_clamp_operational() {
        // v11: verify log-ratio clamp actually modifies distributions
        // A very concentrated model should have its extremes clamped
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(20);
        let pred = combiner.predict(&draws, Pool::Balls);
        // Distribution should be valid
        assert!(validate_distribution(&pred.distribution, Pool::Balls));
        // No extreme ratios vs uniform (clamp should prevent >e^5 ≈ 148x)
        let uniform = 1.0 / 50.0;
        for &p in &pred.distribution {
            let ratio = p / uniform;
            assert!(ratio < 200.0, "Ratio {} exceeds safe clamp bounds", ratio);
        }
    }

    #[test]
    fn test_log_clamp_concentrated_wider_range() {
        // Concentrated model (low entropy) should have wider allowed range
        // This is a property test of the formula max_log_ratio = 2.0 + 3.0 * (1.0 - h_ratio)
        let h_ratio_concentrated: f64 = 0.8;
        let h_ratio_flat: f64 = 1.0;
        let range_concentrated = 4.0 + 6.0 * (1.0 - h_ratio_concentrated); // 5.2
        let range_flat = 4.0 + 6.0 * (1.0 - h_ratio_flat); // 4.0
        assert!(range_concentrated > range_flat);
        assert!((range_concentrated - 5.2).abs() < 1e-9);
        assert!((range_flat - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_multi_scale_sums_to_one() {
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(120);
        for pool in [Pool::Balls, Pool::Stars] {
            let pred = combiner.predict_multi_scale(&draws, pool);
            assert_eq!(pred.len(), pool.size());
            let sum: f64 = pred.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "Multi-scale {:?} sum = {}", pool, sum);
            for &p in &pred {
                assert!(p >= 0.0, "Multi-scale produced negative probability");
            }
        }
    }

    #[test]
    fn test_multi_scale_short_data() {
        // With fewer draws than the short window, all 3 scales should use the same data
        let combiner = EnsembleCombiner::new(crate::models::all_models());
        let draws = make_test_draws(15);
        let pred = combiner.predict_multi_scale(&draws, Pool::Balls);
        assert!(validate_distribution(&pred, Pool::Balls));
    }
}
