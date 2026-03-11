use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// StarSpecialist — modèle dédié aux 12 étoiles avec 5 micro-experts combinés via Hedge.
///
/// Experts :
/// 1. Gap-hazard étoiles : fonction de hasard empirique adaptée (gaps plus courts)
/// 2. Conditionnel ball→star : P(étoile | sum_bin des boules du tirage précédent)
/// 3. Balance haut/bas : persistence étoiles 1-6 vs 7-12, EWMA
/// 4. Mod-4 transition étoiles : (star-1)%4 → 3 classes, matrice de transition 3×3
/// 5. OU mean-reversion : boosts cold stars via Ornstein-Uhlenbeck process (v19 B3)
///
/// Pour Pool::Stars : moyenne pondérée Hedge des 5 experts.
/// Pour Pool::Balls : uniforme.
pub struct StarSpecialistModel {
    smoothing: f64,
    learning_rate: f64,
    min_draws: usize,
}

impl Default for StarSpecialistModel {
    fn default() -> Self {
        Self {
            smoothing: 0.35,
            learning_rate: 0.15,
            min_draws: 30,
        }
    }
}

// ── Expert 1 : Gap-hazard pour étoiles ──────────────────────────────────────

fn gap_hazard_stars(draws: &[Draw]) -> Vec<f64> {
    let size = 12;
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 20 {
        return uniform;
    }

    let prior_weight = 0.20;
    let autocorr_strength = 0.40;

    let mut scores = vec![0.0f64; size];

    for num in 1..=12u8 {
        let idx = (num - 1) as usize;

        // Collecter les gaps pour cette étoile
        let mut gaps = Vec::new();
        let mut current_gap = 0usize;
        let mut seen = false;

        for draw in draws {
            if draw.stars.contains(&num) {
                if seen {
                    gaps.push(current_gap);
                }
                seen = true;
                current_gap = 0;
            } else {
                current_gap += 1;
            }
        }

        // Gap courant
        let mut current = 0usize;
        for draw in draws {
            if draw.stars.contains(&num) {
                break;
            }
            current += 1;
        }

        if gaps.len() < 3 {
            scores[idx] = 1.0 / size as f64;
            continue;
        }

        // Hazard empirique : P(gap=g | gap >= g)
        let n_at_least = gaps.iter().filter(|&&g| g >= current).count();
        let n_exact = gaps.iter().filter(|&&g| g == current).count();

        let empirical_hazard = if n_at_least > 0 {
            n_exact as f64 / n_at_least as f64
        } else {
            0.0
        };

        // Prior géométrique (sans mémoire)
        let mean_gap: f64 = gaps.iter().map(|&g| g as f64).sum::<f64>() / gaps.len() as f64;
        let p_geo = if mean_gap > 0.0 { 1.0 / (mean_gap + 1.0) } else { 0.1 };
        let prior_hazard = p_geo;

        // Mélange prior + empirique
        let hazard = (1.0 - prior_weight) * empirical_hazard + prior_weight * prior_hazard;

        // Autocorrélation lag-1 des gaps
        let autocorr = if gaps.len() >= 4 {
            let mean_g = gaps.iter().map(|&g| g as f64).sum::<f64>() / gaps.len() as f64;
            let var_g: f64 = gaps.iter().map(|&g| (g as f64 - mean_g).powi(2)).sum::<f64>();
            if var_g > 1e-10 {
                let cov: f64 = gaps.windows(2)
                    .map(|w| (w[0] as f64 - mean_g) * (w[1] as f64 - mean_g))
                    .sum();
                (cov / var_g).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Si autocorr négatif et dernier gap long → booster (mean-reversion)
        let last_gap = *gaps.first().unwrap_or(&0);
        let autocorr_factor = if autocorr < 0.0 && last_gap as f64 > mean_gap {
            1.0 + autocorr_strength * autocorr.abs()
        } else if autocorr > 0.0 && last_gap as f64 > mean_gap {
            1.0 - autocorr_strength * autocorr * 0.3
        } else {
            1.0
        };

        scores[idx] = (hazard * autocorr_factor).max(1e-10);
    }

    // Normaliser
    let sum: f64 = scores.iter().sum();
    if sum > 0.0 {
        for s in &mut scores {
            *s /= sum;
        }
    } else {
        return uniform;
    }
    scores
}

// ── Expert 2 : Conditionnel ball→star ────────────────────────────────────────

fn conditional_ball_star(draws: &[Draw]) -> Vec<f64> {
    let size = 12;
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 20 {
        return uniform;
    }

    let laplace_alpha = 0.5;
    let n_bins = 5;

    // Construire table de fréquences P(star | sum_bin)
    // sum_bin : somme des boules du tirage précédent → 5 bins
    let mut counts = vec![vec![laplace_alpha; size]; n_bins];
    let mut bin_totals = vec![laplace_alpha * size as f64; n_bins];

    // draws[0] = plus récent, draws[i+1] = tirage précédent de draws[i]
    for i in 0..draws.len() - 1 {
        let prev_ball_sum: u32 = draws[i + 1].balls.iter().map(|&b| b as u32).sum();
        // 5 boules dans [1,50] → somme dans [5, 250]
        let bin = ((prev_ball_sum as f64 - 5.0) / 245.0 * n_bins as f64)
            .floor()
            .clamp(0.0, (n_bins - 1) as f64) as usize;

        for &s in &draws[i].stars {
            let idx = (s - 1) as usize;
            counts[bin][idx] += 1.0;
            bin_totals[bin] += 1.0;
        }
    }

    // Prédire pour le tirage suivant : conditionné sur draws[0].balls
    let current_sum: u32 = draws[0].balls.iter().map(|&b| b as u32).sum();
    let current_bin = ((current_sum as f64 - 5.0) / 245.0 * n_bins as f64)
        .floor()
        .clamp(0.0, (n_bins - 1) as f64) as usize;

    let mut dist = vec![0.0f64; size];
    let total = bin_totals[current_bin];
    if total > 0.0 {
        for i in 0..size {
            dist[i] = counts[current_bin][i] / total;
        }
    } else {
        return uniform;
    }

    dist
}

// ── Expert 3 : Balance haut/bas ──────────────────────────────────────────────

fn balance_high_low_stars(draws: &[Draw]) -> Vec<f64> {
    let size = 12;
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 10 {
        return uniform;
    }

    let alpha = 0.08;

    // EWMA de la fraction d'étoiles basses (1-6) vs hautes (7-12)
    let mut low_frac = 0.5f64;

    for draw in draws.iter().rev() {
        let low_count = draw.stars.iter().filter(|&&s| s <= 6).count() as f64 / 2.0;
        low_frac = (1.0 - alpha) * low_frac + alpha * low_count;
    }

    // Si beaucoup de basses récemment → booster les hautes, et vice versa
    // Persistence : prédire que le pattern continue (légère tendance)
    let low_boost = low_frac;
    let high_boost = 1.0 - low_frac;

    let mut dist = vec![0.0f64; size];
    for i in 0..6 {
        dist[i] = low_boost / 6.0;
    }
    for i in 6..12 {
        dist[i] = high_boost / 6.0;
    }

    // Normaliser
    let sum: f64 = dist.iter().sum();
    if sum > 0.0 {
        for d in &mut dist {
            *d /= sum;
        }
    }
    dist
}

// ── Expert 4 : Mod-4 transition pour étoiles ────────────────────────────────

fn mod4_transition_stars(draws: &[Draw]) -> Vec<f64> {
    let size = 12;
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 10 {
        return uniform;
    }

    // 3 classes : (star-1) % 4 → classes 0, 1, 2, 3
    // Mais 12 étoiles : 3 par classe
    let n_classes = 4;
    let laplace = 0.5;
    let mut transition = vec![vec![laplace; n_classes]; n_classes];
    let mut from_totals = vec![laplace * n_classes as f64; n_classes];

    // Construire matrice de transition sur les classes mod-4
    for i in 0..draws.len() - 1 {
        // Classe du tirage courant (i) → paire de classes
        for &s_from in &draws[i + 1].stars {
            let class_from = ((s_from - 1) % 4) as usize;
            for &s_to in &draws[i].stars {
                let class_to = ((s_to - 1) % 4) as usize;
                transition[class_from][class_to] += 1.0;
                from_totals[class_from] += 1.0;
            }
        }
    }

    // Prédire la classe suivante basée sur les étoiles du dernier tirage
    let mut class_probs = vec![0.0f64; n_classes];
    for &s in &draws[0].stars {
        let class_from = ((s - 1) % 4) as usize;
        if from_totals[class_from] > 0.0 {
            for c in 0..n_classes {
                class_probs[c] += transition[class_from][c] / from_totals[class_from];
            }
        }
    }

    // Normaliser les probabilités de classe
    let class_sum: f64 = class_probs.iter().sum();
    if class_sum > 0.0 {
        for p in &mut class_probs {
            *p /= class_sum;
        }
    }

    // Redistribuer uniformément dans chaque classe
    let mut dist = vec![0.0f64; size];
    for star in 1..=12u8 {
        let idx = (star - 1) as usize;
        let class = ((star - 1) % 4) as usize;
        let members_in_class = (1..=12u8).filter(|&s| ((s - 1) % 4) as usize == class).count();
        dist[idx] = class_probs[class] / members_in_class as f64;
    }

    dist
}

// ── Expert 5 : OU Mean-Reversion (v19 B3) ──────────────────────────────────

fn ou_mean_reversion_stars(draws: &[Draw]) -> Vec<f64> {
    let size = 12;
    let uniform = vec![1.0 / size as f64; size];

    if draws.len() < 40 {
        return uniform;
    }

    let pick = 2;
    let mu = pick as f64 / size as f64; // theoretical ≈ 0.1667
    let ewma_alpha = 0.05;
    let theta_default = 0.15;

    let mut probs = vec![0.0f64; size];

    for star_idx in 0..size {
        let star_num = (star_idx + 1) as u8;

        // Build slow EWMA frequency series (chronological)
        let mut ewma = mu;
        for d in draws.iter().rev() {
            let val = if d.stars.contains(&star_num) { 1.0 } else { 0.0 };
            ewma = ewma_alpha * val + (1.0 - ewma_alpha) * ewma;
        }
        let f_current = ewma;

        // Estimate θ by OLS on EWMA series
        let mut freq_series: Vec<f64> = Vec::with_capacity(draws.len());
        let mut e = mu;
        for d in draws.iter().rev() {
            let val = if d.stars.contains(&star_num) { 1.0 } else { 0.0 };
            e = ewma_alpha * val + (1.0 - ewma_alpha) * e;
            freq_series.push(e);
        }

        let theta = if freq_series.len() > 10 {
            let mut sum_xy = 0.0f64;
            let mut sum_xx = 0.0f64;
            for t in 0..freq_series.len() - 1 {
                let x = mu - freq_series[t];
                let y = freq_series[t + 1] - freq_series[t];
                sum_xy += x * y;
                sum_xx += x * x;
            }
            if sum_xx > 1e-15 {
                (sum_xy / sum_xx).clamp(0.01, 0.50)
            } else {
                theta_default
            }
        } else {
            theta_default
        };

        // OU prediction
        probs[star_idx] = (f_current + theta * (mu - f_current)).max(1e-6);
    }

    // Normalize
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        return uniform;
    }
    probs
}

impl ForecastModel for StarSpecialistModel {
    fn name(&self) -> &str {
        "StarSpecialist"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Pool::Balls → uniforme (ce modèle ne prédit que les étoiles)
        if pool == Pool::Balls {
            return uniform;
        }

        if draws.len() < self.min_draws {
            return uniform;
        }

        // 5 experts (v19 B3: added OU mean-reversion)
        let expert_preds = [
            gap_hazard_stars(draws),
            conditional_ball_star(draws),
            balance_high_low_stars(draws),
            mod4_transition_stars(draws),
            ou_mean_reversion_stars(draws),
        ];
        let n_experts = expert_preds.len();

        // Hedge : pondération multiplicative sur les tirages récents
        let eta = self.learning_rate;
        let mut weights = vec![1.0 / n_experts as f64; n_experts];

        let n_recent = draws.len().min(50);
        for t in (1..n_recent).rev() {
            // Entraîner sur draws[t..] pour prédire draws[t-1]
            let train_data = &draws[t..];
            if train_data.len() < 10 {
                continue;
            }

            let preds = [
                gap_hazard_stars(train_data),
                conditional_ball_star(train_data),
                balance_high_low_stars(train_data),
                mod4_transition_stars(train_data),
                ou_mean_reversion_stars(train_data),
            ];

            // Loss = -log(prob de l'étoile observée)
            let target = &draws[t - 1];
            for (e, pred) in preds.iter().enumerate() {
                let mut loss = 0.0f64;
                for &s in &target.stars {
                    let idx = (s - 1) as usize;
                    let p = pred[idx].max(1e-15);
                    loss -= p.ln();
                }
                // Clamp loss pour stabilité
                let loss = loss.clamp(0.0, 10.0);
                weights[e] *= (-eta * loss).exp();
            }

            // Normaliser
            let wsum: f64 = weights.iter().sum();
            if wsum > 0.0 {
                for w in &mut weights {
                    *w /= wsum;
                }
            }
        }

        // Combinaison finale
        let mut dist = vec![0.0f64; size];
        for (e, pred) in expert_preds.iter().enumerate() {
            for i in 0..size {
                dist[i] += weights[e] * pred[i];
            }
        }

        // Normaliser
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for d in &mut dist {
                *d /= sum;
            }
        }

        // Lissage uniforme
        let uniform_val = 1.0 / size as f64;
        for d in &mut dist {
            *d = (1.0 - self.smoothing) * *d + self.smoothing * uniform_val;
        }

        // Renormaliser
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for d in &mut dist {
                *d /= sum;
            }
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("learning_rate".into(), self.learning_rate),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn is_stars_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_star_specialist_stars_sums_to_one() {
        let model = StarSpecialistModel::default();
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
    fn test_star_specialist_balls_is_uniform() {
        let model = StarSpecialistModel::default();
        let draws = make_test_draws(60);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_star_specialist_few_draws_uniform() {
        let model = StarSpecialistModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gap_hazard_stars_valid() {
        let draws = make_test_draws(60);
        let dist = gap_hazard_stars(&draws);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "gap_hazard sum = {}", sum);
    }

    #[test]
    fn test_conditional_ball_star_valid() {
        let draws = make_test_draws(60);
        let dist = conditional_ball_star(&draws);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "conditional sum = {}", sum);
    }

    #[test]
    fn test_balance_high_low_valid() {
        let draws = make_test_draws(60);
        let dist = balance_high_low_stars(&draws);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "balance sum = {}", sum);
    }

    #[test]
    fn test_mod4_transition_stars_valid() {
        let draws = make_test_draws(60);
        let dist = mod4_transition_stars(&draws);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "mod4_trans sum = {}", sum);
    }
}
