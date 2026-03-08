use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// BME (Bayesian Mixture of Experts) — mélange adaptatif en ligne.
///
/// 6 experts légers repondérés dynamiquement via l'algorithme Hedge
/// (multiplicative weights update). Les poids s'adaptent au contexte courant.
///
/// Borne de regret prouvée : loss ≤ best_expert + sqrt(T × ln(K))
pub struct MixtureModel {
    learning_rate: f64,
    smoothing: f64,
}

impl MixtureModel {
    pub fn new(learning_rate: f64, smoothing: f64) -> Self {
        Self {
            learning_rate,
            smoothing,
        }
    }
}

impl Default for MixtureModel {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            smoothing: 0.3,
        }
    }
}

const N_EXPERTS: usize = 7;

/// Expert 0 : Fréquence brute (comptage de présences dans la fenêtre)
fn expert_frequency(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let mut counts = vec![0.0f64; size];
    for draw in draws {
        for &n in pool.numbers_from(draw) {
            let idx = (n - 1) as usize;
            if idx < size {
                counts[idx] += 1.0;
            }
        }
    }
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        for c in &mut counts {
            *c /= total;
        }
    } else {
        counts = vec![1.0 / size as f64; size];
    }
    counts
}

/// Expert 1 : Hazard empirique (remplace le gambler's fallacy par la forme de la distribution des gaps)
fn expert_gap(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let geo_hazard = pool.pick_count() as f64 / size as f64;

    let mut scores = vec![geo_hazard; size];

    for num_idx in 0..size {
        let num = (num_idx + 1) as u8;
        // Collect historical gaps and current gap
        let mut completed_gaps = Vec::new();
        let mut current_gap = 0usize;
        let mut seen_first = false;

        for draw in draws.iter() {
            if pool.numbers_from(draw).contains(&num) {
                if seen_first && current_gap > 0 {
                    completed_gaps.push(current_gap);
                }
                seen_first = true;
                current_gap = 0;
            } else {
                current_gap += 1;
            }
        }

        // Current gap from draws[0]
        let mut cur = 0usize;
        for draw in draws.iter() {
            if pool.numbers_from(draw).contains(&num) { break; }
            cur += 1;
        }

        if completed_gaps.len() >= 3 {
            // Empirical hazard at current gap
            let n_survived = completed_gaps.iter().filter(|&&g| g >= cur).count();
            if n_survived > 0 {
                let n_event = completed_gaps.iter().filter(|&&g| g == cur).count();
                let emp_h = n_event as f64 / n_survived as f64;
                // Blend with geometric prior
                scores[num_idx] = 0.7 * emp_h + 0.3 * geo_hazard;
            }
        }
    }

    // Normalize
    let total: f64 = scores.iter().sum();
    if total > 0.0 {
        for s in &mut scores {
            *s /= total;
        }
    }
    scores
}

/// Expert 2 : Parité (favorise les numéros du même type pair/impair que la moyenne récente)
fn expert_parity(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let window = draws.len().min(10);
    let mut odd_count = 0usize;
    let mut total_count = 0usize;
    for draw in &draws[..window] {
        for &n in pool.numbers_from(draw) {
            total_count += 1;
            if n % 2 == 1 {
                odd_count += 1;
            }
        }
    }
    let odd_ratio = if total_count > 0 {
        odd_count as f64 / total_count as f64
    } else {
        0.5
    };
    let mut scores = Vec::with_capacity(size);
    for num in 1..=size as u8 {
        let base = 1.0 / size as f64;
        if num % 2 == 1 {
            scores.push(base * (0.5 + odd_ratio));
        } else {
            scores.push(base * (1.5 - odd_ratio));
        }
    }
    let total: f64 = scores.iter().sum();
    for s in &mut scores {
        *s /= total;
    }
    scores
}

/// Expert 3 : Équilibre décade (favorise les décades sous-représentées récemment)
fn expert_decade(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let n_decades = match pool {
        Pool::Balls => 5,  // [1-10], [11-20], ..., [41-50]
        Pool::Stars => 3,  // [1-4], [5-8], [9-12]
    };
    let decade_size = match pool {
        Pool::Balls => 10,
        Pool::Stars => 4,
    };
    let window = draws.len().min(20);
    let mut decade_counts = vec![0.0f64; n_decades];
    for draw in &draws[..window] {
        for &n in pool.numbers_from(draw) {
            let d = ((n - 1) as usize / decade_size).min(n_decades - 1);
            decade_counts[d] += 1.0;
        }
    }
    let total_dec: f64 = decade_counts.iter().sum();
    let expected = if total_dec > 0.0 {
        total_dec / n_decades as f64
    } else {
        1.0
    };

    let mut scores = Vec::with_capacity(size);
    for i in 0..size {
        let d = (i / decade_size).min(n_decades - 1);
        // Favoriser les décades sous-représentées
        let ratio = if decade_counts[d] > 0.0 {
            expected / decade_counts[d]
        } else {
            2.0
        };
        scores.push(ratio);
    }
    let total: f64 = scores.iter().sum();
    for s in &mut scores {
        *s /= total;
    }
    scores
}

/// Expert 4 : Somme (favorise les numéros qui mèneraient à une somme proche de la moyenne)
fn expert_sum(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let window = draws.len().min(30);
    let mut sum_history: Vec<f64> = Vec::with_capacity(window);
    for draw in &draws[..window] {
        let s: f64 = pool.numbers_from(draw).iter().map(|&n| n as f64).sum();
        sum_history.push(s);
    }
    let mean_sum = if !sum_history.is_empty() {
        sum_history.iter().sum::<f64>() / sum_history.len() as f64
    } else {
        let pick = pool.pick_count() as f64;
        pick * (size as f64 + 1.0) / 2.0
    };

    // Score chaque numéro : plus il est proche de contribuer à la somme moyenne, plus il a de poids
    let pick = pool.pick_count() as f64;
    let target_per_number = mean_sum / pick;
    let mut scores = Vec::with_capacity(size);
    for num in 1..=size as u8 {
        let dist_from_target = (num as f64 - target_per_number).abs();
        let score = (-(dist_from_target / (size as f64 * 0.3)).powi(2)).exp();
        scores.push(score.max(1e-10));
    }
    let total: f64 = scores.iter().sum();
    for s in &mut scores {
        *s /= total;
    }
    scores
}

/// Expert 5 : Co-occurrence (numéros souvent tirés avec ceux du dernier tirage)
fn expert_cooccurrence(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    if draws.is_empty() {
        return vec![1.0 / size as f64; size];
    }

    let latest_numbers = pool.numbers_from(&draws[0]);
    let window = draws.len().min(50);

    // Compter les co-occurrences avec les numéros du dernier tirage
    let mut cooc = vec![0.0f64; size];
    for draw in &draws[..window] {
        let numbers = pool.numbers_from(draw);
        let has_overlap = numbers.iter().any(|n| latest_numbers.contains(n));
        if has_overlap {
            for &n in numbers {
                let idx = (n - 1) as usize;
                if idx < size {
                    cooc[idx] += 1.0;
                }
            }
        }
    }
    let total: f64 = cooc.iter().sum();
    if total > 0.0 {
        for c in &mut cooc {
            *c /= total;
        }
        cooc
    } else {
        vec![1.0 / size as f64; size]
    }
}

/// Expert 6 : Couplage boules/étoiles (exploite la dépendance marginale p=0.014)
///
/// Pour Pool::Stars : P(étoile | sum_bin des boules du dernier tirage)
/// Pour Pool::Balls : uniforme (pas de signal ball→ball via stars)
fn expert_ball_star_coupling(draws: &[Draw], pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let uniform = vec![1.0 / size as f64; size];

    if draws.is_empty() {
        return uniform;
    }

    match pool {
        Pool::Balls => {
            // Pas de signal stars→balls exploitable, retourne uniforme
            uniform
        }
        Pool::Stars => {
            // Construire P(étoile | sum_bin des boules)
            let n_bins = 5;
            let mut counts = vec![vec![0.0f64; size]; n_bins];
            let mut bin_totals = vec![0.0f64; n_bins];

            let window = draws.len().min(200);
            for draw in &draws[..window] {
                let ball_sum: u16 = draw.balls.iter().map(|&b| b as u16).sum();
                // sum range 15-240, bin en 5
                let bin = ((ball_sum as f64 - 15.0) / (240.0 - 15.0) * n_bins as f64)
                    .floor()
                    .clamp(0.0, (n_bins - 1) as f64) as usize;
                bin_totals[bin] += 1.0;
                for &s in &draw.stars {
                    let idx = (s - 1) as usize;
                    if idx < size {
                        counts[bin][idx] += 1.0;
                    }
                }
            }

            // Bin actuel basé sur les boules du dernier tirage
            let current_sum: u16 = draws[0].balls.iter().map(|&b| b as u16).sum();
            let current_bin = ((current_sum as f64 - 15.0) / (240.0 - 15.0) * n_bins as f64)
                .floor()
                .clamp(0.0, (n_bins - 1) as f64) as usize;

            if bin_totals[current_bin] > 0.0 {
                let alpha = 0.5; // Laplace smoothing
                let total_picks = bin_totals[current_bin] * 2.0; // 2 stars per draw
                let mut probs = Vec::with_capacity(size);
                for k in 0..size {
                    probs.push((counts[current_bin][k] + alpha) / (total_picks + alpha * size as f64));
                }
                let sum: f64 = probs.iter().sum();
                for p in &mut probs {
                    *p /= sum;
                }
                probs
            } else {
                uniform
            }
        }
    }
}

/// Retourne les prédictions de tous les experts.
fn all_expert_predictions(draws: &[Draw], pool: Pool) -> Vec<Vec<f64>> {
    vec![
        expert_frequency(draws, pool),
        expert_gap(draws, pool),
        expert_parity(draws, pool),
        expert_decade(draws, pool),
        expert_sum(draws, pool),
        expert_cooccurrence(draws, pool),
        expert_ball_star_coupling(draws, pool),
    ]
}

impl ForecastModel for MixtureModel {
    fn name(&self) -> &str {
        "BME"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 3 {
            return uniform;
        }

        // Algorithme Hedge : repondération online sur la fenêtre de tirages
        let mut weights = vec![1.0f64; N_EXPERTS];

        // Pour chaque tirage passé (du plus ancien au plus récent), évaluer les experts
        // et ajuster les poids
        let window = draws.len().min(100);

        for t in (1..window).rev() {
            // Prédictions des experts basées sur draws[t..]
            let sub_draws = &draws[t..];
            if sub_draws.len() < 2 {
                continue;
            }
            let expert_preds = all_expert_predictions(sub_draws, pool);

            // Évaluer contre le tirage réel draws[t-1]
            let actual = pool.numbers_from(&draws[t - 1]);

            for (e, pred) in expert_preds.iter().enumerate() {
                // Log-loss pour cet expert
                let mut loss = 0.0f64;
                for &num in actual {
                    let idx = (num - 1) as usize;
                    if idx < pred.len() {
                        loss -= pred[idx].max(1e-15).ln();
                    }
                }
                // Hedge update: w_e *= exp(-eta * loss)
                weights[e] *= (-self.learning_rate * loss).exp();
            }

            // Normaliser les poids pour éviter l'underflow
            let w_sum: f64 = weights.iter().sum();
            if w_sum > 0.0 {
                for w in &mut weights {
                    *w /= w_sum;
                }
            } else {
                weights = vec![1.0 / N_EXPERTS as f64; N_EXPERTS];
            }
        }

        // Prédiction finale : mélange pondéré des experts sur draws[0..]
        let expert_preds = all_expert_predictions(draws, pool);

        let mut mixture = vec![0.0f64; size];
        for (e, pred) in expert_preds.iter().enumerate() {
            for (i, &p) in pred.iter().enumerate() {
                mixture[i] += weights[e] * p;
            }
        }

        // Lisser avec l'uniforme
        let uniform_val = 1.0 / size as f64;
        for p in &mut mixture {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        // Normaliser
        let sum: f64 = mixture.iter().sum();
        if sum > 0.0 {
            for p in &mut mixture {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        mixture
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("learning_rate".into(), self.learning_rate),
            ("smoothing".into(), self.smoothing),
            ("n_experts".into(), N_EXPERTS as f64),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_bme_balls_sums_to_one() {
        let model = MixtureModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_bme_stars_sums_to_one() {
        let model = MixtureModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_bme_no_negative() {
        let model = MixtureModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_bme_empty_draws() {
        let model = MixtureModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bme_few_draws() {
        let model = MixtureModel::default();
        let draws = make_test_draws(2);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_expert_frequency_normalized() {
        let draws = make_test_draws(30);
        let dist = expert_frequency(&draws, Pool::Balls);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum = {sum}");
    }

    #[test]
    fn test_expert_gap_normalized() {
        let draws = make_test_draws(30);
        let dist = expert_gap(&draws, Pool::Balls);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum = {sum}");
    }

    #[test]
    fn test_all_experts_same_size() {
        let draws = make_test_draws(30);
        let preds = all_expert_predictions(&draws, Pool::Balls);
        assert_eq!(preds.len(), N_EXPERTS);
        for pred in &preds {
            assert_eq!(pred.len(), 50);
        }
    }

    #[test]
    fn test_bme_deterministic() {
        let model = MixtureModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "BME should be deterministic");
        }
    }
}
