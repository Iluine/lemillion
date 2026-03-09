use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy};

/// GapDynamics — fonction de hasard empirique + correction autocorrélation.
///
/// Exploite le biais de compression des gaps (ratio 0.64 vs 0.78 attendu) :
/// la distribution des gaps n'est PAS géométrique (memoryless).
///
/// Pour chaque numéro :
/// 1. Construire la distribution empirique des gaps historiques
/// 2. Calculer le hazard empirique : P(apparait au gap=g | gap >= g)
/// 3. Régulariser avec un prior géométrique (prior_weight)
/// 4. Ajuster par l'autocorrélation lag-1 des gaps
pub struct GapDynamicsModel {
    smoothing: f64,
    prior_weight: f64,
    min_gaps: usize,
    autocorr_strength: f64,
}

impl GapDynamicsModel {
    pub fn new(smoothing: f64, prior_weight: f64, min_gaps: usize, autocorr_strength: f64) -> Self {
        Self { smoothing, prior_weight, min_gaps, autocorr_strength }
    }
}

impl Default for GapDynamicsModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            prior_weight: 0.20,
            min_gaps: 3,
            autocorr_strength: 0.60,
        }
    }
}

/// Collecte tous les gaps historiques pour un numéro donné.
/// Retourne (gaps complétés, gap courant depuis la dernière apparition).
fn collect_gaps(draws: &[Draw], pool: Pool, num: u8) -> (Vec<usize>, usize) {
    let mut gaps = Vec::new();
    let mut current_gap = 0usize;

    // draws[0] = plus récent
    for draw in draws {
        let numbers = pool.numbers_from(draw);
        if numbers.contains(&num) {
            if current_gap > 0 || !gaps.is_empty() {
                // On ne compte le premier gap que si on a déjà vu le numéro
                gaps.push(current_gap);
            }
            current_gap = 0;
        } else {
            current_gap += 1;
        }
    }

    // Le gap courant = distance depuis la dernière apparition dans draws[0..]
    let mut current = 0usize;
    for draw in draws {
        if pool.numbers_from(draw).contains(&num) {
            break;
        }
        current += 1;
    }

    (gaps, current)
}

/// Calcule le hazard empirique P(gap=g | gap >= g) à partir des gaps observés.
/// Retourne le hazard pour le gap `target_gap`.
fn empirical_hazard(gaps: &[usize], target_gap: usize) -> Option<f64> {
    if gaps.is_empty() {
        return None;
    }

    // Nombre de gaps >= target_gap (survie)
    let n_survived = gaps.iter().filter(|&&g| g >= target_gap).count();
    if n_survived == 0 {
        return None;
    }

    // Nombre de gaps == target_gap (événement au gap exact)
    let n_event = gaps.iter().filter(|&&g| g == target_gap).count();

    Some(n_event as f64 / n_survived as f64)
}

/// Hazard du prior géométrique P(gap=g | gap >= g) = p, constant.
fn geometric_hazard(pool: Pool) -> f64 {
    // p = pick_count / size = probabilité de tirage uniforme
    pool.pick_count() as f64 / pool.size() as f64
}

/// Autocorrélation lag-1 des gaps d'un numéro.
fn gap_autocorrelation(gaps: &[usize]) -> f64 {
    if gaps.len() < 3 {
        return 0.0;
    }

    let n = gaps.len();
    let mean = gaps.iter().sum::<usize>() as f64 / n as f64;
    let var: f64 = gaps.iter().map(|&g| (g as f64 - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-10 {
        return 0.0;
    }

    let mut cov = 0.0;
    for i in 0..n - 1 {
        cov += (gaps[i] as f64 - mean) * (gaps[i + 1] as f64 - mean);
    }
    cov /= (n - 1) as f64;

    (cov / var).clamp(-1.0, 1.0)
}

impl ForecastModel for GapDynamicsModel {
    fn name(&self) -> &str {
        "GapDynamics"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_gaps + 2 {
            return uniform;
        }

        let geo_h = geometric_hazard(pool);
        let mut scores = Vec::with_capacity(size);

        for num in 1..=size as u8 {
            let (gaps, current_gap) = collect_gaps(draws, pool, num);

            if gaps.len() < self.min_gaps {
                // Pas assez de données, utiliser le prior géométrique
                scores.push(geo_h);
                continue;
            }

            // Hazard empirique pour le gap courant+1
            // (probabilité d'apparaitre au prochain tirage sachant qu'on est à current_gap)
            let emp_h = empirical_hazard(&gaps, current_gap)
                .unwrap_or(geo_h);

            // Régulariser avec le prior géométrique
            let h = (1.0 - self.prior_weight) * emp_h + self.prior_weight * geo_h;

            // Correction par autocorrélation lag-1
            let autocorr = gap_autocorrelation(&gaps);
            let last_gap = gaps.first().copied().unwrap_or(0);
            let mean_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;

            // Si autocorr > 0 et dernier gap était court → le prochain sera probablement court aussi
            // → augmenter le hazard (plus de chance d'apparaitre bientôt)
            let gap_deviation = if mean_gap > 0.0 {
                (last_gap as f64 - mean_gap) / mean_gap
            } else {
                0.0
            };
            let autocorr_adjustment = 1.0 - self.autocorr_strength * autocorr * gap_deviation;
            let adjusted_h = (h * autocorr_adjustment).clamp(0.001, 0.999);

            scores.push(adjusted_h);
        }

        // Normaliser
        let total: f64 = scores.iter().sum();
        if total <= 0.0 {
            return uniform;
        }
        for s in &mut scores {
            *s /= total;
        }

        // Lisser avec uniforme
        let uniform_val = 1.0 / size as f64;
        for s in &mut scores {
            *s = (1.0 - self.smoothing) * *s + self.smoothing * uniform_val;
        }

        // Re-normaliser
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in &mut scores {
                *s /= total;
            }
        } else {
            return uniform;
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("prior_weight".into(), self.prior_weight),
            ("min_gaps".into(), self.min_gaps as f64),
            ("autocorr_strength".into(), self.autocorr_strength),
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
    fn test_gap_dynamics_balls_sums_to_one() {
        let model = GapDynamicsModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_gap_dynamics_stars_sums_to_one() {
        let model = GapDynamicsModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_gap_dynamics_no_negative() {
        let model = GapDynamicsModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_gap_dynamics_empty_draws() {
        let model = GapDynamicsModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gap_dynamics_few_draws() {
        let model = GapDynamicsModel::default();
        let draws = make_test_draws(3);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gap_dynamics_deterministic() {
        let model = GapDynamicsModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "GapDynamics should be deterministic");
        }
    }

    #[test]
    fn test_geometric_hazard() {
        let h = geometric_hazard(Pool::Balls);
        assert!((h - 5.0 / 50.0).abs() < 1e-10); // 0.1
        let h = geometric_hazard(Pool::Stars);
        assert!((h - 2.0 / 12.0).abs() < 1e-10); // ~0.167
    }

    #[test]
    fn test_empirical_hazard() {
        let gaps = vec![3, 5, 2, 3, 7, 4, 3];
        // hazard at gap=3: P(gap=3 | gap>=3) = count(==3) / count(>=3)
        // count(>=3) = 5 (3,5,3,7,4,3 -> wait: 3,5,3,7,4,3)
        // gaps >= 3: 3,5,3,7,4,3 → 6
        // gaps == 3: 3,3,3 → 3
        let h = empirical_hazard(&gaps, 3).unwrap();
        assert!((h - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_gap_autocorrelation_zero() {
        // Constant gaps → variance is 0 → autocorr = 0
        let gaps = vec![5, 5, 5, 5, 5];
        let ac = gap_autocorrelation(&gaps);
        assert!((ac - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_collect_gaps() {
        use lemillion_db::models::Draw;
        // Create draws where ball 1 appears in draws 0, 3, 6 (gaps of 3)
        let draws: Vec<Draw> = (0..10)
            .map(|i| {
                let b1 = if i % 3 == 0 { 1 } else { 2 };
                Draw {
                    draw_id: format!("{}", i),
                    day: "MARDI".to_string(),
                    date: format!("2024-01-{:02}", i + 1),
                    balls: [b1, 10, 20, 30, 40],
                    stars: [1, 2],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                    ball_order: None,
                    star_order: None,
                    cycle_number: None,
                }
            })
            .collect();
        let (gaps, current) = collect_gaps(&draws, Pool::Balls, 1);
        assert_eq!(current, 0); // Ball 1 is in draws[0]
        assert!(!gaps.is_empty());
    }
}
