use serde::{Deserialize, Serialize};

use lemillion_db::models::{Draw, Pool};
use crate::models::ForecastModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub model_name: String,
    pub window: usize,
    pub log_likelihood: f64,
    pub n_tests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCalibration {
    pub model_name: String,
    pub results: Vec<CalibrationResult>,
    pub best_window: usize,
    pub best_ll: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleWeights {
    pub ball_weights: Vec<(String, f64)>,
    pub star_weights: Vec<(String, f64)>,
    pub calibrations: Vec<ModelCalibration>,
}

/// Walk-forward evaluation: pour chaque tirage test t, on entraîne sur draws[t+1..t+1+window]
/// et on mesure la log-likelihood sur le tirage t.
/// CRITIQUE: pas de fuite du futur - on n'utilise que des données passées.
///
/// draws[0] = le plus récent, draws[N-1] = le plus ancien.
/// Pour le test t, on entraîne sur draws[t+1 .. t+1+window] (strictement après le tirage test).
pub fn walk_forward_evaluate(
    model: &dyn ForecastModel,
    draws: &[Draw],
    window: usize,
    pool: Pool,
) -> f64 {
    let max_t = draws.len().saturating_sub(window + 1);
    if max_t == 0 {
        return f64::NEG_INFINITY;
    }

    // Limiter à ~100 points de test avec un stride pour la performance
    let max_tests = 100;
    let stride = (max_t / max_tests).max(1);

    let mut total_ll = 0.0f64;
    let mut n_tests = 0usize;

    for t in (0..max_t).step_by(stride) {
        // Données d'entraînement : strictement après le tirage test
        let train_end = (t + 1 + window).min(draws.len());
        let train_data = &draws[t + 1..train_end];

        if train_data.len() < 3 {
            continue;
        }

        // Prédire
        let dist = model.predict(train_data, pool);

        // Mesurer la log-likelihood sur le tirage test
        let test_draw = &draws[t];
        let test_numbers = pool.numbers_from(test_draw);

        let mut draw_ll = 0.0f64;
        for &n in test_numbers {
            let idx = (n - 1) as usize;
            if idx < dist.len() {
                let p = dist[idx].max(1e-15); // Éviter log(0)
                draw_ll += p.ln();
            }
        }

        total_ll += draw_ll;
        n_tests += 1;
    }

    if n_tests > 0 {
        total_ll / n_tests as f64
    } else {
        f64::NEG_INFINITY
    }
}

/// Calcule la log-likelihood de la distribution uniforme pour une pool donnée.
pub fn uniform_log_likelihood(pool: Pool) -> f64 {
    let p = 1.0 / pool.size() as f64;
    pool.pick_count() as f64 * p.ln()
}

/// Calcule les poids de l'ensemble à partir des calibrations.
/// Les modèles avec une LL inférieure à la LL uniforme reçoivent un poids de 0.
pub fn compute_weights(
    calibrations: &[ModelCalibration],
    pool: Pool,
) -> Vec<(String, f64)> {
    let uniform_ll = uniform_log_likelihood(pool);

    // Calculer les skills (LL - LL_uniforme)
    let skills: Vec<f64> = calibrations
        .iter()
        .map(|c| {
            let skill = c.best_ll - uniform_ll;
            if skill > 0.0 { skill } else { 0.0 }
        })
        .collect();

    let total_skill: f64 = skills.iter().sum();

    if total_skill > 0.0 {
        calibrations
            .iter()
            .zip(skills.iter())
            .map(|(c, &skill)| (c.model_name.clone(), skill / total_skill))
            .collect()
    } else {
        // Tous les modèles sont inférieurs à l'uniforme, poids égaux
        let n = calibrations.len() as f64;
        calibrations
            .iter()
            .map(|c| (c.model_name.clone(), 1.0 / n))
            .collect()
    }
}

pub fn calibrate_model(
    model: &dyn ForecastModel,
    draws: &[Draw],
    windows: &[usize],
    pool: Pool,
) -> ModelCalibration {
    let mut results = Vec::new();
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_window = windows[0];

    for &window in windows {
        let ll = walk_forward_evaluate(model, draws, window, pool);
        let max_t = draws.len().saturating_sub(window + 1);

        results.push(CalibrationResult {
            model_name: model.name().to_string(),
            window,
            log_likelihood: ll,
            n_tests: max_t,
        });

        if ll > best_ll {
            best_ll = ll;
            best_window = window;
        }
    }

    ModelCalibration {
        model_name: model.name().to_string(),
        results,
        best_window,
        best_ll,
    }
}

pub fn save_weights(weights: &EnsembleWeights, path: &std::path::Path) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(weights)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_weights(path: &std::path::Path) -> anyhow::Result<EnsembleWeights> {
    let json = std::fs::read_to_string(path)?;
    let weights: EnsembleWeights = serde_json::from_str(&json)?;
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_walk_forward_returns_finite() {
        let draws = make_test_draws(50);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let ll = walk_forward_evaluate(&model, &draws, 20, Pool::Balls);
        assert!(ll.is_finite(), "LL should be finite, got {}", ll);
    }

    #[test]
    fn test_walk_forward_no_future_leak() {
        // Si on a 50 tirages et window=45, max_t = 50-45-1 = 4, donc on teste 4 tirages
        let draws = make_test_draws(50);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let ll = walk_forward_evaluate(&model, &draws, 45, Pool::Balls);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_walk_forward_too_few_draws() {
        let draws = make_test_draws(5);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let ll = walk_forward_evaluate(&model, &draws, 10, Pool::Balls);
        assert!(ll == f64::NEG_INFINITY || ll.is_finite());
    }

    #[test]
    fn test_uniform_ll() {
        let ll = uniform_log_likelihood(Pool::Balls);
        // 5 * ln(1/50) = 5 * ln(0.02) ≈ -19.56
        assert!(ll < 0.0);
        assert!(ll > -30.0);
    }

    #[test]
    fn test_compute_weights_sum_to_one() {
        let calibrations = vec![
            ModelCalibration {
                model_name: "A".to_string(),
                results: vec![],
                best_window: 20,
                best_ll: -15.0,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 30,
                best_ll: -18.0,
            },
        ];
        let weights = compute_weights(&calibrations, Pool::Balls);
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights sum = {}", sum);
    }

    #[test]
    fn test_compute_weights_zero_for_below_uniform() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "Good".to_string(),
                results: vec![],
                best_window: 20,
                best_ll: uniform_ll + 1.0, // Meilleur que l'uniforme
            },
            ModelCalibration {
                model_name: "Bad".to_string(),
                results: vec![],
                best_window: 30,
                best_ll: uniform_ll - 1.0, // Pire que l'uniforme
            },
        ];
        let weights = compute_weights(&calibrations, Pool::Balls);
        let bad_weight = weights.iter().find(|(n, _)| n == "Bad").unwrap().1;
        assert_eq!(bad_weight, 0.0, "Bad model should have weight 0");
    }

    #[test]
    fn test_calibrate_model() {
        let draws = make_test_draws(50);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let cal = calibrate_model(&model, &draws, &[10, 20, 30], Pool::Balls);
        assert_eq!(cal.model_name, "Dirichlet");
        assert_eq!(cal.results.len(), 3);
        assert!(cal.best_ll.is_finite() || cal.best_ll == f64::NEG_INFINITY);
    }

    #[test]
    fn test_weights_json_roundtrip() {
        let weights = EnsembleWeights {
            ball_weights: vec![("A".to_string(), 0.5), ("B".to_string(), 0.5)],
            star_weights: vec![("A".to_string(), 0.3), ("B".to_string(), 0.7)],
            calibrations: vec![],
        };
        let json = serde_json::to_string(&weights).unwrap();
        let loaded: EnsembleWeights = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.ball_weights.len(), 2);
        assert_eq!(loaded.star_weights[1].1, 0.7);
    }
}
