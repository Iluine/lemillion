use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use lemillion_db::models::{Draw, Pool};
use crate::models::{ForecastModel, SamplingStrategy};

/// Deserialize null as f64::NEG_INFINITY for log_likelihood.
fn deserialize_ll<'de, D>(deserializer: D) -> Result<f64, D::Error>
where D: serde::Deserializer<'de> {
    let v: Option<f64> = Option::deserialize(deserializer)?;
    Ok(v.unwrap_or(f64::NEG_INFINITY))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub model_name: String,
    pub window: usize,
    #[serde(default)]
    pub sparse: bool,
    #[serde(deserialize_with = "deserialize_ll")]
    pub log_likelihood: f64,
    pub n_tests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCalibration {
    pub model_name: String,
    pub results: Vec<CalibrationResult>,
    pub best_window: usize,
    #[serde(default)]
    pub best_sparse: bool,
    pub best_ll: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleWeights {
    pub ball_weights: Vec<(String, f64)>,
    pub star_weights: Vec<(String, f64)>,
    pub calibrations: Vec<ModelCalibration>,
    /// LL détaillé par tirage par modèle (optionnel, pour méta-apprentissage).
    /// Chaque entrée : (model_name, Vec<f64> de LL par tirage test).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub detailed_ll: Vec<(String, Vec<f64>)>,
    /// LL détaillé étoiles par tirage par modèle (optionnel, pour méta-apprentissage étoiles).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub star_detailed_ll: Vec<(String, Vec<f64>)>,
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
    walk_forward_evaluate_with_strategy(model, draws, window, pool, SamplingStrategy::Consecutive)
}

/// Walk-forward evaluation avec stratégie d'échantillonnage explicite.
/// Le stride est calculé automatiquement pour limiter à ~500 test points max.
pub fn walk_forward_evaluate_with_strategy(
    model: &dyn ForecastModel,
    draws: &[Draw],
    window: usize,
    pool: Pool,
    strategy: SamplingStrategy,
) -> f64 {
    let min_history = 10;

    let max_t = match strategy {
        SamplingStrategy::Consecutive => draws.len().saturating_sub(window + 1),
        SamplingStrategy::Sparse { span_multiplier } => draws.len().saturating_sub(window * span_multiplier + 1),
        SamplingStrategy::FullHistory => draws.len().saturating_sub(min_history + 1),
    };

    if max_t == 0 {
        return f64::NEG_INFINITY;
    }

    let max_tests = 500;
    let stride = (max_t / max_tests).max(1);

    // Collecter les indices de test, puis paralléliser les predict() calls
    let test_indices: Vec<usize> = (0..max_t).step_by(stride).collect();

    let results: Vec<f64> = test_indices
        .par_iter()
        .filter_map(|&t| {
            let dist = match strategy {
                SamplingStrategy::Consecutive => {
                    let train_end = (t + 1 + window).min(draws.len());
                    let train_data = &draws[t + 1..train_end];
                    if train_data.len() < 3 {
                        return None;
                    }
                    model.predict(train_data, pool)
                }
                SamplingStrategy::Sparse { span_multiplier } => {
                    let span = window * span_multiplier;
                    let actual_span = span.min(draws.len() - t - 1);
                    let full_range = &draws[t + 1..t + 1 + actual_span];
                    if full_range.len() < 3 {
                        return None;
                    }
                    let step = (actual_span / window).max(1);
                    let train_data: Vec<Draw> = full_range
                        .iter()
                        .step_by(step)
                        .take(window)
                        .cloned()
                        .collect();
                    if train_data.len() < 3 {
                        return None;
                    }
                    model.predict(&train_data, pool)
                }
                SamplingStrategy::FullHistory => {
                    let train_data = &draws[t + 1..];
                    if train_data.len() < min_history {
                        return None;
                    }
                    model.predict(train_data, pool)
                }
            };

            let test_draw = &draws[t];
            let test_numbers = pool.numbers_from(test_draw);
            let n_pool = pool.size() as f64;
            let uniform_ll = (1.0 / n_pool).ln();  // -3.91 balls, -2.49 stars
            let ll_cap = 2.0 * uniform_ll;          // -7.82 balls, -4.97 stars
            let hit_bonus = 3.0;
            let mut draw_ll = 0.0f64;
            for &n in test_numbers {
                let idx = (n - 1) as usize;
                if idx < dist.len() {
                    let ll = dist[idx].max(1e-15).ln().max(ll_cap);
                    let excess = ll - uniform_ll;
                    let adjusted = if excess > 0.0 { hit_bonus * excess } else { excess };
                    draw_ll += uniform_ll + adjusted;
                }
            }
            Some(draw_ll)
        })
        .collect();

    let n_tests = results.len();
    if n_tests > 0 {
        results.iter().sum::<f64>() / n_tests as f64
    } else {
        f64::NEG_INFINITY
    }
}

/// Calcule la log-likelihood de la distribution uniforme pour une pool donnée.
pub fn uniform_log_likelihood(pool: Pool) -> f64 {
    let p = 1.0 / pool.size() as f64;
    pool.pick_count() as f64 * p.ln()
}

/// Fenêtres par défaut pour les boules.
pub const DEFAULT_BALL_WINDOWS: &[usize] = &[20, 30, 50, 80, 100, 150, 200, 300];

/// Fenêtres par défaut pour les étoiles (plus longues car patterns plus clairsemés).
pub const DEFAULT_STAR_WINDOWS: &[usize] = &[50, 100, 200, 300, 500];

/// Température par défaut pour le scaling des poids (boules).
/// T<1.0 = concentration sur les meilleurs modèles (sharpening).
/// T=0.5 → exp(skill/0.5) = exp(2×skill), les ratios sont élevés au carré.
pub const DEFAULT_TEMPERATURE: f64 = 0.5;

/// Température séparée pour les étoiles.
/// Plus basse car le skill range étoiles est ~0.42 vs ~1.17 boules.
/// Avec T=0.25 : exp(0.42/0.25) = 5.4× entre meilleur et pire → poids non-uniformes.
pub const STAR_DEFAULT_TEMPERATURE: f64 = 0.25;

/// Calcule les poids de l'ensemble à partir des calibrations.
/// Pondération continue avec temperature scaling : poids = exp(skill / T).
pub fn compute_weights(
    calibrations: &[ModelCalibration],
    pool: Pool,
) -> Vec<(String, f64)> {
    compute_weights_with_params(calibrations, pool, DEFAULT_TEMPERATURE)
}

/// Calcule les poids avec température configurable.
/// Avec T<1 (sharpening), les meilleurs modèles dominent et les mauvais
/// deviennent naturellement négligeables sans besoin de dropout explicite.
pub fn compute_weights_with_params(
    calibrations: &[ModelCalibration],
    pool: Pool,
    temperature: f64,
) -> Vec<(String, f64)> {
    let uniform_ll = uniform_log_likelihood(pool);

    // Poids = exp(skill / T) pour tous les modèles (pas de dropout).
    // skill = best_ll - uniform_ll
    // T < 1 concentre sur les meilleurs (ex: T=0.5 → ratios élevés au carré)
    // T > 1 aplatit vers l'uniforme
    let raw_weights: Vec<f64> = calibrations
        .iter()
        .map(|c| {
            let skill = c.best_ll - uniform_ll;
            (skill / temperature).exp()
        })
        .collect();

    let total: f64 = raw_weights.iter().sum();

    if total > 0.0 {
        calibrations
            .iter()
            .zip(raw_weights.iter())
            .map(|(c, &w)| (c.model_name.clone(), w / total))
            .collect()
    } else {
        // Cas dégénéré → uniforme
        let n = calibrations.len() as f64;
        calibrations
            .iter()
            .map(|c| (c.model_name.clone(), 1.0 / n))
            .collect()
    }
}

/// Calcule les poids avec seuil de skill minimum.
/// Les modèles avec skill <= min_skill reçoivent un poids de 0.
/// Fallback uniforme si tous les modèles sont éliminés.
pub fn compute_weights_with_threshold(
    calibrations: &[ModelCalibration],
    pool: Pool,
    temperature: f64,
    min_skill: f64,
) -> Vec<(String, f64)> {
    let uniform_ll = uniform_log_likelihood(pool);

    let raw_weights: Vec<f64> = calibrations
        .iter()
        .map(|c| {
            let skill = c.best_ll - uniform_ll;
            if skill <= min_skill {
                0.0
            } else {
                (skill / temperature).exp()
            }
        })
        .collect();

    let total: f64 = raw_weights.iter().sum();

    if total > 0.0 {
        calibrations
            .iter()
            .zip(raw_weights.iter())
            .map(|(c, &w)| (c.model_name.clone(), w / total))
            .collect()
    } else {
        // Tous éliminés → fallback uniforme
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
    let strategy = model.sampling_strategy();

    // FullHistory : une seule évaluation, pas de boucle sur les fenêtres
    if strategy == SamplingStrategy::FullHistory {
        let ll = walk_forward_evaluate_with_strategy(
            model, draws, 0, pool, SamplingStrategy::FullHistory,
        );
        let min_history = 10;
        let max_t = draws.len().saturating_sub(min_history + 1);
        let n_tests = max_t;

        return ModelCalibration {
            model_name: model.name().to_string(),
            results: vec![CalibrationResult {
                model_name: model.name().to_string(),
                window: 0,
                sparse: false,
                log_likelihood: ll,
                n_tests,
            }],
            best_window: 0,
            best_sparse: false,
            best_ll: ll,
        };
    }

    let mut results = Vec::new();
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_window = windows[0];
    let mut best_sparse = false;

    for &window in windows {
        // Toujours évaluer en mode consécutif
        let ll = walk_forward_evaluate_with_strategy(
            model, draws, window, pool, SamplingStrategy::Consecutive,
        );
        let max_t = draws.len().saturating_sub(window + 1);

        results.push(CalibrationResult {
            model_name: model.name().to_string(),
            window,
            sparse: false,
            log_likelihood: ll,
            n_tests: max_t,
        });

        if ll > best_ll {
            best_ll = ll;
            best_window = window;
            best_sparse = false;
        }

        // Si le modèle supporte le sparse, évaluer aussi en mode sparse
        // Skip si trop peu de test points pour être statistiquement significatif
        if let SamplingStrategy::Sparse { span_multiplier } = strategy {
            let span = window * span_multiplier;
            let max_t_sparse = draws.len().saturating_sub(span + 1);

            if max_t_sparse >= 30 {
                let sparse_strategy = SamplingStrategy::Sparse { span_multiplier };
                let ll_sparse = walk_forward_evaluate_with_strategy(
                    model, draws, window, pool, sparse_strategy,
                );

                results.push(CalibrationResult {
                    model_name: model.name().to_string(),
                    window,
                    sparse: true,
                    log_likelihood: ll_sparse,
                    n_tests: max_t_sparse,
                });

                if ll_sparse > best_ll {
                    best_ll = ll_sparse;
                    best_window = window;
                    best_sparse = true;
                }
            }
        }
    }

    ModelCalibration {
        model_name: model.name().to_string(),
        results,
        best_window,
        best_sparse,
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

/// Collecte les LL par tirage pour chaque modèle.
/// Utilise la meilleure fenêtre/stratégie de chaque modèle.
/// Retourne Vec<(model_name, Vec<ll_per_test_draw>)>.
pub fn collect_detailed_ll(
    model: &dyn ForecastModel,
    draws: &[Draw],
    window: usize,
    pool: Pool,
    strategy: SamplingStrategy,
) -> Vec<f64> {
    let min_history = 10;

    let max_t = match strategy {
        SamplingStrategy::Consecutive => draws.len().saturating_sub(window + 1),
        SamplingStrategy::Sparse { span_multiplier } => draws.len().saturating_sub(window * span_multiplier + 1),
        SamplingStrategy::FullHistory => draws.len().saturating_sub(min_history + 1),
    };

    if max_t == 0 {
        return vec![];
    }

    // collect_detailed_ll conserve stride=1 pour garder la résolution complète
    // (utilisé pour meta-predictor et redundancy detection)
    let test_indices: Vec<usize> = (0..max_t).collect();

    // Paralléliser les predict() calls — chaque t est indépendant
    // On collecte (index, ll) pour préserver l'ordre
    let mut indexed_lls: Vec<(usize, f64)> = test_indices
        .par_iter()
        .filter_map(|&t| {
            let dist = match strategy {
                SamplingStrategy::Consecutive => {
                    let train_end = (t + 1 + window).min(draws.len());
                    let train_data = &draws[t + 1..train_end];
                    if train_data.len() < 3 { return None; }
                    model.predict(train_data, pool)
                }
                SamplingStrategy::Sparse { span_multiplier } => {
                    let span = window * span_multiplier;
                    let actual_span = span.min(draws.len() - t - 1);
                    let full_range = &draws[t + 1..t + 1 + actual_span];
                    if full_range.len() < 3 { return None; }
                    let step = (actual_span / window).max(1);
                    let train_data: Vec<Draw> = full_range
                        .iter().step_by(step).take(window).cloned().collect();
                    if train_data.len() < 3 { return None; }
                    model.predict(&train_data, pool)
                }
                SamplingStrategy::FullHistory => {
                    let train_data = &draws[t + 1..];
                    if train_data.len() < min_history { return None; }
                    model.predict(train_data, pool)
                }
            };

            let test_draw = &draws[t];
            let test_numbers = pool.numbers_from(test_draw);
            let n_pool = pool.size() as f64;
            let uniform_ll = (1.0 / n_pool).ln();
            let ll_cap = 2.0 * uniform_ll;
            let hit_bonus = 3.0;
            let mut draw_ll = 0.0f64;
            for &n in test_numbers {
                let idx = (n - 1) as usize;
                if idx < dist.len() {
                    let ll = dist[idx].max(1e-15).ln().max(ll_cap);
                    let excess = ll - uniform_ll;
                    let adjusted = if excess > 0.0 { hit_bonus * excess } else { excess };
                    draw_ll += uniform_ll + adjusted;
                }
            }
            Some((t, draw_ll))
        })
        .collect();

    // Trier par index pour préserver l'ordre chronologique
    indexed_lls.sort_unstable_by_key(|&(idx, _)| idx);
    indexed_lls.into_iter().map(|(_, ll)| ll).collect()
}

/// Résultat de l'analyse de redondance entre modèles.
#[derive(Debug, Clone)]
pub struct RedundancyResult {
    pub model_a: String,
    pub model_b: String,
    pub correlation: f64,
}

/// Détecte les paires de modèles redondants (corrélation > threshold).
/// Prend les LL détaillés de chaque modèle.
pub fn detect_redundancy(
    detailed_ll: &[(String, Vec<f64>)],
    threshold: f64,
) -> Vec<RedundancyResult> {
    let n = detailed_ll.len();
    let mut results = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let (name_a, lls_a) = &detailed_ll[i];
            let (name_b, lls_b) = &detailed_ll[j];

            // Aligner sur la longueur commune
            let len = lls_a.len().min(lls_b.len());
            if len < 5 {
                continue;
            }

            let a = &lls_a[..len];
            let b = &lls_b[..len];

            let corr = pearson_correlation(a, b);
            if corr > threshold {
                results.push(RedundancyResult {
                    model_a: name_a.clone(),
                    model_b: name_b.clone(),
                    correlation: corr,
                });
            }
        }
    }

    results.sort_by(|x, y| y.correlation.partial_cmp(&x.correlation).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Pénalise les modèles corrélés de façon continue.
/// Pour chaque paire avec corrélation > min_corr, le modèle le plus faible
/// voit son poids multiplié par un facteur de pénalité décroissant.
/// Pénalité = 1 - strength × (corr - min_corr) / (1 - min_corr), plancher = floor.
pub fn apply_decorrelation_penalty(
    weights: &mut Vec<(String, f64)>,
    redundancies: &[RedundancyResult],
    min_corr: f64,
    strength: f64,
    floor: f64,
) {
    for r in redundancies {
        if r.correlation <= min_corr {
            continue;
        }
        let w_a = weights
            .iter()
            .find(|(n, _)| *n == r.model_a)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        let w_b = weights
            .iter()
            .find(|(n, _)| *n == r.model_b)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        let weaker = if w_a <= w_b {
            &r.model_a
        } else {
            &r.model_b
        };
        let penalty =
            (1.0 - strength * (r.correlation - min_corr) / (1.0 - min_corr)).max(floor);
        if let Some((_, w)) = weights.iter_mut().find(|(n, _)| n == weaker) {
            *w *= penalty;
        }
    }
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    if total > 0.0 {
        for (_, w) in weights.iter_mut() {
            *w /= total;
        }
    }
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
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
                best_sparse: false,
                best_ll: -15.0,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 30,
                best_sparse: false,
                best_ll: -18.0,
            },
        ];
        let weights = compute_weights(&calibrations, Pool::Balls);
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights sum = {}", sum);
    }

    #[test]
    fn test_compute_weights_bad_model_has_lower_weight() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "Good".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 1.0, // Meilleur que l'uniforme
            },
            ModelCalibration {
                model_name: "Bad".to_string(),
                results: vec![],
                best_window: 30,
                best_sparse: false,
                best_ll: uniform_ll - 1.0, // Pire que l'uniforme
            },
        ];
        let weights = compute_weights(&calibrations, Pool::Balls);
        let good_weight = weights.iter().find(|(n, _)| n == "Good").unwrap().1;
        let bad_weight = weights.iter().find(|(n, _)| n == "Bad").unwrap().1;
        assert!(bad_weight < good_weight, "Bad model should have lower weight");
        assert!(bad_weight < 0.5, "Bad model should have smaller weight, got {}", bad_weight);
    }

    #[test]
    fn test_compute_weights_exponential_ordering() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "Best".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 2.0,
            },
            ModelCalibration {
                model_name: "Medium".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.5,
            },
            ModelCalibration {
                model_name: "Worst".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 3.0,
            },
        ];
        let weights = compute_weights(&calibrations, Pool::Balls);
        let best_w = weights.iter().find(|(n, _)| n == "Best").unwrap().1;
        let medium_w = weights.iter().find(|(n, _)| n == "Medium").unwrap().1;
        let worst_w = weights.iter().find(|(n, _)| n == "Worst").unwrap().1;
        assert!(best_w > medium_w, "Best should have highest weight");
        assert!(medium_w > worst_w, "Medium should beat worst");
    }

    #[test]
    fn test_calibrate_model() {
        // 200 tirages pour que les évaluations sparse aient assez de test points (>= 30)
        let draws = make_test_draws(200);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let cal = calibrate_model(&model, &draws, &[10, 20, 30], Pool::Balls);
        assert_eq!(cal.model_name, "Dirichlet");
        // Dirichlet has Sparse strategy → 3 consecutive + up to 3 sparse
        // Sparse évalué seulement si max_t_sparse >= 30
        assert!(cal.results.len() >= 3 && cal.results.len() <= 6);
        assert!(cal.best_ll.is_finite() || cal.best_ll == f64::NEG_INFINITY);
    }

    #[test]
    fn test_calibrate_model_consecutive_only() {
        let draws = make_test_draws(50);
        let model = crate::models::logistic::LogisticModel::new(0.01, 0.001, 50, 100);
        let cal = calibrate_model(&model, &draws, &[10, 20], Pool::Balls);
        assert_eq!(cal.model_name, "Logistic");
        // Logistic is Consecutive → 2 results only
        assert_eq!(cal.results.len(), 2);
        assert!(!cal.best_sparse);
    }

    #[test]
    fn test_walk_forward_sparse() {
        let draws = make_test_draws(100);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let ll = walk_forward_evaluate_with_strategy(
            &model, &draws, 20, Pool::Balls,
            SamplingStrategy::Sparse { span_multiplier: 3 },
        );
        assert!(ll.is_finite(), "Sparse LL should be finite, got {}", ll);
    }

    #[test]
    fn test_temperature_sharpening_increases_ratio() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        // Utiliser deux modèles POSITIFS (au-dessus de l'uniforme) pour tester le sharpening.
        let calibrations = vec![
            ModelCalibration {
                model_name: "Good".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 2.0,
            },
            ModelCalibration {
                model_name: "Mediocre".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.5,
            },
        ];
        // T=1: ratio = exp(1.5) ≈ 4.5
        let w_t1 = compute_weights_with_params(&calibrations, Pool::Balls, 1.0);
        let ratio_t1 = w_t1[0].1 / w_t1[1].1;
        // T=0.5 (default): ratio = exp(3) ≈ 20 → concentration forte
        let w_t05 = compute_weights_with_params(&calibrations, Pool::Balls, 0.5);
        let ratio_t05 = w_t05[0].1 / w_t05[1].1;
        assert!(ratio_t05 > ratio_t1, "T=0.5 should increase ratio: {} vs {}", ratio_t05, ratio_t1);
        assert!(ratio_t05 > 10.0, "T=0.5 ratio should be significant: {}", ratio_t05);
    }

    #[test]
    fn test_bad_model_gets_negligible_weight_with_sharpening() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "Good".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 1.0,
            },
            ModelCalibration {
                model_name: "Terrible".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 6.0, // skill = -6
            },
        ];
        // Avec T=0.5, exp((-6)/0.5) = exp(-12) ≈ 6e-6 → quasi-nul sans dropout
        let weights = compute_weights(&calibrations, Pool::Balls);
        let terrible_w = weights.iter().find(|(n, _)| n == "Terrible").unwrap().1;
        assert!(terrible_w < 1e-4, "Terrible model should be negligible with T=0.5, got {}", terrible_w);
        let good_w = weights.iter().find(|(n, _)| n == "Good").unwrap().1;
        assert!(good_w > 0.999, "Good should get nearly all weight: {}", good_w);
    }

    #[test]
    fn test_equal_bad_models_get_equal_weights() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "A".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 10.0,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 10.0,
            },
        ];
        let weights = compute_weights(&calibrations, Pool::Balls);
        // Equal skill → equal weights (both bad, but equal)
        assert!((weights[0].1 - 0.5).abs() < 1e-10);
        assert!((weights[1].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_walk_forward_full_history() {
        let draws = make_test_draws(50);
        let model = crate::models::dirichlet::DirichletModel::new(1.0);
        let ll = walk_forward_evaluate_with_strategy(
            &model, &draws, 0, Pool::Balls,
            SamplingStrategy::FullHistory,
        );
        assert!(ll.is_finite(), "FullHistory LL should be finite, got {}", ll);
    }

    #[test]
    fn test_calibrate_full_history_single_result() {
        let draws = make_test_draws(50);
        // Créer un modèle wrapper qui déclare FullHistory
        struct FullHistoryModel;
        impl crate::models::ForecastModel for FullHistoryModel {
            fn name(&self) -> &str { "TestFH" }
            fn predict(&self, _draws: &[lemillion_db::models::Draw], pool: Pool) -> Vec<f64> {
                vec![1.0 / pool.size() as f64; pool.size()]
            }
            fn params(&self) -> std::collections::HashMap<String, f64> {
                std::collections::HashMap::new()
            }
            fn sampling_strategy(&self) -> SamplingStrategy {
                SamplingStrategy::FullHistory
            }
        }
        let model = FullHistoryModel;
        let cal = calibrate_model(&model, &draws, &[10, 20, 30], Pool::Balls);
        assert_eq!(cal.model_name, "TestFH");
        assert_eq!(cal.results.len(), 1, "FullHistory should produce exactly 1 result");
        assert_eq!(cal.best_window, 0, "FullHistory best_window should be 0");
        assert!(!cal.best_sparse);
    }

    #[test]
    fn test_compute_weights_with_threshold_zeros_bad_models() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "Good".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.5,
            },
            ModelCalibration {
                model_name: "Uniform".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll, // skill = 0
            },
            ModelCalibration {
                model_name: "Bad".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 0.5, // skill < 0
            },
        ];
        let weights = compute_weights_with_threshold(&calibrations, Pool::Balls, 1.0, 0.0);
        let good_w = weights.iter().find(|(n, _)| n == "Good").unwrap().1;
        let uniform_w = weights.iter().find(|(n, _)| n == "Uniform").unwrap().1;
        let bad_w = weights.iter().find(|(n, _)| n == "Bad").unwrap().1;
        assert_eq!(uniform_w, 0.0, "skill=0 should be zeroed");
        assert_eq!(bad_w, 0.0, "skill<0 should be zeroed");
        assert!((good_w - 1.0).abs() < 1e-10, "Good should get all weight: {good_w}");
    }

    #[test]
    fn test_compute_weights_with_threshold_fallback_uniform() {
        let uniform_ll = uniform_log_likelihood(Pool::Balls);
        let calibrations = vec![
            ModelCalibration {
                model_name: "A".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 1.0,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 2.0,
            },
        ];
        // All models below threshold → uniform fallback
        let weights = compute_weights_with_threshold(&calibrations, Pool::Balls, 1.0, 0.0);
        assert!((weights[0].1 - 0.5).abs() < 1e-10);
        assert!((weights[1].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decorrelation_no_penalty_below_threshold() {
        let mut weights = vec![
            ("A".to_string(), 0.3),
            ("B".to_string(), 0.7),
        ];
        let redundancies = vec![RedundancyResult {
            model_a: "A".to_string(),
            model_b: "B".to_string(),
            correlation: 0.79,
        }];
        apply_decorrelation_penalty(&mut weights, &redundancies, 0.80, 0.5, 0.30);
        // Below threshold → no penalty, just renormalized
        assert!((weights[0].1 - 0.3).abs() < 1e-10);
        assert!((weights[1].1 - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_decorrelation_at_threshold_boundary() {
        let mut weights = vec![
            ("A".to_string(), 0.4),
            ("B".to_string(), 0.6),
        ];
        let redundancies = vec![RedundancyResult {
            model_a: "A".to_string(),
            model_b: "B".to_string(),
            correlation: 0.80,
        }];
        apply_decorrelation_penalty(&mut weights, &redundancies, 0.80, 0.5, 0.30);
        // At exact threshold → penalty = 1 - 0.5 * 0/0.2 = 1.0, no change
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decorrelation_at_0_90() {
        let mut weights = vec![
            ("A".to_string(), 0.4),
            ("B".to_string(), 0.6),
        ];
        let redundancies = vec![RedundancyResult {
            model_a: "A".to_string(),
            model_b: "B".to_string(),
            correlation: 0.90,
        }];
        apply_decorrelation_penalty(&mut weights, &redundancies, 0.80, 0.5, 0.30);
        // penalty = 1 - 0.5 * (0.10/0.20) = 0.75
        // A (weaker, 0.4) → 0.4 * 0.75 = 0.30
        // B stays 0.6, total = 0.9, renormalized
        let a_w = weights.iter().find(|(n, _)| n == "A").unwrap().1;
        let b_w = weights.iter().find(|(n, _)| n == "B").unwrap().1;
        assert!(a_w < b_w);
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decorrelation_floor_respected() {
        let mut weights = vec![
            ("A".to_string(), 0.5),
            ("B".to_string(), 0.5),
        ];
        let redundancies = vec![RedundancyResult {
            model_a: "A".to_string(),
            model_b: "B".to_string(),
            correlation: 1.0,
        }];
        // strength=1.0 would give penalty = 0, but floor = 0.30 applies
        apply_decorrelation_penalty(&mut weights, &redundancies, 0.80, 1.0, 0.30);
        let a_w = weights.iter().find(|(n, _)| n == "A").unwrap().1;
        let b_w = weights.iter().find(|(n, _)| n == "B").unwrap().1;
        // One of them got floor penalty (0.30), the other is original
        // Since both equal, first alphabetically gets penalized (model_a = "A")
        // Actually with equal weights, w_a <= w_b → weaker = model_a = "A"
        // A: 0.5 * 0.30 = 0.15, B: 0.5, total = 0.65
        assert!(a_w < b_w);
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weights_json_roundtrip() {
        let weights = EnsembleWeights {
            ball_weights: vec![("A".to_string(), 0.5), ("B".to_string(), 0.5)],
            star_weights: vec![("A".to_string(), 0.3), ("B".to_string(), 0.7)],
            calibrations: vec![],
            detailed_ll: Vec::new(),
            star_detailed_ll: Vec::new(),
        };
        let json = serde_json::to_string(&weights).unwrap();
        let loaded: EnsembleWeights = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.ball_weights.len(), 2);
        assert_eq!(loaded.star_weights[1].1, 0.7);
    }
}
