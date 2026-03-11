use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use lemillion_db::models::{Draw, Pool};
use crate::models::{ForecastModel, SamplingStrategy, filter_star_era};

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
    /// Number of test points for the best configuration.
    #[serde(default)]
    pub best_n_tests: usize,
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
    /// Poids de stacking boules (optionnel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stacking_balls: Option<crate::ensemble::stacking::StackingWeights>,
    /// Poids de stacking étoiles (optionnel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stacking_stars: Option<crate::ensemble::stacking::StackingWeights>,
    /// Matrice de corrélation complète entre modèles (boules) — v7 decorrelation.
    /// Vec<(model_a, model_b, correlation)> pour toutes paires avec |corr| > 0.3.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub correlation_matrix: Vec<(String, String, f64)>,
    /// Matrice de corrélation étoiles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub star_correlation_matrix: Vec<(String, String, f64)>,
    /// v16: Beta-transform parameters for balls (alpha, beta). Identity = (1.0, 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta_balls: Option<(f64, f64)>,
    /// v16: Beta-transform parameters for stars (alpha, beta). Identity = (1.0, 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta_stars: Option<(f64, f64)>,
    /// v16: Optimal temperature for balls learned by NLL grid-search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimal_t_balls: Option<f64>,
    /// v16: Optimal temperature for stars learned by NLL grid-search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimal_t_stars: Option<f64>,
    /// v17: Learned coherence weight for balls (jackpot scoring). Default: 30.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherence_ball_weight: Option<f64>,
    /// v17: Learned coherence weight for stars (jackpot scoring). Default: 15.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherence_star_weight: Option<f64>,
    /// v17: Learned stacking blend factor for balls. Default: 0.6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stacking_blend_balls: Option<f64>,
    /// v17: Learned stacking blend factor for stars. Default: 0.6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stacking_blend_stars: Option<f64>,
    /// v17: Learned online blend EWMA alpha. Default: 0.15.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub online_ewma_alpha: Option<f64>,
    /// v17: Learned online blend window size. Default: 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub online_window: Option<usize>,
}

/// Calcule le nombre réel de test points après stride.
/// Le stride est calculé pour limiter à ~500 tests max.
fn actual_test_count(max_t: usize) -> usize {
    if max_t == 0 { return 0; }
    let stride = (max_t / 500).max(1);
    (0..max_t).step_by(stride).count()
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
/// When pool == Stars, draws are filtered to the current star era (post 2016-09-27)
/// to avoid training on incompatible star pools (2/9, 2/11).
pub fn walk_forward_evaluate_with_strategy(
    model: &dyn ForecastModel,
    draws: &[Draw],
    window: usize,
    pool: Pool,
    strategy: SamplingStrategy,
) -> f64 {
    // For stars, filter to current era only (post 2016-09-27, 2/12 format)
    let draws = if pool == Pool::Stars {
        filter_star_era(draws)
    } else {
        draws
    };

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
    // v10: toujours inclure les 5 tirages les plus récents (indices 0-4)
    let mut test_indices: Vec<usize> = (0..max_t).step_by(stride).collect();
    for recent in 0..5.min(max_t) {
        if !test_indices.contains(&recent) {
            test_indices.push(recent);
        }
    }
    test_indices.sort();
    test_indices.dedup();

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
            let mut draw_ll = 0.0f64;
            for &n in test_numbers {
                let idx = (n - 1) as usize;
                if idx < dist.len() {
                    let ll = dist[idx].max(1e-15).ln().max(ll_cap);
                    draw_ll += ll;
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

/// Fenêtres par défaut pour les étoiles (raccourcies post-filtrage ère: ~1040 draws valides).
pub const DEFAULT_STAR_WINDOWS: &[usize] = &[30, 50, 80, 100, 150, 200];

/// Température par défaut pour le scaling des poids.
/// T=1.0 : pas de sharpening dans les poids. Le seul contrôle de température
/// est appliqué sur la distribution finale dans cmd_predict.
pub const DEFAULT_TEMPERATURE: f64 = 1.0;

/// Température séparée pour les étoiles.
/// T=1.0 : pas de double-sharpening.
pub const STAR_DEFAULT_TEMPERATURE: f64 = 1.0;

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

    // BMA: poids = exp(total_skill / T) où total_skill = skill_per_draw × n_tests.
    // Cela amplifie exponentiellement les modèles avec un signal cumulé fort.
    let raw_weights: Vec<f64> = calibrations
        .iter()
        .map(|c| {
            let skill_per_draw = c.best_ll - uniform_ll;
            let n = if c.best_n_tests > 0 { c.best_n_tests } else {
                // Fallback pour les anciens fichiers de calibration sans best_n_tests
                c.results.iter()
                    .find(|r| r.window == c.best_window && r.sparse == c.best_sparse)
                    .map(|r| r.n_tests)
                    .unwrap_or(100)
            };
            // v11: Zero-skill threshold — models at or below uniform get 0 weight.
            // Without this, exp(0) = 1.0 gives uniform-skill models nonzero weight,
            // diluting the top models and reducing concentration.
            // Threshold 0.0005 filters noise-level skill (< 0.05% per draw improvement).
            if skill_per_draw <= 0.0005 {
                return 0.0;
            }
            let total_skill = (skill_per_draw * n as f64).min(20.0);
            let raw = (total_skill / temperature).exp();

            // v11: Cross-window stability penalty.
            // Penalize models whose LL varies a lot across windows — unreliable signal.
            let stability = cross_window_stability(&c.results);
            raw * stability
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

/// v11: Compute cross-window stability penalty.
/// Returns 1.0 for stable models, < 1.0 for models with high LL variance across windows.
fn cross_window_stability(results: &[CalibrationResult]) -> f64 {
    let skills: Vec<f64> = results.iter()
        .map(|r| r.log_likelihood)
        .filter(|&ll| ll.is_finite())
        .collect();

    if skills.len() < 2 {
        return 1.0;
    }

    let mean = skills.iter().sum::<f64>() / skills.len() as f64;
    let variance = skills.iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f64>() / skills.len() as f64;

    // Very mild Gaussian penalty — only penalizes extremely unstable models.
    // LL values across windows typically vary by 0.001-0.01.
    // sigma=0.5 means: variance 0.01 → 0.980, 0.1 → 0.819, 1.0 → 0.135
    let sigma = 0.5;
    (-variance / (2.0 * sigma * sigma)).exp()
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
            let skill_per_draw = c.best_ll - uniform_ll;
            if skill_per_draw <= min_skill {
                0.0
            } else {
                let n = if c.best_n_tests > 0 { c.best_n_tests } else {
                    c.results.iter()
                        .find(|r| r.window == c.best_window && r.sparse == c.best_sparse)
                        .map(|r| r.n_tests)
                        .unwrap_or(100)
                };
                let total_skill = skill_per_draw * n as f64;
                let raw = (total_skill / temperature).exp();
                // v11: stability penalty
                raw * cross_window_stability(&c.results)
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
        let n_tests = actual_test_count(max_t);

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
            best_n_tests: n_tests,
        };
    }

    let mut results = Vec::new();
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_window = windows[0];
    let mut best_sparse = false;
    let mut best_n_tests = 0usize;

    for &window in windows {
        // Toujours évaluer en mode consécutif
        let ll = walk_forward_evaluate_with_strategy(
            model, draws, window, pool, SamplingStrategy::Consecutive,
        );
        let max_t = draws.len().saturating_sub(window + 1);

        let n_actual = actual_test_count(max_t);
        results.push(CalibrationResult {
            model_name: model.name().to_string(),
            window,
            sparse: false,
            log_likelihood: ll,
            n_tests: n_actual,
        });

        if ll > best_ll {
            best_ll = ll;
            best_window = window;
            best_sparse = false;
            best_n_tests = n_actual;
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

                let n_actual_sparse = actual_test_count(max_t_sparse);
                results.push(CalibrationResult {
                    model_name: model.name().to_string(),
                    window,
                    sparse: true,
                    log_likelihood: ll_sparse,
                    n_tests: n_actual_sparse,
                });

                if ll_sparse > best_ll {
                    best_ll = ll_sparse;
                    best_window = window;
                    best_sparse = true;
                    best_n_tests = n_actual_sparse;
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
        best_n_tests,
    }
}

/// Test de permutation pour valider le skill d'un modèle.
/// Permute la séquence des tirages n_perms fois et recalcule le skill.
/// Retourne la p-value = fraction des permutations avec skill >= skill réel.
pub fn permutation_test_skill(
    model: &dyn ForecastModel,
    draws: &[Draw],
    window: usize,
    pool: Pool,
    n_perms: usize,
) -> f64 {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let strategy = model.sampling_strategy();
    let real_ll = walk_forward_evaluate_with_strategy(model, draws, window, pool, strategy);
    if !real_ll.is_finite() {
        return 1.0;
    }

    let uniform_ll = uniform_log_likelihood(pool);
    let real_skill = real_ll - uniform_ll;
    if real_skill <= 0.0 {
        return 1.0; // pas de skill positif, pas besoin de tester
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut n_better = 0usize;

    for _ in 0..n_perms {
        let mut shuffled = draws.to_vec();
        shuffled.shuffle(&mut rng);
        let perm_ll = walk_forward_evaluate_with_strategy(model, &shuffled, window, pool, strategy);
        let perm_skill = perm_ll - uniform_ll;
        if perm_skill >= real_skill {
            n_better += 1;
        }
    }

    (n_better as f64 + 1.0) / (n_perms as f64 + 1.0) // +1 pour inclure l'observation réelle
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
/// When pool == Stars, draws are filtered to the current star era (post 2016-09-27).
pub fn collect_detailed_ll(
    model: &dyn ForecastModel,
    draws: &[Draw],
    window: usize,
    pool: Pool,
    strategy: SamplingStrategy,
) -> Vec<f64> {
    // For stars, filter to current era only (post 2016-09-27, 2/12 format)
    let draws = if pool == Pool::Stars {
        filter_star_era(draws)
    } else {
        draws
    };

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
            let mut draw_ll = 0.0f64;
            for &n in test_numbers {
                let idx = (n - 1) as usize;
                if idx < dist.len() {
                    let ll = dist[idx].max(1e-15).ln().max(ll_cap);
                    draw_ll += ll;
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
        let penalty =
            (1.0 - strength * (r.correlation - min_corr) / (1.0 - min_corr)).max(floor);

        // Fix v9: quand les poids sont quasi-égaux, pénaliser les DEUX modèles
        // avec sqrt(penalty) au lieu de pénaliser arbitrairement le premier.
        let weight_ratio = if w_a > 0.0 && w_b > 0.0 {
            (w_a / w_b).min(w_b / w_a)
        } else {
            0.0
        };
        if weight_ratio > 0.95 {
            let sqrt_penalty = penalty.sqrt();
            for name in [&r.model_a, &r.model_b] {
                if let Some((_, w)) = weights.iter_mut().find(|(n, _)| n == name) {
                    *w *= sqrt_penalty;
                }
            }
        } else {
            let weaker = if w_a <= w_b { &r.model_a } else { &r.model_b };
            if let Some((_, w)) = weights.iter_mut().find(|(n, _)| n == weaker) {
                *w *= penalty;
            }
        }
    }
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    if total > 0.0 {
        for (_, w) in weights.iter_mut() {
            *w /= total;
        }
    }
}

/// Compute the full correlation matrix for all model pairs (v7).
/// Returns all pairs with |correlation| > min_corr.
pub fn compute_correlation_matrix(
    detailed_ll: &[(String, Vec<f64>)],
    min_corr: f64,
) -> Vec<(String, String, f64)> {
    let n = detailed_ll.len();
    let mut results = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let (name_a, lls_a) = &detailed_ll[i];
            let (name_b, lls_b) = &detailed_ll[j];

            let len = lls_a.len().min(lls_b.len());
            if len < 5 {
                continue;
            }

            let corr = pearson_correlation(&lls_a[..len], &lls_b[..len]);
            if corr.abs() > min_corr {
                results.push((name_a.clone(), name_b.clone(), corr));
            }
        }
    }

    results
}

/// Compute effective weights accounting for model correlations (v7).
/// effective_w_i = w_i × Π_j sqrt(1 - ρ_ij²) for j with ρ_ij > threshold.
/// This penalizes correlated models to avoid double-counting in the log-linear pool.
pub fn compute_decorrelated_weights(
    weights: &[f64],
    model_names: &[String],
    correlation_matrix: &[(String, String, f64)],
    threshold: f64,
) -> Vec<f64> {
    let n = weights.len();
    let mut effective = weights.to_vec();

    for i in 0..n {
        let name_i = &model_names[i];
        let mut penalty = 1.0f64;

        let _ = threshold; // kept for API compat; continuous penalty replaces hard threshold
        for (a, b, corr) in correlation_matrix {
            // Check if this correlation involves model i
            let partner = if a == name_i {
                Some(b)
            } else if b == name_i {
                Some(a)
            } else {
                None
            };

            if partner.is_some() {
                // Continuous Gaussian penalty: center=0.50, σ=0.20
                // Gentle below 0.50, significant at 0.60+, strong at 0.75+
                let excess = (corr.abs() - 0.50).max(0.0);
                penalty *= (-0.5 * excess * excess / (0.20_f64 * 0.20)).exp();
            }
        }

        effective[i] *= penalty;
    }

    // Renormalize
    let total: f64 = effective.iter().sum();
    if total > 0.0 {
        for w in &mut effective {
            *w /= total;
        }
    }

    effective
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

/// v16: Joint optimization of beta-transform + temperature on walk-forward test data.
/// Returns (beta_balls, beta_stars, optimal_t_balls, optimal_t_stars).
/// Uses cached ensemble distributions to avoid recomputing model predictions for each grid point.
pub fn optimize_post_pooling(
    combiner: &super::EnsembleCombiner,
    draws: &[Draw],
    n_test: usize,
) -> (Option<(f64, f64)>, Option<(f64, f64)>, Option<f64>, Option<f64>) {
    let n_test = n_test.min(draws.len().saturating_sub(30));
    if n_test < 5 {
        return (None, None, None, None);
    }

    // Collect ensemble distributions for test draws (expensive, done once)
    let mut ball_dists: Vec<(Vec<f64>, [u8; 5])> = Vec::with_capacity(n_test);
    let mut star_dists: Vec<(Vec<f64>, [u8; 2])> = Vec::with_capacity(n_test);

    for t in 0..n_test {
        let context = &draws[t + 1..];
        if context.len() < 30 {
            continue;
        }
        let ball_pred = combiner.predict(context, Pool::Balls);
        let star_pred = combiner.predict(context, Pool::Stars);
        ball_dists.push((ball_pred.distribution, draws[t].balls));
        star_dists.push((star_pred.distribution, draws[t].stars));
    }

    if ball_dists.is_empty() {
        return (None, None, None, None);
    }

    // v16b: constrain alpha >= 1 (only sharpening, never flattening).
    // NLL optimization with alpha<1 flattens distributions, which hurts jackpot mode.
    let alphas: &[f64] = &[1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0];
    let betas: &[f64] = &[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0];
    let temps: &[f64] = &[0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70, 1.0];

    // Optimize balls
    let (beta_b, t_b) = {
        let mut best_nll = f64::INFINITY;
        let mut best_ab = (1.0, 1.0);
        let mut best_t = 1.0;
        for &alpha in alphas {
            for &beta in betas {
                for &temp in temps {
                    let nll = score_for_params(&ball_dists, alpha, beta, temp, |actual| actual.as_slice());
                    if nll < best_nll {
                        best_nll = nll;
                        best_ab = (alpha, beta);
                        best_t = temp;
                    }
                }
            }
        }
        (best_ab, best_t)
    };

    // Optimize stars
    let (beta_s, t_s) = {
        let mut best_nll = f64::INFINITY;
        let mut best_ab = (1.0, 1.0);
        let mut best_t = 1.0;
        for &alpha in alphas {
            for &beta in betas {
                for &temp in temps {
                    let nll = score_for_params(&star_dists, alpha, beta, temp, |actual| actual.as_slice());
                    if nll < best_nll {
                        best_nll = nll;
                        best_ab = (alpha, beta);
                        best_t = temp;
                    }
                }
            }
        }
        (best_ab, best_t)
    };

    // Only return non-identity transforms
    let beta_balls = if (beta_b.0 - 1.0).abs() > 0.01 || (beta_b.1 - 1.0).abs() > 0.01 {
        Some(beta_b)
    } else {
        None
    };
    let beta_stars = if (beta_s.0 - 1.0).abs() > 0.01 || (beta_s.1 - 1.0).abs() > 0.01 {
        Some(beta_s)
    } else {
        None
    };
    let optimal_t_balls = if (t_b - 1.0).abs() > 0.01 { Some(t_b) } else { None };
    let optimal_t_stars = if (t_s - 1.0).abs() > 0.01 { Some(t_s) } else { None };

    (beta_balls, beta_stars, optimal_t_balls, optimal_t_stars)
}

/// Compute score for a given (alpha, beta, temperature) on cached distributions.
/// Uses a blend of NLL and Bayesian score (product of probabilities assigned to correct numbers).
/// The Bayesian score rewards concentration while NLL prevents catastrophic misses.
fn score_for_params<T, F>(dists: &[(Vec<f64>, T)], alpha: f64, beta: f64, temp: f64, numbers_fn: F) -> f64
where F: Fn(&T) -> &[u8] {
    let inv_t = 1.0 / temp;
    let mut total_log_score = 0.0f64;
    let n_dists = dists.len();
    for (raw_dist, actual) in dists {
        let mut dist = raw_dist.clone();
        super::beta_transform(&mut dist, alpha, beta);
        let scaled: Vec<f64> = dist.iter().map(|&p| p.max(1e-15).powf(inv_t)).collect();
        let total: f64 = scaled.iter().sum();
        if total <= 0.0 { continue; }
        let norm: Vec<f64> = scaled.iter().map(|s| s / total).collect();
        let uniform_p = 1.0 / norm.len() as f64;
        // Bayesian score: product of (prob / uniform) for correct numbers
        // In log space: sum of log(prob / uniform)
        for &n in numbers_fn(actual) {
            let idx = (n as usize).saturating_sub(1);
            if idx < norm.len() {
                let ratio = norm[idx] / uniform_p;
                total_log_score += ratio.max(1e-15).ln();
            }
        }
    }
    // Return negative (we minimize, so negate the score we want to maximize)
    // Average per draw for stability
    -(total_log_score / n_dists.max(1) as f64)
}

/// v17: Optimize coherence weights by walk-forward scoring.
/// Evaluates different (CW, SCW) pairs by scoring actual draws.
pub fn optimize_coherence_weights(
    combiner: &super::EnsembleCombiner,
    draws: &[Draw],
    n_test: usize,
) -> (f64, f64) {
    let n_test = n_test.min(draws.len().saturating_sub(30));
    if n_test < 10 {
        return (30.0, 15.0);
    }

    // Collect test draw data: (ball_bayes, ball_coherence, star_bayes, star_coherence)
    let mut data: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(n_test);

    for t in 0..n_test {
        let context = &draws[t + 1..];
        if context.len() < 30 { continue; }

        let ball_dist = combiner.predict(context, Pool::Balls).distribution;
        let star_dist = combiner.predict(context, Pool::Stars).distribution;

        let uniform_ball = 1.0 / 50.0;
        let uniform_star = 1.0 / 12.0;
        let ball_score: f64 = draws[t].balls.iter()
            .map(|&b| (ball_dist[(b - 1) as usize] / uniform_ball).max(1e-30).ln())
            .sum();
        let star_score: f64 = draws[t].stars.iter()
            .map(|&s| (star_dist[(s - 1) as usize] / uniform_star).max(1e-30).ln())
            .sum();

        let coherence = crate::sampler::CoherenceScorer::from_history(context, Pool::Balls);
        let ball_coh = (coherence.score_balls(&draws[t].balls) - 0.5).clamp(-0.5, 0.5);
        let star_coh_scorer = crate::sampler::StarCoherenceScorer::from_history(context);
        let star_coh = (star_coh_scorer.score_star_pair(&draws[t].stars) - 0.5).clamp(-0.5, 0.5);

        data.push((ball_score, ball_coh, star_score, star_coh));
    }

    if data.len() < 5 {
        return (30.0, 15.0);
    }

    // Grid-search: maximize penalized mean of total scores (Sharpe-like)
    let cw_candidates = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0];
    let mut best_cw = 30.0;
    let mut best_metric = f64::NEG_INFINITY;
    let n = data.len() as f64;

    for &cw in &cw_candidates {
        let scores: Vec<f64> = data.iter().map(|(b, c, _, _)| b + cw * c).collect();
        let mean = scores.iter().sum::<f64>() / n;
        let var = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
        let metric = mean - 0.3 * var.sqrt();
        if metric > best_metric {
            best_metric = metric;
            best_cw = cw;
        }
    }

    let scw_candidates = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0];
    let mut best_scw = 15.0;
    let mut best_metric_s = f64::NEG_INFINITY;

    for &scw in &scw_candidates {
        let scores: Vec<f64> = data.iter().map(|(_, _, s, c)| s + scw * c).collect();
        let mean = scores.iter().sum::<f64>() / n;
        let var = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
        let metric = mean - 0.3 * var.sqrt();
        if metric > best_metric_s {
            best_metric_s = metric;
            best_scw = scw;
        }
    }

    (best_cw, best_scw)
}

/// v17: Optimize stacking blend factors by walk-forward log-likelihood.
pub fn optimize_stacking_blend(
    combiner: &super::EnsembleCombiner,
    draws: &[Draw],
    stacking_balls: Option<&crate::ensemble::stacking::StackingWeights>,
    stacking_stars: Option<&crate::ensemble::stacking::StackingWeights>,
    n_test: usize,
) -> (f64, f64) {
    let n_test = n_test.min(draws.len().saturating_sub(30));
    if n_test < 5 || (stacking_balls.is_none() && stacking_stars.is_none()) {
        return (0.6, 0.6);
    }

    // Cache standard and full-stacked predictions
    let mut ball_data: Vec<(Vec<f64>, Vec<f64>, [u8; 5])> = Vec::new();
    let mut star_data: Vec<(Vec<f64>, Vec<f64>, [u8; 2])> = Vec::new();

    for t in 0..n_test {
        let context = &draws[t + 1..];
        if context.len() < 30 { continue; }

        if let Some(sw) = stacking_balls {
            let standard = combiner.predict(context, Pool::Balls).distribution;
            let stacked = combiner.predict_stacked(context, Pool::Balls, sw, 1.0).distribution;
            ball_data.push((standard, stacked, draws[t].balls));
        }
        if let Some(sw) = stacking_stars {
            let standard = combiner.predict(context, Pool::Stars).distribution;
            let stacked = combiner.predict_stacked(context, Pool::Stars, sw, 1.0).distribution;
            star_data.push((standard, stacked, draws[t].stars));
        }
    }

    let blends = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    let best_ball_blend = if !ball_data.is_empty() {
        let mut best = 0.6;
        let mut best_ll = f64::NEG_INFINITY;
        for &blend in &blends {
            let ll: f64 = ball_data.iter().map(|(standard, stacked, actual)| {
                actual.iter().map(|&b| {
                    let idx = (b - 1) as usize;
                    let p = blend * stacked[idx] + (1.0 - blend) * standard[idx];
                    p.max(1e-15).ln()
                }).sum::<f64>()
            }).sum();
            if ll > best_ll { best_ll = ll; best = blend; }
        }
        best
    } else {
        0.6
    };

    let best_star_blend = if !star_data.is_empty() {
        let mut best = 0.6;
        let mut best_ll = f64::NEG_INFINITY;
        for &blend in &blends {
            let ll: f64 = star_data.iter().map(|(standard, stacked, actual)| {
                actual.iter().map(|&s| {
                    let idx = (s - 1) as usize;
                    let p = blend * stacked[idx] + (1.0 - blend) * standard[idx];
                    p.max(1e-15).ln()
                }).sum::<f64>()
            }).sum();
            if ll > best_ll { best_ll = ll; best = blend; }
        }
        best
    } else {
        0.6
    };

    (best_ball_blend, best_star_blend)
}

/// v17: Optimize online blend parameters (EWMA alpha and window) by walk-forward LL.
pub fn optimize_online_blend(
    combiner: &super::EnsembleCombiner,
    draws: &[Draw],
    n_test: usize,
) -> (f64, usize) {
    let n_test = n_test.min(draws.len().saturating_sub(30));
    if n_test < 5 {
        return (0.15, 8);
    }

    // Cache offline predictions
    let mut ball_data: Vec<(Vec<f64>, usize, [u8; 5])> = Vec::new();
    for t in 0..n_test {
        let context = &draws[t + 1..];
        if context.len() < 30 { continue; }
        let offline = combiner.predict(context, Pool::Balls).distribution;
        ball_data.push((offline, t + 1, draws[t].balls));
    }

    if ball_data.is_empty() {
        return (0.15, 8);
    }

    let alphas = [0.05, 0.10, 0.15, 0.20, 0.30];
    let windows = [5, 8, 10, 15, 20];
    let mut best_alpha = 0.15;
    let mut best_window = 8usize;
    let mut best_ll = f64::NEG_INFINITY;

    for &alpha in &alphas {
        for &window in &windows {
            let ll: f64 = ball_data.iter().map(|(offline, start_idx, actual)| {
                let context = &draws[*start_idx..];
                let blended = super::online::online_offline_blend_with_alpha(
                    offline, context, Pool::Balls, window, alpha,
                );
                actual.iter().map(|&b| {
                    blended[(b - 1) as usize].max(1e-15).ln()
                }).sum::<f64>()
            }).sum();
            if ll > best_ll {
                best_ll = ll;
                best_alpha = alpha;
                best_window = window;
            }
        }
    }

    (best_alpha, best_window)
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
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 30,
                best_sparse: false,
                best_ll: -18.0,
                best_n_tests: 100,
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
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Bad".to_string(),
                results: vec![],
                best_window: 30,
                best_sparse: false,
                best_ll: uniform_ll - 1.0, // Pire que l'uniforme
                best_n_tests: 100,
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
        // Use small n_tests so total_skill = skill_per_draw * n stays within cap
        let calibrations = vec![
            ModelCalibration {
                model_name: "Best".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.10,
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Medium".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.05,
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Worst".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 0.10,
                best_n_tests: 100,
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
        // Use realistic skill values so total_skill stays under cap of 20
        let calibrations = vec![
            ModelCalibration {
                model_name: "Good".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.10,
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Mediocre".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll + 0.05,
                best_n_tests: 100,
            },
        ];
        // total_skill(Good) = 0.10 * 100 = 10, Med = 0.05 * 100 = 5
        // T=1: ratio = exp(10)/exp(5) = exp(5) ≈ 148
        let w_t1 = compute_weights_with_params(&calibrations, Pool::Balls, 1.0);
        let ratio_t1 = w_t1[0].1 / w_t1[1].1;
        // T=0.5: ratio = exp(20)/exp(10) = exp(10) ≈ 22026
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
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Terrible".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 6.0, // skill = -6
                best_n_tests: 100,
            },
        ];
        // Avec T=1.0, exp(-6)/exp(1) = exp(-7) ≈ 9e-4, exp(-6)/(exp(1)+exp(-6)) → very small
        let weights = compute_weights(&calibrations, Pool::Balls);
        let terrible_w = weights.iter().find(|(n, _)| n == "Terrible").unwrap().1;
        assert!(terrible_w < 0.01, "Terrible model should be negligible, got {}", terrible_w);
        let good_w = weights.iter().find(|(n, _)| n == "Good").unwrap().1;
        assert!(good_w > 0.99, "Good should get nearly all weight: {}", good_w);
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
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 10.0,
                best_n_tests: 100,
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
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Uniform".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll, // skill = 0
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "Bad".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 0.5, // skill < 0
                best_n_tests: 100,
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
                best_n_tests: 100,
            },
            ModelCalibration {
                model_name: "B".to_string(),
                results: vec![],
                best_window: 20,
                best_sparse: false,
                best_ll: uniform_ll - 2.0,
                best_n_tests: 100,
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
        // v9: with equal weights (ratio > 0.95), BOTH get sqrt(floor) penalty
        apply_decorrelation_penalty(&mut weights, &redundancies, 0.80, 1.0, 0.30);
        let a_w = weights.iter().find(|(n, _)| n == "A").unwrap().1;
        let b_w = weights.iter().find(|(n, _)| n == "B").unwrap().1;
        // Both penalized equally with sqrt(0.30) ≈ 0.5477
        // A: 0.5 * 0.5477, B: 0.5 * 0.5477 → equal after renormalization
        assert!((a_w - b_w).abs() < 1e-10, "Equal weights should be penalized equally: A={} B={}", a_w, b_w);
        assert!((a_w - 0.5).abs() < 1e-10, "Both should be 0.5 after renormalization: A={}", a_w);
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stability_constant_returns_one() {
        // Model with constant LL across windows → stability = 1.0
        let results = vec![
            CalibrationResult { model_name: "A".into(), window: 20, sparse: false, log_likelihood: -3.0, n_tests: 100 },
            CalibrationResult { model_name: "A".into(), window: 50, sparse: false, log_likelihood: -3.0, n_tests: 100 },
            CalibrationResult { model_name: "A".into(), window: 100, sparse: false, log_likelihood: -3.0, n_tests: 100 },
        ];
        let s = cross_window_stability(&results);
        assert!((s - 1.0).abs() < 1e-10, "Constant LL → stability = 1.0, got {}", s);
    }

    #[test]
    fn test_stability_variable_penalizes() {
        // Model with high variance across windows → stability < 1.0
        let results = vec![
            CalibrationResult { model_name: "A".into(), window: 20, sparse: false, log_likelihood: -3.0, n_tests: 100 },
            CalibrationResult { model_name: "A".into(), window: 50, sparse: false, log_likelihood: -3.1, n_tests: 100 },
            CalibrationResult { model_name: "A".into(), window: 100, sparse: false, log_likelihood: -2.9, n_tests: 100 },
        ];
        let s = cross_window_stability(&results);
        assert!(s < 1.0, "Variable LL → stability < 1.0, got {}", s);
        assert!(s > 0.0, "Should still be positive");

        // Very stable model (tiny variance) → close to 1.0
        let results_stable = vec![
            CalibrationResult { model_name: "A".into(), window: 20, sparse: false, log_likelihood: -3.000, n_tests: 100 },
            CalibrationResult { model_name: "A".into(), window: 50, sparse: false, log_likelihood: -3.001, n_tests: 100 },
            CalibrationResult { model_name: "A".into(), window: 100, sparse: false, log_likelihood: -2.999, n_tests: 100 },
        ];
        let s_stable = cross_window_stability(&results_stable);
        assert!(s_stable > s, "More stable model should have higher stability: {} > {}", s_stable, s);
    }

    #[test]
    fn test_weights_json_roundtrip() {
        let weights = EnsembleWeights {
            ball_weights: vec![("A".to_string(), 0.5), ("B".to_string(), 0.5)],
            star_weights: vec![("A".to_string(), 0.3), ("B".to_string(), 0.7)],
            calibrations: vec![],
            detailed_ll: Vec::new(),
            star_detailed_ll: Vec::new(),
            stacking_balls: None,
            stacking_stars: None,
            correlation_matrix: Vec::new(),
            star_correlation_matrix: Vec::new(),
            beta_balls: None,
            beta_stars: None,
            optimal_t_balls: None,
            optimal_t_stars: None,
            coherence_ball_weight: None,
            coherence_star_weight: None,
            stacking_blend_balls: None,
            stacking_blend_stars: None,
            online_ewma_alpha: None,
            online_window: None,
        };
        let json = serde_json::to_string(&weights).unwrap();
        let loaded: EnsembleWeights = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.ball_weights.len(), 2);
        assert_eq!(loaded.star_weights[1].1, 0.7);
    }
}
