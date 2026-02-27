use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use ndarray::{Array1, Array2};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, base_models};
use super::reduction::{ReductionStrategy, FeatureStandardizer, PcaTransform, apply_reduction};

fn default_version() -> u32 { 0 }

/// Poids sérialisables de l'Oracle.
#[derive(Serialize, Deserialize)]
pub struct OracleWeights {
    ball_rff_w: Vec<Vec<f64>>,   // [n_rff × input_dim_balls]
    ball_rff_b: Vec<f64>,         // [n_rff]
    ball_readout: Vec<Vec<f64>>,  // [50 × n_rff]
    star_rff_w: Vec<Vec<f64>>,   // [n_rff × input_dim_stars]
    star_rff_b: Vec<f64>,         // [n_rff]
    star_readout: Vec<Vec<f64>>,  // [12 × n_rff]
    n_rff_features: usize,
    bandwidth: f64,
    ridge_lambda: f64,
    n_base_models: usize,
    #[serde(default = "default_version")]
    pub version: u32,              // 2 = MetaOracle, 1 ou absent = legacy
    #[serde(default)]
    pub ball_input_dim: usize,     // 957 pour v2
    #[serde(default)]
    pub star_input_dim: usize,     // 235 pour v2

    // v3 fields — per-pool hyperparams
    #[serde(default)]
    pub ball_n_rff: usize,
    #[serde(default)]
    pub ball_bandwidth: f64,
    #[serde(default)]
    pub ball_ridge_lambda: f64,
    #[serde(default)]
    pub star_n_rff: usize,
    #[serde(default)]
    pub star_bandwidth: f64,
    #[serde(default)]
    pub star_ridge_lambda: f64,

    // v3 fields — reduction
    #[serde(default)]
    pub ball_reduction: ReductionStrategy,
    #[serde(default)]
    pub star_reduction: ReductionStrategy,
    #[serde(default)]
    pub ball_standardizer: Option<FeatureStandardizer>,
    #[serde(default)]
    pub star_standardizer: Option<FeatureStandardizer>,

    // v3 fields — mode
    #[serde(default)]
    pub hybrid: bool,
    #[serde(default)]
    pub ball_cv_ll: f64,
    #[serde(default)]
    pub star_cv_ll: f64,

    // v3 fields — ensemble members (multi-seed)
    #[serde(default)]
    pub ensemble_members: Vec<OracleWeights>,
}

pub struct OracleModel {
    weights: OracleWeights,
}

/// Configuration du grid search.
pub struct GridSearchConfig {
    pub n_rff_values: Vec<usize>,
    pub bandwidth_values: Vec<f64>,
    pub lambda_values: Vec<f64>,
}

impl GridSearchConfig {
    /// Grille par défaut (~125 combos).
    pub fn default_grid() -> Self {
        Self {
            n_rff_values: vec![100, 200, 400, 800, 1500],
            bandwidth_values: vec![0.1, 0.3, 1.0, 3.0, 10.0],
            lambda_values: vec![1e-1, 1e-2, 1e-3, 1e-4, 1e-6],
        }
    }

    /// Grille étendue (~648 combos).
    pub fn extended() -> Self {
        Self {
            n_rff_values: vec![50, 75, 100, 150, 200, 400, 800, 1500],
            bandwidth_values: vec![0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0, 30.0, 50.0],
            lambda_values: vec![1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0],
        }
    }

    /// Grille fine autour d'un point central (~125 combos).
    pub fn fine_around(n_rff: usize, bw: f64, lambda: f64) -> Self {
        let n_rff_values = vec![
            (n_rff as f64 * 0.5) as usize,
            (n_rff as f64 * 0.75) as usize,
            n_rff,
            (n_rff as f64 * 1.25) as usize,
            (n_rff as f64 * 1.5) as usize,
        ];
        let bandwidth_values = vec![
            bw * 0.5, bw * 0.7, bw, bw * 1.5, bw * 2.0,
        ];
        let lambda_values = vec![
            lambda * 0.3, lambda * 0.5, lambda, lambda * 2.0, lambda * 5.0,
        ];
        Self { n_rff_values, bandwidth_values, lambda_values }
    }

    pub fn total_combos(&self) -> usize {
        self.n_rff_values.len() * self.bandwidth_values.len() * self.lambda_values.len()
    }
}

/// Configuration de l'entraînement MetaOracle v3.
#[derive(Default)]
pub struct TrainConfig {
    pub grid_mode: GridMode,
    pub separate_pools: bool,
    pub pca_balls: Option<PcaMode>,
    pub pca_stars: Option<PcaMode>,
    pub standardize: bool,
    pub label_smoothing: f64,
    pub cv_mode: CvMode,
    pub temporal_decay: f64,
    pub ensemble_seeds: usize,
    pub n_folds: usize,
    pub seed: u64,
}

/// Mode de grille pour le grid search.
#[derive(Default, Clone, Debug)]
pub enum GridMode {
    #[default]
    Default,
    Extended,
    TwoStage,
}

/// Mode PCA.
#[derive(Clone, Debug)]
pub enum PcaMode {
    Auto,
    Fixed(usize),
}

/// Mode de cross-validation.
#[derive(Default, Clone, Debug)]
pub enum CvMode {
    #[default]
    Fold,
    WalkForward,
}

/// Résultat d'une combinaison d'hyperparamètres lors du grid search CV.
pub struct GridSearchEntry {
    pub n_rff: usize,
    pub bandwidth: f64,
    pub ridge_lambda: f64,
    pub mean_ball_ll: f64,
    pub mean_star_ll: f64,
    pub combined_ll: f64, // (5*ball + 2*star) / 7
}

/// Résultat complet du grid search CV (v1 — backward compat).
pub struct GridSearchResult {
    pub best: GridSearchEntry,
    pub entries: Vec<GridSearchEntry>,
}

/// Résultat complet du grid search CV v2 — avec best per-pool.
pub struct GridSearchResultV2 {
    pub best_combined: GridSearchEntry,
    pub best_balls: GridSearchEntry,
    pub best_stars: GridSearchEntry,
    pub entries: Vec<GridSearchEntry>,
}

/// Données collectées par collect_all_meta_features.
pub struct MetaFeatures {
    pub ball_features: Vec<Vec<f64>>,
    pub star_features: Vec<Vec<f64>>,
    pub ball_targets: Vec<Vec<f64>>,
    pub star_targets: Vec<Vec<f64>>,
    pub ball_input_dim: usize,
    pub star_input_dim: usize,
}

/// Transformation Random Fourier Features.
/// Φ(x) = sqrt(2/D) * cos(Wx + b)
struct RffTransform {
    w: Array2<f64>,
    b: Array1<f64>,
    scale: f64,
}

impl RffTransform {
    fn new(input_dim: usize, n_features: usize, bandwidth: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let sigma = 1.0 / bandwidth;

        let w = Array2::from_shape_fn((n_features, input_dim), |_| {
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            sigma * (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });

        let b = Array1::from_shape_fn(n_features, |_| {
            rng.random::<f64>() * 2.0 * std::f64::consts::PI
        });

        let scale = (2.0 / n_features as f64).sqrt();

        Self { w, b, scale }
    }

    fn transform(&self, x: &Array1<f64>) -> Array1<f64> {
        let z = self.w.dot(x) + &self.b;
        z.mapv(|v| self.scale * v.cos())
    }

    fn from_weights(w: &[Vec<f64>], b: &[f64]) -> Self {
        let n_features = w.len();
        let input_dim = if n_features > 0 { w[0].len() } else { 0 };

        let w_arr = Array2::from_shape_fn((n_features, input_dim), |(i, j)| w[i][j]);
        let b_arr = Array1::from_vec(b.to_vec());
        let scale = (2.0 / n_features as f64).sqrt();

        Self { w: w_arr, b: b_arr, scale }
    }
}

/// Collecte les prédictions des 18 modèles de base pour un ensemble de tirages.
/// Retourne (ball_features, star_features) — chacun un Vec<f64> de taille n_models*pool_size.
fn collect_base_predictions(draws: &[Draw]) -> (Vec<f64>, Vec<f64>) {
    let models = base_models();
    let mut ball_features = Vec::with_capacity(models.len() * 50);
    let mut star_features = Vec::with_capacity(models.len() * 12);

    for model in &models {
        let ball_pred = model.predict(draws, Pool::Balls);
        ball_features.extend_from_slice(&ball_pred);

        let star_pred = model.predict(draws, Pool::Stars);
        star_features.extend_from_slice(&star_pred);
    }

    (ball_features, star_features)
}

/// Encode la date en 7 features cycliques :
/// sin/cos(day_of_week/7), sin/cos(day_of_year/366), sin/cos(month/12), (year-2004)/30
fn encode_date(date_str: &str) -> Vec<f64> {
    let pi2 = 2.0 * std::f64::consts::PI;
    match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        Ok(d) => {
            let dow = d.weekday().num_days_from_monday() as f64; // 0=lundi
            let doy = d.ordinal() as f64;
            let month = d.month() as f64;
            let year = d.year() as f64;
            vec![
                (pi2 * dow / 7.0).sin(),
                (pi2 * dow / 7.0).cos(),
                (pi2 * doy / 366.0).sin(),
                (pi2 * doy / 366.0).cos(),
                (pi2 * month / 12.0).sin(),
                (pi2 * month / 12.0).cos(),
                (year - 2004.0) / 30.0,
            ]
        }
        Err(_) => vec![0.0; 7],
    }
}

/// Encode le tirage précédent en one-hot (50 pour boules, 12 pour étoiles).
/// Retourne un vecteur de zéros si `prev_draw` est None.
fn encode_feedback(prev_draw: Option<&Draw>, pool: Pool) -> Vec<f64> {
    let size = pool.size();
    let mut encoding = vec![0.0; size];
    if let Some(draw) = prev_draw {
        let numbers = pool.numbers_from(draw);
        for &n in numbers {
            encoding[(n - 1) as usize] = 1.0;
        }
    }
    encoding
}

/// Collecte les meta-features pour un tirage :
/// - 18 modèles de base × pool_size
/// - 7 features de date
/// - pool_size features de feedback (one-hot du tirage précédent)
fn collect_meta_features(
    context: &[Draw],
    target_date: &str,
    prev_draw: Option<&Draw>,
) -> (Vec<f64>, Vec<f64>) {
    let (ball_base, star_base) = collect_base_predictions(context);
    let date_feat = encode_date(target_date);
    let ball_feedback = encode_feedback(prev_draw, Pool::Balls);
    let star_feedback = encode_feedback(prev_draw, Pool::Stars);

    let mut ball_features = ball_base;
    ball_features.extend_from_slice(&date_feat);
    ball_features.extend_from_slice(&ball_feedback);

    let mut star_features = star_base;
    star_features.extend_from_slice(&date_feat);
    star_features.extend_from_slice(&star_feedback);

    (ball_features, star_features)
}

/// Calcule la prochaine date de tirage (MARDI ou VENDREDI) après `last_date`.
/// Retourne (date_str "YYYY-MM-DD", jour "MARDI"/"VENDREDI").
pub fn next_draw_date(last_date: &str) -> Result<(String, String)> {
    let d = NaiveDate::parse_from_str(last_date, "%Y-%m-%d")
        .context("Format de date invalide")?;

    // Avancer jour par jour jusqu'à trouver un mardi (1) ou vendredi (4)
    let mut next = d + chrono::Duration::days(1);
    loop {
        let wd = next.weekday().num_days_from_monday(); // 0=lun, 1=mar, 4=ven
        if wd == 1 {
            return Ok((next.format("%Y-%m-%d").to_string(), "MARDI".to_string()));
        }
        if wd == 4 {
            return Ok((next.format("%Y-%m-%d").to_string(), "VENDREDI".to_string()));
        }
        next += chrono::Duration::days(1);
    }
}

/// Collecte les meta-features et targets pour tous les tirages.
/// C'est la partie coûteuse (~40 min) car elle appelle les 18 modèles de base.
/// `label_smoothing`: 0.0 = one-hot pur, >0 = soft targets (ex: 0.1).
pub fn collect_all_meta_features(
    draws: &[Draw],
    label_smoothing: f64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<MetaFeatures> {
    let n_base = base_models().len();
    let ball_input_dim = n_base * 50 + 7 + 50;
    let star_input_dim = n_base * 12 + 7 + 12;

    let n_samples = draws.len();
    if n_samples < 2 {
        anyhow::bail!("Pas assez de tirages ({}) pour entraîner le MetaOracle", draws.len());
    }

    if let Some(pb) = progress {
        pb.set_length(n_samples as u64);
        pb.set_message("Collecte des meta-features...");
    }

    let alpha = label_smoothing.max(0.0);

    let mut ball_features = Vec::with_capacity(n_samples);
    let mut star_features = Vec::with_capacity(n_samples);
    let mut ball_targets = Vec::with_capacity(n_samples);
    let mut star_targets = Vec::with_capacity(n_samples);

    for t in 0..n_samples {
        let context = &draws[t + 1..];
        let target_date = &draws[t].date;
        let prev_draw = draws.get(t + 1);

        let (ball_feat, star_feat) = collect_meta_features(context, target_date, prev_draw);
        ball_features.push(ball_feat);
        star_features.push(star_feat);

        // Label smoothing for balls
        let ball_target = if alpha > 0.0 {
            let smooth_off = alpha / 50.0;
            let smooth_on = 1.0 - alpha + alpha / 50.0;
            let mut target = vec![smooth_off; 50];
            for &b in &draws[t].balls {
                target[(b - 1) as usize] = smooth_on;
            }
            target
        } else {
            let mut target = vec![0.0f64; 50];
            for &b in &draws[t].balls {
                target[(b - 1) as usize] = 1.0;
            }
            target
        };
        ball_targets.push(ball_target);

        // Label smoothing for stars
        let star_target = if alpha > 0.0 {
            let smooth_off = alpha / 12.0;
            let smooth_on = 1.0 - alpha + alpha / 12.0;
            let mut target = vec![smooth_off; 12];
            for &s in &draws[t].stars {
                target[(s - 1) as usize] = smooth_on;
            }
            target
        } else {
            let mut target = vec![0.0f64; 12];
            for &s in &draws[t].stars {
                target[(s - 1) as usize] = 1.0;
            }
            target
        };
        star_targets.push(star_target);

        if let Some(pb) = progress {
            pb.inc(1);
        }
    }

    if ball_features.len() < 2 {
        anyhow::bail!("Pas assez de samples collectés ({})", ball_features.len());
    }

    Ok(MetaFeatures {
        ball_features,
        star_features,
        ball_targets,
        star_targets,
        ball_input_dim,
        star_input_dim,
    })
}

/// Construit les RFF et entraîne la ridge regression.
/// C'est la partie rapide, O(n_rff² × N).
pub fn fit_rff_ridge(
    meta: &MetaFeatures,
    n_rff_features: usize,
    bandwidth: f64,
    ridge_lambda: f64,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<OracleWeights> {
    let n_base = base_models().len();
    let actual_samples = meta.ball_features.len();

    if let Some(pb) = progress {
        pb.set_message("Transformation RFF boules...");
    }

    let ball_rff = RffTransform::new(meta.ball_input_dim, n_rff_features, bandwidth, seed);
    let star_rff = RffTransform::new(meta.star_input_dim, n_rff_features, bandwidth, seed + 1);

    // Matrice H boules [n_rff × n_samples]
    let mut h_balls = Array2::<f64>::zeros((n_rff_features, actual_samples));
    for (s, feat) in meta.ball_features.iter().enumerate() {
        let input = Array1::from_vec(feat.clone());
        let phi = ball_rff.transform(&input);
        for (f, &val) in phi.iter().enumerate() {
            h_balls[[f, s]] = val;
        }
    }

    // Targets boules [50 × n_samples]
    let mut y_balls = Array2::<f64>::zeros((50, actual_samples));
    for (s, target) in meta.ball_targets.iter().enumerate() {
        for (i, &v) in target.iter().enumerate() {
            y_balls[[i, s]] = v;
        }
    }

    if let Some(pb) = progress {
        pb.set_message("Ridge regression boules...");
    }

    let w_balls = lemillion_esn::linalg::ridge_regression(&h_balls, &y_balls, ridge_lambda)
        .context("Ridge regression boules échouée")?;

    if let Some(pb) = progress {
        pb.set_message("Transformation RFF étoiles...");
    }

    // Matrice H étoiles [n_rff × n_samples]
    let mut h_stars = Array2::<f64>::zeros((n_rff_features, actual_samples));
    for (s, feat) in meta.star_features.iter().enumerate() {
        let input = Array1::from_vec(feat.clone());
        let phi = star_rff.transform(&input);
        for (f, &val) in phi.iter().enumerate() {
            h_stars[[f, s]] = val;
        }
    }

    // Targets étoiles [12 × n_samples]
    let mut y_stars = Array2::<f64>::zeros((12, actual_samples));
    for (s, target) in meta.star_targets.iter().enumerate() {
        for (i, &v) in target.iter().enumerate() {
            y_stars[[i, s]] = v;
        }
    }

    if let Some(pb) = progress {
        pb.set_message("Ridge regression étoiles...");
    }

    let w_stars = lemillion_esn::linalg::ridge_regression(&h_stars, &y_stars, ridge_lambda)
        .context("Ridge regression étoiles échouée")?;

    if let Some(pb) = progress {
        pb.set_message("Sérialisation...");
    }

    // Convertir en vecteurs pour la sérialisation
    let ball_rff_w_vec: Vec<Vec<f64>> = (0..n_rff_features)
        .map(|i| ball_rff.w.row(i).to_vec())
        .collect();
    let ball_rff_b_vec: Vec<f64> = ball_rff.b.to_vec();
    let ball_readout_vec: Vec<Vec<f64>> = (0..50)
        .map(|i| w_balls.row(i).to_vec())
        .collect();

    let star_rff_w_vec: Vec<Vec<f64>> = (0..n_rff_features)
        .map(|i| star_rff.w.row(i).to_vec())
        .collect();
    let star_rff_b_vec: Vec<f64> = star_rff.b.to_vec();
    let star_readout_vec: Vec<Vec<f64>> = (0..12)
        .map(|i| w_stars.row(i).to_vec())
        .collect();

    Ok(OracleWeights {
        ball_rff_w: ball_rff_w_vec,
        ball_rff_b: ball_rff_b_vec,
        ball_readout: ball_readout_vec,
        star_rff_w: star_rff_w_vec,
        star_rff_b: star_rff_b_vec,
        star_readout: star_readout_vec,
        n_rff_features,
        bandwidth,
        ridge_lambda,
        n_base_models: n_base,
        version: 2,
        ball_input_dim: meta.ball_input_dim,
        star_input_dim: meta.star_input_dim,
        ball_n_rff: 0,
        ball_bandwidth: 0.0,
        ball_ridge_lambda: 0.0,
        star_n_rff: 0,
        star_bandwidth: 0.0,
        star_ridge_lambda: 0.0,
        ball_reduction: ReductionStrategy::None,
        star_reduction: ReductionStrategy::None,
        ball_standardizer: None,
        star_standardizer: None,
        hybrid: false,
        ball_cv_ll: 0.0,
        star_cv_ll: 0.0,
        ensemble_members: Vec::new(),
    })
}

/// Construit les RFF et entraîne la ridge regression, avec hyperparamètres séparés par pool,
/// pondération temporelle, et support pour standardization/PCA.
pub fn fit_rff_ridge_v2(
    meta: &MetaFeatures,
    ball_params: (usize, f64, f64),  // (n_rff, bw, lambda)
    star_params: (usize, f64, f64),
    temporal_decay: f64,
    seed: u64,
    ball_standardizer: Option<&FeatureStandardizer>,
    star_standardizer: Option<&FeatureStandardizer>,
    ball_reduction: &ReductionStrategy,
    star_reduction: &ReductionStrategy,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<OracleWeights> {
    let n_base = base_models().len();
    let actual_samples = meta.ball_features.len();

    // Apply standardization + reduction to features
    let ball_features_processed: Vec<Vec<f64>> = meta.ball_features.iter().map(|f| {
        let mut x = f.clone();
        if let Some(std) = ball_standardizer { x = std.transform(&x); }
        apply_reduction(&x, ball_reduction)
    }).collect();

    let star_features_processed: Vec<Vec<f64>> = meta.star_features.iter().map(|f| {
        let mut x = f.clone();
        if let Some(std) = star_standardizer { x = std.transform(&x); }
        apply_reduction(&x, star_reduction)
    }).collect();

    let ball_dim = ball_features_processed[0].len();
    let star_dim = star_features_processed[0].len();

    let (ball_n_rff, ball_bw, ball_lambda) = ball_params;
    let (star_n_rff, star_bw, star_lambda) = star_params;

    if let Some(pb) = progress {
        pb.set_message("Transformation RFF boules...");
    }

    let ball_rff = RffTransform::new(ball_dim, ball_n_rff, ball_bw, seed);
    let star_rff = RffTransform::new(star_dim, star_n_rff, star_bw, seed + 1);

    // Temporal weights
    let use_temporal = temporal_decay < 1.0 && temporal_decay > 0.0;
    let weights: Vec<f64> = if use_temporal {
        (0..actual_samples).map(|t| temporal_decay.powi(t as i32)).collect()
    } else {
        vec![1.0; actual_samples]
    };

    // --- Balls ---
    let mut h_balls = Array2::<f64>::zeros((ball_n_rff, actual_samples));
    for (s, feat) in ball_features_processed.iter().enumerate() {
        let input = Array1::from_vec(feat.clone());
        let phi = ball_rff.transform(&input);
        let w_sqrt = weights[s].sqrt();
        for (f, &val) in phi.iter().enumerate() {
            h_balls[[f, s]] = val * w_sqrt;
        }
    }

    let mut y_balls = Array2::<f64>::zeros((50, actual_samples));
    for (s, target) in meta.ball_targets.iter().enumerate() {
        let w_sqrt = weights[s].sqrt();
        for (i, &v) in target.iter().enumerate() {
            y_balls[[i, s]] = v * w_sqrt;
        }
    }

    if let Some(pb) = progress {
        pb.set_message("Ridge regression boules...");
    }

    let w_balls = lemillion_esn::linalg::ridge_regression(&h_balls, &y_balls, ball_lambda)
        .context("Ridge regression boules échouée")?;

    // --- Stars ---
    if let Some(pb) = progress {
        pb.set_message("Transformation RFF étoiles...");
    }

    let mut h_stars = Array2::<f64>::zeros((star_n_rff, actual_samples));
    for (s, feat) in star_features_processed.iter().enumerate() {
        let input = Array1::from_vec(feat.clone());
        let phi = star_rff.transform(&input);
        let w_sqrt = weights[s].sqrt();
        for (f, &val) in phi.iter().enumerate() {
            h_stars[[f, s]] = val * w_sqrt;
        }
    }

    let mut y_stars = Array2::<f64>::zeros((12, actual_samples));
    for (s, target) in meta.star_targets.iter().enumerate() {
        let w_sqrt = weights[s].sqrt();
        for (i, &v) in target.iter().enumerate() {
            y_stars[[i, s]] = v * w_sqrt;
        }
    }

    if let Some(pb) = progress {
        pb.set_message("Ridge regression étoiles...");
    }

    let w_stars = lemillion_esn::linalg::ridge_regression(&h_stars, &y_stars, star_lambda)
        .context("Ridge regression étoiles échouée")?;

    if let Some(pb) = progress {
        pb.set_message("Sérialisation...");
    }

    // Serialize
    let ball_rff_w_vec: Vec<Vec<f64>> = (0..ball_n_rff).map(|i| ball_rff.w.row(i).to_vec()).collect();
    let ball_rff_b_vec: Vec<f64> = ball_rff.b.to_vec();
    let ball_readout_vec: Vec<Vec<f64>> = (0..50).map(|i| w_balls.row(i).to_vec()).collect();

    let star_rff_w_vec: Vec<Vec<f64>> = (0..star_n_rff).map(|i| star_rff.w.row(i).to_vec()).collect();
    let star_rff_b_vec: Vec<f64> = star_rff.b.to_vec();
    let star_readout_vec: Vec<Vec<f64>> = (0..12).map(|i| w_stars.row(i).to_vec()).collect();

    Ok(OracleWeights {
        ball_rff_w: ball_rff_w_vec,
        ball_rff_b: ball_rff_b_vec,
        ball_readout: ball_readout_vec,
        star_rff_w: star_rff_w_vec,
        star_rff_b: star_rff_b_vec,
        star_readout: star_readout_vec,
        n_rff_features: ball_n_rff.max(star_n_rff),
        bandwidth: ball_bw,
        ridge_lambda: ball_lambda,
        n_base_models: n_base,
        version: 3,
        ball_input_dim: ball_dim,
        star_input_dim: star_dim,
        ball_n_rff,
        ball_bandwidth: ball_bw,
        ball_ridge_lambda: ball_lambda,
        star_n_rff,
        star_bandwidth: star_bw,
        star_ridge_lambda: star_lambda,
        ball_reduction: ball_reduction.clone(),
        star_reduction: star_reduction.clone(),
        ball_standardizer: ball_standardizer.cloned(),
        star_standardizer: star_standardizer.cloned(),
        hybrid: false,
        ball_cv_ll: 0.0,
        star_cv_ll: 0.0,
        ensemble_members: Vec::new(),
    })
}

/// Entraîne le MetaOracle (v2) par walk-forward sur tous les tirages.
/// Appelle collect_all_meta_features puis fit_rff_ridge.
pub fn train_oracle(
    draws: &[Draw],
    n_rff_features: usize,
    bandwidth: f64,
    ridge_lambda: f64,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<OracleWeights> {
    let meta = collect_all_meta_features(draws, 0.0, progress)?;
    fit_rff_ridge(&meta, n_rff_features, bandwidth, ridge_lambda, seed, progress)
}

/// Calcule la log-likelihood de validation pour un fold.
/// w_out: [pool_size × n_rff], h_val: [n_rff × n_val], targets: [n_val][pool_size]
fn compute_validation_ll(
    w_out: &Array2<f64>,
    h_val: &Array2<f64>,
    targets: &[Vec<f64>],
    pool_size: usize,
) -> f64 {
    let n_val = h_val.ncols();
    let mut total_ll = 0.0;

    for s in 0..n_val {
        let h_col = h_val.column(s);
        let prediction = w_out.dot(&h_col);

        // Normaliser en probabilités
        let mut probs: Vec<f64> = prediction.iter().map(|&x| x.max(1e-15)).collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            probs = vec![1.0 / pool_size as f64; pool_size];
        }

        // LL pour les numéros tirés (one-hot > 0.5)
        for i in 0..pool_size {
            if targets[s][i] > 0.5 {
                total_ll += probs[i].max(1e-30).ln();
            }
        }
    }

    total_ll / n_val as f64
}

/// Grid search CV pour trouver les meilleurs hyperparamètres RFF.
/// Collecte des meta-features déjà faite, itère uniquement sur les hyperparamètres.
pub fn cv_grid_search(
    ball_features: &[Vec<f64>],
    star_features: &[Vec<f64>],
    ball_targets: &[Vec<f64>],
    star_targets: &[Vec<f64>],
    ball_input_dim: usize,
    star_input_dim: usize,
    n_folds: usize,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<GridSearchResult> {
    let n = ball_features.len();
    let fold_size = n / n_folds;

    // Grille d'hyperparamètres
    let n_rff_values = [100, 200, 400, 800, 1500];
    let bandwidth_values = [0.1, 0.3, 1.0, 3.0, 10.0];
    let lambda_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6];

    let total_combos = n_rff_values.len() * bandwidth_values.len() * lambda_values.len();

    if let Some(pb) = progress {
        pb.set_length((total_combos * n_folds) as u64);
        pb.set_position(0);
        pb.set_message("Grid search CV...");
    }

    let mut entries = Vec::with_capacity(total_combos);

    for &n_rff in &n_rff_values {
        for &bw in &bandwidth_values {
            for &lambda in &lambda_values {
                let mut fold_ball_lls = Vec::with_capacity(n_folds);
                let mut fold_star_lls = Vec::with_capacity(n_folds);

                for fold in 0..n_folds {
                    // Blocs contigus chronologiques
                    let val_start = fold * fold_size;
                    let val_end = if fold == n_folds - 1 { n } else { (fold + 1) * fold_size };

                    // Indices d'entraînement (tout sauf le fold de validation)
                    let train_indices: Vec<usize> = (0..val_start).chain(val_end..n).collect();
                    let val_indices: Vec<usize> = (val_start..val_end).collect();
                    let n_train = train_indices.len();
                    let n_val = val_indices.len();

                    if n_train == 0 || n_val == 0 {
                        continue;
                    }

                    // Créer les RFF
                    let ball_rff = RffTransform::new(ball_input_dim, n_rff, bw, seed);
                    let star_rff = RffTransform::new(star_input_dim, n_rff, bw, seed + 1);

                    // H_train boules [n_rff × n_train]
                    let mut h_train_balls = Array2::<f64>::zeros((n_rff, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        let input = Array1::from_vec(ball_features[idx].clone());
                        let phi = ball_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() {
                            h_train_balls[[f, col]] = val;
                        }
                    }

                    // Y_train boules [50 × n_train]
                    let mut y_train_balls = Array2::<f64>::zeros((50, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        for (i, &v) in ball_targets[idx].iter().enumerate() {
                            y_train_balls[[i, col]] = v;
                        }
                    }

                    // Ridge boules
                    let w_balls = match lemillion_esn::linalg::ridge_regression(&h_train_balls, &y_train_balls, lambda) {
                        Ok(w) => w,
                        Err(_) => {
                            if let Some(pb) = progress { pb.inc(1); }
                            continue;
                        }
                    };

                    // H_val boules [n_rff × n_val]
                    let mut h_val_balls = Array2::<f64>::zeros((n_rff, n_val));
                    for (col, &idx) in val_indices.iter().enumerate() {
                        let input = Array1::from_vec(ball_features[idx].clone());
                        let phi = ball_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() {
                            h_val_balls[[f, col]] = val;
                        }
                    }

                    let val_ball_targets: Vec<Vec<f64>> = val_indices.iter().map(|&i| ball_targets[i].clone()).collect();
                    let ball_ll = compute_validation_ll(&w_balls, &h_val_balls, &val_ball_targets, 50);
                    fold_ball_lls.push(ball_ll);

                    // Idem étoiles
                    let mut h_train_stars = Array2::<f64>::zeros((n_rff, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        let input = Array1::from_vec(star_features[idx].clone());
                        let phi = star_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() {
                            h_train_stars[[f, col]] = val;
                        }
                    }

                    let mut y_train_stars = Array2::<f64>::zeros((12, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        for (i, &v) in star_targets[idx].iter().enumerate() {
                            y_train_stars[[i, col]] = v;
                        }
                    }

                    let w_stars = match lemillion_esn::linalg::ridge_regression(&h_train_stars, &y_train_stars, lambda) {
                        Ok(w) => w,
                        Err(_) => {
                            if let Some(pb) = progress { pb.inc(1); }
                            continue;
                        }
                    };

                    let mut h_val_stars = Array2::<f64>::zeros((n_rff, n_val));
                    for (col, &idx) in val_indices.iter().enumerate() {
                        let input = Array1::from_vec(star_features[idx].clone());
                        let phi = star_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() {
                            h_val_stars[[f, col]] = val;
                        }
                    }

                    let val_star_targets: Vec<Vec<f64>> = val_indices.iter().map(|&i| star_targets[i].clone()).collect();
                    let star_ll = compute_validation_ll(&w_stars, &h_val_stars, &val_star_targets, 12);
                    fold_star_lls.push(star_ll);

                    if let Some(pb) = progress {
                        pb.inc(1);
                    }
                }

                if !fold_ball_lls.is_empty() && !fold_star_lls.is_empty() {
                    let mean_ball_ll = fold_ball_lls.iter().sum::<f64>() / fold_ball_lls.len() as f64;
                    let mean_star_ll = fold_star_lls.iter().sum::<f64>() / fold_star_lls.len() as f64;
                    let combined_ll = (5.0 * mean_ball_ll + 2.0 * mean_star_ll) / 7.0;

                    entries.push(GridSearchEntry {
                        n_rff,
                        bandwidth: bw,
                        ridge_lambda: lambda,
                        mean_ball_ll,
                        mean_star_ll,
                        combined_ll,
                    });
                }
            }
        }
    }

    if entries.is_empty() {
        anyhow::bail!("Aucune combinaison n'a convergé dans le grid search");
    }

    // Trier par combined_ll décroissant
    entries.sort_by(|a, b| b.combined_ll.partial_cmp(&a.combined_ll).unwrap_or(std::cmp::Ordering::Equal));

    let best = GridSearchEntry {
        n_rff: entries[0].n_rff,
        bandwidth: entries[0].bandwidth,
        ridge_lambda: entries[0].ridge_lambda,
        mean_ball_ll: entries[0].mean_ball_ll,
        mean_star_ll: entries[0].mean_star_ll,
        combined_ll: entries[0].combined_ll,
    };

    Ok(GridSearchResult { best, entries })
}

/// Grid search CV v2 : supporte GridSearchConfig personnalisée et traque best per-pool.
/// Les features doivent déjà être standardisées/réduites si demandé.
pub fn cv_grid_search_v2(
    ball_features: &[Vec<f64>],
    star_features: &[Vec<f64>],
    ball_targets: &[Vec<f64>],
    star_targets: &[Vec<f64>],
    ball_input_dim: usize,
    star_input_dim: usize,
    config: &GridSearchConfig,
    n_folds: usize,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<GridSearchResultV2> {
    let n = ball_features.len();
    let fold_size = n / n_folds;

    let total_combos = config.total_combos();

    if let Some(pb) = progress {
        pb.set_length((total_combos * n_folds) as u64);
        pb.set_position(0);
        pb.set_message("Grid search CV v2...");
    }

    let mut entries = Vec::with_capacity(total_combos);

    for &n_rff in &config.n_rff_values {
        for &bw in &config.bandwidth_values {
            for &lambda in &config.lambda_values {
                let mut fold_ball_lls = Vec::with_capacity(n_folds);
                let mut fold_star_lls = Vec::with_capacity(n_folds);

                for fold in 0..n_folds {
                    let val_start = fold * fold_size;
                    let val_end = if fold == n_folds - 1 { n } else { (fold + 1) * fold_size };

                    let train_indices: Vec<usize> = (0..val_start).chain(val_end..n).collect();
                    let val_indices: Vec<usize> = (val_start..val_end).collect();
                    let n_train = train_indices.len();
                    let n_val = val_indices.len();

                    if n_train == 0 || n_val == 0 { continue; }

                    let ball_rff = RffTransform::new(ball_input_dim, n_rff, bw, seed);
                    let star_rff = RffTransform::new(star_input_dim, n_rff, bw, seed + 1);

                    // Balls
                    let mut h_train_balls = Array2::<f64>::zeros((n_rff, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        let input = Array1::from_vec(ball_features[idx].clone());
                        let phi = ball_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() { h_train_balls[[f, col]] = val; }
                    }

                    let mut y_train_balls = Array2::<f64>::zeros((50, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        for (i, &v) in ball_targets[idx].iter().enumerate() { y_train_balls[[i, col]] = v; }
                    }

                    let w_balls = match lemillion_esn::linalg::ridge_regression(&h_train_balls, &y_train_balls, lambda) {
                        Ok(w) => w,
                        Err(_) => { if let Some(pb) = progress { pb.inc(1); } continue; }
                    };

                    let mut h_val_balls = Array2::<f64>::zeros((n_rff, n_val));
                    for (col, &idx) in val_indices.iter().enumerate() {
                        let input = Array1::from_vec(ball_features[idx].clone());
                        let phi = ball_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() { h_val_balls[[f, col]] = val; }
                    }

                    let val_ball_targets: Vec<Vec<f64>> = val_indices.iter().map(|&i| ball_targets[i].clone()).collect();
                    let ball_ll = compute_validation_ll(&w_balls, &h_val_balls, &val_ball_targets, 50);
                    fold_ball_lls.push(ball_ll);

                    // Stars
                    let mut h_train_stars = Array2::<f64>::zeros((n_rff, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        let input = Array1::from_vec(star_features[idx].clone());
                        let phi = star_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() { h_train_stars[[f, col]] = val; }
                    }

                    let mut y_train_stars = Array2::<f64>::zeros((12, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        for (i, &v) in star_targets[idx].iter().enumerate() { y_train_stars[[i, col]] = v; }
                    }

                    let w_stars = match lemillion_esn::linalg::ridge_regression(&h_train_stars, &y_train_stars, lambda) {
                        Ok(w) => w,
                        Err(_) => { if let Some(pb) = progress { pb.inc(1); } continue; }
                    };

                    let mut h_val_stars = Array2::<f64>::zeros((n_rff, n_val));
                    for (col, &idx) in val_indices.iter().enumerate() {
                        let input = Array1::from_vec(star_features[idx].clone());
                        let phi = star_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() { h_val_stars[[f, col]] = val; }
                    }

                    let val_star_targets: Vec<Vec<f64>> = val_indices.iter().map(|&i| star_targets[i].clone()).collect();
                    let star_ll = compute_validation_ll(&w_stars, &h_val_stars, &val_star_targets, 12);
                    fold_star_lls.push(star_ll);

                    if let Some(pb) = progress { pb.inc(1); }
                }

                if !fold_ball_lls.is_empty() && !fold_star_lls.is_empty() {
                    let mean_ball_ll = fold_ball_lls.iter().sum::<f64>() / fold_ball_lls.len() as f64;
                    let mean_star_ll = fold_star_lls.iter().sum::<f64>() / fold_star_lls.len() as f64;
                    let combined_ll = (5.0 * mean_ball_ll + 2.0 * mean_star_ll) / 7.0;

                    entries.push(GridSearchEntry {
                        n_rff, bandwidth: bw, ridge_lambda: lambda,
                        mean_ball_ll, mean_star_ll, combined_ll,
                    });
                }
            }
        }
    }

    if entries.is_empty() {
        anyhow::bail!("Aucune combinaison n'a convergé dans le grid search v2");
    }

    // Sort by combined_ll descending
    entries.sort_by(|a, b| b.combined_ll.partial_cmp(&a.combined_ll).unwrap_or(std::cmp::Ordering::Equal));

    let best_combined = GridSearchEntry {
        n_rff: entries[0].n_rff, bandwidth: entries[0].bandwidth, ridge_lambda: entries[0].ridge_lambda,
        mean_ball_ll: entries[0].mean_ball_ll, mean_star_ll: entries[0].mean_star_ll, combined_ll: entries[0].combined_ll,
    };

    // Best balls
    let best_ball_entry = entries.iter().max_by(|a, b| a.mean_ball_ll.partial_cmp(&b.mean_ball_ll).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let best_balls = GridSearchEntry {
        n_rff: best_ball_entry.n_rff, bandwidth: best_ball_entry.bandwidth, ridge_lambda: best_ball_entry.ridge_lambda,
        mean_ball_ll: best_ball_entry.mean_ball_ll, mean_star_ll: best_ball_entry.mean_star_ll, combined_ll: best_ball_entry.combined_ll,
    };

    // Best stars
    let best_star_entry = entries.iter().max_by(|a, b| a.mean_star_ll.partial_cmp(&b.mean_star_ll).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let best_stars = GridSearchEntry {
        n_rff: best_star_entry.n_rff, bandwidth: best_star_entry.bandwidth, ridge_lambda: best_star_entry.ridge_lambda,
        mean_ball_ll: best_star_entry.mean_ball_ll, mean_star_ll: best_star_entry.mean_star_ll, combined_ll: best_star_entry.combined_ll,
    };

    Ok(GridSearchResultV2 { best_combined, best_balls, best_stars, entries })
}

/// Walk-forward CV temporel : expanding window, pas de fuite temporelle.
/// Les données sont ordonnées index 0 = plus récent.
pub fn cv_walk_forward(
    ball_features: &[Vec<f64>],
    star_features: &[Vec<f64>],
    ball_targets: &[Vec<f64>],
    star_targets: &[Vec<f64>],
    ball_input_dim: usize,
    star_input_dim: usize,
    config: &GridSearchConfig,
    min_train_size: usize,
    n_eval_points: usize,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<GridSearchResultV2> {
    let n = ball_features.len();
    if n < min_train_size + 1 {
        anyhow::bail!("Pas assez de données ({}) pour walk-forward CV (min_train={})", n, min_train_size);
    }

    // Eval points distributed between min_train_size and N
    let n_eval = n_eval_points.min(n - min_train_size);
    let step = if n_eval > 1 { (n - min_train_size) / n_eval } else { 1 };
    let eval_points: Vec<usize> = (0..n_eval).map(|i| n - min_train_size - 1 - i * step).filter(|&t| t < n).collect();

    let total_combos = config.total_combos();

    if let Some(pb) = progress {
        pb.set_length((total_combos * eval_points.len()) as u64);
        pb.set_position(0);
        pb.set_message("Walk-forward CV...");
    }

    let mut entries = Vec::with_capacity(total_combos);

    for &n_rff in &config.n_rff_values {
        for &bw in &config.bandwidth_values {
            for &lambda in &config.lambda_values {
                let mut eval_ball_lls = Vec::with_capacity(eval_points.len());
                let mut eval_star_lls = Vec::with_capacity(eval_points.len());

                for &t in &eval_points {
                    // Train on samples[t+1..n] (past), test on sample[t]
                    let train_indices: Vec<usize> = ((t + 1)..n).collect();
                    let n_train = train_indices.len();
                    if n_train < min_train_size { continue; }

                    let ball_rff = RffTransform::new(ball_input_dim, n_rff, bw, seed);
                    let star_rff = RffTransform::new(star_input_dim, n_rff, bw, seed + 1);

                    // Balls
                    let mut h_train = Array2::<f64>::zeros((n_rff, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        let input = Array1::from_vec(ball_features[idx].clone());
                        let phi = ball_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() { h_train[[f, col]] = val; }
                    }

                    let mut y_train = Array2::<f64>::zeros((50, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        for (i, &v) in ball_targets[idx].iter().enumerate() { y_train[[i, col]] = v; }
                    }

                    let w_balls = match lemillion_esn::linalg::ridge_regression(&h_train, &y_train, lambda) {
                        Ok(w) => w,
                        Err(_) => { if let Some(pb) = progress { pb.inc(1); } continue; }
                    };

                    // Eval on single test point
                    let test_input = Array1::from_vec(ball_features[t].clone());
                    let test_phi = ball_rff.transform(&test_input);
                    let h_test = test_phi.insert_axis(ndarray::Axis(1));
                    let ball_ll = compute_validation_ll(&w_balls, &h_test, &[ball_targets[t].clone()], 50);
                    eval_ball_lls.push(ball_ll);

                    // Stars
                    let mut h_train_s = Array2::<f64>::zeros((n_rff, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        let input = Array1::from_vec(star_features[idx].clone());
                        let phi = star_rff.transform(&input);
                        for (f, &val) in phi.iter().enumerate() { h_train_s[[f, col]] = val; }
                    }

                    let mut y_train_s = Array2::<f64>::zeros((12, n_train));
                    for (col, &idx) in train_indices.iter().enumerate() {
                        for (i, &v) in star_targets[idx].iter().enumerate() { y_train_s[[i, col]] = v; }
                    }

                    let w_stars = match lemillion_esn::linalg::ridge_regression(&h_train_s, &y_train_s, lambda) {
                        Ok(w) => w,
                        Err(_) => { if let Some(pb) = progress { pb.inc(1); } continue; }
                    };

                    let test_input_s = Array1::from_vec(star_features[t].clone());
                    let test_phi_s = star_rff.transform(&test_input_s);
                    let h_test_s = test_phi_s.insert_axis(ndarray::Axis(1));
                    let star_ll = compute_validation_ll(&w_stars, &h_test_s, &[star_targets[t].clone()], 12);
                    eval_star_lls.push(star_ll);

                    if let Some(pb) = progress { pb.inc(1); }
                }

                if !eval_ball_lls.is_empty() && !eval_star_lls.is_empty() {
                    let mean_ball_ll = eval_ball_lls.iter().sum::<f64>() / eval_ball_lls.len() as f64;
                    let mean_star_ll = eval_star_lls.iter().sum::<f64>() / eval_star_lls.len() as f64;
                    let combined_ll = (5.0 * mean_ball_ll + 2.0 * mean_star_ll) / 7.0;

                    entries.push(GridSearchEntry {
                        n_rff, bandwidth: bw, ridge_lambda: lambda,
                        mean_ball_ll, mean_star_ll, combined_ll,
                    });
                }
            }
        }
    }

    if entries.is_empty() {
        anyhow::bail!("Aucune combinaison n'a convergé dans le walk-forward CV");
    }

    entries.sort_by(|a, b| b.combined_ll.partial_cmp(&a.combined_ll).unwrap_or(std::cmp::Ordering::Equal));

    let best_combined = GridSearchEntry {
        n_rff: entries[0].n_rff, bandwidth: entries[0].bandwidth, ridge_lambda: entries[0].ridge_lambda,
        mean_ball_ll: entries[0].mean_ball_ll, mean_star_ll: entries[0].mean_star_ll, combined_ll: entries[0].combined_ll,
    };

    let best_ball_entry = entries.iter().max_by(|a, b| a.mean_ball_ll.partial_cmp(&b.mean_ball_ll).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let best_balls = GridSearchEntry {
        n_rff: best_ball_entry.n_rff, bandwidth: best_ball_entry.bandwidth, ridge_lambda: best_ball_entry.ridge_lambda,
        mean_ball_ll: best_ball_entry.mean_ball_ll, mean_star_ll: best_ball_entry.mean_star_ll, combined_ll: best_ball_entry.combined_ll,
    };

    let best_star_entry = entries.iter().max_by(|a, b| a.mean_star_ll.partial_cmp(&b.mean_star_ll).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let best_stars = GridSearchEntry {
        n_rff: best_star_entry.n_rff, bandwidth: best_star_entry.bandwidth, ridge_lambda: best_star_entry.ridge_lambda,
        mean_ball_ll: best_star_entry.mean_ball_ll, mean_star_ll: best_star_entry.mean_star_ll, combined_ll: best_star_entry.combined_ll,
    };

    Ok(GridSearchResultV2 { best_combined, best_balls, best_stars, entries })
}

/// Grid search 2-stage : grille étendue → grille fine autour du best.
pub fn two_stage_grid_search(
    ball_features: &[Vec<f64>],
    star_features: &[Vec<f64>],
    ball_targets: &[Vec<f64>],
    star_targets: &[Vec<f64>],
    ball_input_dim: usize,
    star_input_dim: usize,
    n_folds: usize,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<GridSearchResultV2> {
    // Stage 1: extended grid
    if let Some(pb) = progress {
        pb.set_message("Stage 1 : grille étendue...");
    }
    let stage1 = cv_grid_search_v2(
        ball_features, star_features, ball_targets, star_targets,
        ball_input_dim, star_input_dim,
        &GridSearchConfig::extended(), n_folds, seed, progress,
    )?;

    // Stage 2: fine grid around best combined
    if let Some(pb) = progress {
        pb.set_message("Stage 2 : grille fine...");
    }
    let fine_config = GridSearchConfig::fine_around(
        stage1.best_combined.n_rff,
        stage1.best_combined.bandwidth,
        stage1.best_combined.ridge_lambda,
    );
    let stage2 = cv_grid_search_v2(
        ball_features, star_features, ball_targets, star_targets,
        ball_input_dim, star_input_dim,
        &fine_config, n_folds, seed, progress,
    )?;

    // Merge entries and pick bests
    let mut all_entries = stage1.entries;
    all_entries.extend(stage2.entries);
    all_entries.sort_by(|a, b| b.combined_ll.partial_cmp(&a.combined_ll).unwrap_or(std::cmp::Ordering::Equal));

    let best_combined = GridSearchEntry {
        n_rff: all_entries[0].n_rff, bandwidth: all_entries[0].bandwidth, ridge_lambda: all_entries[0].ridge_lambda,
        mean_ball_ll: all_entries[0].mean_ball_ll, mean_star_ll: all_entries[0].mean_star_ll, combined_ll: all_entries[0].combined_ll,
    };

    let best_ball_entry = all_entries.iter().max_by(|a, b| a.mean_ball_ll.partial_cmp(&b.mean_ball_ll).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let best_balls = GridSearchEntry {
        n_rff: best_ball_entry.n_rff, bandwidth: best_ball_entry.bandwidth, ridge_lambda: best_ball_entry.ridge_lambda,
        mean_ball_ll: best_ball_entry.mean_ball_ll, mean_star_ll: best_ball_entry.mean_star_ll, combined_ll: best_ball_entry.combined_ll,
    };

    let best_star_entry = all_entries.iter().max_by(|a, b| a.mean_star_ll.partial_cmp(&b.mean_star_ll).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let best_stars = GridSearchEntry {
        n_rff: best_star_entry.n_rff, bandwidth: best_star_entry.bandwidth, ridge_lambda: best_star_entry.ridge_lambda,
        mean_ball_ll: best_star_entry.mean_ball_ll, mean_star_ll: best_star_entry.mean_star_ll, combined_ll: best_star_entry.combined_ll,
    };

    Ok(GridSearchResultV2 { best_combined, best_balls, best_stars, entries: all_entries })
}

/// Entraîne le MetaOracle avec grid search CV pour trouver les meilleurs hyperparamètres.
/// 1. Collecte des meta-features (une seule fois)
/// 2. Grid search CV
/// 3. Entraîne le modèle final avec les meilleurs params
pub fn train_oracle_cv(
    draws: &[Draw],
    n_folds: usize,
    seed: u64,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<(OracleWeights, GridSearchResult)> {
    let meta = collect_all_meta_features(draws, 0.0, progress)?;

    if let Some(pb) = progress {
        pb.set_message("Grid search CV...");
    }

    let grid_result = cv_grid_search(
        &meta.ball_features,
        &meta.star_features,
        &meta.ball_targets,
        &meta.star_targets,
        meta.ball_input_dim,
        meta.star_input_dim,
        n_folds,
        seed,
        progress,
    )?;

    if let Some(pb) = progress {
        pb.set_message("Entraînement final avec les meilleurs hyperparamètres...");
    }

    let weights = fit_rff_ridge(
        &meta,
        grid_result.best.n_rff,
        grid_result.best.bandwidth,
        grid_result.best.ridge_lambda,
        seed,
        progress,
    )?;

    Ok((weights, grid_result))
}

/// Entraîne M MetaOracles avec des seeds différentes et retourne le vecteur de poids.
pub fn train_oracle_ensemble(
    meta: &MetaFeatures,
    ball_params: (usize, f64, f64),
    star_params: (usize, f64, f64),
    temporal_decay: f64,
    n_members: usize,
    base_seed: u64,
    ball_standardizer: Option<&FeatureStandardizer>,
    star_standardizer: Option<&FeatureStandardizer>,
    ball_reduction: &ReductionStrategy,
    star_reduction: &ReductionStrategy,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<Vec<OracleWeights>> {
    let mut members = Vec::with_capacity(n_members);
    for i in 0..n_members {
        if let Some(pb) = progress {
            pb.set_message(format!("Ensemble member {}/{}...", i + 1, n_members));
        }
        let member = fit_rff_ridge_v2(
            meta,
            ball_params, star_params,
            temporal_decay,
            base_seed + i as u64,
            ball_standardizer, star_standardizer,
            ball_reduction, star_reduction,
            None,
        )?;
        members.push(member);
    }
    Ok(members)
}

/// Entraîne le MetaOracle v3 complet : standardize → PCA → grid search → fit.
pub fn train_oracle_cv_v2(
    draws: &[Draw],
    config: TrainConfig,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<(OracleWeights, GridSearchResultV2)> {
    // 1. Collect meta-features
    let meta = collect_all_meta_features(draws, config.label_smoothing, progress)?;

    // 2. Standardize if requested
    let ball_std = if config.standardize {
        Some(FeatureStandardizer::fit(&meta.ball_features))
    } else {
        None
    };
    let star_std = if config.standardize {
        Some(FeatureStandardizer::fit(&meta.star_features))
    } else {
        None
    };

    let ball_feats = if let Some(ref std) = ball_std {
        std.transform_batch(&meta.ball_features)
    } else {
        meta.ball_features.clone()
    };
    let star_feats = if let Some(ref std) = star_std {
        std.transform_batch(&meta.star_features)
    } else {
        meta.star_features.clone()
    };

    // 3. PCA if requested
    let ball_pca = match &config.pca_balls {
        Some(PcaMode::Auto) => Some(PcaTransform::fit_auto(&ball_feats, 0.95)?),
        Some(PcaMode::Fixed(k)) => Some(PcaTransform::fit(&ball_feats, *k)?),
        None => None,
    };
    let star_pca = match &config.pca_stars {
        Some(PcaMode::Auto) => Some(PcaTransform::fit_auto(&star_feats, 0.95)?),
        Some(PcaMode::Fixed(k)) => Some(PcaTransform::fit(&star_feats, *k)?),
        None => None,
    };

    let ball_feats_reduced = if let Some(ref pca) = ball_pca {
        pca.transform_batch(&ball_feats)
    } else {
        ball_feats
    };
    let star_feats_reduced = if let Some(ref pca) = star_pca {
        pca.transform_batch(&star_feats)
    } else {
        star_feats
    };

    let ball_dim = ball_feats_reduced[0].len();
    let star_dim = star_feats_reduced[0].len();

    // Display reduction info
    if let Some(ref pca) = ball_pca {
        println!(
            "  PCA boules : {} → {} composantes ({:.1}% variance)",
            meta.ball_input_dim, pca.n_components, pca.explained_variance_ratio * 100.0
        );
    }
    if let Some(ref pca) = star_pca {
        println!(
            "  PCA étoiles : {} → {} composantes ({:.1}% variance)",
            meta.star_input_dim, pca.n_components, pca.explained_variance_ratio * 100.0
        );
    }

    // 4. Grid search
    let grid_config = match config.grid_mode {
        GridMode::Default => GridSearchConfig::default_grid(),
        GridMode::Extended => GridSearchConfig::extended(),
        GridMode::TwoStage => GridSearchConfig::extended(), // handled separately
    };

    let grid_result = match config.cv_mode {
        CvMode::Fold => {
            match config.grid_mode {
                GridMode::TwoStage => two_stage_grid_search(
                    &ball_feats_reduced, &star_feats_reduced,
                    &meta.ball_targets, &meta.star_targets,
                    ball_dim, star_dim,
                    config.n_folds, config.seed, progress,
                )?,
                _ => cv_grid_search_v2(
                    &ball_feats_reduced, &star_feats_reduced,
                    &meta.ball_targets, &meta.star_targets,
                    ball_dim, star_dim,
                    &grid_config, config.n_folds, config.seed, progress,
                )?,
            }
        }
        CvMode::WalkForward => {
            let min_train = 200;
            let n_eval = 100;
            cv_walk_forward(
                &ball_feats_reduced, &star_feats_reduced,
                &meta.ball_targets, &meta.star_targets,
                ball_dim, star_dim,
                &grid_config, min_train, n_eval, config.seed, progress,
            )?
        }
    };

    // 5. Choose params
    let ball_params = if config.separate_pools {
        (grid_result.best_balls.n_rff, grid_result.best_balls.bandwidth, grid_result.best_balls.ridge_lambda)
    } else {
        (grid_result.best_combined.n_rff, grid_result.best_combined.bandwidth, grid_result.best_combined.ridge_lambda)
    };

    let star_params = if config.separate_pools {
        (grid_result.best_stars.n_rff, grid_result.best_stars.bandwidth, grid_result.best_stars.ridge_lambda)
    } else {
        (grid_result.best_combined.n_rff, grid_result.best_combined.bandwidth, grid_result.best_combined.ridge_lambda)
    };

    let ball_reduction = ball_pca.as_ref().map(|p| p.to_strategy()).unwrap_or(ReductionStrategy::None);
    let star_reduction = star_pca.as_ref().map(|p| p.to_strategy()).unwrap_or(ReductionStrategy::None);

    // 6. Train final model(s)
    if let Some(pb) = progress {
        pb.set_message("Entraînement final...");
    }

    // Create meta with reduced features for fit_rff_ridge_v2
    // Note: standardization + PCA are already applied, so pass None/None to fit_rff_ridge_v2
    let reduced_meta = MetaFeatures {
        ball_features: ball_feats_reduced,
        star_features: star_feats_reduced,
        ball_targets: meta.ball_targets,
        star_targets: meta.star_targets,
        ball_input_dim: ball_dim,
        star_input_dim: star_dim,
    };

    let n_seeds = config.ensemble_seeds.max(1);

    let mut weights = if n_seeds > 1 {
        let members = train_oracle_ensemble(
            &reduced_meta, ball_params, star_params,
            config.temporal_decay, n_seeds, config.seed,
            None, None, &ReductionStrategy::None, &ReductionStrategy::None,
            progress,
        )?;

        // Use first member as base, store others
        let mut base = members.into_iter().next().unwrap();
        // Re-train all as members
        let all_members = train_oracle_ensemble(
            &reduced_meta, ball_params, star_params,
            config.temporal_decay, n_seeds, config.seed,
            None, None, &ReductionStrategy::None, &ReductionStrategy::None,
            None,
        )?;
        base.ensemble_members = all_members;
        base
    } else {
        fit_rff_ridge_v2(
            &reduced_meta, ball_params, star_params,
            config.temporal_decay, config.seed,
            None, None, &ReductionStrategy::None, &ReductionStrategy::None,
            progress,
        )?
    };

    // Store reduction and standardization info for inference
    weights.ball_reduction = ball_reduction;
    weights.star_reduction = star_reduction;
    weights.ball_standardizer = ball_std;
    weights.star_standardizer = star_std;
    weights.ball_n_rff = ball_params.0;
    weights.ball_bandwidth = ball_params.1;
    weights.ball_ridge_lambda = ball_params.2;
    weights.star_n_rff = star_params.0;
    weights.star_bandwidth = star_params.1;
    weights.star_ridge_lambda = star_params.2;
    weights.ball_cv_ll = grid_result.best_balls.mean_ball_ll;
    weights.star_cv_ll = grid_result.best_stars.mean_star_ll;
    weights.version = 3;

    // Detect hybrid mode: ball_gain < 0 and star_gain > 0
    let uniform_ball_ll = 5.0 * (1.0 / 50.0f64).ln();
    let uniform_star_ll = 2.0 * (1.0 / 12.0f64).ln();
    let ball_gain = grid_result.best_balls.mean_ball_ll - uniform_ball_ll;
    let star_gain = grid_result.best_stars.mean_star_ll - uniform_star_ll;
    if ball_gain < 0.0 && star_gain > 0.0 {
        weights.hybrid = true;
        println!("  Mode hybride détecté : balls gain={:+.3} < 0, stars gain={:+.3} > 0", ball_gain, star_gain);
    }

    Ok((weights, grid_result))
}

/// Sauvegarde les poids dans un fichier JSON.
pub fn save_oracle(weights: &OracleWeights, path: &Path) -> Result<()> {
    let json = serde_json::to_string(weights)
        .context("Échec de la sérialisation Oracle")?;
    std::fs::write(path, json)
        .context("Échec de l'écriture du fichier Oracle")?;
    Ok(())
}

/// Charge les poids depuis un fichier JSON.
pub fn load_oracle(path: &Path) -> Result<OracleWeights> {
    let json = std::fs::read_to_string(path)
        .context("Échec de la lecture du fichier Oracle")?;
    let weights: OracleWeights = serde_json::from_str(&json)
        .context("Échec de la désérialisation Oracle")?;
    Ok(weights)
}

impl OracleModel {
    pub fn new(weights: OracleWeights) -> Self {
        Self { weights }
    }

    /// Tente de charger l'Oracle depuis le fichier par défaut.
    pub fn load() -> Option<Self> {
        let path = Path::new("oracle.json");
        load_oracle(path).ok().map(|w| Self::new(w))
    }

    /// Retourne true si c'est un MetaOracle v2+ (remplace l'ensemble).
    pub fn is_meta(&self) -> bool {
        self.weights.version >= 2
    }

    /// Retourne true si le mode hybride est actif.
    pub fn is_hybrid(&self) -> bool {
        self.weights.hybrid
    }

    /// Transforme des features brutes en probabilités via RFF + readout.
    fn features_to_probs(&self, ball_features: Vec<f64>, star_features: Vec<f64>, pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        let (rff, readout, features) = match pool {
            Pool::Balls => {
                let n_rff = if self.weights.ball_n_rff > 0 { self.weights.ball_n_rff } else { self.weights.n_rff_features };
                let rff = RffTransform::from_weights(&self.weights.ball_rff_w, &self.weights.ball_rff_b);
                let readout_arr = Array2::from_shape_fn(
                    (50, n_rff),
                    |(i, j)| self.weights.ball_readout[i][j],
                );
                (rff, readout_arr, ball_features)
            }
            Pool::Stars => {
                let n_rff = if self.weights.star_n_rff > 0 { self.weights.star_n_rff } else { self.weights.n_rff_features };
                let rff = RffTransform::from_weights(&self.weights.star_rff_w, &self.weights.star_rff_b);
                let readout_arr = Array2::from_shape_fn(
                    (12, n_rff),
                    |(i, j)| self.weights.star_readout[i][j],
                );
                (rff, readout_arr, star_features)
            }
        };

        let input = Array1::from_vec(features);
        let phi = rff.transform(&input);
        let prediction = readout.dot(&phi);

        let mut probs: Vec<f64> = prediction.iter().map(|&x| x.max(1e-15)).collect();
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

    fn predict_internal(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if self.weights.version >= 2 {
            // MetaOracle v2 : utiliser meta-features avec date/feedback estimés
            // On estime la date cible depuis le premier tirage connu
            let target_date = if let Some(first) = draws.first() {
                match next_draw_date(&first.date) {
                    Ok((d, _)) => d,
                    Err(_) => return uniform,
                }
            } else {
                return uniform;
            };
            let prev_draw = draws.first();
            return self.predict_for_date(draws, &target_date, prev_draw, pool);
        }

        // Legacy v0/v1 : collect_base_predictions seul
        if draws.len() < 10 {
            return uniform;
        }

        let expected_ball_dim = self.weights.n_base_models * 50;
        let expected_star_dim = self.weights.n_base_models * 12;
        let (ball_features, star_features) = collect_base_predictions(draws);
        if ball_features.len() != expected_ball_dim || star_features.len() != expected_star_dim {
            return uniform;
        }

        self.features_to_probs(ball_features, star_features, pool)
    }

    /// Prédit pour une date cible spécifique (utilisé par cmd_predict).
    pub fn predict_for_date(&self, draws: &[Draw], target_date: &str, prev_draw: Option<&Draw>, pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if self.weights.version >= 2 {
            let (ball_features_raw, star_features_raw) = collect_meta_features(draws, target_date, prev_draw);

            // v3: Apply standardization + reduction if available
            let (ball_features, star_features) = if self.weights.version >= 3 {
                let bf = match (&self.weights.ball_standardizer, pool) {
                    (Some(std), Pool::Balls) | (Some(std), _) => {
                        let standardized = std.transform(&ball_features_raw);
                        apply_reduction(&standardized, &self.weights.ball_reduction)
                    }
                    _ => apply_reduction(&ball_features_raw, &self.weights.ball_reduction),
                };
                let sf = match (&self.weights.star_standardizer, pool) {
                    (Some(std), Pool::Stars) | (Some(std), _) => {
                        let standardized = std.transform(&star_features_raw);
                        apply_reduction(&standardized, &self.weights.star_reduction)
                    }
                    _ => apply_reduction(&star_features_raw, &self.weights.star_reduction),
                };
                (bf, sf)
            } else {
                (ball_features_raw.clone(), star_features_raw.clone())
            };

            // Dimension check
            let expected_ball_dim = self.weights.ball_input_dim;
            let expected_star_dim = self.weights.star_input_dim;
            if expected_ball_dim > 0 && ball_features.len() != expected_ball_dim {
                return uniform;
            }
            if expected_star_dim > 0 && star_features.len() != expected_star_dim {
                return uniform;
            }

            // Multi-seed ensemble averaging
            if !self.weights.ensemble_members.is_empty() {
                let mut avg_probs = vec![0.0; size];
                let n_members = self.weights.ensemble_members.len();
                for member in &self.weights.ensemble_members {
                    let m = OracleModel { weights: clone_weights(member) };
                    let probs = m.features_to_probs(ball_features.clone(), star_features.clone(), pool);
                    for (i, &p) in probs.iter().enumerate() {
                        avg_probs[i] += p;
                    }
                }
                for p in &mut avg_probs {
                    *p /= n_members as f64;
                }
                return avg_probs;
            }

            self.features_to_probs(ball_features, star_features, pool)
        } else {
            // Legacy : ignore date/feedback
            self.predict_internal(draws, pool)
        }
    }

    /// Produit une EnsemblePrediction complète (distribution + détails par modèle + spread).
    /// Collecte les 18 distributions individuelles pour l'affichage, puis utilise MetaOracle pour la distribution finale.
    pub fn predict_ensemble(&self, draws: &[Draw], target_date: &str, pool: Pool) -> crate::ensemble::EnsemblePrediction {
        let models = base_models();
        let size = pool.size();

        // Collecter les distributions individuelles (pour display)
        let mut model_distributions = Vec::with_capacity(models.len());
        for model in &models {
            let dist = model.predict(draws, pool);
            model_distributions.push((model.name().to_string(), dist));
        }

        // Distribution MetaOracle
        let prev_draw = draws.first();
        let distribution = self.predict_for_date(draws, target_date, prev_draw, pool);

        // Spread (dispersion des modèles de base)
        let spread = crate::ensemble::compute_spread(&model_distributions, size);

        crate::ensemble::EnsemblePrediction {
            distribution,
            model_distributions,
            spread,
        }
    }
}

impl ForecastModel for OracleModel {
    fn name(&self) -> &str {
        "Oracle"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        self.predict_internal(draws, pool)
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_rff_features".into(), self.weights.n_rff_features as f64),
            ("bandwidth".into(), self.weights.bandwidth),
            ("ridge_lambda".into(), self.weights.ridge_lambda),
            ("n_base_models".into(), self.weights.n_base_models as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::FullHistory
    }
}

fn clone_weights(w: &OracleWeights) -> OracleWeights {
    OracleWeights {
        ball_rff_w: w.ball_rff_w.clone(),
        ball_rff_b: w.ball_rff_b.clone(),
        ball_readout: w.ball_readout.clone(),
        star_rff_w: w.star_rff_w.clone(),
        star_rff_b: w.star_rff_b.clone(),
        star_readout: w.star_readout.clone(),
        n_rff_features: w.n_rff_features,
        bandwidth: w.bandwidth,
        ridge_lambda: w.ridge_lambda,
        n_base_models: w.n_base_models,
        version: w.version,
        ball_input_dim: w.ball_input_dim,
        star_input_dim: w.star_input_dim,
        ball_n_rff: w.ball_n_rff,
        ball_bandwidth: w.ball_bandwidth,
        ball_ridge_lambda: w.ball_ridge_lambda,
        star_n_rff: w.star_n_rff,
        star_bandwidth: w.star_bandwidth,
        star_ridge_lambda: w.star_ridge_lambda,
        ball_reduction: w.ball_reduction.clone(),
        star_reduction: w.star_reduction.clone(),
        ball_standardizer: w.ball_standardizer.clone(),
        star_standardizer: w.star_standardizer.clone(),
        hybrid: w.hybrid,
        ball_cv_ll: w.ball_cv_ll,
        star_cv_ll: w.star_cv_ll,
        ensemble_members: Vec::new(), // Don't deep-clone ensemble members for inference
    }
}

/// Vérifie la mémorisation de l'Oracle sur tous les tirages.
/// Retourne (n_perfect, n_total, misses) où misses contient les détails des échecs.
pub fn verify_oracle(
    draws: &[Draw],
    weights: &OracleWeights,
    progress: Option<&indicatif::ProgressBar>,
) -> (usize, usize, Vec<VerifyMiss>) {
    let model = OracleModel::new(clone_weights(weights));
    let is_v2 = weights.version >= 2;

    // v2 : couvrir tous les tirages, v1 : min_history = 10
    let n_test = if is_v2 { draws.len() } else { draws.len().saturating_sub(10) };

    if let Some(pb) = progress {
        pb.set_length(n_test as u64);
        pb.set_message("Vérification...");
    }

    let mut n_perfect = 0usize;
    let mut misses = Vec::new();

    for t in 0..n_test {
        let context = &draws[t + 1..];
        if !is_v2 && context.len() < 10 {
            break;
        }

        let (ball_pred, star_pred) = if is_v2 {
            let target_date = &draws[t].date;
            let prev_draw = draws.get(t + 1);
            (
                model.predict_for_date(context, target_date, prev_draw, Pool::Balls),
                model.predict_for_date(context, target_date, prev_draw, Pool::Stars),
            )
        } else {
            (
                model.predict(context, Pool::Balls),
                model.predict(context, Pool::Stars),
            )
        };

        let optimal = crate::sampler::optimal_grid(&ball_pred, &star_pred);

        let mut actual_balls = draws[t].balls;
        actual_balls.sort();
        let mut actual_stars = draws[t].stars;
        actual_stars.sort();

        let ball_match = actual_balls.iter().filter(|b| optimal.balls.contains(b)).count();
        let star_match = actual_stars.iter().filter(|s| optimal.stars.contains(s)).count();

        if ball_match == 5 && star_match == 2 {
            n_perfect += 1;
        } else {
            misses.push(VerifyMiss {
                index: t,
                date: draws[t].date.clone(),
                actual_balls,
                actual_stars,
                predicted_balls: optimal.balls,
                predicted_stars: optimal.stars,
                ball_match: ball_match as u8,
                star_match: star_match as u8,
            });
        }

        if let Some(pb) = progress {
            pb.inc(1);
        }
    }

    (n_perfect, n_test, misses)
}

/// Résultat de l'évaluation Oracle sur un seul tirage.
#[derive(Debug)]
pub struct OracleEvalResult {
    pub date: String,
    pub actual_balls: [u8; 5],
    pub actual_stars: [u8; 2],
    pub ball_match: u8,
    pub star_match: u8,
    pub ball_ll: f64,
    pub star_ll: f64,
}

/// Évalue Oracle sur un seul tirage (target) en utilisant context comme historique.
/// Retourne les log-likelihoods et matches.
pub fn eval_oracle_single(
    weights: &OracleWeights,
    target: &Draw,
    context: &[Draw],
) -> OracleEvalResult {
    let model = OracleModel::new(clone_weights(weights));

    let (ball_probs, star_probs) = if weights.version >= 2 {
        let prev_draw = context.first();
        (
            model.predict_for_date(context, &target.date, prev_draw, Pool::Balls),
            model.predict_for_date(context, &target.date, prev_draw, Pool::Stars),
        )
    } else {
        (
            model.predict(context, Pool::Balls),
            model.predict(context, Pool::Stars),
        )
    };

    // Log-likelihood du tirage réel
    let ball_ll: f64 = target.balls.iter()
        .map(|&b| ball_probs[(b - 1) as usize].max(1e-30).ln())
        .sum();
    let star_ll: f64 = target.stars.iter()
        .map(|&s| star_probs[(s - 1) as usize].max(1e-30).ln())
        .sum();

    // Match optimal (top-5 boules, top-2 étoiles)
    let optimal = crate::sampler::optimal_grid(&ball_probs, &star_probs);

    let mut actual_balls = target.balls;
    actual_balls.sort();
    let mut actual_stars = target.stars;
    actual_stars.sort();

    let ball_match = actual_balls.iter().filter(|b| optimal.balls.contains(b)).count() as u8;
    let star_match = actual_stars.iter().filter(|s| optimal.stars.contains(s)).count() as u8;

    OracleEvalResult {
        date: target.date.clone(),
        actual_balls,
        actual_stars,
        ball_match,
        star_match,
        ball_ll,
        star_ll,
    }
}

#[derive(Debug)]
pub struct VerifyMiss {
    pub index: usize,
    pub date: String,
    pub actual_balls: [u8; 5],
    pub actual_stars: [u8; 2],
    pub predicted_balls: [u8; 5],
    pub predicted_stars: [u8; 2],
    pub ball_match: u8,
    pub star_match: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_rff_transform_deterministic() {
        let rff = RffTransform::new(10, 100, 1.0, 42);
        let x = Array1::from_vec(vec![0.1; 10]);
        let phi1 = rff.transform(&x);
        let phi2 = rff.transform(&x);
        for (a, b) in phi1.iter().zip(phi2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_rff_from_weights_roundtrip() {
        let rff = RffTransform::new(10, 50, 1.5, 123);
        let w_vec: Vec<Vec<f64>> = (0..50).map(|i| rff.w.row(i).to_vec()).collect();
        let b_vec: Vec<f64> = rff.b.to_vec();

        let rff2 = RffTransform::from_weights(&w_vec, &b_vec);
        let x = Array1::from_vec(vec![0.5; 10]);
        let phi1 = rff.transform(&x);
        let phi2 = rff2.transform(&x);

        for (a, b) in phi1.iter().zip(phi2.iter()) {
            assert!((a - b).abs() < 1e-12, "RFF roundtrip mismatch");
        }
    }

    #[test]
    fn test_collect_base_predictions_dimensions() {
        let draws = make_test_draws(50);
        let (ball_feat, star_feat) = collect_base_predictions(&draws);
        let n_base = base_models().len();
        assert_eq!(ball_feat.len(), n_base * 50);
        assert_eq!(star_feat.len(), n_base * 12);
    }

    #[test]
    fn test_oracle_model_with_load_fallback() {
        // Si oracle.json n'existe pas, load() retourne None
        let model = OracleModel::load();
        // On ne peut pas garantir l'existence du fichier en test
        if model.is_none() {
            // C'est OK, le fichier n'existe pas en environnement de test
        }
    }
}
