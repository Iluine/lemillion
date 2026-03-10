pub mod dirichlet;
pub mod logistic;
pub mod random_forest;
pub mod markov;
pub mod esn;
pub mod spectral;
pub mod ctw;
pub mod mixture;
pub mod transformer;
pub mod tda;
pub mod physics;
pub mod mod4;
pub mod triplet;
pub mod conditional;
pub mod conditional_v2;
pub mod gap_dynamics;
pub mod joint;
pub mod mod4_profile;
pub mod summary_predictor;
pub mod star_specialist;
pub mod stresa;
pub mod transfer_entropy;
pub mod star_pair;
pub mod star_recency;
pub mod context_knn;
pub mod neural_scorer;
pub mod jackpot_context;
pub mod max_entropy;
pub mod hmm;
pub mod boltzmann;
pub mod hawkes;
pub mod bocpd;
pub mod decade_persist;
pub mod modular_balls;
pub mod compression;
pub mod star_momentum;
pub mod spread;
pub mod gap_model;
pub mod unit_digit;
pub mod delayed_mi;
pub mod community;
pub mod rqa_predictability;
pub mod wavelet;
pub mod copula;
pub mod renewal;
pub mod draw_order;
pub mod tlr;
pub mod particle_stresa;
pub mod forbidden_patterns;
pub mod renyi_te;
pub mod cross_te;
pub mod te_order2;
pub mod spectral_graph;
pub mod evt;

use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};

/// Date after which the current star format (2/12) applies.
/// Before 2016-09-27, EuroMillions used different star pools (2/9 or 2/11),
/// making pre-era star data incompatible with current models.
pub const STAR_ERA_DATE: &str = "2016-09-27";

/// Filter draws to only include those from the current star era (post Sep 2016).
/// Returns a slice of the draws that are in the current era.
/// Since draws[0] = most recent, we just need to find where the old era starts.
pub fn filter_star_era(draws: &[Draw]) -> &[Draw] {
    let cutoff = draws.iter().position(|d| d.date.as_str() < STAR_ERA_DATE);
    match cutoff {
        Some(pos) => &draws[..pos],
        None => draws, // all draws are in current era
    }
}

/// Stratégie d'échantillonnage pour la calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Fenêtre consécutive (comportement actuel). Le modèle reçoit `window` tirages consécutifs.
    Consecutive,
    /// Fenêtre éparse : on pioche `window` tirages uniformément dans un span de `span_multiplier * window` tirages.
    /// Préserve l'ordre chronologique.
    Sparse { span_multiplier: usize },
    /// Historique complet : draws[t+1..] pour les modèles entraînés walk-forward.
    FullHistory,
}

pub trait ForecastModel: Send + Sync {
    fn name(&self) -> &str;
    /// draws[0] = tirage le plus récent. Retourne Vec<f64> de taille pool.size(), somme = 1.0
    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64>;
    fn params(&self) -> HashMap<String, f64>;

    /// Stratégie d'échantillonnage préférée pour la calibration.
    /// Par défaut: fenêtre consécutive.
    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Consecutive
    }

    /// Stride de calibration : les modèles lourds peuvent sauter des test points.
    /// stride=1 → tous les points, stride=3 → 1 sur 3.
    fn calibration_stride(&self) -> usize {
        1
    }

    /// Retourne true si le modèle ne prédit que les étoiles (uniform pour les boules).
    /// Les modèles star-only auront leur poids boules mis à 0.
    fn is_stars_only(&self) -> bool {
        false
    }
}

/// Numerical safety floor — prevents log(0) = -inf, zero signal impact.
pub const PROB_FLOOR_BALLS: f64 = 1e-15;
pub const PROB_FLOOR_STARS: f64 = 1e-15;

/// Numerical safety only: floor at epsilon to prevent log(0), then renormalize.
/// NOTE: includes James-Stein shrinkage — use floor_only() for models where
/// the ensemble already provides sufficient shrinkage.
pub fn floor_and_normalize(probs: &mut Vec<f64>, floor: f64) {
    // Apply James-Stein shrinkage before flooring (provably optimal for p >= 3)
    james_stein_shrink(probs);

    for p in probs.iter_mut() {
        if *p < floor {
            *p = floor;
        }
    }
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Floor-only normalization: prevents log(0) without James-Stein shrinkage.
/// Use this for models whose output feeds into the log-linear ensemble pool,
/// which already acts as a shrinkage mechanism.
pub fn floor_only(probs: &mut Vec<f64>, floor: f64) {
    for p in probs.iter_mut() {
        if *p < floor {
            *p = floor;
        }
    }
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// James-Stein shrinkage towards uniform: provably dominates MLE for simultaneous
/// estimation of p >= 3 parameters. Shrinks extreme probability estimates toward the mean.
pub fn james_stein_shrink(probs: &mut Vec<f64>) {
    let n = probs.len();
    if n < 3 { return; }

    let p_bar = 1.0 / n as f64;
    let ss: f64 = probs.iter().map(|&p| (p - p_bar).powi(2)).sum();
    if ss < 1e-15 { return; } // already uniform

    // B = max(0, 1 - (n-2) × sigma² / (n × ss))
    // sigma² = p_bar × (1 - p_bar) is the variance under the uniform null
    let sigma_sq = p_bar * (1.0 - p_bar);
    let b = (1.0 - (n as f64 - 2.0) * sigma_sq / (n as f64 * ss)).max(0.0);

    for p in probs.iter_mut() {
        *p = p_bar + b * (*p - p_bar);
    }

    // Ensure no negatives and renormalize
    for p in probs.iter_mut() {
        if *p < 0.0 { *p = 1e-15; }
    }
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

pub fn validate_distribution(dist: &[f64], pool: Pool) -> bool {
    if dist.len() != pool.size() {
        return false;
    }
    if dist.iter().any(|&p| p < 0.0) {
        return false;
    }
    let sum: f64 = dist.iter().sum();
    (sum - 1.0).abs() < 1e-9
}

/// Modèles de base de l'ensemble (27 modèles actifs).
/// v14: SpectralGraph, EVT ajoutés (signaux orthogonaux non-TE). TEOrder2 retiré (dilution BMA via corrélation TE).
/// v13: RényiTE, CrossTE ajoutés (signaux orthogonaux aux TE existants).
/// v11: TLR, ParticleStresa, ForbiddenPatterns ajoutés puis exclus (dilution sans signal).
/// Retirés v9: Copula, Wavelet, Renewal (0% poids boules+étoiles).
/// Retirés v7: RqaPredictability, UnitDigit, DelayedMI, Community, GapModel (0% poids boules+étoiles).
/// Retirés v5: RandomForest, ModProfile, StresaSMC, GapDynamics, ModTrans.
/// Retirés v4: CTW, Spectral, StarRecency, BME/Mixture.
/// Retirés avant: Dirichlet, Markov, ESN, CondSummary, JackpotContext.
pub fn base_models() -> Vec<Box<dyn ForecastModel>> {
    vec![
        Box::new(logistic::LogisticModel::new(0.01, 0.0001, 200, 100)),
        Box::new(transformer::TransformerModel::default()),
        Box::new(tda::TdaModel::default()),
        Box::new(physics::PhysicsModel::default()),
        Box::new(stresa::StresaSgdModel::default()),
        Box::new(stresa::StresaChaosModel::default()),
        Box::new(conditional_v2::CondSummaryV2Model::default()),
        Box::new(star_specialist::StarSpecialistModel::default()),
        Box::new(transfer_entropy::TransferEntropyModel::default()),
        Box::new(star_pair::StarPairModel::default()),
        Box::new(context_knn::ContextKnnModel::default()),
        Box::new(max_entropy::MaxEntropyModel::default()),
        Box::new(hmm::HmmModel::default()),
        Box::new(boltzmann::BoltzmannModel::default()),
        Box::new(hawkes::HawkesModel::default()),
        Box::new(bocpd::BocpdModel::default()),
        Box::new(decade_persist::DecadePersistModel::default()),
        Box::new(modular_balls::ModularBallsModel::default()),
        Box::new(compression::CompressionModel::default()),
        Box::new(triplet::TripletBoostModel::default()),
        Box::new(star_momentum::StarMomentumModel::default()),
        Box::new(spread::SpreadModel::default()),
        Box::new(draw_order::DrawOrderModel::default()),
        Box::new(renyi_te::RenyiTEModel::default()),        // v13
        Box::new(cross_te::CrossTEModel::default()),          // v13
        //Box::new(te_order2::TEOrder2Model::default()),        // v14
        Box::new(spectral_graph::SpectralGraphModel::default()), // v14
        Box::new(evt::EvtModel::default()),                    // v14
        //Box::new(tlr::TlrModel::default()),
        //Box::new(particle_stresa::ParticleStresaModel::default()),
        //Box::new(forbidden_patterns::ForbiddenPatternsModel::default()),
    ]
}

/// Tous les modèles de l'ensemble.
pub fn all_models() -> Vec<Box<dyn ForecastModel>> {
    base_models()
}

pub fn make_test_draws(n: usize) -> Vec<Draw> {
    (0..n)
        .map(|i| {
            let base = (i % 10) as u8;
            Draw {
                draw_id: format!("{:03}", i),
                day: if i % 2 == 0 { "MARDI".to_string() } else { "VENDREDI".to_string() },
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: [
                    (base * 5 + 1).clamp(1, 50),
                    (base * 5 + 2).clamp(1, 50),
                    (base * 5 + 3).clamp(1, 50),
                    (base * 5 + 4).clamp(1, 50),
                    (base * 5 + 5).clamp(1, 50),
                ],
                stars: [
                    (base % 12 + 1).min(12),
                    ((base + 1) % 12 + 1).min(12),
                ],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_distribution_valid() {
        let dist = vec![1.0 / 50.0; 50];
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_validate_distribution_wrong_size() {
        let dist = vec![1.0 / 50.0; 49];
        assert!(!validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_validate_distribution_negative() {
        let mut dist = vec![1.0 / 50.0; 50];
        dist[0] = -0.1;
        assert!(!validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_filter_star_era_all_current() {
        // All draws are post-2016-09-27
        let draws: Vec<Draw> = (0..5)
            .map(|i| Draw {
                draw_id: format!("{}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", i + 1),
                balls: [1, 2, 3, 4, 5],
                stars: [1, 2],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
            })
            .collect();
        let filtered = filter_star_era(&draws);
        assert_eq!(filtered.len(), 5, "All current-era draws should be kept");
    }

    #[test]
    fn test_filter_star_era_mixed() {
        // draws[0..3] are current era (newest first), draws[3..5] are old era
        let draws = vec![
            Draw {
                draw_id: "4".to_string(), day: "MARDI".to_string(),
                date: "2024-01-04".to_string(),
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
            Draw {
                draw_id: "3".to_string(), day: "MARDI".to_string(),
                date: "2020-06-15".to_string(),
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
            Draw {
                draw_id: "2".to_string(), day: "MARDI".to_string(),
                date: "2016-09-27".to_string(),
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
            Draw {
                draw_id: "1".to_string(), day: "MARDI".to_string(),
                date: "2016-09-26".to_string(),  // one day before cutoff
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
            Draw {
                draw_id: "0".to_string(), day: "MARDI".to_string(),
                date: "2010-01-01".to_string(),
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
        ];
        let filtered = filter_star_era(&draws);
        assert_eq!(filtered.len(), 3, "Should keep 3 current-era draws");
        assert_eq!(filtered[0].date, "2024-01-04");
        assert_eq!(filtered[2].date, "2016-09-27");
    }

    #[test]
    fn test_filter_star_era_all_old() {
        let draws = vec![
            Draw {
                draw_id: "1".to_string(), day: "MARDI".to_string(),
                date: "2015-01-01".to_string(),
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
            Draw {
                draw_id: "0".to_string(), day: "MARDI".to_string(),
                date: "2010-01-01".to_string(),
                balls: [1,2,3,4,5], stars: [1,2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
                ball_order: None, star_order: None, cycle_number: None,
            },
        ];
        let filtered = filter_star_era(&draws);
        assert_eq!(filtered.len(), 0, "All old-era draws should be filtered out");
    }

    #[test]
    fn test_filter_star_era_empty() {
        let draws: Vec<Draw> = vec![];
        let filtered = filter_star_era(&draws);
        assert_eq!(filtered.len(), 0);
    }
}
