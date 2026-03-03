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

use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};

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
}

/// Numerical safety floor — prevents log(0) = -inf, zero signal impact.
pub const PROB_FLOOR_BALLS: f64 = 1e-15;
pub const PROB_FLOOR_STARS: f64 = 1e-15;

/// Numerical safety only: floor at epsilon to prevent log(0), then renormalize.
pub fn floor_and_normalize(probs: &mut Vec<f64>, floor: f64) {
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

/// Les 14 modèles de base (Dirichlet, Markov, ESN, CondSummary supprimés — skill négatif).
pub fn base_models() -> Vec<Box<dyn ForecastModel>> {
    vec![
        Box::new(logistic::LogisticModel::new(0.01, 0.0001, 200, 100)),
        Box::new(random_forest::RandomForestModel::new(100, 3, 200)),
        Box::new(spectral::SpectralModel::default()),
        Box::new(ctw::CtwModel::default()),
        Box::new(mixture::MixtureModel::default()),
        Box::new(transformer::TransformerModel::default()),
        Box::new(tda::TdaModel::default()),
        Box::new(physics::PhysicsModel::default()),
        Box::new(mod4::Mod4TransitionModel::default()),
        Box::new(mod4_profile::Mod4ProfileModel::default()),
        Box::new(triplet::TripletBoostModel::default()),
        Box::new(stresa::StresaSgdModel::default()),
        Box::new(stresa::StresaSmcModel::default()),
        Box::new(stresa::StresaChaosModel::default()),
        Box::new(conditional_v2::CondSummaryV2Model::default()),
        Box::new(gap_dynamics::GapDynamicsModel::default()),
        Box::new(star_specialist::StarSpecialistModel::default()),
        Box::new(transfer_entropy::TransferEntropyModel::default()),
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
}
