pub mod dirichlet;
pub mod ewma;
pub mod logistic;
pub mod random_forest;
pub mod markov;
pub mod retard;
pub mod hot_streak;
pub mod esn;
pub mod takens;
pub mod spectral;
pub mod ctw;
pub mod nvar;
pub mod nvar_memo;
pub mod mixture;

use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};

pub trait ForecastModel: Send + Sync {
    fn name(&self) -> &str;
    /// draws[0] = tirage le plus récent. Retourne Vec<f64> de taille pool.size(), somme = 1.0
    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64>;
    fn params(&self) -> HashMap<String, f64>;
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

pub fn all_models() -> Vec<Box<dyn ForecastModel>> {
    vec![
        Box::new(dirichlet::DirichletModel::with_window(0.1, Some(30))),
        Box::new(ewma::EwmaModel::new(0.9)),
        Box::new(logistic::LogisticModel::new(0.01, 0.001, 50, 100)),
        Box::new(random_forest::RandomForestModel::new(50, 5, 100)),
        Box::new(markov::MarkovModel::new()),
        Box::new(retard::RetardModel::new(1.5)),
        Box::new(hot_streak::HotStreakModel::new(5)),
        Box::new(esn::EsnModel::new(lemillion_esn::config::EsnConfig {
            reservoir_size: 1000,
            spectral_radius: 0.95,
            sparsity: 0.8,
            leaking_rate: 1.0,
            ridge_lambda: 1e-6,
            input_scaling: 0.1,
            encoding: lemillion_esn::config::Encoding::Normalized,
            washout: 100,
            noise_amplitude: 1e-4,
            seed: 42,
        })),
        Box::new(takens::TakensKnnModel::default()),
        Box::new(spectral::SpectralModel::default()),
        Box::new(ctw::CtwModel::default()),
        Box::new(nvar::NvarModel::default()),
        Box::new(nvar_memo::NvarMemoModel::default()),
        Box::new(mixture::MixtureModel::default()),
    ]
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
                    (base * 5 + 1).min(50).max(1),
                    (base * 5 + 2).min(50).max(1),
                    (base * 5 + 3).min(50).max(1),
                    (base * 5 + 4).min(50).max(1),
                    (base * 5 + 5).min(50).max(1),
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
