use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    #[clap(name = "onehot")]
    OneHot,
    Normalized,
}

impl Encoding {
    pub fn input_dim(&self) -> usize {
        match self {
            Encoding::OneHot => 62,       // Pool::Balls.size() + Pool::Stars.size()
            Encoding::Normalized => 7,    // 5 balls + 2 stars
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnConfig {
    pub reservoir_size: usize,
    pub spectral_radius: f64,
    pub sparsity: f64,
    pub leaking_rate: f64,
    pub ridge_lambda: f64,
    pub input_scaling: f64,
    pub encoding: Encoding,
    pub washout: usize,
    pub noise_amplitude: f64,
    pub seed: u64,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 500,
            spectral_radius: 0.95,
            sparsity: 0.9,
            leaking_rate: 0.3,
            ridge_lambda: 1e-4,
            input_scaling: 0.1,
            encoding: Encoding::OneHot,
            washout: 50,
            noise_amplitude: 1e-4,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnResult {
    pub config: EsnConfig,
    pub val_ball_hit_rate: f64,
    pub val_star_hit_rate: f64,
    pub val_ball_topk: f64,
    pub val_star_topk: f64,
    pub test_ball_hit_rate: f64,
    pub test_star_hit_rate: f64,
    pub test_ball_topk: f64,
    pub test_star_topk: f64,
    pub lyapunov_exponent: f64,
    pub train_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchResults {
    pub results: Vec<EsnResult>,
    pub best_config: EsnConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_input_dim() {
        assert_eq!(Encoding::OneHot.input_dim(), 62);
        assert_eq!(Encoding::Normalized.input_dim(), 7);
    }

    #[test]
    fn test_default_config() {
        let config = EsnConfig::default();
        assert_eq!(config.reservoir_size, 500);
        assert!((config.spectral_radius - 0.95).abs() < 1e-10);
        assert_eq!(config.encoding, Encoding::OneHot);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = EsnConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: EsnConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.reservoir_size, config.reservoir_size);
        assert_eq!(restored.encoding, config.encoding);
    }
}
