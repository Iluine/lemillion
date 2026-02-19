pub mod compute;

use lemillion_db::models::{Draw, Pool};

pub const FEATURE_NAMES: &[&str] = &[
    "freq_5", "freq_10", "freq_20",
    "retard", "retard_norm",
    "trend",
    "mean_gap", "std_gap",
    "is_odd",
    "decade",
    "decade_density",
    "day_of_week",
    "recent_sum_norm",
    "recent_even_count",
];

#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub number: u8,
    pub features: Vec<f64>,
    pub label: f64,
}

pub fn extract_features_for_draw(draws: &[Draw], pool: Pool, target_draw_idx: usize) -> Vec<FeatureRow> {
    compute::extract_features_for_draw(draws, pool, target_draw_idx)
}
