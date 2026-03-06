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
    "freq_3",
    "pair_freq",
    "gap_acceleration",
    "low_half",
    "mod4_class",
    "mod4_class_freq",
    "mod4_transition",
    "month_sin",
    "month_cos",
    "quarter",
    "day_of_year_sin",
    "day_of_year_cos",
    "draw_position",
    "days_since_last",
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
