/// WearDrift v21: Modèle de dérive mécanique lente.
///
/// L'usure des trappes de la machine Stresa crée une dérive lente des probabilités
/// sur des centaines de tirages. Ce modèle capture ces tendances séculaires.
///
/// Algorithme:
/// 1. EWMA ultra-lent (α=0.005) sur fréquences par numéro, fenêtre 500+ tirages
/// 2. Régression linéaire sur fréquences pour détecter tendances
/// 3. Score = fréquence EWMA ajustée par tendance extrapolée (1 tirage)
/// 4. Gate: seuls les numéros avec tendance significative (|t-stat| > 1.645, p<0.10) contribuent

use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS};

pub struct WearDriftModel {
    ewma_alpha: f64,
    smoothing: f64,
    min_draws: usize,
    trend_window: usize,
}

impl Default for WearDriftModel {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.005,
            smoothing: 0.20,
            min_draws: 100,
            trend_window: 500,
        }
    }
}

impl ForecastModel for WearDriftModel {
    fn name(&self) -> &str { "WearDrift" }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 5 }
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("ewma_alpha".into(), self.ewma_alpha);
        m.insert("smoothing".into(), self.smoothing);
        m.insert("trend_window".into(), self.trend_window as f64);
        m
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;

        if draws.len() < self.min_draws {
            return vec![uniform; size];
        }

        let n = draws.len().min(self.trend_window);
        let numbers: Vec<Vec<u8>> = draws[..n].iter()
            .map(|d| pool.numbers_from(d).to_vec())
            .collect();

        // 1. EWMA ultra-lente sur fréquences (chronological: iterate from oldest to newest)
        let mut ewma = vec![uniform; size];
        let alpha = self.ewma_alpha;
        for t in (0..n).rev() {
            let mut indicator = vec![0.0_f64; size];
            for &num in &numbers[t] {
                indicator[(num - 1) as usize] = 1.0 / pool.pick_count() as f64;
            }
            for i in 0..size {
                ewma[i] = (1.0 - alpha) * ewma[i] + alpha * indicator[i];
            }
        }

        // 2. Régression linéaire sur fréquences glissantes pour détecter tendances
        // Compute rolling frequency at multiple timepoints
        let n_points = 10.min(n / 20); // at least 20 draws per point
        if n_points < 3 {
            let mut probs = ewma;
            for p in probs.iter_mut() {
                *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform;
            }
            floor_only(&mut probs, PROB_FLOOR_BALLS);
            return probs;
        }

        let chunk_size = n / n_points;
        let mut freq_series: Vec<Vec<f64>> = vec![Vec::with_capacity(n_points); size];

        for point in 0..n_points {
            let start = point * chunk_size;
            let end = ((point + 1) * chunk_size).min(n);
            let chunk_len = (end - start) as f64;
            let mut freq = vec![0.0_f64; size];
            for t in start..end {
                for &num in &numbers[t] {
                    freq[(num - 1) as usize] += 1.0;
                }
            }
            for i in 0..size {
                freq[i] /= chunk_len;
                freq_series[i].push(freq[i]);
            }
        }

        // 3. Linear regression per number: freq = a + b*t
        // t values: 0, 1, ..., n_points-1 (0 = most recent chunk)
        let t_mean = (n_points as f64 - 1.0) / 2.0;
        let t_var: f64 = (0..n_points).map(|t| (t as f64 - t_mean).powi(2)).sum::<f64>();

        let mut probs = vec![uniform; size];
        let mut has_signal = false;

        for i in 0..size {
            let series = &freq_series[i];
            let y_mean: f64 = series.iter().sum::<f64>() / n_points as f64;

            // slope b = Σ(t-t̄)(y-ȳ) / Σ(t-t̄)²
            let cov: f64 = series.iter().enumerate()
                .map(|(t, &y)| (t as f64 - t_mean) * (y - y_mean))
                .sum();
            let slope = if t_var > 1e-15 { cov / t_var } else { 0.0 };

            // Residual standard error
            let intercept = y_mean - slope * t_mean;
            let residual_var: f64 = series.iter().enumerate()
                .map(|(t, &y)| {
                    let predicted = intercept + slope * t as f64;
                    (y - predicted).powi(2)
                })
                .sum::<f64>() / (n_points as f64 - 2.0).max(1.0);
            let se_slope = if t_var > 1e-15 { (residual_var / t_var).sqrt() } else { f64::INFINITY };

            // t-statistic
            let t_stat = if se_slope > 1e-15 { slope / se_slope } else { 0.0 };

            // Gate: only use trend if significant (|t| > 1.645 for p<0.10 one-sided)
            if t_stat.abs() > 1.645 {
                // Extrapolate 1 step ahead (t = -1, i.e., next draw)
                let extrapolated = intercept + slope * (-1.0_f64);
                // Blend EWMA with trend-adjusted frequency
                probs[i] = 0.5 * ewma[i] + 0.5 * extrapolated.max(0.0);
                has_signal = true;
            } else {
                probs[i] = ewma[i];
            }
        }

        // If no significant trends, use pure EWMA
        if !has_signal {
            probs = ewma;
        }

        // Apply smoothing
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * (*p).max(0.0) + self.smoothing * uniform;
        }

        floor_only(&mut probs, PROB_FLOOR_BALLS);
        probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_wear_drift_balls_sums_to_one() {
        let model = WearDriftModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum = {}", sum);
    }

    #[test]
    fn test_wear_drift_stars_sums_to_one() {
        let model = WearDriftModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Stars);
        assert_eq!(dist.len(), 12);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum = {}", sum);
    }

    #[test]
    fn test_wear_drift_short_history() {
        let model = WearDriftModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        // Should return near-uniform for short history
        let uniform = 1.0 / 50.0;
        assert!((dist[0] - uniform).abs() < 0.01);
    }

    #[test]
    fn test_wear_drift_name() {
        let model = WearDriftModel::default();
        assert_eq!(model.name(), "WearDrift");
    }

    #[test]
    fn test_wear_drift_params() {
        let model = WearDriftModel::default();
        let params = model.params();
        assert_eq!(params["ewma_alpha"], 0.005);
        assert_eq!(params["smoothing"], 0.20);
    }
}
