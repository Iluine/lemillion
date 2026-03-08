use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{filter_star_era, floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_STARS};

/// StarMomentum — exploite la persistance/anti-persistance par étoile via DFA Hurst.
///
/// Signal: Hurst H=0.529 pour étoiles (p=0.037, persistance marginale).
/// - Étoiles persistantes (H>0.52): boost fréquence récente EWMA
/// - Étoiles anti-persistantes (H<0.48): mean-reversion basée sur le gap
/// - Neutres: fréquence globale
///
/// Retourne uniforme pour Pool::Balls.
pub struct StarMomentumModel {
    smoothing: f64,
    min_draws: usize,
    ewma_alpha: f64,
    momentum_boost: f64,
    reversion_boost: f64,
    persistence_threshold: f64,
    reversion_threshold: f64,
}

impl Default for StarMomentumModel {
    fn default() -> Self {
        Self {
            smoothing: 0.30,
            min_draws: 80,
            ewma_alpha: 0.10,
            momentum_boost: 0.15,
            reversion_boost: 0.10,
            persistence_threshold: 0.52,
            reversion_threshold: 0.48,
        }
    }
}

impl ForecastModel for StarMomentumModel {
    fn name(&self) -> &str {
        "StarMomentum"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Balls: retourner uniforme
        if pool == Pool::Balls {
            return uniform;
        }

        // Stars: filtrer l'ère actuelle
        let draws = filter_star_era(draws);
        if draws.len() < self.min_draws {
            return uniform;
        }

        let n_stars = size; // 12
        let pick = pool.pick_count(); // 2
        let base_freq = pick as f64 / n_stars as f64;

        // Pour chaque étoile: calculer Hurst, EWMA freq, gap
        let mut probs = vec![0.0f64; n_stars];

        for star_idx in 0..n_stars {
            let star_num = (star_idx + 1) as u8;

            // Construire série binaire (ordre chronologique = reversed)
            let series: Vec<f64> = draws.iter().rev()
                .map(|d| if d.stars.contains(&star_num) { 1.0 } else { 0.0 })
                .collect();

            // Calculer Hurst
            let hurst = crate::research::dfa::compute_hurst_exponent(&series)
                .unwrap_or(0.5);

            // Fréquence EWMA récente (draws[0] = plus récent)
            let mut ewma_freq = base_freq;
            for d in draws.iter().rev() {
                let val = if d.stars.contains(&star_num) { 1.0 } else { 0.0 };
                ewma_freq = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * ewma_freq;
            }

            // Gap actuel (nombre de tirages depuis la dernière apparition)
            let current_gap = draws.iter()
                .position(|d| d.stars.contains(&star_num))
                .unwrap_or(draws.len());

            // Gap moyen
            let total_appearances = series.iter().sum::<f64>();
            let mean_gap = if total_appearances > 1.0 {
                draws.len() as f64 / total_appearances
            } else {
                draws.len() as f64
            };

            // Fréquence globale
            let global_freq = total_appearances / draws.len() as f64;

            if hurst > self.persistence_threshold {
                // Persistant: boost la fréquence récente EWMA
                probs[star_idx] = ewma_freq * (1.0 + self.momentum_boost);
            } else if hurst < self.reversion_threshold {
                // Anti-persistant: mean-reversion
                if current_gap as f64 > mean_gap {
                    // Gap long → boost proportionnel au retard
                    let gap_ratio = (current_gap as f64 / mean_gap).min(2.0);
                    probs[star_idx] = global_freq * (1.0 + self.reversion_boost * gap_ratio);
                } else {
                    // Gap court → réduction
                    let gap_ratio = (current_gap as f64 / mean_gap).max(0.1);
                    probs[star_idx] = global_freq * gap_ratio;
                }
            } else {
                // Neutre: fréquence globale
                probs[star_idx] = global_freq;
            }
        }

        // Normaliser
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        // Smoothing avec uniforme
        let uniform_val = 1.0 / n_stars as f64;
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        floor_only(&mut probs, PROB_FLOOR_STARS);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("min_draws".into(), self.min_draws as f64),
            ("ewma_alpha".into(), self.ewma_alpha),
            ("momentum_boost".into(), self.momentum_boost),
            ("reversion_boost".into(), self.reversion_boost),
            ("persistence_threshold".into(), self.persistence_threshold),
            ("reversion_threshold".into(), self.reversion_threshold),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }

    fn is_stars_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_star_momentum_stars_valid() {
        let model = StarMomentumModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_star_momentum_balls_uniform() {
        let model = StarMomentumModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Balls should be uniform");
        }
    }

    #[test]
    fn test_star_momentum_few_draws() {
        let model = StarMomentumModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Stars);
        let expected = 1.0 / 12.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10, "Too few draws → uniform");
        }
    }

    #[test]
    fn test_star_momentum_no_negative() {
        let model = StarMomentumModel::default();
        let draws = make_test_draws(200);
        let dist = model.predict(&draws, Pool::Stars);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_star_momentum_deterministic() {
        let model = StarMomentumModel::default();
        let draws = make_test_draws(200);
        let dist1 = model.predict(&draws, Pool::Stars);
        let dist2 = model.predict(&draws, Pool::Stars);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_star_momentum_is_stars_only() {
        let model = StarMomentumModel::default();
        assert!(model.is_stars_only());
    }

    #[test]
    fn test_star_momentum_sparse_strategy() {
        let model = StarMomentumModel::default();
        assert!(matches!(model.sampling_strategy(), SamplingStrategy::Sparse { span_multiplier: 3 }));
    }
}
