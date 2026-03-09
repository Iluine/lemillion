use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{ForecastModel, SamplingStrategy, floor_only, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// ForbiddenPatternsModel — Détection de structure via permutation entropy (Bandt & Pompe 2002).
///
/// Les patterns ordinaux interdits dans la série de fréquences d'un numéro
/// signalent de la structure déterministe. Quand PE < seuil et que le pattern
/// courant est rare, le modèle ajuste les probabilités.
pub struct ForbiddenPatternsModel {
    pattern_length: usize,  // d=3 (3! = 6 patterns)
    freq_window: usize,     // rolling frequency window
    smoothing: f64,
    min_draws: usize,
}

impl Default for ForbiddenPatternsModel {
    fn default() -> Self {
        Self {
            pattern_length: 3,
            freq_window: 8,
            smoothing: 0.30,
            min_draws: 60,
        }
    }
}

/// Compute ordinal pattern index for a subsequence of length d.
/// Maps the rank ordering to a unique index in [0, d!).
fn ordinal_pattern_index(values: &[f64]) -> usize {
    let d = values.len();
    // Compute rank of each element
    let mut ranks = vec![0usize; d];
    for i in 0..d {
        for j in 0..d {
            if j != i && (values[j] < values[i] || (values[j] == values[i] && j < i)) {
                ranks[i] += 1;
            }
        }
    }

    // Convert rank permutation to Lehmer code → factorial number system index
    let mut index = 0;
    let mut factorial = 1;
    for i in (0..d).rev() {
        let mut count = 0;
        for j in (i + 1)..d {
            if ranks[j] < ranks[i] {
                count += 1;
            }
        }
        index += count * factorial;
        if i > 0 {
            factorial *= d - i;
        }
    }
    index
}

/// Factorial of n.
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Rolling presence frequency for a given number.
/// Returns a time series of frequency values (oldest-first).
fn rolling_presence(draws: &[Draw], number: u8, pool: Pool, window: usize) -> Vec<f64> {
    let n = draws.len();
    if n < window {
        return vec![];
    }

    // draws[0] = most recent; we process oldest-first
    let mut series = Vec::with_capacity(n - window + 1);
    for start in (0..n - window + 1).rev() {
        let count = draws[start..start + window].iter()
            .filter(|d| {
                let nums = pool.numbers_from(d);
                nums.contains(&number)
            })
            .count();
        series.push(count as f64 / window as f64);
    }
    series
}

impl ForecastModel for ForbiddenPatternsModel {
    fn name(&self) -> &str {
        "ForbiddenPatterns"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = 1.0 / size as f64;
        if draws.len() < self.min_draws {
            return vec![uniform; size];
        }

        let d = self.pattern_length;
        let n_patterns = factorial(d); // 6 for d=3
        let mut probs = vec![uniform; size];

        for num in 0..size {
            // Rolling frequency series for number (num+1)
            let freq_series = rolling_presence(draws, (num + 1) as u8, pool, self.freq_window);
            if freq_series.len() < d + 5 {
                continue;
            }

            // Count ordinal patterns
            let mut counts = vec![0u32; n_patterns];
            let mut total = 0u32;
            for i in 0..freq_series.len() - d + 1 {
                let idx = ordinal_pattern_index(&freq_series[i..i + d]);
                if idx < n_patterns {
                    counts[idx] += 1;
                }
                total += 1;
            }
            if total < 10 {
                continue;
            }

            // Permutation entropy (normalized)
            let pe: f64 = counts.iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total as f64;
                    -p * p.ln()
                }).sum();
            let pe_norm = pe / (n_patterns as f64).ln();

            // Current pattern
            let current_slice = &freq_series[freq_series.len() - d..];
            let current = ordinal_pattern_index(current_slice);
            let expected = total as f64 / n_patterns as f64;
            let rarity = if expected > 0.0 {
                1.0 - (counts[current.min(n_patterns - 1)] as f64 / expected).min(1.0)
            } else {
                0.0
            };

            // Trend (last two frequency values)
            let trend = freq_series[freq_series.len() - 1] - freq_series[freq_series.len() - 2];

            // Signal: si PE < 0.85 (structure détectée) ET pattern rare
            if pe_norm < 0.85 && rarity > 0.3 {
                let signal = 0.2 * rarity * (1.0 - pe_norm);
                if trend > 0.0 {
                    // Rising + rare pattern → likely to reverse → less probable
                    probs[num] = uniform * (1.0 - signal);
                } else {
                    // Falling + rare pattern → likely to bounce → more probable
                    probs[num] = uniform * (1.0 + signal);
                }
            }
        }

        // Normalize + smooth
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
        for p in &mut probs {
            *p = *p * (1.0 - self.smoothing) + uniform * self.smoothing;
        }

        let floor = if pool == Pool::Balls { PROB_FLOOR_BALLS } else { PROB_FLOOR_STARS };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("pattern_length".to_string(), self.pattern_length as f64);
        m.insert("freq_window".to_string(), self.freq_window as f64);
        m.insert("smoothing".to_string(), self.smoothing);
        m.insert("min_draws".to_string(), self.min_draws as f64);
        m
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_forbidden_patterns_valid_distribution() {
        let model = ForbiddenPatternsModel::default();
        let draws = make_test_draws(100);
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            assert_eq!(dist.len(), pool.size());
            let sum: f64 = dist.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "Sum = {} for {:?}", sum, pool);
            assert!(dist.iter().all(|&p| p >= 0.0));
        }
    }

    #[test]
    fn test_forbidden_patterns_insufficient_data() {
        let model = ForbiddenPatternsModel::default();
        let draws = make_test_draws(10);
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ordinal_pattern_index_identity() {
        // [1, 2, 3] = identity permutation → index 0
        let idx = ordinal_pattern_index(&[1.0, 2.0, 3.0]);
        assert_eq!(idx, 0, "Identity permutation should be index 0");
    }

    #[test]
    fn test_ordinal_pattern_index_reverse() {
        // [3, 2, 1] = reverse permutation → last index
        let idx = ordinal_pattern_index(&[3.0, 2.0, 1.0]);
        assert_eq!(idx, 5, "Reverse permutation should be index 5 (3! - 1)");
    }

    #[test]
    fn test_forbidden_patterns_detects_periodic() {
        // Strongly periodic data: balls alternate between two sets
        let draws: Vec<Draw> = (0..120).map(|i| {
            let set = i % 2;
            Draw {
                draw_id: format!("{:03}", i),
                day: "MARDI".to_string(),
                date: format!("2024-01-{:02}", (i % 28) + 1),
                balls: if set == 0 {
                    [1, 2, 3, 4, 5]
                } else {
                    [46, 47, 48, 49, 50]
                },
                stars: [1, 2],
                winner_count: 0,
                winner_prize: 0.0,
                my_million: String::new(),
                ball_order: None,
                star_order: None,
                cycle_number: None,
            }
        }).collect();

        let model = ForbiddenPatternsModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
