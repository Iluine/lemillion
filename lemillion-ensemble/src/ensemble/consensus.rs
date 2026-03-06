use super::EnsemblePrediction;
use lemillion_db::models::Pool;

#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusCategory {
    StrongPick,
    DivisivePick,
    StrongAvoid,
    Uncertain,
}

impl std::fmt::Display for ConsensusCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsensusCategory::StrongPick => write!(f, "STRONG PICK"),
            ConsensusCategory::DivisivePick => write!(f, "DIVISIVE"),
            ConsensusCategory::StrongAvoid => write!(f, "AVOID"),
            ConsensusCategory::Uncertain => write!(f, "UNCERTAIN"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsensusEntry {
    pub number: u8,
    pub probability: f64,
    pub spread: f64,
    pub category: ConsensusCategory,
    /// Score continu : deviation × confidence. Positif = favorable, négatif = défavorable.
    pub consensus_value: f64,
}

pub fn build_consensus_map(prediction: &EnsemblePrediction, pool: Pool) -> Vec<ConsensusEntry> {
    let uniform = 1.0 / pool.size() as f64;

    let mut spreads: Vec<f64> = prediction.spread.clone();
    spreads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_spread = if spreads.is_empty() {
        0.0
    } else {
        spreads[spreads.len() / 2]
    };

    prediction
        .distribution
        .iter()
        .enumerate()
        .map(|(i, &prob)| {
            let spread = prediction.spread[i];
            let high_prob = prob > uniform;
            let low_spread = spread <= median_spread;

            let category = match (high_prob, low_spread) {
                (true, true) => ConsensusCategory::StrongPick,
                (true, false) => ConsensusCategory::DivisivePick,
                (false, true) => ConsensusCategory::StrongAvoid,
                (false, false) => ConsensusCategory::Uncertain,
            };

            // Score continu : déviation relative × confiance (inverse du spread)
            let deviation = (prob - uniform) / uniform; // -1 à +∞
            let confidence = if median_spread > 0.0 {
                1.0 / (1.0 + spread / median_spread)
            } else {
                1.0
            };
            let consensus_value = deviation * confidence;

            ConsensusEntry {
                number: (i + 1) as u8,
                probability: prob,
                spread,
                category,
                consensus_value,
            }
        })
        .collect()
}

/// Score continu d'une grille contre la consensus map.
/// Somme des consensus_value pour chaque numéro sélectionné.
pub fn consensus_score(
    balls: &[u8; 5],
    stars: &[u8; 2],
    ball_consensus: &[ConsensusEntry],
    star_consensus: &[ConsensusEntry],
) -> f64 {
    let score_entry = |number: u8, entries: &[ConsensusEntry]| -> f64 {
        entries
            .iter()
            .find(|e| e.number == number)
            .map(|e| e.consensus_value)
            .unwrap_or(0.0)
    };
    balls.iter().map(|&b| score_entry(b, ball_consensus)).sum::<f64>()
        + stars.iter().map(|&s| score_entry(s, star_consensus)).sum::<f64>()
}

/// Compute a set of ball numbers to exclude from jackpot enumeration.
/// Returns numbers classified as StrongAvoid with consensus_value < threshold,
/// sorted by most negative first, capped at max_excluded.
pub fn compute_exclusion_set(
    consensus: &[ConsensusEntry],
    threshold: f64,
    max_excluded: usize,
) -> Vec<u8> {
    let mut candidates: Vec<&ConsensusEntry> = consensus
        .iter()
        .filter(|e| e.category == ConsensusCategory::StrongAvoid && e.consensus_value < threshold)
        .collect();

    candidates.sort_by(|a, b| {
        a.consensus_value
            .partial_cmp(&b.consensus_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
        .iter()
        .take(max_excluded)
        .map(|e| e.number)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_map_length() {
        let pred = EnsemblePrediction {
            distribution: vec![1.0 / 50.0; 50],
            model_distributions: vec![],
            spread: vec![0.001; 50],
        };
        let map = build_consensus_map(&pred, Pool::Balls);
        assert_eq!(map.len(), 50);
    }

    #[test]
    fn test_consensus_value_positive_for_high_prob() {
        // Prob above uniform, low spread → positive consensus_value
        let mut dist = vec![1.0 / 50.0; 50];
        dist[0] = 0.05; // 2.5× uniform
        // Renormalize
        let total: f64 = dist.iter().sum();
        let dist: Vec<f64> = dist.iter().map(|&p| p / total).collect();

        let pred = EnsemblePrediction {
            distribution: dist,
            model_distributions: vec![],
            spread: vec![0.001; 50],
        };
        let map = build_consensus_map(&pred, Pool::Balls);
        assert!(map[0].consensus_value > 0.0, "High prob should have positive value: {}", map[0].consensus_value);
    }

    #[test]
    fn test_consensus_value_negative_for_low_prob() {
        let mut dist = vec![1.0 / 50.0; 50];
        dist[0] = 0.005; // 0.25× uniform
        let total: f64 = dist.iter().sum();
        let dist: Vec<f64> = dist.iter().map(|&p| p / total).collect();

        let pred = EnsemblePrediction {
            distribution: dist,
            model_distributions: vec![],
            spread: vec![0.001; 50],
        };
        let map = build_consensus_map(&pred, Pool::Balls);
        assert!(map[0].consensus_value < 0.0, "Low prob should have negative value: {}", map[0].consensus_value);
    }

    #[test]
    fn test_consensus_score_positive_for_good_picks() {
        // Create a prediction where numbers 1-5 have high prob, rest low
        let mut dist = vec![0.01; 50];
        for i in 0..5 { dist[i] = 0.05; }
        let total: f64 = dist.iter().sum();
        let dist: Vec<f64> = dist.iter().map(|&p| p / total).collect();

        let pred = EnsemblePrediction {
            distribution: dist,
            model_distributions: vec![],
            spread: vec![0.001; 50],
        };
        let ball_consensus = build_consensus_map(&pred, Pool::Balls);

        let star_dist = vec![1.0 / 12.0; 12];
        let star_pred = EnsemblePrediction {
            distribution: star_dist,
            model_distributions: vec![],
            spread: vec![0.001; 12],
        };
        let star_consensus = build_consensus_map(&star_pred, Pool::Stars);

        let score = consensus_score(&[1, 2, 3, 4, 5], &[1, 2], &ball_consensus, &star_consensus);
        assert!(score > 0.0, "Good picks should have positive score: {}", score);
    }

    #[test]
    fn test_consensus_score_negative_for_bad_picks() {
        let mut dist = vec![0.025; 50];
        for i in 0..5 { dist[i] = 0.005; } // These are bad
        let total: f64 = dist.iter().sum();
        let dist: Vec<f64> = dist.iter().map(|&p| p / total).collect();

        let pred = EnsemblePrediction {
            distribution: dist,
            model_distributions: vec![],
            spread: vec![0.001; 50],
        };
        let ball_consensus = build_consensus_map(&pred, Pool::Balls);

        let star_dist = vec![1.0 / 12.0; 12];
        let star_pred = EnsemblePrediction {
            distribution: star_dist,
            model_distributions: vec![],
            spread: vec![0.001; 12],
        };
        let star_consensus = build_consensus_map(&star_pred, Pool::Stars);

        let score = consensus_score(&[1, 2, 3, 4, 5], &[1, 2], &ball_consensus, &star_consensus);
        assert!(score < 0.0, "Bad picks should have negative score: {}", score);
    }

    #[test]
    fn test_high_spread_reduces_confidence() {
        // Two numbers with same prob above uniform, but different spreads
        let mut dist = vec![1.0 / 50.0; 50];
        dist[0] = 0.04;
        dist[1] = 0.04;
        let total: f64 = dist.iter().sum();
        let dist: Vec<f64> = dist.iter().map(|&p| p / total).collect();

        let mut spread = vec![0.001; 50];
        spread[0] = 0.001; // Low spread → high confidence
        spread[1] = 0.010; // High spread → low confidence

        let pred = EnsemblePrediction {
            distribution: dist,
            model_distributions: vec![],
            spread,
        };
        let map = build_consensus_map(&pred, Pool::Balls);
        assert!(map[0].consensus_value > map[1].consensus_value,
            "Low spread should give higher consensus value: {} vs {}",
            map[0].consensus_value, map[1].consensus_value);
    }

    #[test]
    fn test_exclusion_set_empty_if_all_positive() {
        let entries: Vec<ConsensusEntry> = (1..=50).map(|i| ConsensusEntry {
            number: i,
            probability: 1.0 / 50.0,
            spread: 0.001,
            category: ConsensusCategory::StrongPick,
            consensus_value: 0.5,
        }).collect();
        let excluded = compute_exclusion_set(&entries, -0.3, 10);
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_exclusion_set_respects_max() {
        let mut entries: Vec<ConsensusEntry> = (1..=50).map(|i| ConsensusEntry {
            number: i,
            probability: 0.01,
            spread: 0.001,
            category: ConsensusCategory::StrongAvoid,
            consensus_value: -0.5 - i as f64 * 0.01,
        }).collect();
        let _ = &mut entries; // suppress
        let excluded = compute_exclusion_set(&entries, -0.3, 5);
        assert_eq!(excluded.len(), 5);
    }

    #[test]
    fn test_exclusion_set_sorted_most_negative_first() {
        let entries = vec![
            ConsensusEntry { number: 10, probability: 0.01, spread: 0.001, category: ConsensusCategory::StrongAvoid, consensus_value: -0.4 },
            ConsensusEntry { number: 20, probability: 0.01, spread: 0.001, category: ConsensusCategory::StrongAvoid, consensus_value: -0.8 },
            ConsensusEntry { number: 30, probability: 0.01, spread: 0.001, category: ConsensusCategory::StrongAvoid, consensus_value: -0.6 },
            ConsensusEntry { number: 5, probability: 0.03, spread: 0.001, category: ConsensusCategory::StrongPick, consensus_value: 0.5 },
        ];
        let excluded = compute_exclusion_set(&entries, -0.3, 10);
        assert_eq!(excluded, vec![20, 30, 10]);
    }
}
