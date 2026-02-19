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
}

pub fn build_consensus_map(prediction: &EnsemblePrediction, pool: Pool) -> Vec<ConsensusEntry> {
    // Stub - sera implémenté en Phase 4
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

            ConsensusEntry {
                number: (i + 1) as u8,
                probability: prob,
                spread,
                category,
            }
        })
        .collect()
}

/// Score une grille contre la consensus map.
/// StrongPick=+2, DivisivePick=+1, Uncertain=0, StrongAvoid=-1.
/// Plage : -7 (tout avoid) à +14 (tout strong pick).
pub fn consensus_score(
    balls: &[u8; 5],
    stars: &[u8; 2],
    ball_consensus: &[ConsensusEntry],
    star_consensus: &[ConsensusEntry],
) -> i32 {
    let score_entry = |number: u8, entries: &[ConsensusEntry]| -> i32 {
        entries
            .iter()
            .find(|e| e.number == number)
            .map(|e| match e.category {
                ConsensusCategory::StrongPick => 2,
                ConsensusCategory::DivisivePick => 1,
                ConsensusCategory::Uncertain => 0,
                ConsensusCategory::StrongAvoid => -1,
            })
            .unwrap_or(0)
    };
    balls.iter().map(|&b| score_entry(b, ball_consensus)).sum::<i32>()
        + stars.iter().map(|&s| score_entry(s, star_consensus)).sum::<i32>()
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

    fn make_entries(numbers: &[u8], category: ConsensusCategory) -> Vec<ConsensusEntry> {
        numbers
            .iter()
            .map(|&n| ConsensusEntry {
                number: n,
                probability: 0.03,
                spread: 0.001,
                category: category.clone(),
            })
            .collect()
    }

    #[test]
    fn test_consensus_score_all_strong_pick() {
        let ball_entries = make_entries(&(1..=50).collect::<Vec<u8>>(), ConsensusCategory::StrongPick);
        let star_entries = make_entries(&(1..=12).collect::<Vec<u8>>(), ConsensusCategory::StrongPick);
        let score = consensus_score(&[1, 2, 3, 4, 5], &[1, 2], &ball_entries, &star_entries);
        assert_eq!(score, 14); // 5×2 + 2×2
    }

    #[test]
    fn test_consensus_score_all_avoid() {
        let ball_entries = make_entries(&(1..=50).collect::<Vec<u8>>(), ConsensusCategory::StrongAvoid);
        let star_entries = make_entries(&(1..=12).collect::<Vec<u8>>(), ConsensusCategory::StrongAvoid);
        let score = consensus_score(&[1, 2, 3, 4, 5], &[1, 2], &ball_entries, &star_entries);
        assert_eq!(score, -7); // 5×(-1) + 2×(-1)
    }

    #[test]
    fn test_consensus_score_mixed() {
        let mut entries = make_entries(&[1, 2, 3], ConsensusCategory::StrongPick);
        entries.extend(make_entries(&[4, 5], ConsensusCategory::StrongAvoid));
        let star_entries = make_entries(&[1, 2], ConsensusCategory::DivisivePick);
        let score = consensus_score(&[1, 2, 3, 4, 5], &[1, 2], &entries, &star_entries);
        // 3×2 + 2×(-1) + 2×1 = 6 - 2 + 2 = 6
        assert_eq!(score, 6);
    }
}
