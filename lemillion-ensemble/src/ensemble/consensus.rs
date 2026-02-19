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
}
