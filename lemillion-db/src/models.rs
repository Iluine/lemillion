use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Prize tier data for a single tier (rang) of EuroMillions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrizeTier {
    /// Tier rank (1 = jackpot 5+2, 13 = lowest 2+0)
    pub rank: u8,
    /// Number of winners in France
    pub winners_fr: i32,
    /// Number of winners in Europe
    pub winners_eu: i32,
    /// Prize per winner (euros)
    pub prize: f64,
}

#[derive(Debug, Clone)]
pub struct Draw {
    pub draw_id: String,
    pub day: String,
    pub date: String,
    pub balls: [u8; 5],
    pub stars: [u8; 2],
    pub winner_count: i32,
    pub winner_prize: f64,
    pub my_million: String,
    /// Ordre d'extraction physique des boules (machine Stresa). None pour les draws ajoutés manuellement.
    pub ball_order: Option<[u8; 5]>,
    /// Ordre d'extraction physique des étoiles. None pour les draws ajoutés manuellement.
    pub star_order: Option<[u8; 2]>,
    /// Numéro de tirage dans le cycle (colonne 3 du CSV FDJ).
    pub cycle_number: Option<u8>,
    /// v23: Full prize tier data (13 tiers) — winners and prizes per tier.
    pub prize_tiers: Option<Vec<PrizeTier>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    Balls,
    Stars,
}

impl Pool {
    pub fn size(&self) -> usize {
        match self {
            Pool::Balls => 50,
            Pool::Stars => 12,
        }
    }

    pub fn pick_count(&self) -> usize {
        match self {
            Pool::Balls => 5,
            Pool::Stars => 2,
        }
    }

    pub fn numbers_from<'a>(&self, draw: &'a Draw) -> &'a [u8] {
        match self {
            Pool::Balls => &draw.balls,
            Pool::Stars => &draw.stars,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumberStats {
    pub number: u8,
    pub frequency: u32,
    pub gap: u32,
}

#[derive(Debug, Clone)]
pub struct NumberProbability {
    pub number: u8,
    pub probability: f64,
    pub tag: ProbabilityTag,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProbabilityTag {
    Hot,
    Cold,
    Normal,
}

impl std::fmt::Display for ProbabilityTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbabilityTag::Hot => write!(f, "HOT"),
            ProbabilityTag::Cold => write!(f, "COLD"),
            ProbabilityTag::Normal => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Suggestion {
    pub balls: [u8; 5],
    pub stars: [u8; 2],
    pub score: f64,
}

impl Draw {
    /// Trie balls et stars en ordre croissant (défensif contre les données DB non triées).
    pub fn normalize(&mut self) {
        self.balls.sort();
        self.stars.sort();
    }
}

pub fn validate_draw(balls: &[u8; 5], stars: &[u8; 2]) -> Result<()> {
    for &b in balls {
        if !(1..=50).contains(&b) {
            bail!("Boule {} hors limites (1-50)", b);
        }
    }
    for &s in stars {
        if !(1..=12).contains(&s) {
            bail!("Étoile {} hors limites (1-12)", s);
        }
    }
    for i in 0..balls.len() {
        for j in (i + 1)..balls.len() {
            if balls[i] == balls[j] {
                bail!("Boule en double : {}", balls[i]);
            }
        }
    }
    if stars[0] == stars[1] {
        bail!("Étoile en double : {}", stars[0]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_draw_ok() {
        assert!(validate_draw(&[1, 2, 3, 4, 5], &[1, 2]).is_ok());
        assert!(validate_draw(&[50, 49, 48, 47, 46], &[11, 12]).is_ok());
    }

    #[test]
    fn test_validate_draw_ball_out_of_range() {
        assert!(validate_draw(&[0, 2, 3, 4, 5], &[1, 2]).is_err());
        assert!(validate_draw(&[1, 2, 3, 4, 51], &[1, 2]).is_err());
    }

    #[test]
    fn test_validate_draw_star_out_of_range() {
        assert!(validate_draw(&[1, 2, 3, 4, 5], &[0, 2]).is_err());
        assert!(validate_draw(&[1, 2, 3, 4, 5], &[1, 13]).is_err());
    }

    #[test]
    fn test_validate_draw_duplicate_balls() {
        assert!(validate_draw(&[1, 1, 3, 4, 5], &[1, 2]).is_err());
    }

    #[test]
    fn test_validate_draw_duplicate_stars() {
        assert!(validate_draw(&[1, 2, 3, 4, 5], &[3, 3]).is_err());
    }

    #[test]
    fn test_pool_size() {
        assert_eq!(Pool::Balls.size(), 50);
        assert_eq!(Pool::Stars.size(), 12);
    }

    #[test]
    fn test_pool_pick_count() {
        assert_eq!(Pool::Balls.pick_count(), 5);
        assert_eq!(Pool::Stars.pick_count(), 2);
    }

    #[test]
    fn test_pool_numbers_from() {
        let draw = Draw {
            draw_id: "001".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-01".to_string(),
            balls: [1, 2, 3, 4, 5],
            stars: [6, 7],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        prize_tiers: None,
        };
        assert_eq!(Pool::Balls.numbers_from(&draw), &[1, 2, 3, 4, 5]);
        assert_eq!(Pool::Stars.numbers_from(&draw), &[6, 7]);
    }
}
