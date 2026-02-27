use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::RngExt;

use crate::expected_value::{PopularityModel, anti_popularity, compute_ev, PRIZE_TIERS};
use crate::sampler::{StructuralFilter, ScoredSuggestion};

/// Statistiques de couverture pour un ensemble de tickets.
pub struct CoverageStats {
    pub unique_balls: usize,
    pub unique_stars: usize,
    pub any_win_probability: f64,
    pub tier_probabilities: [f64; 13],
    pub total_ev: f64,
    pub total_cost: f64,
}

/// Optimise un ensemble de tickets pour maximiser la couverture et l'anti-popularite.
///
/// Algorithme glouton : chaque nouveau ticket maximise :
/// - Couverture de numeros non encore couverts
/// - Anti-popularite individuelle
pub fn optimize_coverage(
    n_tickets: usize,
    popularity: &PopularityModel,
    jackpot: f64,
    draws: &[lemillion_db::models::Draw],
    seed: u64,
) -> Result<Vec<ScoredSuggestion>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let filter = if !draws.is_empty() {
        Some(StructuralFilter::from_history(draws, lemillion_db::models::Pool::Balls))
    } else {
        None
    };

    // Generer un grand pool de candidats
    let n_candidates = n_tickets * 200;
    let candidates = generate_candidate_pool(n_candidates, popularity, filter.as_ref(), &mut rng);

    // Selection gloutonne maximisant couverture + anti-popularite
    let mut selected: Vec<ScoredSuggestion> = Vec::with_capacity(n_tickets);
    let mut covered_balls = [false; 50];
    let mut covered_stars = [false; 12];

    for _ in 0..n_tickets {
        if candidates.is_empty() {
            break;
        }

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, cand) in candidates.iter().enumerate() {
            // Ignorer les candidats deja selectionnes (memes boules+etoiles)
            if selected.iter().any(|s| s.balls == cand.balls && s.stars == cand.stars) {
                continue;
            }

            // Compter les nouveaux numeros couverts
            let new_balls = cand.balls.iter().filter(|&&b| !covered_balls[(b - 1) as usize]).count();
            let new_stars = cand.stars.iter().filter(|&&s| !covered_stars[(s - 1) as usize]).count();

            // Score composite : couverture + anti-popularite
            let coverage_score = new_balls as f64 + new_stars as f64 * 2.0; // etoiles plus rares
            let score = coverage_score * 0.6 + cand.anti_popularity * 0.4;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let chosen = candidates[best_idx].clone();

        // Mettre a jour la couverture
        for &b in &chosen.balls {
            covered_balls[(b - 1) as usize] = true;
        }
        for &s in &chosen.stars {
            covered_stars[(s - 1) as usize] = true;
        }

        selected.push(chosen);
    }

    // Recalculer l'EV pour chaque ticket selectionne
    for s in &mut selected {
        let ev = compute_ev(&s.balls, &s.stars, popularity, jackpot);
        s.ev_per_euro = ev.ev_per_euro;
    }

    Ok(selected)
}

/// Calcule les statistiques de couverture d'un ensemble de tickets.
pub fn compute_coverage_stats(
    tickets: &[ScoredSuggestion],
    popularity: &PopularityModel,
    jackpot: f64,
) -> CoverageStats {
    let mut ball_covered = [false; 50];
    let mut star_covered = [false; 12];

    for t in tickets {
        for &b in &t.balls {
            ball_covered[(b - 1) as usize] = true;
        }
        for &s in &t.stars {
            star_covered[(s - 1) as usize] = true;
        }
    }

    let unique_balls = ball_covered.iter().filter(|&&c| c).count();
    let unique_stars = star_covered.iter().filter(|&&c| c).count();

    // Probabilite de gagner au moins quelque chose avec N tickets
    // P(au moins 1 gain) = 1 - P(0 gain)^N
    // P(0 gain pour 1 ticket) = 1 - sum(prob_tier_i)
    let prob_any_win_single: f64 = PRIZE_TIERS.iter().map(|t| t.probability).sum();
    let prob_no_win_all = (1.0 - prob_any_win_single).powi(tickets.len() as i32);
    let any_win_probability = 1.0 - prob_no_win_all;

    // Probabilite par rang (independante pour chaque ticket, approximation)
    let mut tier_probabilities = [0.0f64; 13];
    for (i, tier) in PRIZE_TIERS.iter().enumerate() {
        tier_probabilities[i] = 1.0 - (1.0 - tier.probability).powi(tickets.len() as i32);
    }

    // EV total
    let total_ev: f64 = tickets
        .iter()
        .map(|t| compute_ev(&t.balls, &t.stars, popularity, jackpot).total_ev)
        .sum();
    let total_cost = tickets.len() as f64 * crate::expected_value::TICKET_PRICE;

    CoverageStats {
        unique_balls,
        unique_stars,
        any_win_probability,
        tier_probabilities,
        total_ev,
        total_cost,
    }
}

/// Genere un pool de candidats diversifies avec scoring anti-popularite.
fn generate_candidate_pool(
    count: usize,
    popularity: &PopularityModel,
    filter: Option<&StructuralFilter>,
    rng: &mut StdRng,
) -> Vec<ScoredSuggestion> {
    let mut candidates = Vec::with_capacity(count);
    let max_attempts = count * 3;
    let mut attempts = 0;

    while candidates.len() < count && attempts < max_attempts {
        attempts += 1;

        // Generer une grille aleatoire uniforme
        let mut balls = [0u8; 5];
        let mut used = [false; 51]; // 1-indexed
        for b in &mut balls {
            loop {
                let num = rng.random_range(1u8..=50);
                if !used[num as usize] {
                    used[num as usize] = true;
                    *b = num;
                    break;
                }
            }
        }
        balls.sort();

        // Filtre structurel
        if let Some(f) = filter
            && !f.accept_balls(&balls)
        {
            continue;
        }

        let mut stars = [0u8; 2];
        let s1: u8 = rng.random_range(1..=12);
        let mut s2: u8 = rng.random_range(1..=12);
        while s2 == s1 {
            s2 = rng.random_range(1..=12);
        }
        stars[0] = s1.min(s2);
        stars[1] = s1.max(s2);

        let ap = anti_popularity(&balls, &stars, popularity);

        candidates.push(ScoredSuggestion {
            balls,
            stars,
            bayesian_score: 0.0, // pas pertinent pour la couverture
            anti_popularity: ap,
            ev_per_euro: 0.0, // calcule apres selection
        });
    }

    // Trier par anti-popularite decroissante
    candidates.sort_by(|a, b| {
        b.anti_popularity
            .partial_cmp(&a.anti_popularity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_coverage_returns_correct_count() {
        let draws = crate::models::make_test_draws(50);
        let model = PopularityModel::from_history(&draws);
        let tickets = optimize_coverage(5, &model, 130_000_000.0, &draws, 42).unwrap();
        assert_eq!(tickets.len(), 5);
    }

    #[test]
    fn test_optimize_coverage_no_duplicates() {
        let draws = crate::models::make_test_draws(50);
        let model = PopularityModel::from_history(&draws);
        let tickets = optimize_coverage(10, &model, 130_000_000.0, &draws, 42).unwrap();

        for i in 0..tickets.len() {
            for j in (i + 1)..tickets.len() {
                assert!(
                    tickets[i].balls != tickets[j].balls || tickets[i].stars != tickets[j].stars,
                    "Tickets {} et {} identiques", i, j
                );
            }
        }
    }

    #[test]
    fn test_coverage_stats_computation() {
        let draws = crate::models::make_test_draws(50);
        let model = PopularityModel::from_history(&draws);
        let tickets = optimize_coverage(10, &model, 130_000_000.0, &draws, 42).unwrap();
        let stats = compute_coverage_stats(&tickets, &model, 130_000_000.0);

        assert!(stats.unique_balls > 0);
        assert!(stats.unique_balls <= 50);
        assert!(stats.unique_stars > 0);
        assert!(stats.unique_stars <= 12);
        assert!(stats.any_win_probability > 0.0);
        assert!(stats.any_win_probability <= 1.0);
        assert!(stats.total_ev > 0.0);
        assert!(stats.total_cost > 0.0);
    }
}
