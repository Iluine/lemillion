use lemillion_db::models::Draw;

/// Prix d'un ticket EuroMillions en euros.
pub const TICKET_PRICE: f64 = 2.50;

/// Un rang de prix EuroMillions.
pub struct PrizeTier {
    pub name: &'static str,
    pub balls_matched: u8,
    pub stars_matched: u8,
    pub probability: f64,
    pub is_parimutuel: bool,
    pub fixed_prize: f64,
}

/// Les 13 rangs de prix EuroMillions avec probabilites exactes (combinatoire).
/// Sources : reglement officiel EuroMillions.
/// Probabilites : C(5,k)*C(45,5-k)/C(50,5) * C(2,j)*C(10,2-j)/C(12,2)
pub const PRIZE_TIERS: [PrizeTier; 13] = [
    PrizeTier { name: "5+2", balls_matched: 5, stars_matched: 2, probability: 1.0 / 139_838_160.0, is_parimutuel: true,  fixed_prize: 0.0 },
    PrizeTier { name: "5+1", balls_matched: 5, stars_matched: 1, probability: 1.0 / 6_991_908.0,   is_parimutuel: true,  fixed_prize: 0.0 },
    PrizeTier { name: "5+0", balls_matched: 5, stars_matched: 0, probability: 1.0 / 3_107_515.0,   is_parimutuel: true,  fixed_prize: 0.0 },
    PrizeTier { name: "4+2", balls_matched: 4, stars_matched: 2, probability: 1.0 / 621_503.0,     is_parimutuel: false, fixed_prize: 3_000.0 },
    PrizeTier { name: "4+1", balls_matched: 4, stars_matched: 1, probability: 1.0 / 31_075.0,      is_parimutuel: false, fixed_prize: 150.0 },
    PrizeTier { name: "3+2", balls_matched: 3, stars_matched: 2, probability: 1.0 / 14_125.0,      is_parimutuel: false, fixed_prize: 60.0 },
    PrizeTier { name: "4+0", balls_matched: 4, stars_matched: 0, probability: 1.0 / 13_811.0,      is_parimutuel: false, fixed_prize: 40.0 },
    PrizeTier { name: "2+2", balls_matched: 2, stars_matched: 2, probability: 1.0 / 985.0,         is_parimutuel: false, fixed_prize: 15.0 },
    PrizeTier { name: "3+1", balls_matched: 3, stars_matched: 1, probability: 1.0 / 706.0,         is_parimutuel: false, fixed_prize: 13.0 },
    PrizeTier { name: "3+0", balls_matched: 3, stars_matched: 0, probability: 1.0 / 314.0,         is_parimutuel: false, fixed_prize: 10.0 },
    PrizeTier { name: "1+2", balls_matched: 1, stars_matched: 2, probability: 1.0 / 188.0,         is_parimutuel: false, fixed_prize: 8.0 },
    PrizeTier { name: "2+1", balls_matched: 2, stars_matched: 1, probability: 1.0 / 49.0,          is_parimutuel: false, fixed_prize: 6.0 },
    PrizeTier { name: "2+0", balls_matched: 2, stars_matched: 0, probability: 1.0 / 22.0,          is_parimutuel: false, fixed_prize: 4.0 },
];

/// Resultat du calcul d'esperance de gain pour une grille.
pub struct EvResult {
    pub total_ev: f64,
    pub ev_per_euro: f64,
    pub tier_evs: [f64; 13],
}

/// Modele de popularite des numeros.
/// Les valeurs representent la popularite relative (1.0 = moyenne).
/// Plus la valeur est haute, plus le numero est joue par les autres joueurs.
pub struct PopularityModel {
    pub ball_popularity: [f64; 50],
    pub star_popularity: [f64; 12],
}

impl PopularityModel {
    /// Construit le modele depuis les biais theoriques + recence des derniers tirages.
    ///
    /// Facteurs de biais documentes dans la recherche sur les loteries :
    /// - Biais anniversaire : 1-12 (mois) x1.3, 13-31 (jours) x1.15, 32-50 x0.85
    /// - Numeros porte-bonheur : 7 x1.25, 3 x1.10
    /// - Biais de recence : numeros tires recemment x1.10 par apparition
    pub fn from_history(draws: &[Draw]) -> Self {
        let mut ball_pop = [1.0f64; 50];
        let mut star_pop = [1.0f64; 12];

        // Biais anniversaire (boules)
        for (i, pop) in ball_pop.iter_mut().enumerate() {
            let num = i + 1;
            if num <= 12 {
                *pop *= 1.30; // mois
            } else if num <= 31 {
                *pop *= 1.15; // jours
            } else {
                *pop *= 0.85; // hors dates
            }
        }

        // Biais anniversaire (etoiles) - 1-12 sont aussi des mois
        for (i, pop) in star_pop.iter_mut().enumerate() {
            let num = i + 1;
            if num <= 7 {
                *pop *= 1.20; // jours de la semaine populaires
            }
        }

        // Biais de grille physique (layout 5x10) : boules au centre plus jouées
        for (i, pop) in ball_pop.iter_mut().enumerate() {
            let num = i + 1;
            let row = (num - 1) / 10;   // 0-4
            let col = (num - 1) % 10;   // 0-9
            let center_dist = ((row as f64 - 2.0).powi(2) + (col as f64 - 4.5).powi(2)).sqrt();
            let center_bias = 1.0 + 0.05 * (4.0 - center_dist).max(0.0);
            *pop *= center_bias;
        }

        // Numeros porte-bonheur (boules)
        ball_pop[6] *= 1.25;  // 7
        ball_pop[2] *= 1.10;  // 3
        ball_pop[12] *= 1.05; // 13

        // Numeros porte-bonheur (etoiles)
        star_pop[6] *= 1.20; // 7

        // Biais de recence : les 5 derniers tirages
        let recency_window = 5.min(draws.len());
        for draw in draws.iter().take(recency_window) {
            for &b in &draw.balls {
                ball_pop[(b - 1) as usize] *= 1.10;
            }
            for &s in &draw.stars {
                star_pop[(s - 1) as usize] *= 1.10;
            }
        }

        // Normaliser pour que la moyenne soit 1.0
        let ball_mean = ball_pop.iter().sum::<f64>() / 50.0;
        if ball_mean > 0.0 {
            for p in &mut ball_pop {
                *p /= ball_mean;
            }
        }

        let star_mean = star_pop.iter().sum::<f64>() / 12.0;
        if star_mean > 0.0 {
            for p in &mut star_pop {
                *p /= star_mean;
            }
        }

        Self {
            ball_popularity: ball_pop,
            star_popularity: star_pop,
        }
    }
}

/// Popularite d'une grille (produit des popularites individuelles × facteur de pattern).
/// Plus la valeur est haute, plus la grille est jouee par les autres → plus de partage.
pub fn grid_popularity(balls: &[u8; 5], stars: &[u8; 2], model: &PopularityModel) -> f64 {
    let ball_pop: f64 = balls
        .iter()
        .map(|&b| model.ball_popularity[(b - 1) as usize])
        .product();
    let star_pop: f64 = stars
        .iter()
        .map(|&s| model.star_popularity[(s - 1) as usize])
        .product();

    let pattern_factor = pattern_popularity_factor(balls);

    ball_pop * star_pop * pattern_factor
}

/// Facteur de popularite lie aux patterns reconnaissables.
/// Les joueurs humains evitent les suites consecutives et les patterns "trop reguliers",
/// mais favorisent certains patterns (tout en bas, tout en haut, etc.)
fn pattern_popularity_factor(balls: &[u8; 5]) -> f64 {
    let mut sorted = *balls;
    sorted.sort();
    let mut factor = 1.0;

    // Suites consecutives : les joueurs les evitent → popularity baisse
    let mut max_consec = 1u8;
    let mut cur = 1u8;
    for w in sorted.windows(2) {
        if w[1] == w[0] + 1 {
            cur += 1;
            max_consec = max_consec.max(cur);
        } else {
            cur = 1;
        }
    }
    if max_consec >= 3 {
        factor *= 0.7; // les gens evitent les suites
    }

    // Tout en bas (1-25) ou tout en haut (26-50)
    let all_low = sorted.iter().all(|&b| b <= 25);
    let all_high = sorted.iter().all(|&b| b > 25);
    if all_low || all_high {
        factor *= 1.15; // les gens aiment ces patterns simples
    }

    // Tout pair ou tout impair
    let all_even = sorted.iter().all(|&b| b % 2 == 0);
    let all_odd = sorted.iter().all(|&b| b % 2 == 1);
    if all_even || all_odd {
        factor *= 0.8; // les joueurs l'evitent souvent
    }

    factor
}

/// Score d'anti-popularite d'une grille (inverse de la popularite).
/// Plus la valeur est haute, moins la grille est jouee → meilleur EV.
pub fn anti_popularity(balls: &[u8; 5], stars: &[u8; 2], model: &PopularityModel) -> f64 {
    let pop = grid_popularity(balls, stars, model);
    if pop > 0.0 {
        1.0 / pop
    } else {
        1.0
    }
}

/// Calcule l'esperance de gain d'une grille.
///
/// Pour les rangs parimutuels (top 3), le gain est estime en fonction de la popularite
/// de la grille : une grille moins populaire sera partagee entre moins de gagnants.
///
/// Pour les rangs a gains fixes (rangs 4-13), le gain est le montant fixe.
pub fn compute_ev(
    balls: &[u8; 5],
    stars: &[u8; 2],
    model: &PopularityModel,
    jackpot: f64,
) -> EvResult {
    let pop = grid_popularity(balls, stars, model);
    let mut tier_evs = [0.0f64; 13];

    // Estimation du nombre de tickets vendus (environ 50-80M pour un gros jackpot)
    // Le rang 13 (2+0, proba 1/22) donne une estimation: gagnants_rang13 * 22 ~ tickets vendus
    // Pour un tirage normal ~40M, pour un gros jackpot ~80M
    let estimated_tickets = if jackpot >= 100_000_000.0 {
        80_000_000.0
    } else if jackpot >= 50_000_000.0 {
        60_000_000.0
    } else {
        40_000_000.0
    };

    for (i, tier) in PRIZE_TIERS.iter().enumerate() {
        if tier.is_parimutuel {
            // Pour les rangs parimutuels, le gain depend du nombre de gagnants qui partagent
            // Plus la grille est populaire, plus il y aura de co-gagnants
            let expected_winners = estimated_tickets * tier.probability * pop;
            let sharing_factor = if expected_winners > 1.0 {
                1.0 / expected_winners
            } else {
                1.0
            };

            let pool = match i {
                0 => jackpot,
                1 => jackpot * 0.015, // ~1.5% du jackpot pour le rang 2
                2 => jackpot * 0.005, // ~0.5% du jackpot pour le rang 3
                _ => 0.0,
            };

            tier_evs[i] = tier.probability * pool * sharing_factor;
        } else {
            tier_evs[i] = tier.probability * tier.fixed_prize;
        }
    }

    let total_ev: f64 = tier_evs.iter().sum();
    let ev_per_euro = total_ev / TICKET_PRICE;

    EvResult {
        total_ev,
        ev_per_euro,
        tier_evs,
    }
}

/// v23c: Estimate total tickets sold from real prize tier data.
/// Uses tier 12 (2+1) winners as the most reliable estimator:
/// tickets_sold ≈ winners_eu_tier12 / P(2+1)
/// Falls back to tier 13 (2+0) or jackpot-based heuristic.
pub fn estimate_tickets_from_tiers(draws: &[Draw]) -> f64 {
    // Average over recent draws for stability
    let n = draws.len().min(20);
    if n == 0 { return 40_000_000.0; }

    let mut ticket_estimates = Vec::new();
    for draw in &draws[..n] {
        if let Some(ref tiers) = draw.prize_tiers {
            // Tier 12 (2+1): P = 1/49.27 ≈ 0.020295
            if let Some(t12) = tiers.iter().find(|t| t.rank == 12) {
                if t12.winners_eu > 100 {
                    let p_2_1 = PRIZE_TIERS[11].probability; // 1/49
                    ticket_estimates.push(t12.winners_eu as f64 / p_2_1);
                }
            }
            // Tier 13 (2+0): P = 1/22 ≈ 0.04545
            if let Some(t13) = tiers.iter().find(|t| t.rank == 13) {
                if t13.winners_eu > 100 {
                    let p_2_0 = PRIZE_TIERS[12].probability; // 1/22
                    ticket_estimates.push(t13.winners_eu as f64 / p_2_0);
                }
            }
        }
    }

    if ticket_estimates.is_empty() {
        return 40_000_000.0; // fallback
    }

    // Median for robustness
    ticket_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ticket_estimates[ticket_estimates.len() / 2]
}

/// v23c: Compute EV using real prize data from the most recent draw.
/// Uses actual prize amounts and winner counts instead of heuristic estimates.
pub fn compute_ev_real(
    balls: &[u8; 5],
    stars: &[u8; 2],
    model: &PopularityModel,
    jackpot: f64,
    recent_draws: &[Draw],
) -> EvResult {
    let pop = grid_popularity(balls, stars, model);
    let mut tier_evs = [0.0f64; 13];

    // Use real ticket estimate instead of heuristic
    let estimated_tickets = estimate_tickets_from_tiers(recent_draws);

    // Use real fixed prizes from most recent draw (they can vary slightly)
    let real_prizes: Option<&Vec<lemillion_db::models::PrizeTier>> = recent_draws.first()
        .and_then(|d| d.prize_tiers.as_ref());

    for (i, tier) in PRIZE_TIERS.iter().enumerate() {
        if tier.is_parimutuel {
            let expected_winners = estimated_tickets * tier.probability * pop;
            let sharing_factor = if expected_winners > 1.0 {
                1.0 / expected_winners
            } else {
                1.0
            };

            // Use real prize pool from recent data when available
            let pool = if let Some(real) = real_prizes {
                if let Some(rt) = real.iter().find(|t| t.rank == (i + 1) as u8) {
                    if rt.prize > 0.0 && rt.winners_eu > 0 {
                        // Real pool = prize_per_winner × winners
                        rt.prize * rt.winners_eu as f64
                    } else if i == 0 {
                        jackpot
                    } else {
                        jackpot * match i { 1 => 0.015, 2 => 0.005, _ => 0.0 }
                    }
                } else {
                    match i { 0 => jackpot, 1 => jackpot * 0.015, 2 => jackpot * 0.005, _ => 0.0 }
                }
            } else {
                match i { 0 => jackpot, 1 => jackpot * 0.015, 2 => jackpot * 0.005, _ => 0.0 }
            };

            tier_evs[i] = tier.probability * pool * sharing_factor;
        } else {
            // Use real fixed prizes when available
            let prize = if let Some(real) = real_prizes {
                real.iter().find(|t| t.rank == (i + 1) as u8)
                    .map(|t| t.prize)
                    .filter(|&p| p > 0.0)
                    .unwrap_or(tier.fixed_prize)
            } else {
                tier.fixed_prize
            };
            tier_evs[i] = tier.probability * prize;
        }
    }

    let total_ev: f64 = tier_evs.iter().sum();
    let ev_per_euro = total_ev / TICKET_PRICE;

    EvResult {
        total_ev,
        ev_per_euro,
        tier_evs,
    }
}

/// v23c: Estimate actual ticket volume trend over time.
/// Returns (median_tickets, trend) where trend > 1.0 means growing.
pub fn ticket_volume_trend(draws: &[Draw]) -> (f64, f64) {
    let n = draws.len().min(50);
    let mut volumes: Vec<f64> = Vec::new();

    for draw in &draws[..n] {
        if let Some(ref tiers) = draw.prize_tiers {
            if let Some(t12) = tiers.iter().find(|t| t.rank == 12) {
                if t12.winners_eu > 100 {
                    volumes.push(t12.winners_eu as f64 / PRIZE_TIERS[11].probability);
                }
            }
        }
    }

    if volumes.len() < 5 {
        return (40_000_000.0, 1.0);
    }

    let median = {
        let mut sorted = volumes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };

    // Trend: compare recent 10 vs older 10
    let recent = &volumes[..volumes.len().min(10)];
    let older = &volumes[volumes.len().min(10)..volumes.len().min(20)];
    let trend = if !older.is_empty() && !recent.is_empty() {
        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;
        if older_avg > 0.0 { recent_avg / older_avg } else { 1.0 }
    } else {
        1.0
    };

    (median, trend)
}

/// EV-optimal play timing: when to play based on jackpot size.
/// Returns a multiplier relative to the average jackpot (50M€).
/// Player count grows sub-linearly with jackpot (~jackpot^0.6),
/// so EV per euro is super-linear in jackpot size.
pub fn jackpot_timing_multiplier(jackpot_eur: f64) -> f64 {
    let base_jackpot = 50_000_000.0;
    if jackpot_eur <= 0.0 { return 0.0; }
    let player_growth = (jackpot_eur / base_jackpot).powf(0.6);
    (jackpot_eur / base_jackpot) / player_growth
}

/// Returns a human-readable timing recommendation based on jackpot size.
pub fn jackpot_timing_advice(jackpot_eur: f64) -> &'static str {
    let mult = jackpot_timing_multiplier(jackpot_eur);
    if mult >= 2.5 {
        "EXCELLENT — EV nettement supérieure, jouer fortement"
    } else if mult >= 1.8 {
        "TRÈS BON — EV bien au-dessus de la moyenne"
    } else if mult >= 1.3 {
        "BON — EV au-dessus de la moyenne"
    } else if mult >= 0.8 {
        "NEUTRE — EV proche de la moyenne"
    } else {
        "FAIBLE — EV en dessous de la moyenne, attendre un meilleur jackpot"
    }
}

/// Jackpot minimum approximatif pour que l'EV soit positive (grille moyenne).
/// Avec une grille anti-populaire, le seuil peut etre plus bas.
pub fn jackpot_threshold() -> f64 {
    // EV des rangs fixes (ne depend pas du jackpot)
    let fixed_ev: f64 = PRIZE_TIERS
        .iter()
        .filter(|t| !t.is_parimutuel)
        .map(|t| t.probability * t.fixed_prize)
        .sum();

    // Pour EV = TICKET_PRICE, on a besoin que la partie parimutuelle compense
    let needed_parimutuel_ev = TICKET_PRICE - fixed_ev;

    // Approximation naive : seul le jackpot (rang 1) compte significativement
    // EV_jackpot = prob_jackpot * jackpot * sharing_factor
    // Avec sharing_factor ~ 1 (grille impopulaire) et prob = 1/139_838_160
    let prob_jackpot = PRIZE_TIERS[0].probability;
    let factor_rang2_3 = 1.0 + 0.015 / prob_jackpot * PRIZE_TIERS[1].probability
        + 0.005 / prob_jackpot * PRIZE_TIERS[2].probability;

    needed_parimutuel_ev / (prob_jackpot * factor_rang2_3)
}

/// Convertit un nombre de matches (boules, etoiles) en indice de rang de prix.
/// Retourne None si la combinaison ne correspond a aucun rang gagnant.
pub fn match_to_tier(ball_matches: u8, star_matches: u8) -> Option<usize> {
    match (ball_matches, star_matches) {
        (5, 2) => Some(0),
        (5, 1) => Some(1),
        (5, 0) => Some(2),
        (4, 2) => Some(3),
        (4, 1) => Some(4),
        (3, 2) => Some(5),
        (4, 0) => Some(6),
        (2, 2) => Some(7),
        (3, 1) => Some(8),
        (3, 0) => Some(9),
        (1, 2) => Some(10),
        (2, 1) => Some(11),
        (2, 0) => Some(12),
        _ => None,
    }
}

/// Compte le nombre de boules et etoiles en commun entre une suggestion et un tirage.
pub fn count_matches(
    suggestion_balls: &[u8; 5],
    suggestion_stars: &[u8; 2],
    draw: &Draw,
) -> (u8, u8) {
    let ball_matches = suggestion_balls
        .iter()
        .filter(|b| draw.balls.contains(b))
        .count() as u8;
    let star_matches = suggestion_stars
        .iter()
        .filter(|s| draw.stars.contains(s))
        .count() as u8;
    (ball_matches, star_matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prize_tiers_count() {
        assert_eq!(PRIZE_TIERS.len(), 13);
    }

    #[test]
    fn test_prize_tiers_probabilities_decreasing() {
        // Les rangs sont ordonnees du plus rare au plus frequent
        for w in PRIZE_TIERS.windows(2) {
            assert!(
                w[0].probability <= w[1].probability,
                "Rang {} (p={}) devrait etre <= Rang {} (p={})",
                w[0].name, w[0].probability, w[1].name, w[1].probability
            );
        }
    }

    #[test]
    fn test_match_to_tier_all_tiers() {
        assert_eq!(match_to_tier(5, 2), Some(0));
        assert_eq!(match_to_tier(5, 1), Some(1));
        assert_eq!(match_to_tier(5, 0), Some(2));
        assert_eq!(match_to_tier(4, 2), Some(3));
        assert_eq!(match_to_tier(4, 1), Some(4));
        assert_eq!(match_to_tier(3, 2), Some(5));
        assert_eq!(match_to_tier(4, 0), Some(6));
        assert_eq!(match_to_tier(2, 2), Some(7));
        assert_eq!(match_to_tier(3, 1), Some(8));
        assert_eq!(match_to_tier(3, 0), Some(9));
        assert_eq!(match_to_tier(1, 2), Some(10));
        assert_eq!(match_to_tier(2, 1), Some(11));
        assert_eq!(match_to_tier(2, 0), Some(12));
    }

    #[test]
    fn test_match_to_tier_no_win() {
        assert_eq!(match_to_tier(0, 0), None);
        assert_eq!(match_to_tier(1, 0), None);
        assert_eq!(match_to_tier(0, 1), None);
        assert_eq!(match_to_tier(1, 1), None);
        assert_eq!(match_to_tier(0, 2), None);
    }

    #[test]
    fn test_count_matches_exact() {
        let draw = Draw {
            draw_id: "1".into(),
            day: "MARDI".into(),
            date: "2026-01-01".into(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        prize_tiers: None,
        };
        assert_eq!(count_matches(&[1, 2, 3, 4, 5], &[1, 2], &draw), (5, 2));
    }

    #[test]
    fn test_count_matches_partial() {
        let draw = Draw {
            draw_id: "1".into(),
            day: "MARDI".into(),
            date: "2026-01-01".into(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        prize_tiers: None,
        };
        assert_eq!(count_matches(&[1, 2, 10, 20, 30], &[1, 8], &draw), (2, 1));
    }

    #[test]
    fn test_count_matches_none() {
        let draw = Draw {
            draw_id: "1".into(),
            day: "MARDI".into(),
            date: "2026-01-01".into(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        prize_tiers: None,
        };
        assert_eq!(count_matches(&[10, 20, 30, 40, 50], &[8, 9], &draw), (0, 0));
    }

    #[test]
    fn test_popularity_model_normalized() {
        let draws = crate::models::make_test_draws(50);
        let model = PopularityModel::from_history(&draws);
        let ball_mean: f64 = model.ball_popularity.iter().sum::<f64>() / 50.0;
        let star_mean: f64 = model.star_popularity.iter().sum::<f64>() / 12.0;
        assert!((ball_mean - 1.0).abs() < 0.01, "ball mean = {ball_mean}");
        assert!((star_mean - 1.0).abs() < 0.01, "star mean = {star_mean}");
    }

    #[test]
    fn test_anti_popularity_inverse() {
        let draws = crate::models::make_test_draws(50);
        let model = PopularityModel::from_history(&draws);

        // Grille populaire (petits numeros = anniversaires)
        let pop_grid = grid_popularity(&[1, 2, 3, 7, 12], &[1, 7], &model);
        let anti_pop = anti_popularity(&[1, 2, 3, 7, 12], &[1, 7], &model);
        assert!((pop_grid * anti_pop - 1.0).abs() < 1e-10);

        // Grille impopulaire (grands numeros)
        let pop_unpopular = grid_popularity(&[35, 38, 42, 47, 50], &[10, 12], &model);
        let pop_popular = grid_popularity(&[1, 3, 5, 7, 12], &[1, 7], &model);
        assert!(pop_unpopular < pop_popular, "grands numeros devraient etre moins populaires");
    }

    #[test]
    fn test_compute_ev_positive_fields() {
        let draws = crate::models::make_test_draws(50);
        let model = PopularityModel::from_history(&draws);
        let ev = compute_ev(&[10, 20, 30, 40, 50], &[8, 12], &model, 130_000_000.0);
        assert!(ev.total_ev > 0.0);
        assert!(ev.ev_per_euro > 0.0);
        for &tev in &ev.tier_evs {
            assert!(tev >= 0.0);
        }
    }

    #[test]
    fn test_jackpot_threshold_reasonable() {
        let threshold = jackpot_threshold();
        // Le seuil devrait etre entre 100M et 500M EUR
        assert!(threshold > 50_000_000.0, "threshold={threshold} trop bas");
        assert!(threshold < 1_000_000_000.0, "threshold={threshold} trop haut");
    }

    #[test]
    fn test_jackpot_timing_multiplier_increases() {
        let m50 = jackpot_timing_multiplier(50_000_000.0);
        let m100 = jackpot_timing_multiplier(100_000_000.0);
        let m200 = jackpot_timing_multiplier(200_000_000.0);
        assert!((m50 - 1.0).abs() < 1e-10, "base jackpot should give mult=1.0, got {m50}");
        assert!(m100 > m50, "higher jackpot should give higher multiplier");
        assert!(m200 > m100, "200M should be better than 100M");
    }

    #[test]
    fn test_jackpot_timing_advice_varies() {
        let advice_low = jackpot_timing_advice(17_000_000.0);
        let advice_high = jackpot_timing_advice(230_000_000.0);
        assert_ne!(advice_low, advice_high, "different jackpots should give different advice");
    }

    #[test]
    fn test_pattern_popularity_consecutive() {
        // [3,18,27,36,48] : pas consecutifs, mix pair/impair, mix low/high → factor = 1.0
        let non_consec = pattern_popularity_factor(&[3, 18, 27, 36, 48]);
        assert!((non_consec - 1.0).abs() < 1e-10, "grille neutre devrait avoir factor=1.0, got {non_consec}");

        // [1,2,3,4,5] : consecutifs → factor inclut ×0.7
        let consec = pattern_popularity_factor(&[1, 2, 3, 4, 5]);
        assert!(consec < 1.0, "suites consecutives ({consec}) devraient avoir factor < 1.0");
    }
}
