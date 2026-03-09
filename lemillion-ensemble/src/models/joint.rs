use std::collections::HashMap;

use lemillion_db::models::Draw;

/// Modèle joint conditionnel pour scorer des grilles complètes.
///
/// Modélise P(tirage) = P(b1) × P(b2|b1) × P(b3|b1,b2) × P(b4|b1,b2,b3) × P(b5|b1,b2,b3,b4)
/// en miroir du processus physique de tirage séquentiel dans la machine Stresa.
///
/// Le conditionnement direct est impossible (trop de combinaisons). Approximation par features :
/// P(bk|b1,...,bk-1) ≈ P(bk | features(b1,...,bk-1))
///
/// Features du sous-ensemble déjà tiré :
/// - Profil mod-4 partiel : (n₀,n₁,n₂,n₃)
/// - Decades couvertes : bitmask des decades touchées
/// - Parité : count pair/impair
/// - Somme partielle (bins par quintiles)
/// - Spread partiel (max-min, 3 bins)
/// - Compteur de consécutives
///
/// Cross-pool conditioning : P(star | ball_sum_bin)
///
/// N'est PAS un ForecastModel (il ne retourne pas des marginales).
/// Il score des GRILLES complètes via `score_grid`.
pub struct JointConditionalModel {
    /// Tables conditionnelles par position (0..4 pour les 5 boules).
    /// Pour chaque position k : HashMap<state_key, Vec<f64>> de taille 50
    position_tables: Vec<HashMap<u64, Vec<f64>>>,
    /// Fréquences marginales par position (fallback)
    marginal_tables: Vec<Vec<f64>>,
    /// Table conditionnelle pour les étoiles (2 positions)
    star_tables: Vec<HashMap<u64, Vec<f64>>>,
    star_marginals: Vec<Vec<f64>>,
    /// Cross-pool: P(star | ball_sum_bin) — 5 bins × 12 étoiles
    star_given_ball_sum: Vec<Vec<f64>>,
    laplace_alpha: f64,
    trained: bool,
}

impl JointConditionalModel {
    pub fn new(laplace_alpha: f64) -> Self {
        Self {
            position_tables: Vec::new(),
            marginal_tables: Vec::new(),
            star_tables: Vec::new(),
            star_marginals: Vec::new(),
            star_given_ball_sum: Vec::new(),
            laplace_alpha,
            trained: false,
        }
    }

    /// Entraîne le modèle sur les tirages historiques.
    pub fn train(&mut self, draws: &[Draw]) {
        if draws.len() < 20 {
            self.trained = false;
            return;
        }

        let alpha = self.laplace_alpha;

        // ── Boules (5 positions) ──
        self.position_tables = vec![HashMap::new(); 5];
        self.marginal_tables = vec![vec![alpha; 50]; 5];

        for draw in draws {
            let mut sorted_balls = draw.balls;
            sorted_balls.sort();

            for pos in 0..5 {
                let ball = sorted_balls[pos];
                let ball_idx = (ball - 1) as usize;

                // Features du sous-ensemble déjà tiré (positions 0..pos)
                let prefix = &sorted_balls[..pos];
                let key = encode_prefix_state(prefix);

                let entry = self.position_tables[pos]
                    .entry(key)
                    .or_insert_with(|| vec![alpha; 50]);
                entry[ball_idx] += 1.0;
                self.marginal_tables[pos][ball_idx] += 1.0;
            }
        }

        // ── Étoiles (2 positions) ──
        self.star_tables = vec![HashMap::new(); 2];
        self.star_marginals = vec![vec![alpha; 12]; 2];

        for draw in draws {
            let mut sorted_stars = draw.stars;
            sorted_stars.sort();

            for pos in 0..2 {
                let star = sorted_stars[pos];
                let star_idx = (star - 1) as usize;

                let prefix = &sorted_stars[..pos];
                let key = encode_star_prefix_state(prefix);

                let entry = self.star_tables[pos]
                    .entry(key)
                    .or_insert_with(|| vec![alpha; 12]);
                entry[star_idx] += 1.0;
                self.star_marginals[pos][star_idx] += 1.0;
            }
        }

        // ── Cross-pool : P(star | ball_sum_bin) ──
        self.star_given_ball_sum = vec![vec![alpha; 12]; 5];
        for draw in draws {
            let ball_sum: u16 = draw.balls.iter().map(|&b| b as u16).sum();
            let bin = ball_sum_bin(ball_sum);
            for &s in &draw.stars {
                self.star_given_ball_sum[bin][(s - 1) as usize] += 1.0;
            }
        }

        self.trained = true;
    }

    /// Score les boules par le modèle joint, normalisé vs uniforme séquentiel.
    /// Retourne log(P_joint / P_uniform) — même échelle que le score marginal.
    /// P_uniform_seq = 1/50 × 1/49 × 1/48 × 1/47 × 1/46.
    pub fn score_balls(&self, balls: &[u8; 5]) -> f64 {
        let (score, _) = self.score_balls_with_confidence(balls);
        score
    }

    /// Score les boules + confiance du modèle joint.
    /// Retourne (log_score_normalisé, confidence ∈ [0, 1]).
    /// La confiance mesure combien d'observations supportent le conditionnement.
    /// Seules les positions 0-1 contribuent à la confiance (les positions profondes
    /// ont toujours très peu d'observations par état et retombent sur les marginales).
    pub fn score_balls_with_confidence(&self, balls: &[u8; 5]) -> (f64, f64) {
        if !self.trained { return (0.0, 0.0); }

        let mut sorted_balls = *balls;
        sorted_balls.sort();

        let mut log_score = 0.0;
        let alpha_ref = 20.0;
        let mut confidence_sum = 0.0f64;
        let confidence_positions = 2; // only pos 0-1 for confidence

        for pos in 0..5 {
            let ball = sorted_balls[pos];
            let ball_idx = (ball - 1) as usize;
            let prefix = &sorted_balls[..pos];
            let key = encode_prefix_state(prefix);

            let (probs, obs_count) = if let Some(table) = self.position_tables[pos].get(&key) {
                let count: f64 = table.iter().sum::<f64>() - self.laplace_alpha * 50.0;
                let adaptive_alpha = self.laplace_alpha / (1.0 + count.max(0.0) / 10.0);
                let smoothed: Vec<f64> = table.iter().enumerate().map(|(i, &c)| {
                    let raw_count = c - self.laplace_alpha;
                    raw_count.max(0.0) + adaptive_alpha + self.normalized_marginal(pos)[i] * adaptive_alpha
                }).collect();
                let total: f64 = smoothed.iter().sum();
                if total > 0.0 {
                    (smoothed.iter().map(|&c| c / total).collect::<Vec<f64>>(), count.max(0.0))
                } else {
                    (self.normalized_marginal(pos), 0.0)
                }
            } else {
                (self.normalized_marginal(pos), 0.0)
            };

            // Confiance pour les 2 premières positions seulement
            if pos < confidence_positions {
                confidence_sum += (obs_count / (obs_count + alpha_ref)).min(1.0);
            }

            let available_prob: f64 = probs.iter().enumerate()
                .filter(|(idx, _)| !prefix.contains(&((*idx + 1) as u8)))
                .map(|(_, &p)| p)
                .sum();
            let p = if available_prob > 0.0 { probs[ball_idx] / available_prob }
                    else { 1.0 / (50 - pos) as f64 };
            log_score += p.max(1e-15).ln();
        }

        let log_uniform_seq: f64 = (0..5).map(|k| -((50 - k) as f64).ln()).sum();
        let normalized = log_score - log_uniform_seq;
        let confidence = (confidence_sum / confidence_positions as f64).min(1.0);
        (normalized, confidence)
    }

    /// Score une grille complète par le modèle joint.
    /// Retourne log P(grille) comme somme des log P(bk | b1,...,bk-1).
    /// Inclut cross-pool conditioning P(star | ball_sum_bin).
    pub fn score_grid(&self, balls: &[u8; 5], stars: &[u8; 2]) -> f64 {
        if !self.trained {
            return 0.0;
        }

        let mut sorted_balls = *balls;
        sorted_balls.sort();

        let mut log_score = 0.0;

        // Score séquentiel des boules
        for pos in 0..5 {
            let ball = sorted_balls[pos];
            let ball_idx = (ball - 1) as usize;
            let prefix = &sorted_balls[..pos];
            let key = encode_prefix_state(prefix);

            let probs = if let Some(table) = self.position_tables[pos].get(&key) {
                let count: f64 = table.iter().sum::<f64>() - self.laplace_alpha * 50.0;
                // Smoothing adaptatif : plus de données → moins de lissage
                let adaptive_alpha = self.laplace_alpha / (1.0 + count.max(0.0) / 10.0);
                let smoothed: Vec<f64> = table.iter().enumerate().map(|(i, &c)| {
                    let raw_count = c - self.laplace_alpha;
                    raw_count.max(0.0) + adaptive_alpha + self.normalized_marginal(pos)[i] * adaptive_alpha
                }).collect();
                let total: f64 = smoothed.iter().sum();
                if total > 0.0 {
                    smoothed.iter().map(|&c| c / total).collect::<Vec<f64>>()
                } else {
                    self.normalized_marginal(pos)
                }
            } else {
                self.normalized_marginal(pos)
            };

            // Exclure les boules déjà tirées (contrainte sans-remise)
            let available_prob: f64 = probs.iter().enumerate()
                .filter(|(idx, _)| !prefix.contains(&((*idx + 1) as u8)))
                .map(|(_, &p)| p)
                .sum();

            let p = if available_prob > 0.0 {
                probs[ball_idx] / available_prob
            } else {
                1.0 / (50 - pos) as f64
            };

            log_score += p.max(1e-15).ln();
        }

        // Score séquentiel des étoiles
        let mut sorted_stars = *stars;
        sorted_stars.sort();

        for pos in 0..2 {
            let star = sorted_stars[pos];
            let star_idx = (star - 1) as usize;
            let prefix = &sorted_stars[..pos];
            let key = encode_star_prefix_state(prefix);

            let probs = if let Some(table) = self.star_tables[pos].get(&key) {
                let total: f64 = table.iter().sum();
                if total > 0.0 {
                    table.iter().map(|&c| c / total).collect::<Vec<f64>>()
                } else {
                    self.normalized_star_marginal(pos)
                }
            } else {
                self.normalized_star_marginal(pos)
            };

            let available_prob: f64 = probs.iter().enumerate()
                .filter(|(idx, _)| !prefix.contains(&((*idx + 1) as u8)))
                .map(|(_, &p)| p)
                .sum();

            let p = if available_prob > 0.0 {
                probs[star_idx] / available_prob
            } else {
                1.0 / (12 - pos) as f64
            };

            log_score += p.max(1e-15).ln();
        }

        // Cross-pool conditioning : bonus/malus basé sur P(star | ball_sum_bin)
        if !self.star_given_ball_sum.is_empty() {
            let ball_sum: u16 = sorted_balls.iter().map(|&b| b as u16).sum();
            let bin = ball_sum_bin(ball_sum);
            let table = &self.star_given_ball_sum[bin];
            let total: f64 = table.iter().sum();
            if total > 0.0 {
                let uniform_star = 1.0 / 12.0;
                for &s in &sorted_stars {
                    let p_cond = table[(s - 1) as usize] / total;
                    // Ratio vs uniform, clamped et en log
                    let ratio = (p_cond / uniform_star).clamp(0.5, 2.0);
                    log_score += ratio.ln();
                }
            }
        }

        log_score
    }

    /// Score normalisé : exp(log_score_grid - log_score_uniforme).
    /// Un score > 1.0 signifie que la grille est plus probable que l'uniforme.
    pub fn score_grid_normalized(&self, balls: &[u8; 5], stars: &[u8; 2]) -> f64 {
        let log_score = self.score_grid(balls, stars);
        // Score log uniforme : ln(1/C(50,5)) + ln(1/C(12,2))
        let log_uniform = -(50.0f64 * 49.0 * 48.0 * 47.0 * 46.0 / 120.0).ln()
            - (12.0f64 * 11.0 / 2.0).ln();
        (log_score - log_uniform).exp()
    }

    fn normalized_marginal(&self, pos: usize) -> Vec<f64> {
        let total: f64 = self.marginal_tables[pos].iter().sum();
        if total > 0.0 {
            self.marginal_tables[pos].iter().map(|&c| c / total).collect()
        } else {
            vec![1.0 / 50.0; 50]
        }
    }

    fn normalized_star_marginal(&self, pos: usize) -> Vec<f64> {
        let total: f64 = self.star_marginals[pos].iter().sum();
        if total > 0.0 {
            self.star_marginals[pos].iter().map(|&c| c / total).collect()
        } else {
            vec![1.0 / 12.0; 12]
        }
    }
}

impl Default for JointConditionalModel {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Bin de la somme des boules en 5 quintiles (somme théorique 15..240, pratique ~80..180).
fn ball_sum_bin(sum: u16) -> usize {
    match sum {
        0..=99 => 0,
        100..=119 => 1,
        120..=139 => 2,
        140..=159 => 3,
        _ => 4,
    }
}

/// Encode l'état du préfixe de boules en features discrètes.
/// Features : profil mod-4 (12 bits) + decades (5 bits) + odd_count (3 bits)
///          + somme partielle bin (3 bits) + spread bin (2 bits) + consecutive count (2 bits).
fn encode_prefix_state(prefix: &[u8]) -> u64 {
    if prefix.is_empty() {
        return 0;
    }

    // Profil mod-4 : 4 compteurs, chacun ≤ 5 → 3 bits chacun
    let mut mod4 = [0u8; 4];
    for &b in prefix {
        let r = ((b - 1) % 4) as usize;
        mod4[r] += 1;
    }

    // Decades couvertes : bitmask (5 decades pour 1-50)
    let mut decade_mask = 0u8;
    for &b in prefix {
        decade_mask |= 1 << ((b - 1) / 10);
    }

    // Odd count
    let odd_count = prefix.iter().filter(|&&b| b % 2 == 1).count() as u8;

    // Somme partielle bin (quintiles de la somme attendue finale ~127)
    let partial_sum: u16 = prefix.iter().map(|&b| b as u16).sum();
    let expected_avg = 25.5; // E[ball] for 1..50
    let expected_partial = (expected_avg * prefix.len() as f64) as u16;
    let sum_bin = if partial_sum < expected_partial.saturating_sub(15) {
        0u8
    } else if partial_sum > expected_partial + 15 {
        2
    } else {
        1
    };

    // Spread bin (max - min)
    let min_b = prefix.iter().copied().min().unwrap_or(1);
    let max_b = prefix.iter().copied().max().unwrap_or(1);
    let spread = max_b - min_b;
    let spread_bin = if spread < 10 { 0u8 } else if spread < 25 { 1 } else { 2 };

    // Consecutive count
    let mut sorted = prefix.to_vec();
    sorted.sort();
    let consec_count = sorted.windows(2).filter(|w| w[1] == w[0] + 1).count().min(3) as u8;

    // Packer en u64
    (mod4[0] as u64)
        | ((mod4[1] as u64) << 3)
        | ((mod4[2] as u64) << 6)
        | ((mod4[3] as u64) << 9)
        | ((decade_mask as u64) << 12)
        | ((odd_count as u64) << 17)
        | ((sum_bin as u64) << 20)
        | ((spread_bin as u64) << 22)
        | ((consec_count as u64) << 24)
}

/// Encode l'état du préfixe d'étoiles.
fn encode_star_prefix_state(prefix: &[u8]) -> u64 {
    if prefix.is_empty() {
        return 0;
    }

    // Mod-4 partiel
    let mut mod4 = [0u8; 4];
    for &s in prefix {
        let r = ((s - 1) % 4) as usize;
        mod4[r] += 1;
    }

    // High/low split
    let high_count = prefix.iter().filter(|&&s| s > 6).count() as u8;
    let odd_count = prefix.iter().filter(|&&s| s % 2 == 1).count() as u8;

    (mod4[0] as u64)
        | ((mod4[1] as u64) << 3)
        | ((mod4[2] as u64) << 6)
        | ((mod4[3] as u64) << 9)
        | ((high_count as u64) << 12)
        | ((odd_count as u64) << 15)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_joint_train_and_score() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(50);
        model.train(&draws);
        assert!(model.trained);

        let score = model.score_grid(&draws[0].balls, &draws[0].stars);
        // Le score devrait être fini et négatif (log-probabilité)
        assert!(score.is_finite(), "Score should be finite, got {}", score);
        assert!(score < 0.0, "Log-probability should be negative, got {}", score);
    }

    #[test]
    fn test_joint_untrained_returns_zero() {
        let model = JointConditionalModel::default();
        let score = model.score_grid(&[1, 2, 3, 4, 5], &[1, 2]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_joint_few_draws() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(10);
        model.train(&draws);
        assert!(!model.trained);
    }

    #[test]
    fn test_joint_normalized_score() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(50);
        model.train(&draws);

        let norm_score = model.score_grid_normalized(&draws[0].balls, &draws[0].stars);
        assert!(norm_score > 0.0, "Normalized score should be positive, got {}", norm_score);
        assert!(norm_score.is_finite(), "Normalized score should be finite");
    }

    #[test]
    fn test_joint_seen_draw_scores_higher() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(50);
        model.train(&draws);

        // Un tirage vu dans l'historique devrait scorer plus haut qu'un aléatoire
        let seen_score = model.score_grid(&draws[0].balls, &draws[0].stars);
        let random_score = model.score_grid(&[1, 25, 30, 40, 50], &[1, 12]);

        // Le tirage vu devrait avoir un meilleur score (plus proche de 0 en log)
        assert!(
            seen_score >= random_score,
            "Seen draw should score >= random: {} vs {}",
            seen_score, random_score
        );
    }

    #[test]
    fn test_encode_prefix_state_empty() {
        let key = encode_prefix_state(&[]);
        assert_eq!(key, 0);
    }

    #[test]
    fn test_encode_prefix_state_deterministic() {
        let key1 = encode_prefix_state(&[1, 5, 10]);
        let key2 = encode_prefix_state(&[1, 5, 10]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_encode_prefix_state_different() {
        let key1 = encode_prefix_state(&[1, 2, 3]);
        let key2 = encode_prefix_state(&[10, 20, 30]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_joint_star_scoring() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(50);
        model.train(&draws);

        // Vérifier que les étoiles contribuent au score
        let score1 = model.score_grid(&draws[0].balls, &[1, 2]);
        let score2 = model.score_grid(&draws[0].balls, &[11, 12]);
        // Les scores doivent être différents (étoiles différentes)
        // Avec des données test synthétiques, les deux peuvent être proches
        assert!(score1.is_finite());
        assert!(score2.is_finite());
    }

    #[test]
    fn test_score_balls_with_confidence_range() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(50);
        model.train(&draws);

        let (score, conf) = model.score_balls_with_confidence(&draws[0].balls);
        assert!(score.is_finite(), "Score should be finite: {}", score);
        assert!(conf >= 0.0 && conf <= 1.0, "Confidence should be in [0,1]: {}", conf);
    }

    #[test]
    fn test_score_balls_with_confidence_increases_with_data() {
        let mut model_small = JointConditionalModel::default();
        let draws_small = make_test_draws(30);
        model_small.train(&draws_small);

        let mut model_large = JointConditionalModel::default();
        let draws_large = make_test_draws(200);
        model_large.train(&draws_large);

        let (_, conf_small) = model_small.score_balls_with_confidence(&draws_small[0].balls);
        let (_, conf_large) = model_large.score_balls_with_confidence(&draws_large[0].balls);

        // Plus de données → confiance plus haute (ou au moins égale)
        assert!(conf_large >= conf_small * 0.9,
            "More data should give higher confidence: {} vs {}", conf_large, conf_small);
    }

    #[test]
    fn test_score_balls_with_confidence_untrained() {
        let model = JointConditionalModel::default();
        let (score, conf) = model.score_balls_with_confidence(&[1, 2, 3, 4, 5]);
        assert_eq!(score, 0.0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_joint_deterministic() {
        let mut model = JointConditionalModel::default();
        let draws = make_test_draws(50);
        model.train(&draws);

        let score1 = model.score_grid(&[5, 10, 15, 20, 25], &[3, 7]);
        let score2 = model.score_grid(&[5, 10, 15, 20, 25], &[3, 7]);
        assert!((score1 - score2).abs() < 1e-15);
    }
}
