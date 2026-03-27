use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering as CmpOrdering;

use anyhow::Result;
use rand::SeedableRng;
use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use serde::{Serialize, Deserialize};

use lemillion_db::models::{Draw, Pool, Suggestion};
use crate::expected_value::{PopularityModel, anti_popularity, compute_ev};
use crate::models::summary_predictor::SummaryPredictor;

fn default_joint_blend() -> f64 { 0.30 }
fn default_k_balls() -> usize { 15 }
fn default_anti_pop_weight() -> f64 { 8.0 }

/// Hyperparamètres optimisables pour le pipeline de prédiction.
/// Sauvegardés/chargés depuis `hyperparams.json` via BayesOpt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    /// Température boules en mode few-grid [0.05, 1.0]
    pub t_balls: f64,
    /// Température étoiles en mode few-grid [0.10, 0.50]
    pub t_stars: f64,
    /// Poids cohérence boules dans le scoring jackpot [5.0, 60.0]
    pub coherence_weight: f64,
    /// Poids cohérence étoiles dans le scoring jackpot [3.0, 40.0]
    pub star_coherence_weight: f64,
    /// Learning rate Hedge [0.01, 0.50]
    pub hedge_eta: f64,
    /// Blend marginal vs joint conditional [0.0, 0.6] — v21
    #[serde(default = "default_joint_blend")]
    pub joint_blend: f64,
    /// K balls subset for jackpot enumeration [12, 35] — v21
    #[serde(default = "default_k_balls")]
    pub k_balls: usize,
    /// v24: Anti-popularity weight in jackpot scoring [0.0, 20.0]
    #[serde(default = "default_anti_pop_weight")]
    pub anti_pop_weight: f64,
}

impl Default for HyperParams {
    fn default() -> Self {
        Self {
            t_balls: 0.55,
            t_stars: 0.25,
            coherence_weight: 30.0,
            star_coherence_weight: 15.0,
            hedge_eta: 0.10,
            joint_blend: 0.30,
            k_balls: 15,
            anti_pop_weight: 8.0,
        }
    }
}

impl HyperParams {
    /// Load from JSON file, or return default if not found.
    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Save to JSON file.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Create from a parameter vector (for BayesOpt).
    /// Order: [t_balls, t_stars, coherence_weight, star_coherence_weight, hedge_eta, joint_blend, k_balls]
    pub fn from_vec(params: &[f64]) -> Self {
        Self {
            t_balls: params.get(0).copied().unwrap_or(0.55),
            t_stars: params.get(1).copied().unwrap_or(0.25),
            coherence_weight: params.get(2).copied().unwrap_or(30.0),
            star_coherence_weight: params.get(3).copied().unwrap_or(15.0),
            hedge_eta: params.get(4).copied().unwrap_or(0.10),
            joint_blend: params.get(5).copied().unwrap_or(0.30),
            k_balls: params.get(6).map(|v| (*v as usize).clamp(12, 35)).unwrap_or(15),
            anti_pop_weight: params.get(7).copied().unwrap_or(8.0),
        }
    }

    /// Convert to parameter vector (for BayesOpt).
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.t_balls,
            self.t_stars,
            self.coherence_weight,
            self.star_coherence_weight,
            self.hedge_eta,
            self.joint_blend,
            self.k_balls as f64,
            self.anti_pop_weight,
        ]
    }
}

/// Filtre structurel rejetant les candidats implausibles avant scoring.
pub struct StructuralFilter {
    pub sum_range: (u16, u16),
    pub max_consecutive: u8,
    pub odd_range: (u8, u8),
    pub spread_range: (u8, u8),
}

impl StructuralFilter {
    /// Construit un filtre depuis l'historique (percentile 2-98%).
    pub fn from_history(draws: &[Draw], pool: Pool) -> Self {
        match pool {
            Pool::Balls => Self::from_ball_history(draws),
            Pool::Stars => Self {
                sum_range: (2, 24),
                max_consecutive: 2,
                odd_range: (0, 2),
                spread_range: (0, 11),
            },
        }
    }

    fn from_ball_history(draws: &[Draw]) -> Self {
        if draws.is_empty() {
            return Self {
                sum_range: (50, 200),
                max_consecutive: 3,
                odd_range: (0, 5),
                spread_range: (10, 49),
            };
        }

        let mut sums: Vec<u16> = draws
            .iter()
            .map(|d| d.balls.iter().map(|&b| b as u16).sum())
            .collect();
        sums.sort();

        let mut odd_counts: Vec<u8> = draws
            .iter()
            .map(|d| d.balls.iter().filter(|&&b| b % 2 == 1).count() as u8)
            .collect();
        odd_counts.sort();

        let mut max_consecs: Vec<u8> = draws.iter().map(|d| {
            let mut sorted = d.balls;
            sorted.sort();
            let mut max_c = 1u8;
            let mut cur = 1u8;
            for w in sorted.windows(2) {
                if w[1] == w[0] + 1 {
                    cur += 1;
                    max_c = max_c.max(cur);
                } else {
                    cur = 1;
                }
            }
            max_c
        }).collect();
        max_consecs.sort();

        let mut spreads: Vec<u8> = draws.iter().map(|d| {
            let mut sorted = d.balls;
            sorted.sort();
            sorted[4] - sorted[0]
        }).collect();
        spreads.sort();

        let p2 = |v: &[u16]| v[v.len() * 2 / 100];
        let p98 = |v: &[u16]| v[v.len() * 98 / 100];
        let p2_u8 = |v: &[u8]| v[v.len() * 2 / 100];
        let p98_u8 = |v: &[u8]| v[v.len() * 98 / 100];

        Self {
            sum_range: (p2(&sums), p98(&sums)),
            max_consecutive: p98_u8(&max_consecs),
            odd_range: (p2_u8(&odd_counts), p98_u8(&odd_counts)),
            spread_range: (p2_u8(&spreads).min(10), p98_u8(&spreads).max(45)),
        }
    }

    /// Construit un filtre adaptatif combinant filtres historiques et prédictions résumées.
    /// Les bornes adaptatives resserrent les bornes historiques quand la prédiction est fiable.
    pub fn adaptive(draws: &[Draw]) -> Self {
        let base = Self::from_history(draws, Pool::Balls);
        let predictor = SummaryPredictor::default();

        match predictor.predict_bounds(draws, 0.80) {
            Some(bounds) => {
                // Intersecter bornes historiques et bornes adaptatives
                let sum_range = (
                    base.sum_range.0.max(bounds.sum_range.0),
                    base.sum_range.1.min(bounds.sum_range.1),
                );
                // Sécurité : si l'intersection est vide, garder les bornes historiques
                let sum_range = if sum_range.0 >= sum_range.1 {
                    base.sum_range
                } else {
                    sum_range
                };

                let spread_range = (
                    base.spread_range.0.max(bounds.spread_range.0),
                    base.spread_range.1.min(bounds.spread_range.1),
                );
                let spread_range = if spread_range.0 >= spread_range.1 {
                    base.spread_range
                } else {
                    spread_range
                };

                // Odd range : intersecter avec les valeurs prédites
                let odd_min = bounds.odd_values.iter().copied().min().unwrap_or(0).max(base.odd_range.0);
                let odd_max = bounds.odd_values.iter().copied().max().unwrap_or(5).min(base.odd_range.1);
                let odd_range = if odd_min > odd_max {
                    base.odd_range
                } else {
                    (odd_min, odd_max)
                };

                Self {
                    sum_range,
                    max_consecutive: base.max_consecutive,
                    odd_range,
                    spread_range,
                }
            }
            None => base,
        }
    }

    /// Vérifie si une grille de boules passe le filtre.
    pub fn accept_balls(&self, balls: &[u8; 5]) -> bool {
        let sum: u16 = balls.iter().map(|&b| b as u16).sum();
        if sum < self.sum_range.0 || sum > self.sum_range.1 {
            return false;
        }

        let odd_count = balls.iter().filter(|&&b| b % 2 == 1).count() as u8;
        if odd_count < self.odd_range.0 || odd_count > self.odd_range.1 {
            return false;
        }

        let mut sorted = *balls;
        sorted.sort();

        let spread = sorted[4] - sorted[0];
        if spread < self.spread_range.0 || spread > self.spread_range.1 {
            return false;
        }

        let mut cur_consec = 1u8;
        for w in sorted.windows(2) {
            if w[1] == w[0] + 1 {
                cur_consec += 1;
                if cur_consec > self.max_consecutive {
                    return false;
                }
            } else {
                cur_consec = 1;
            }
        }

        true
    }
}

/// Génère un seed déterministe basé sur la date du jour (YYYYMMDD).
pub fn date_seed() -> u64 {
    let today = chrono::Local::now().date_naive();
    let y = today.year() as u64;
    let m = today.month() as u64;
    let d = today.day() as u64;
    y * 10_000 + m * 100 + d
}

use chrono::Datelike;

/// Nombre de boules dans `a` absentes de `b`.
fn ball_distance(a: &[u8; 5], b: &[u8; 5]) -> usize {
    a.iter().filter(|x| !b.contains(x)).count()
}

/// Nombre d'étoiles dans `a` absentes de `b`.
fn star_distance(a: &[u8; 2], b: &[u8; 2]) -> usize {
    a.iter().filter(|x| !b.contains(x)).count()
}

/// Distance combinée balls+stars. Les étoiles comptent double (seulement 2 à choisir).
fn combined_distance(a: &Suggestion, b: &Suggestion) -> usize {
    ball_distance(&a.balls, &b.balls) + 2 * star_distance(&a.stars, &b.stars)
}

/// Sélection gloutonne : meilleur score + diversité minimum entre paires.
/// Le seuil `min_ball_diff` s'applique à la distance combinée (balls + 2×stars).
fn select_diverse(candidates: &[Suggestion], count: usize, min_ball_diff: usize) -> Vec<Suggestion> {
    // candidates doit déjà être trié par score décroissant
    let mut selected: Vec<Suggestion> = Vec::with_capacity(count);

    for candidate in candidates {
        if selected.len() >= count {
            break;
        }
        let dominated = selected.iter().any(|s| combined_distance(candidate, s) < min_ball_diff);
        if !dominated {
            selected.push(candidate.clone());
        }
    }

    // Fallback : si pas assez de candidats diversifiés, remplir avec les meilleurs restants
    if selected.len() < count {
        for candidate in candidates {
            if selected.len() >= count {
                break;
            }
            if !selected.iter().any(|s| s.balls == candidate.balls && s.stars == candidate.stars) {
                selected.push(candidate.clone());
            }
        }
    }

    selected
}

/// Applique un scaling température aux probabilités.
/// T < 1 : concentration (sharpening), T > 1 : aplatissement vers l'uniforme.
pub fn apply_temperature(probs: &[f64], temperature: f64) -> Vec<f64> {
    let inv_t = 1.0 / temperature;
    let scaled: Vec<f64> = probs.iter().map(|&p| p.powf(inv_t)).collect();
    let total: f64 = scaled.iter().sum();
    if total > 0.0 {
        scaled.iter().map(|&s| s / total).collect()
    } else {
        vec![1.0 / probs.len() as f64; probs.len()]
    }
}

/// v23: Entropic tilting — concentrates distribution SELECTIVELY based on structural constraints.
/// Unlike temperature scaling (which sharpens uniformly), entropic tilting amplifies numbers
/// that have high PAIR co-occurrence with other high-probability numbers.
/// Formula: q(i) ∝ p(i) × exp(λ × pair_affinity(i))
/// where pair_affinity(i) = Σ_j p(j) × pair_bonus(i,j) for j≠i.
pub fn entropic_tilt(
    probs: &[f64],
    pair_freq: &std::collections::HashMap<(u8, u8), f64>,
    strength: f64, // tilting strength ∈ [0, 50], 0=no tilt, higher=more concentrated
) -> Vec<f64> {
    let n = probs.len();
    if n == 0 || strength < 1e-6 {
        return probs.to_vec();
    }

    let uniform = 1.0 / n as f64;
    let expected_pair = if n == 50 { 5.0 * 4.0 / (50.0 * 49.0) } else { 2.0 * 1.0 / (12.0 * 11.0) };

    // Compute pair affinity for each number:
    // How well does number i "fit" with the other high-probability numbers?
    let mut affinity = vec![0.0f64; n];
    for i in 0..n {
        let ni = (i + 1) as u8;
        for j in 0..n {
            if i == j { continue; }
            let nj = (j + 1) as u8;
            let (a, b) = if ni < nj { (ni, nj) } else { (nj, ni) };
            let freq = pair_freq.get(&(a, b)).copied().unwrap_or(0.0);
            // Pair bonus: log-ratio of observed to expected frequency
            let bonus = if expected_pair > 0.0 && freq > 0.0 {
                (freq / expected_pair).ln().clamp(-2.0, 2.0)
            } else {
                0.0
            };
            // Weight by partner's probability (focus on likely partners)
            affinity[i] += probs[j] * bonus;
        }
    }

    // Normalize affinity to [-1, 1] range
    let max_aff = affinity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_aff = affinity.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max_aff - min_aff).max(1e-10);
    for a in &mut affinity {
        *a = (*a - min_aff) / range * 2.0 - 1.0; // ∈ [-1, 1]
    }

    // Apply tilting: q(i) ∝ p(i) × exp(strength × affinity(i))
    let mut tilted: Vec<f64> = probs.iter().enumerate()
        .map(|(i, &p)| p * (strength * affinity[i]).exp())
        .collect();

    // Normalize
    let total: f64 = tilted.iter().sum();
    if total > 0.0 {
        for t in &mut tilted { *t /= total; }
    } else {
        return probs.to_vec();
    }

    tilted
}

/// Grille déterministe : top 5 boules + top 2 étoiles par probabilité ensemble.
pub fn optimal_grid(ball_probs: &[f64], star_probs: &[f64]) -> Suggestion {
    let mut ball_indices: Vec<usize> = (0..ball_probs.len()).collect();
    ball_indices.sort_by(|&a, &b| ball_probs[b].partial_cmp(&ball_probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut star_indices: Vec<usize> = (0..star_probs.len()).collect();
    star_indices.sort_by(|&a, &b| star_probs[b].partial_cmp(&star_probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut balls = [0u8; 5];
    for (i, &idx) in ball_indices.iter().take(5).enumerate() {
        balls[i] = (idx + 1) as u8;
    }
    balls.sort();

    let mut stars = [0u8; 2];
    for (i, &idx) in star_indices.iter().take(2).enumerate() {
        stars[i] = (idx + 1) as u8;
    }
    stars.sort();

    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;
    let score: f64 = balls.iter().map(|&b| ball_probs[(b - 1) as usize] / uniform_ball).product::<f64>()
        * stars.iter().map(|&s| star_probs[(s - 1) as usize] / uniform_star).product::<f64>();

    Suggestion { balls, stars, score }
}

pub fn generate_suggestions_from_probs(
    ball_probs: &[f64],
    star_probs: &[f64],
    count: usize,
    seed: u64,
    oversample: usize,
    min_ball_diff: usize,
) -> Result<Vec<Suggestion>> {
    generate_suggestions_filtered(ball_probs, star_probs, count, seed, oversample, min_ball_diff, None)
}

pub fn generate_suggestions_filtered(
    ball_probs: &[f64],
    star_probs: &[f64],
    count: usize,
    seed: u64,
    oversample: usize,
    min_ball_diff: usize,
    filter: Option<&StructuralFilter>,
) -> Result<Vec<Suggestion>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    let n_candidates = count * oversample;
    let max_attempts = n_candidates * 3; // éviter boucle infinie
    let mut candidates = Vec::with_capacity(n_candidates);
    let mut attempts = 0;

    while candidates.len() < n_candidates && attempts < max_attempts {
        attempts += 1;
        let (balls, ball_score) = sample_without_replacement(ball_probs, 5, uniform_ball, &mut rng)?;
        let (stars, star_score) = sample_without_replacement(star_probs, 2, uniform_star, &mut rng)?;

        let mut balls_arr = [0u8; 5];
        for (i, &b) in balls.iter().enumerate() {
            balls_arr[i] = b;
        }
        balls_arr.sort();

        // Appliquer le filtre structurel
        if let Some(f) = filter
            && !f.accept_balls(&balls_arr)
        {
            continue;
        }

        let mut stars_arr = [0u8; 2];
        for (i, &s) in stars.iter().enumerate() {
            stars_arr[i] = s;
        }
        stars_arr.sort();

        let score = ball_score * star_score;

        candidates.push(Suggestion {
            balls: balls_arr,
            stars: stars_arr,
            score,
        });
    }

    // Trier par score décroissant
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Sélection diversifiée
    let suggestions = select_diverse(&candidates, count, min_ball_diff);

    Ok(suggestions)
}

/// Calcule le score bayésien d'une grille arbitraire contre une distribution ensemble.
/// Score = produit des (prob_i / prob_uniforme) pour chaque boule et étoile.
pub fn compute_bayesian_score(
    balls: &[u8; 5],
    stars: &[u8; 2],
    ball_probs: &[f64],
    star_probs: &[f64],
) -> f64 {
    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    let ball_score: f64 = balls
        .iter()
        .map(|&b| ball_probs[(b - 1) as usize] / uniform_ball)
        .product();
    let star_score: f64 = stars
        .iter()
        .map(|&s| star_probs[(s - 1) as usize] / uniform_star)
        .product();

    ball_score * star_score
}

// ═══════════════════════════════════════════════════════════════════
// v19 G2: Exponential Tilting Unifié
// ═══════════════════════════════════════════════════════════════════

/// Multi-tilt scoring: P_tilted(x) ∝ P(x) × exp(λ₁·h_coherence + λ₂·h_joint + λ₃·h_anti_pop)
/// The λ weights can be learned from backtest (future work H8 BayesOpt).
pub struct ExponentialTilt {
    pub lambda_coherence: f64,
    pub lambda_joint: f64,
    pub lambda_anti_pop: f64,
}

impl Default for ExponentialTilt {
    fn default() -> Self {
        Self {
            lambda_coherence: 1.0,
            lambda_joint: 0.5,
            lambda_anti_pop: 0.3,
        }
    }
}

impl ExponentialTilt {
    /// Apply exponential tilting to a base score.
    pub fn tilt_score(
        &self,
        base_log_score: f64,
        coherence_score: f64,     // [0, 1]
        joint_log_score: f64,     // log P(joint)
        anti_popularity: f64,      // [0, 1]
    ) -> f64 {
        base_log_score
            + self.lambda_coherence * (coherence_score - 0.5)
            + self.lambda_joint * joint_log_score
            + self.lambda_anti_pop * anti_popularity
    }
}

// ════════════════════════════════════════════════════════════════
// Conditioning ball→star
// ════════════════════════════════════════════════════════════════

/// Conditionne les probabilités de paires d'étoiles sur le contexte des boules.
/// v10: 3 sum_bins × 3 spread_bins × 3 odd_bins = 27 contextes (terciles adaptatifs).
/// Pour chaque contexte, distribution sur les 66 paires d'étoiles (Laplace smoothed).
pub struct BallStarConditioner {
    table: Vec<[f64; 66]>,
    context_counts: Vec<f64>,
    /// v21: Seuils adaptatifs calculés par quintiles (was terciles v9)
    sum_thresholds: [u32; 4],
    spread_thresholds: [u8; 2],
}

impl BallStarConditioner {
    /// v21: 15 contextes (5 sum × 3 spread) — was 9 (3×3, v12).
    /// Quintile sum bins give finer conditioning (~42 obs/ctx with 630 draws),
    /// blend ratio ~42/47 = 0.89. Kernel smoothing compensates for data dilution.
    const N_CONTEXTS: usize = 15; // 5 sum_bins × 3 spread_bins
    const LAPLACE_ALPHA: f64 = 0.3;

    pub fn from_history(draws: &[Draw]) -> Self {
        // v9: calculer les seuils adaptatifs par terciles
        let (sum_thresholds, spread_thresholds) = Self::compute_adaptive_bins(draws);

        let mut counts = vec![[0.0f64; 66]; Self::N_CONTEXTS];
        let mut global_counts = [0.0f64; 66];

        // Crée une instance temporaire pour utiliser ball_context_with_thresholds
        let tmp = Self {
            table: vec![],
            context_counts: vec![],
            sum_thresholds,
            spread_thresholds,
        };

        for (t, draw) in draws.iter().enumerate() {
            let weight = (-0.02 * t as f64).exp();
            let mut balls = draw.balls;
            balls.sort();
            let ctx = tmp.ball_context(&balls);
            let (s1, s2) = if draw.stars[0] < draw.stars[1] {
                (draw.stars[0], draw.stars[1])
            } else {
                (draw.stars[1], draw.stars[0])
            };
            let pidx = crate::models::star_pair::pair_index(s1, s2);
            counts[ctx][pidx] += weight;
            global_counts[pidx] += weight;
        }

        // Normaliser la distribution globale
        let global_total: f64 = global_counts.iter().sum::<f64>() + Self::LAPLACE_ALPHA * 66.0;
        let mut global_probs = [0.0f64; 66];
        for p in 0..66 {
            global_probs[p] = (global_counts[p] + Self::LAPLACE_ALPHA) / global_total;
        }

        // v13: Smooth blend instead of hard threshold at 10 observations.
        // local_weight = ctx_total / (ctx_total + 5.0)
        // 0 obs → 0% local, 5 obs → 50%, 10 obs → 67%, 30 obs → 86%, 70 obs → 93%
        let mut table = vec![[0.0f64; 66]; Self::N_CONTEXTS];
        for ctx in 0..Self::N_CONTEXTS {
            let ctx_total: f64 = counts[ctx].iter().sum::<f64>();
            if ctx_total < 1.0 {
                table[ctx] = global_probs;
            } else {
                let local_weight = ctx_total / (ctx_total + 5.0);
                // Ratio multiplicatif local
                let expected_per_pair = ctx_total / 66.0;
                let mut local_probs = [0.0f64; 66];
                for p in 0..66 {
                    let ratio = (counts[ctx][p] + Self::LAPLACE_ALPHA)
                        / (expected_per_pair + Self::LAPLACE_ALPHA);
                    local_probs[p] = global_probs[p] * ratio;
                }
                // Renormaliser local
                let local_total: f64 = local_probs.iter().sum();
                if local_total > 0.0 {
                    for p in &mut local_probs { *p /= local_total; }
                }
                // Smooth blend global/local
                for p in 0..66 {
                    table[ctx][p] = local_weight * local_probs[p] + (1.0 - local_weight) * global_probs[p];
                }
                // Renormaliser
                let total: f64 = table[ctx].iter().sum();
                if total > 0.0 {
                    for p in &mut table[ctx] { *p /= total; }
                }
            }
        }

        // Compute context observation counts for adaptive blending
        let mut context_counts = vec![0.0f64; Self::N_CONTEXTS];
        for ctx in 0..Self::N_CONTEXTS {
            context_counts[ctx] = counts[ctx].iter().sum::<f64>();
        }

        Self { table, context_counts, sum_thresholds, spread_thresholds }
    }

    /// v21: Calcule les quintiles adaptatifs pour sum et terciles pour spread depuis l'historique.
    fn compute_adaptive_bins(draws: &[Draw]) -> ([u32; 4], [u8; 2]) {
        if draws.is_empty() {
            return ([105, 120, 135, 150], [20, 35]); // fallback hardcodé
        }
        let mut sums: Vec<u32> = draws.iter()
            .map(|d| d.balls.iter().map(|&b| b as u32).sum())
            .collect();
        sums.sort();
        let mut spreads: Vec<u8> = draws.iter()
            .map(|d| {
                let mut b = d.balls;
                b.sort();
                b[4] - b[0]
            })
            .collect();
        spreads.sort();

        // v21: 4 thresholds for 5 quintile bins (was 2 for 3 tercile bins)
        let n = sums.len();
        let sum_t = [
            sums[n / 5],
            sums[2 * n / 5],
            sums[3 * n / 5],
            sums[4 * n / 5],
        ];
        let spread_t = [spreads[spreads.len() / 3], spreads[2 * spreads.len() / 3]];
        (sum_t, spread_t)
    }

    #[inline]
    pub fn ball_context(&self, balls: &[u8; 5]) -> usize {
        let sum: u32 = balls.iter().map(|&b| b as u32).sum();
        let spread = balls[4] - balls[0];
        // v21: 5 sum bins (quintiles) × 3 spread bins = 15 contexts
        let sum_bin = if sum < self.sum_thresholds[0] { 0 }
            else if sum < self.sum_thresholds[1] { 1 }
            else if sum < self.sum_thresholds[2] { 2 }
            else if sum < self.sum_thresholds[3] { 3 }
            else { 4 };
        let spread_bin = if spread < self.spread_thresholds[0] { 0 } else if spread <= self.spread_thresholds[1] { 1 } else { 2 };
        sum_bin * 3 + spread_bin
    }

    /// v21: Kernel-smoothed conditioned pair probabilities.
    /// Blends adjacent sum contexts with Gaussian kernel (sigma=0.5)
    /// to reduce noise from context boundary effects.
    pub fn conditioned_pair_probs(&self, balls: &[u8; 5]) -> [f64; 66] {
        let primary_ctx = self.ball_context(balls);
        let primary_sum_bin = primary_ctx / 3;
        let spread_bin = primary_ctx % 3;

        let mut blended = [0.0_f64; 66];
        let mut total_weight = 0.0_f64;

        for sum_offset in -1i32..=1 {
            let neighbor_sum_bin = primary_sum_bin as i32 + sum_offset;
            if neighbor_sum_bin < 0 || neighbor_sum_bin >= 5 { continue; }
            let neighbor_ctx = neighbor_sum_bin as usize * 3 + spread_bin;
            let kernel_weight = (-0.5 * (sum_offset as f64).powi(2) / (0.5_f64).powi(2)).exp();
            let ctx_blend = self.context_counts[neighbor_ctx] / (self.context_counts[neighbor_ctx] + 5.0);
            for pair_idx in 0..66 {
                blended[pair_idx] += kernel_weight * ctx_blend * self.table[neighbor_ctx][pair_idx];
            }
            total_weight += kernel_weight * ctx_blend;
        }

        // Normalize
        if total_weight > 0.0 {
            for p in blended.iter_mut() { *p /= total_weight; }
        } else {
            // Fallback to primary context
            return self.table[primary_ctx];
        }

        blended
    }

    /// Adaptive blend ratio: 0 when few observations, ~0.78 with many.
    /// v12: seuil 20 (was 30) — 9 contexts give ~70 obs/ctx, blend = 70/90 = 0.78.
    #[inline]
    pub fn adaptive_blend(&self, balls: &[u8; 5]) -> f64 {
        let ctx = self.ball_context(balls);
        let obs = self.context_counts[ctx];
        obs / (obs + 20.0)
    }
}

// Scoring de cohérence conjointe
// ════════════════════════════════════════════════════════════════

/// Statistiques historiques pour le scoring de cohérence des grilles.
pub struct CoherenceScorer {
    pub mean_sum: f64,
    pub std_sum: f64,
    pub mean_spread: f64,
    pub std_spread: f64,
    pub pair_freq: HashMap<(u8, u8), f64>,
    pub triplet_freq: HashMap<(u8, u8, u8), f64>,
}

impl CoherenceScorer {
    /// v23: Construct from pre-computed stats (stored in calibration.json).
    pub fn from_stats(stats: &crate::ensemble::calibration::CoherenceStats) -> Self {
        Self {
            mean_sum: stats.mean_sum,
            std_sum: stats.std_sum,
            mean_spread: stats.mean_spread,
            std_spread: stats.std_spread,
            pair_freq: stats.pair_freq.iter().cloned().collect(),
            triplet_freq: stats.triplet_freq.iter().cloned().collect(),
        }
    }

    /// v23: Export stats for serialization in calibration.json.
    pub fn to_stats(&self) -> crate::ensemble::calibration::CoherenceStats {
        crate::ensemble::calibration::CoherenceStats {
            mean_sum: self.mean_sum,
            std_sum: self.std_sum,
            mean_spread: self.mean_spread,
            std_spread: self.std_spread,
            pair_freq: self.pair_freq.iter().map(|(&k, &v)| (k, v)).collect(),
            triplet_freq: self.triplet_freq.iter().map(|(&k, &v)| (k, v)).collect(),
        }
    }

    /// Construit un scorer depuis l'historique des tirages.
    pub fn from_history(draws: &[Draw], pool: Pool) -> Self {
        let (sums, spreads) = Self::compute_sum_spread_stats(draws, pool);
        let pair_freq = Self::compute_pair_frequencies(draws, pool);
        let triplet_freq = Self::compute_triplet_frequencies(draws, pool);

        let mean_sum = if sums.is_empty() {
            125.0
        } else {
            sums.iter().sum::<f64>() / sums.len() as f64
        };
        let std_sum = if sums.len() < 2 {
            30.0
        } else {
            let var = sums.iter().map(|&s| (s - mean_sum).powi(2)).sum::<f64>() / sums.len() as f64;
            var.sqrt().max(1.0)
        };
        let mean_spread = if spreads.is_empty() {
            30.0
        } else {
            spreads.iter().sum::<f64>() / spreads.len() as f64
        };
        let std_spread = if spreads.len() < 2 {
            10.0
        } else {
            let var = spreads
                .iter()
                .map(|&s| (s - mean_spread).powi(2))
                .sum::<f64>()
                / spreads.len() as f64;
            var.sqrt().max(1.0)
        };

        Self {
            mean_sum,
            std_sum,
            mean_spread,
            std_spread,
            pair_freq,
            triplet_freq,
        }
    }

    fn compute_sum_spread_stats(draws: &[Draw], pool: Pool) -> (Vec<f64>, Vec<f64>) {
        let mut sums = Vec::with_capacity(draws.len());
        let mut spreads = Vec::with_capacity(draws.len());

        for draw in draws {
            let nums = pool.numbers_from(draw);
            let sum: f64 = nums.iter().map(|&n| n as f64).sum();
            let min = *nums.iter().min().unwrap_or(&0) as f64;
            let max = *nums.iter().max().unwrap_or(&0) as f64;
            sums.push(sum);
            spreads.push(max - min);
        }

        (sums, spreads)
    }

    fn compute_pair_frequencies(draws: &[Draw], pool: Pool) -> HashMap<(u8, u8), f64> {
        let mut pair_counts: HashMap<(u8, u8), u32> = HashMap::new();
        let total = draws.len() as f64;

        for draw in draws {
            let nums = pool.numbers_from(draw);
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    let a = nums[i].min(nums[j]);
                    let b = nums[i].max(nums[j]);
                    *pair_counts.entry((a, b)).or_insert(0) += 1;
                }
            }
        }

        pair_counts
            .into_iter()
            .map(|(k, v)| (k, v as f64 / total))
            .collect()
    }

    fn compute_triplet_frequencies(draws: &[Draw], pool: Pool) -> HashMap<(u8, u8, u8), f64> {
        let mut triplet_counts: HashMap<(u8, u8, u8), u32> = HashMap::new();
        let total = draws.len() as f64;

        for draw in draws {
            let nums = pool.numbers_from(draw);
            for i in 0..nums.len() {
                for j in (i + 1)..nums.len() {
                    for k in (j + 1)..nums.len() {
                        let mut triple = [nums[i], nums[j], nums[k]];
                        triple.sort();
                        *triplet_counts.entry((triple[0], triple[1], triple[2])).or_insert(0) += 1;
                    }
                }
            }
        }

        triplet_counts
            .into_iter()
            .map(|(k, v)| (k, v as f64 / total))
            .collect()
    }

    /// Score de cohérence d'une grille de boules (0.0 = incohérent, ~1.0+ = cohérent).
    pub fn score_balls(&self, balls: &[u8; 5]) -> f64 {
        let sum: f64 = balls.iter().map(|&b| b as f64).sum();
        let min = *balls.iter().min().unwrap() as f64;
        let max = *balls.iter().max().unwrap() as f64;
        let spread = max - min;

        // Gaussien : exp(-z²/2)
        let z_sum = (sum - self.mean_sum) / self.std_sum;
        let sum_score = (-z_sum * z_sum / 2.0).exp();

        let z_spread = (spread - self.mean_spread) / self.std_spread;
        let spread_score = (-z_spread * z_spread / 2.0).exp();

        // Fréquence moyenne des paires
        let mut pair_total = 0.0;
        let mut pair_count = 0;
        for i in 0..5 {
            for j in (i + 1)..5 {
                let a = balls[i].min(balls[j]);
                let b = balls[i].max(balls[j]);
                pair_total += self.pair_freq.get(&(a, b)).copied().unwrap_or(0.0);
                pair_count += 1;
            }
        }
        let pair_score = if pair_count > 0 {
            let expected_pair_freq = (5.0 * 4.0) / (50.0 * 49.0);
            let avg_pair = pair_total / pair_count as f64;
            (1.0 + (avg_pair / expected_pair_freq).ln().max(0.0)).min(3.0) / 3.0
        } else {
            0.5
        };

        // Fréquence moyenne des triplets
        let mut triplet_total = 0.0;
        let mut triplet_count = 0;
        for i in 0..5 {
            for j in (i + 1)..5 {
                for k in (j + 1)..5 {
                    let mut triple = [balls[i], balls[j], balls[k]];
                    triple.sort();
                    triplet_total += self.triplet_freq.get(&(triple[0], triple[1], triple[2])).copied().unwrap_or(0.0);
                    triplet_count += 1;
                }
            }
        }
        let triplet_score = if triplet_count > 0 {
            // Fréquence attendue d'un triplet : C(47,2)/C(50,5)
            let expected_triplet_freq = (5.0 * 4.0 * 3.0) / (50.0 * 49.0 * 48.0);
            let avg_triplet = triplet_total / triplet_count as f64;
            (1.0 + (avg_triplet / expected_triplet_freq).ln().max(0.0)).min(3.0) / 3.0
        } else {
            0.5
        };

        // v23b: Quadruplet proxy — product of pair frequencies for each 4-subset
        // C(5,4) = 5 quadruplets per grid. Uses geometric mean of pair frequencies as proxy.
        let mut quad_score = 0.5;
        {
            let expected_pair_freq = (5.0 * 4.0) / (50.0 * 49.0);
            let mut quad_total = 0.0f64;
            let mut quad_count = 0;
            for skip in 0..5 {
                // 4-subset = all balls except balls[skip]
                let mut pair_product = 1.0f64;
                let mut n_pairs = 0;
                for i in 0..5 {
                    if i == skip { continue; }
                    for j in (i + 1)..5 {
                        if j == skip { continue; }
                        let a = balls[i].min(balls[j]);
                        let b = balls[i].max(balls[j]);
                        let freq = self.pair_freq.get(&(a, b)).copied().unwrap_or(0.0);
                        let ratio = (freq / expected_pair_freq).max(0.01);
                        pair_product *= ratio;
                        n_pairs += 1;
                    }
                }
                // Geometric mean of pair ratios within this quadruplet
                if n_pairs > 0 {
                    quad_total += pair_product.powf(1.0 / n_pairs as f64);
                    quad_count += 1;
                }
            }
            if quad_count > 0 {
                let avg_quad = quad_total / quad_count as f64;
                quad_score = (avg_quad / 3.0).clamp(0.0, 1.0); // normalize to [0, 1]
            }
        }

        // Combinaison pondérée
        // v23b: Added quadruplet proxy for higher-order structural coherence
        0.20 * sum_score + 0.15 * spread_score + 0.25 * pair_score + 0.25 * triplet_score + 0.15 * quad_score
    }
}

// ═══════════════════════════════════════════════════════════════════
// v19 F1: StarBallConditioner — conditionneur inverse star→ball
// ═══════════════════════════════════════════════════════════════════

/// Models P(ball | star_context) — inverse of BallStarConditioner.
/// If certain star pairs are predicted with high confidence, this adjusts
/// ball probabilities based on historical co-occurrence.
pub struct StarBallConditioner {
    /// table[ctx][ball_idx] = probability of ball given star context
    table: Vec<Vec<f64>>,
    /// Star sum thresholds for 3 bins (adaptive terciles)
    sum_thresholds: [u32; 2],
    /// Star spread thresholds for 3 bins
    spread_thresholds: [u8; 2],
}

impl StarBallConditioner {
    const N_CONTEXTS: usize = 9; // 3 star_sum × 3 star_spread

    pub fn from_history(draws: &[Draw]) -> Self {
        let n_balls = 50;

        // Compute adaptive tercile thresholds for star features
        let mut star_sums: Vec<u32> = draws.iter()
            .map(|d| d.stars[0] as u32 + d.stars[1] as u32)
            .collect();
        star_sums.sort();
        let sum_thresholds = if star_sums.len() >= 3 {
            [star_sums[star_sums.len() / 3], star_sums[2 * star_sums.len() / 3]]
        } else {
            [8, 16]
        };

        let mut star_spreads: Vec<u8> = draws.iter()
            .map(|d| d.stars[1].saturating_sub(d.stars[0]))
            .collect();
        star_spreads.sort();
        let spread_thresholds = if star_spreads.len() >= 3 {
            [star_spreads[star_spreads.len() / 3], star_spreads[2 * star_spreads.len() / 3]]
        } else {
            [3, 7]
        };

        let mut counts = vec![vec![0.5f64; n_balls]; Self::N_CONTEXTS]; // Laplace prior
        let mut global_counts = vec![0.5f64; n_balls];

        for (t, draw) in draws.iter().enumerate() {
            let weight = (-0.02 * t as f64).exp();
            let star_sum = draw.stars[0] as u32 + draw.stars[1] as u32;
            let star_spread = draw.stars[1].saturating_sub(draw.stars[0]);

            let sum_bin = if star_sum <= sum_thresholds[0] { 0 }
                else if star_sum <= sum_thresholds[1] { 1 }
                else { 2 };
            let spread_bin = if star_spread <= spread_thresholds[0] { 0 }
                else if star_spread <= spread_thresholds[1] { 1 }
                else { 2 };
            let ctx = sum_bin * 3 + spread_bin;

            for &b in &draw.balls {
                let idx = (b - 1) as usize;
                if idx < n_balls {
                    counts[ctx][idx] += weight;
                    global_counts[idx] += weight;
                }
            }
        }

        // Normalize and smooth blend
        let global_total: f64 = global_counts.iter().sum();
        let global_probs: Vec<f64> = global_counts.iter().map(|&c| c / global_total).collect();

        let mut table = vec![vec![0.0f64; n_balls]; Self::N_CONTEXTS];
        for ctx in 0..Self::N_CONTEXTS {
            let ctx_total: f64 = counts[ctx].iter().sum();
            let local_weight = ctx_total / (ctx_total + 10.0);
            let local_probs: Vec<f64> = counts[ctx].iter().map(|&c| c / ctx_total).collect();
            // Blend
            for i in 0..n_balls {
                table[ctx][i] = local_weight * local_probs[i] + (1.0 - local_weight) * global_probs[i];
            }
            // Normalize
            let total: f64 = table[ctx].iter().sum();
            if total > 0.0 {
                for p in &mut table[ctx] { *p /= total; }
            }
        }

        Self { table, sum_thresholds, spread_thresholds }
    }

    /// Given expected stars, return adjusted ball probabilities.
    pub fn adjust_balls(&self, stars: &[u8; 2], ball_probs: &[f64]) -> Vec<f64> {
        let star_sum = stars[0] as u32 + stars[1] as u32;
        let star_spread = stars[1].saturating_sub(stars[0]);

        let sum_bin = if star_sum <= self.sum_thresholds[0] { 0 }
            else if star_sum <= self.sum_thresholds[1] { 1 }
            else { 2 };
        let spread_bin = if star_spread <= self.spread_thresholds[0] { 0 }
            else if star_spread <= self.spread_thresholds[1] { 1 }
            else { 2 };
        let ctx = sum_bin * 3 + spread_bin;

        let cond = &self.table[ctx];
        // Blend 70% original + 30% conditional
        let n = ball_probs.len().min(cond.len());
        let mut adjusted = vec![0.0f64; n];
        for i in 0..n {
            adjusted[i] = 0.70 * ball_probs[i] + 0.30 * cond[i];
        }
        // Normalize
        let sum: f64 = adjusted.iter().sum();
        if sum > 0.0 {
            for p in &mut adjusted { *p /= sum; }
        }
        adjusted
    }
}

/// Poids du bonus de cohérence étoiles dans le scoring jackpot (v23b: increased from 25 to 35)
const STAR_COHERENCE_WEIGHT: f64 = 35.0;

/// Score de cohérence des paires d'étoiles (v15).
///
/// Blende fréquence historique (60%) et fréquence EWMA récente (40%)
/// pour chaque paire. Retourne un score [0, 1] centré à 0.5.
pub struct StarCoherenceScorer {
    pair_freq: [f64; 66],
    recent_pair_freq: [f64; 66],
}

impl StarCoherenceScorer {
    /// Construit le scorer depuis l'historique des tirages.
    pub fn from_history(draws: &[Draw]) -> Self {
        let mut pair_counts = [0.0f64; 66];
        let total = draws.len() as f64;

        // Fréquence historique
        for draw in draws {
            let pidx = crate::models::star_pair::pair_index(draw.stars[0], draw.stars[1]);
            pair_counts[pidx] += 1.0;
        }

        let mut pair_freq = [0.0f64; 66];
        if total > 0.0 {
            for i in 0..66 {
                pair_freq[i] = pair_counts[i] / total;
            }
        } else {
            for p in &mut pair_freq {
                *p = 1.0 / 66.0;
            }
        }

        // EWMA récente (α=0.05, itération chronologique = inverse)
        let ewma_alpha = 0.05;
        let mut recent = [1.0 / 66.0; 66];

        for draw in draws.iter().rev() {
            let pidx = crate::models::star_pair::pair_index(draw.stars[0], draw.stars[1]);
            for i in 0..66 {
                let present = if i == pidx { 1.0 } else { 0.0 };
                recent[i] = ewma_alpha * present + (1.0 - ewma_alpha) * recent[i];
            }
        }

        // Normaliser la distribution EWMA
        let sum: f64 = recent.iter().sum();
        if sum > 0.0 {
            for p in &mut recent {
                *p /= sum;
            }
        }

        Self {
            pair_freq,
            recent_pair_freq: recent,
        }
    }

    /// Score de cohérence pour une paire d'étoiles [0, 1] centré 0.5.
    pub fn score_star_pair(&self, stars: &[u8; 2]) -> f64 {
        let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
        let historical = self.pair_freq[pidx];
        let recent = self.recent_pair_freq[pidx];

        // Blend 60% historique + 40% récent
        let blended = 0.60 * historical + 0.40 * recent;
        let expected: f64 = 1.0 / 66.0;
        let ratio = blended / expected.max(1e-15);

        // Map ratio → [0, 1] centré 0.5 via log-ratio
        let log_ratio = ratio.max(1e-6).ln();
        (0.5 + log_ratio / 3.0).clamp(0.0, 1.0)
    }
}

/// Génère des suggestions avec scoring conjoint (cohérence historique + modèle joint conditionnel).
#[allow(clippy::too_many_arguments)]
pub fn generate_suggestions_joint(
    ball_probs: &[f64],
    star_probs: &[f64],
    draws: &[Draw],
    count: usize,
    seed: u64,
    oversample: usize,
    min_ball_diff: usize,
    filter: Option<&StructuralFilter>,
) -> Result<Vec<Suggestion>> {
    let coherence = CoherenceScorer::from_history(draws, Pool::Balls);

    // Entraîner le modèle joint conditionnel
    let mut joint_model = crate::models::joint::JointConditionalModel::default();
    joint_model.train(draws);

    let mut rng = StdRng::seed_from_u64(seed);

    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    let n_candidates = count * oversample;
    let max_attempts = n_candidates * 3;
    let mut candidates = Vec::with_capacity(n_candidates);

    // Phase 1 : Candidats par échantillonnage marginal (50%)
    let n_marginal = n_candidates / 2;
    let mut attempts = 0;
    while candidates.len() < n_marginal && attempts < max_attempts {
        attempts += 1;
        let (balls, ball_score) =
            sample_without_replacement(ball_probs, 5, uniform_ball, &mut rng)?;
        let (stars, star_score) =
            sample_without_replacement(star_probs, 2, uniform_star, &mut rng)?;

        let mut balls_arr = [0u8; 5];
        for (i, &b) in balls.iter().enumerate() {
            balls_arr[i] = b;
        }
        balls_arr.sort();

        if let Some(f) = filter
            && !f.accept_balls(&balls_arr)
        {
            continue;
        }

        let mut stars_arr = [0u8; 2];
        for (i, &s) in stars.iter().enumerate() {
            stars_arr[i] = s;
        }
        stars_arr.sort();

        let bayesian_score = ball_score * star_score;
        let log_bayesian = bayesian_score.max(1e-15).ln();
        let coherence_score = coherence.score_balls(&balls_arr);
        // Score joint conditionnel (bonus si le modèle est entraîné)
        let joint_bonus = if joint_model.score_grid(&balls_arr, &stars_arr) != 0.0 {
            // Normaliser le score joint : ratio vs uniforme, clamped
            let norm = joint_model.score_grid_normalized(&balls_arr, &stars_arr);
            norm.max(0.01).ln() * 0.3 // Poids 30% pour le joint
        } else {
            0.0
        };
        // Score en log-espace pour le tri : tout en log
        let sort_score = log_bayesian + coherence_score.max(1e-6).ln() + joint_bonus;

        candidates.push(Suggestion {
            balls: balls_arr,
            stars: stars_arr,
            score: sort_score,
        });
    }

    // Phase 2 : Candidats par recombinaison de templates historiques (50%)
    let n_templates = (count * oversample).saturating_sub(candidates.len());
    if !draws.is_empty() && n_templates > 0 {
        let template_candidates =
            generate_template_candidates(draws, ball_probs, star_probs, &coherence, &joint_model, n_templates, &mut rng, filter);
        candidates.extend(template_candidates);
    }

    // Trier par sort_score (log-espace) décroissant
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = select_diverse(&candidates, count, min_ball_diff);

    // Recalculer le score linéaire (bayésien) pour l'affichage
    for s in &mut selected {
        s.score = compute_bayesian_score(&s.balls, &s.stars, ball_probs, star_probs);
    }

    Ok(selected)
}

/// Génère des candidats par recombinaison de templates historiques.
#[allow(clippy::too_many_arguments)]
fn generate_template_candidates(
    draws: &[Draw],
    ball_probs: &[f64],
    star_probs: &[f64],
    coherence: &CoherenceScorer,
    joint_model: &crate::models::joint::JointConditionalModel,
    count: usize,
    rng: &mut StdRng,
    filter: Option<&StructuralFilter>,
) -> Vec<Suggestion> {
    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    // Scorer les tirages historiques et prendre les top K comme templates
    let k_templates = 20.min(draws.len());
    let mut draw_scores: Vec<(usize, f64)> = draws
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let ball_score: f64 = d
                .balls
                .iter()
                .map(|&b| ball_probs[(b - 1) as usize] / uniform_ball)
                .product();
            (i, ball_score)
        })
        .collect();
    draw_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    draw_scores.truncate(k_templates);

    let ball_dist = match WeightedIndex::new(ball_probs) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    let star_dist = match WeightedIndex::new(star_probs) {
        Ok(d) => d,
        Err(_) => return vec![],
    };

    let mut candidates = Vec::with_capacity(count);
    let max_attempts = count * 3;
    let mut attempts = 0;

    while candidates.len() < count && attempts < max_attempts {
        attempts += 1;

        // Choisir un template aléatoire parmi les top K
        let template_idx = draw_scores[ball_dist.sample(rng) % draw_scores.len()].0;
        let template = &draws[template_idx];

        // Recombinaison : garder 3-4 boules, remplacer 1-2
        let mut balls = template.balls;
        balls.sort();
        let n_replace = 1 + (ball_dist.sample(rng) % 2); // 1 ou 2 remplacements

        for _ in 0..n_replace {
            let replace_pos = ball_dist.sample(rng) % 5;
            loop {
                let new_ball = (ball_dist.sample(rng) + 1) as u8;
                if (1..=50).contains(&new_ball) && !balls.contains(&new_ball) {
                    balls[replace_pos] = new_ball;
                    break;
                }
            }
        }
        balls.sort();

        if let Some(f) = filter
            && !f.accept_balls(&balls)
        {
            continue;
        }

        // Étoiles : échantillonnage marginal
        let mut stars = [0u8; 2];
        let s1 = (star_dist.sample(rng) + 1) as u8;
        let mut s2 = s1;
        while s2 == s1 {
            s2 = (star_dist.sample(rng) + 1) as u8;
        }
        stars[0] = s1.min(s2);
        stars[1] = s1.max(s2);

        let ball_score: f64 = balls
            .iter()
            .map(|&b| ball_probs[(b - 1) as usize] / uniform_ball)
            .product();
        let star_score: f64 = stars
            .iter()
            .map(|&s| star_probs[(s - 1) as usize] / uniform_star)
            .product();
        let bayesian_score = ball_score * star_score;
        let log_bayesian = bayesian_score.max(1e-15).ln();
        let coherence_score = coherence.score_balls(&balls);
        let joint_bonus = if joint_model.score_grid(&balls, &stars) != 0.0 {
            let norm = joint_model.score_grid_normalized(&balls, &stars);
            norm.max(0.01).ln() * 0.3
        } else {
            0.0
        };
        let sort_score = log_bayesian + coherence_score + joint_bonus;

        candidates.push(Suggestion {
            balls,
            stars,
            score: sort_score,
        });
    }

    candidates
}

/// Suggestion enrichie avec scores multiples pour le mode EV.
#[derive(Debug, Clone)]
pub struct ScoredSuggestion {
    pub balls: [u8; 5],
    pub stars: [u8; 2],
    pub bayesian_score: f64,
    pub anti_popularity: f64,
    pub ev_per_euro: f64,
}

/// Genere des suggestions optimisees pour l'esperance de gain (anti-popularite).
///
/// Utilise les distributions ensemble avec T=2.0 (quasi-uniforme) pour la generation,
/// puis score par anti-popularite au lieu de score bayesien.
#[allow(clippy::too_many_arguments)]
pub fn generate_suggestions_ev(
    ball_probs: &[f64],
    star_probs: &[f64],
    draws: &[Draw],
    count: usize,
    seed: u64,
    oversample: usize,
    min_ball_diff: usize,
    popularity: &PopularityModel,
    jackpot: f64,
) -> Result<Vec<ScoredSuggestion>> {
    // Utiliser directement les distributions (la température est déjà appliquée en amont par cmd_predict)
    let flat_ball = ball_probs.to_vec();
    let flat_star = star_probs.to_vec();

    let filter = if !draws.is_empty() {
        Some(StructuralFilter::from_history(draws, Pool::Balls))
    } else {
        None
    };

    let uniform_ball = 1.0 / flat_ball.len() as f64;
    let uniform_star = 1.0 / flat_star.len() as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let n_candidates = count * oversample;
    let max_attempts = n_candidates * 3;
    let mut candidates: Vec<ScoredSuggestion> = Vec::with_capacity(n_candidates);
    let mut attempts = 0;

    while candidates.len() < n_candidates && attempts < max_attempts {
        attempts += 1;
        let (balls, ball_score) = sample_without_replacement(&flat_ball, 5, uniform_ball, &mut rng)?;
        let (stars, star_score) = sample_without_replacement(&flat_star, 2, uniform_star, &mut rng)?;

        let mut balls_arr = [0u8; 5];
        for (i, &b) in balls.iter().enumerate() {
            balls_arr[i] = b;
        }
        balls_arr.sort();

        if let Some(ref f) = filter
            && !f.accept_balls(&balls_arr)
        {
            continue;
        }

        let mut stars_arr = [0u8; 2];
        for (i, &s) in stars.iter().enumerate() {
            stars_arr[i] = s;
        }
        stars_arr.sort();

        let bayesian_score = ball_score * star_score;
        let ap = anti_popularity(&balls_arr, &stars_arr, popularity);
        let ev = compute_ev(&balls_arr, &stars_arr, popularity, jackpot);

        candidates.push(ScoredSuggestion {
            balls: balls_arr,
            stars: stars_arr,
            bayesian_score,
            anti_popularity: ap,
            ev_per_euro: ev.ev_per_euro,
        });
    }

    // Trier par anti-popularite decroissante (= grilles les moins jouees)
    candidates.sort_by(|a, b| {
        b.anti_popularity
            .partial_cmp(&a.anti_popularity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Selection diversifiee via conversion temporaire en Suggestion
    let as_suggestions: Vec<Suggestion> = candidates
        .iter()
        .map(|c| Suggestion {
            balls: c.balls,
            stars: c.stars,
            score: c.anti_popularity,
        })
        .collect();

    let diverse = select_diverse(&as_suggestions, count, min_ball_diff);

    // Reconstruire les ScoredSuggestion correspondantes
    let result: Vec<ScoredSuggestion> = diverse
        .iter()
        .map(|s| {
            candidates
                .iter()
                .find(|c| c.balls == s.balls && c.stars == s.stars)
                .cloned()
                .unwrap_or(ScoredSuggestion {
                    balls: s.balls,
                    stars: s.stars,
                    bayesian_score: s.score,
                    anti_popularity: s.score,
                    ev_per_euro: 0.0,
                })
        })
        .collect();

    Ok(result)
}

fn sample_without_replacement(
    probs: &[f64],
    count: usize,
    _uniform_prob: f64,
    rng: &mut StdRng,
) -> Result<(Vec<u8>, f64)> {
    let mut available: Vec<(u8, f64)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| ((i + 1) as u8, p))
        .collect();
    let mut selected = Vec::with_capacity(count);
    let mut score = 1.0f64;

    for _ in 0..count {
        let remaining = available.len() as f64;
        let current_uniform = 1.0 / remaining;
        let weights: Vec<f64> = available.iter().map(|(_, w)| *w).collect();
        let dist = WeightedIndex::new(&weights)?;
        let idx = dist.sample(rng);

        let (number, prob) = available.remove(idx);
        selected.push(number);
        score *= prob / current_uniform;
    }

    Ok((selected, score))
}

// ════════════════════════════════════════════════════════════════
// Mode Jackpot : énumération exhaustive top-N par P(5+2)
// ════════════════════════════════════════════════════════════════

/// Résultat du mode jackpot : top-N combinaisons par probabilité de 5+2.
pub struct JackpotResult {
    /// Top-N suggestions triées par score bayésien décroissant.
    pub suggestions: Vec<Suggestion>,
    /// Somme des P(5+2) pour tous les tickets retournés.
    pub total_jackpot_probability: f64,
    /// Nombre total de combinaisons énumérées.
    pub enumeration_size: u64,
    /// Nombre de combinaisons passant le filtre structurel.
    pub filtered_size: u64,
    /// Facteur d'amélioration vs N tickets uniformes (raw).
    pub improvement_factor: f64,
    /// v19 D2: Facteur d'amélioration ajusté pour l'overlap entre grilles.
    pub adjusted_improvement_factor: f64,
}

/// v19 D2: Compute overlap-adjusted improvement factor.
/// The raw improvement_factor assumes grids are independent.
/// This adjusts for shared balls/stars between selected grids.
fn compute_overlap_adjusted_factor(suggestions: &[Suggestion], raw_improvement: f64) -> f64 {
    if suggestions.len() <= 1 {
        return raw_improvement;
    }

    let n = suggestions.len();
    let mut total_overlap = 0.0f64;
    let mut pair_count = 0;

    for i in 0..n.min(100) {  // Cap at 100 to avoid O(n²) for large sets
        for j in (i + 1)..n.min(100) {
            let common_balls = suggestions[i].balls.iter()
                .filter(|b| suggestions[j].balls.contains(b))
                .count();
            let common_stars = suggestions[i].stars.iter()
                .filter(|s| suggestions[j].stars.contains(s))
                .count();
            total_overlap += (common_balls + common_stars) as f64 / 7.0;
            pair_count += 1;
        }
    }

    let avg_overlap = if pair_count > 0 { total_overlap / pair_count as f64 } else { 0.0 };
    // effective_n = n * (1 - avg_overlap) — fewer effective independent grids
    let effective_ratio = (1.0 - avg_overlap).max(0.1);
    raw_improvement * effective_ratio
}

/// Coefficient binomial C(n, k).
fn comb(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / (i as u64 + 1);
    }
    result
}

/// v11: Calcule la taille de sous-ensemble optimale m* qui maximise la couverture
/// probable du jackpot. Inspiré du Sirius Code / VNAE (Pereira 2025).
///
/// Pour chaque taille m ∈ [12, 50], on évalue :
/// - La probabilité que les 5 gagnantes soient dans le top-m
/// - La densité de probabilité par combinaison dans le sous-ensemble
/// - Le score total = effective_grids × density × coverage_prob
pub fn optimal_subset_k(ball_probs: &[f64], n_grids: usize) -> usize {
    let n = ball_probs.len();
    if n == 0 { return 50; }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| ball_probs[b].partial_cmp(&ball_probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_k = n;
    let mut best_score = 0.0f64;

    for m in 12..=n {
        // Sum of probabilities in top-m
        let subset_prob: f64 = indices[..m].iter().map(|&i| ball_probs[i]).sum();

        // Approximate probability that all 5 winning balls are in top-m
        let p_5_in_m = subset_prob.powi(5);

        // Number of combinations in the subset
        let n_combos = comb(m, 5) as f64;
        if n_combos < 1.0 { continue; }

        // v12: Fixed double-counting of p_5_in_m.
        // P(at least one grid wins) ≈ P(winner in top-m) × coverage_fraction
        let coverage = (n_grids as f64).min(n_combos) / n_combos;
        let total = p_5_in_m * coverage;

        if total > best_score {
            best_score = total;
            best_k = m;
        }
    }

    best_k
}

/// v17: Conformal-adaptive K selection.
/// Falls back to optimal_subset_k when no calibration scores are available.
pub fn conformal_subset_k(
    ball_probs: &[f64],
    n_grids: usize,
    calibration_scores: Option<&[f64]>,
) -> usize {
    if let Some(scores) = calibration_scores {
        if !scores.is_empty() {
            let k = crate::conformal::conformal_k(ball_probs, scores, Pool::Balls);
            // Never go below the original optimal_subset_k
            return k.max(optimal_subset_k(ball_probs, n_grids).min(k + 10));
        }
    }
    optimal_subset_k(ball_probs, n_grids)
}

/// v22: Conformal K from max-rank data.
/// Given historical max_ranks (1-indexed rank of worst-ranked winning ball),
/// returns K such that the coverage guarantee is met at level 1-alpha.
pub fn conformal_k_from_ranks(max_ranks: &[usize], alpha: f64) -> usize {
    if max_ranks.is_empty() {
        return 50; // fallback: full pool
    }
    let mut sorted = max_ranks.to_vec();
    sorted.sort();
    let n = sorted.len();
    // Conformal quantile: ceil((n+1)*(1-alpha))/n
    let quantile_idx = (((n + 1) as f64 * (1.0 - alpha)).ceil() as usize).min(n) - 1;
    let k = sorted[quantile_idx];
    // Add margin of 2 for safety
    (k + 2).min(50)
}

/// Calcule K_balls et K_stars adaptatifs pour que C(K_b,5)×C(K_s,2) >= 3×count.
fn compute_adaptive_k(count: usize) -> (usize, usize) {
    let target = 3 * count as u64;

    // Commencer avec K_stars=6, K_balls=10 et augmenter
    for k_stars in 2..=12usize {
        let star_combs = comb(k_stars, 2);
        for k_balls in 10..=50usize {
            let ball_combs = comb(k_balls, 5);
            if ball_combs.saturating_mul(star_combs) >= target {
                return (k_balls, k_stars);
            }
        }
    }
    (50, 12) // fallback : tout énumérer
}

/// Wrapper pour le min-heap : on veut garder les N plus grands scores,
/// donc on utilise un min-heap et on éjecte le plus petit quand on dépasse N.
#[derive(PartialEq)]
struct MinHeapEntry {
    score: f64,
    balls: [u8; 5],
    stars: [u8; 2],
}

impl Eq for MinHeapEntry {}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Inverser pour min-heap (BinaryHeap est un max-heap par défaut)
        other.score.partial_cmp(&self.score).unwrap_or(CmpOrdering::Equal)
    }
}

/// Génère les top-N combinaisons par probabilité de jackpot (5+2).
///
/// Poids du bonus de cohérence dans le scoring jackpot (v23b: increased from 45 to 60)
/// Coherence scoring is the primary driver of improvement factor (90% of log_score).
const COHERENCE_WEIGHT: f64 = 60.00;

/// Énumère exhaustivement les combinaisons dans la zone haute probabilité,
/// triées par score bayésien, sans diversité.
pub fn generate_suggestions_jackpot(
    ball_probs: &[f64],
    star_probs: &[f64],
    count: usize,
    filter: Option<&StructuralFilter>,
    coherence: Option<&CoherenceScorer>,
    joint_model: Option<&crate::models::joint::JointConditionalModel>,
    star_pair_probs: Option<&[f64; 66]>,
    excluded_balls: Option<&[u8]>,
    conditioner: Option<&BallStarConditioner>,
    _neural_scorer: Option<&crate::models::neural_scorer::NeuralScorer>,
    star_coherence: Option<&StarCoherenceScorer>,
    coherence_ball_w: Option<f64>,
    coherence_star_w: Option<f64>,
    conformal_max_ranks: Option<&[usize]>,
    popularity: Option<&crate::expected_value::PopularityModel>,
    anti_pop_weight: f64,
) -> Result<JackpotResult> {
    let cw = coherence_ball_w.unwrap_or(COHERENCE_WEIGHT);
    let scw = coherence_star_w.unwrap_or(STAR_COHERENCE_WEIGHT);
    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    // Trier les boules par probabilité décroissante
    let mut ball_indices: Vec<usize> = (0..ball_probs.len()).collect();
    ball_indices.sort_by(|&a, &b| {
        ball_probs[b].partial_cmp(&ball_probs[a]).unwrap_or(CmpOrdering::Equal)
    });

    // Appliquer l'exclusion de boules
    if let Some(excluded) = excluded_balls {
        ball_indices.retain(|&idx| !excluded.contains(&((idx + 1) as u8)));
    }

    // v10: K adaptatif par consensus — basé sur l'entropie ET le nombre de favoris
    let (k_balls_base, _k_stars) = compute_adaptive_k(count);
    let h: f64 = ball_probs.iter()
        .filter(|&&p| p > 1e-30)
        .map(|&p| -p * p.ln())
        .sum();
    let h_max = (ball_probs.len() as f64).ln();
    let entropy_ratio = h / h_max;

    // Combiner les deux bornes inférieures (v7 entropie + v10 favoris)
    let uniform_ball_threshold = 1.3 / ball_probs.len() as f64;
    let n_favored = ball_probs.iter().filter(|&&p| p > uniform_ball_threshold).count();
    let k_min_entropy = (25.0 + 20.0 * entropy_ratio).round() as usize; // v7 formula
    let k_min_favored = (n_favored + 8).max(15);
    let k_min = k_min_entropy.max(k_min_favored); // prendre la plus haute des deux bornes
    // v22: Use conformal K from max-rank data when available
    let k_optimal = if let Some(ranks) = conformal_max_ranks {
        if !ranks.is_empty() {
            conformal_k_from_ranks(ranks, 0.05)
        } else {
            conformal_subset_k(ball_probs, count, None)
        }
    } else {
        conformal_subset_k(ball_probs, count, None)
    };
    let k_balls = k_balls_base.max(k_min);
    // Apply optimal cap only if it wouldn't restrict below k_min
    let k_balls = if k_optimal >= k_min { k_balls.min(k_optimal) } else { k_balls };
    let top_balls = &ball_indices[..k_balls.min(ball_indices.len())];

    // Énumérer exhaustivement les 66 paires d'étoiles avec score direct
    let uniform_pair = 1.0 / 66.0;
    let mut star_pairs: Vec<([u8; 2], f64)> = Vec::with_capacity(66);
    for s1 in 1u8..=11 {
        for s2 in (s1 + 1)..=12u8 {
            let stars = [s1, s2];
            let star_score = if let Some(pp) = star_pair_probs {
                let pidx = crate::models::star_pair::pair_index(s1, s2);
                pp[pidx] / uniform_pair
            } else {
                (star_probs[(s1 - 1) as usize] / uniform_star)
                    * (star_probs[(s2 - 1) as usize] / uniform_star)
            };
            star_pairs.push((stars, star_score));
        }
    }
    // Trier par score décroissant et garder les top paires (toutes 66 si star_pair_probs disponible)
    star_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));
    let top_star_pairs = if star_pair_probs.is_some() { 66 } else { 30 };
    star_pairs.truncate(top_star_pairs);

    // Blend ratio conditionné vs marginal pour les étoiles (adaptive per context)

    // Pré-calcul en log-espace pour stabilité numérique
    let log_ball_probs: Vec<f64> = (0..ball_probs.len())
        .map(|i| (ball_probs[i] / uniform_ball).max(1e-30).ln())
        .collect();

    // Pré-calcul des log star scores (évite ln() redondant dans les boucles internes)
    let log_base_star_scores: Vec<f64> = star_pairs.iter()
        .map(|&(_, score)| score.max(1e-30).ln())
        .collect();

    let total_enum = comb(top_balls.len(), 5) * star_pairs.len() as u64;
    let use_heap = total_enum > 10 * count as u64;

    let mut enumeration_size: u64 = 0;
    let mut filtered_size: u64 = 0;

    if use_heap {
        // Mode heap : garder seulement les top-N (scores en log-espace)
        let mut heap: BinaryHeap<MinHeapEntry> = BinaryHeap::with_capacity(count + 1);
        let mut min_log_score = f64::NEG_INFINITY;

        for i0 in 0..top_balls.len() {
            for i1 in (i0 + 1)..top_balls.len() {
                for i2 in (i1 + 1)..top_balls.len() {
                    for i3 in (i2 + 1)..top_balls.len() {
                        for i4 in (i3 + 1)..top_balls.len() {
                            let mut balls = [
                                (top_balls[i0] + 1) as u8,
                                (top_balls[i1] + 1) as u8,
                                (top_balls[i2] + 1) as u8,
                                (top_balls[i3] + 1) as u8,
                                (top_balls[i4] + 1) as u8,
                            ];
                            balls.sort();

                            if let Some(f) = filter
                                && !f.accept_balls(&balls)
                            {
                                enumeration_size += star_pairs.len() as u64;
                                continue;
                            }

                            let marginal_log_ball_score: f64 = [i0,i1,i2,i3,i4].iter()
                                .map(|&i| log_ball_probs[top_balls[i]])
                                .sum();

                            // v10: blend adaptatif basé sur la confiance du modèle joint
                            let log_ball_score = if let Some(jm) = joint_model {
                                let (joint_log, joint_conf) = jm.score_balls_with_confidence(&balls);
                                if joint_log.is_finite() && joint_log > -100.0 {
                                    let joint_weight = 0.15 + 0.55 * joint_conf; // v22: [0.15, 0.70]
                                    let marginal_weight = 1.0 - joint_weight;
                                    marginal_weight * marginal_log_ball_score + joint_weight * joint_log
                                } else {
                                    marginal_log_ball_score
                                }
                            } else {
                                marginal_log_ball_score
                            };

                            let cond_probs = conditioner.map(|c| c.conditioned_pair_probs(&balls));
                            let cond_blend = conditioner.map(|c| c.adaptive_blend(&balls)).unwrap_or(0.0);

                            // Coherence ne dépend que des boules → hors boucle étoiles
                            let coherence_bonus = if let Some(cs) = coherence {
                                let c = cs.score_balls(&balls);
                                cw * (c - 0.5).clamp(-0.5, 0.5)
                            } else {
                                0.0
                            };

                            for (star_idx, &(stars, base_star_score)) in star_pairs.iter().enumerate() {
                                enumeration_size += 1;
                                let log_star_score = if let Some(cp) = cond_probs {
                                    let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
                                    let conditioned = cp[pidx] / uniform_pair;
                                    let blended = cond_blend * conditioned + (1.0 - cond_blend) * base_star_score;
                                    blended.max(1e-30).ln()
                                } else {
                                    log_base_star_scores[star_idx]
                                };
                                let star_coherence_bonus = if let Some(scs) = star_coherence {
                                    let sc = scs.score_star_pair(&stars);
                                    scw * (sc - 0.5).clamp(-0.5, 0.5)
                                } else {
                                    0.0
                                };
                                // v24: Anti-popularity bonus — prefer grids that fewer players pick
                                let anti_pop_bonus = if let Some(pop) = popularity {
                                    let grid_pop = crate::expected_value::grid_popularity(&balls, &stars, pop);
                                    anti_pop_weight * (-grid_pop.max(0.01).ln()).clamp(0.0, 3.0)
                                } else { 0.0 };

                                let log_score = log_ball_score + log_star_score + coherence_bonus + star_coherence_bonus + anti_pop_bonus;

                                // Élagage rapide : si le score est inférieur au min du heap plein, skip
                                if heap.len() >= count && log_score <= min_log_score {
                                    continue;
                                }

                                filtered_size += 1;
                                heap.push(MinHeapEntry { score: log_score, balls, stars });

                                if heap.len() > count {
                                    heap.pop(); // éjecter le plus petit
                                    if let Some(top) = heap.peek() {
                                        min_log_score = top.score;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extraire du heap, convertir log-scores en linéaires, trier par score décroissant
        let mut suggestions: Vec<Suggestion> = heap
            .into_iter()
            .map(|e| Suggestion {
                balls: e.balls,
                stars: e.stars,
                score: e.score.exp(), // log → linéaire
            })
            .collect();
        suggestions.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal)
        });

        // Recalculer le score bayésien pur (sans coherence bonus) pour les métriques
        for s in &mut suggestions {
            s.score = compute_bayesian_score(&s.balls, &s.stars, ball_probs, star_probs);
        }

        // score = prod(prob/uniform) = facteur d'amélioration vs uniforme par grille
        // P(5+2) par grille sous le modèle = score / 139_838_160
        let n_sugg = suggestions.len() as f64;
        let mean_score = suggestions.iter().map(|s| s.score).sum::<f64>() / n_sugg;
        let total_prob = mean_score * n_sugg / 139_838_160.0;
        let improvement = mean_score;

        // v19 D2: Overlap-adjusted improvement factor
        let adjusted = compute_overlap_adjusted_factor(&suggestions, improvement);

        Ok(JackpotResult {
            suggestions,
            total_jackpot_probability: total_prob,
            enumeration_size,
            filtered_size,
            improvement_factor: improvement,
            adjusted_improvement_factor: adjusted,
        })
    } else {
        // Mode collecte : tout garder puis trier
        let mut all: Vec<Suggestion> = Vec::with_capacity(total_enum as usize);

        for i0 in 0..top_balls.len() {
            for i1 in (i0 + 1)..top_balls.len() {
                for i2 in (i1 + 1)..top_balls.len() {
                    for i3 in (i2 + 1)..top_balls.len() {
                        for i4 in (i3 + 1)..top_balls.len() {
                            let mut balls = [
                                (top_balls[i0] + 1) as u8,
                                (top_balls[i1] + 1) as u8,
                                (top_balls[i2] + 1) as u8,
                                (top_balls[i3] + 1) as u8,
                                (top_balls[i4] + 1) as u8,
                            ];
                            balls.sort();

                            let passes_filter = filter.is_none_or(|f| f.accept_balls(&balls));

                            let marginal_log_ball_score: f64 = [i0,i1,i2,i3,i4].iter()
                                .map(|&i| log_ball_probs[top_balls[i]])
                                .sum();

                            // v10: blend adaptatif basé sur la confiance du modèle joint
                            let log_ball_score = if let Some(jm) = joint_model {
                                let (joint_log, joint_conf) = jm.score_balls_with_confidence(&balls);
                                if joint_log.is_finite() && joint_log > -100.0 {
                                    let joint_weight = 0.15 + 0.55 * joint_conf; // v22: [0.15, 0.70]
                                    let marginal_weight = 1.0 - joint_weight;
                                    marginal_weight * marginal_log_ball_score + joint_weight * joint_log
                                } else {
                                    marginal_log_ball_score
                                }
                            } else {
                                marginal_log_ball_score
                            };

                            let cond_probs = conditioner.map(|c| c.conditioned_pair_probs(&balls));
                            let cond_blend = conditioner.map(|c| c.adaptive_blend(&balls)).unwrap_or(0.0);

                            let coherence_bonus = if let Some(cs) = coherence {
                                let c = cs.score_balls(&balls);
                                cw * (c - 0.5).clamp(-0.5, 0.5)
                            } else {
                                0.0
                            };

                            for (star_idx, &(stars, base_star_score)) in star_pairs.iter().enumerate() {
                                enumeration_size += 1;
                                if passes_filter {
                                    filtered_size += 1;
                                    let log_star_score = if let Some(cp) = cond_probs {
                                        let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
                                        let conditioned = cp[pidx] / uniform_pair;
                                        let blended = cond_blend * conditioned + (1.0 - cond_blend) * base_star_score;
                                        blended.max(1e-30).ln()
                                    } else {
                                        log_base_star_scores[star_idx]
                                    };
                                    let star_coherence_bonus = if let Some(scs) = star_coherence {
                                        let sc = scs.score_star_pair(&stars);
                                        scw * (sc - 0.5).clamp(-0.5, 0.5)
                                    } else {
                                        0.0
                                    };
                                    let score = (log_ball_score + log_star_score + coherence_bonus + star_coherence_bonus).exp();
                                    all.push(Suggestion { balls, stars, score });
                                }
                            }
                        }
                    }
                }
            }
        }

        all.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal)
        });
        all.truncate(count);

        // Recalculer le score bayésien pur (sans coherence bonus) pour les métriques
        for s in &mut all {
            s.score = compute_bayesian_score(&s.balls, &s.stars, ball_probs, star_probs);
        }

        // score = prod(prob/uniform) = facteur d'amélioration vs uniforme par grille
        // P(5+2) par grille sous le modèle = score / 139_838_160
        let n_sugg = all.len() as f64;
        let mean_score = all.iter().map(|s| s.score).sum::<f64>() / n_sugg;
        let total_prob = mean_score * n_sugg / 139_838_160.0;
        let improvement = mean_score;

        // v19 D2: Overlap-adjusted improvement factor
        let adjusted = compute_overlap_adjusted_factor(&all, improvement);

        Ok(JackpotResult {
            suggestions: all,
            total_jackpot_probability: total_prob,
            enumeration_size,
            filtered_size,
            improvement_factor: improvement,
            adjusted_improvement_factor: adjusted,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// Mode Gibbs : exploration MCMC de la distribution jointe P(b1,...,b5,s1,s2)
// ═══════════════════════════════════════════════════════════════

/// Generates top-N grids via Gibbs sampling of the joint distribution P(b1,...,b5,s1,s2).
///
/// Complements `generate_suggestions_jackpot` (exhaustive enumeration) by exploring the
/// joint distribution more fluidly via MCMC. Each chain starts from the optimal grid
/// (perturbed for diversity), then performs component-wise Gibbs sweeps.
///
/// Scoring matches `generate_suggestions_jackpot`:
/// `log_ball_score + log_star_score + coherence_bonus + star_coherence_bonus`
/// with joint model blending (marginal/joint adaptive weight).
#[allow(clippy::too_many_arguments)]
pub fn generate_suggestions_gibbs(
    ball_probs: &[f64],
    star_probs: &[f64],
    count: usize,
    filter: Option<&StructuralFilter>,
    coherence: Option<&CoherenceScorer>,
    joint_model: Option<&crate::models::joint::JointConditionalModel>,
    star_pair_probs: Option<&[f64; 66]>,
    conditioner: Option<&BallStarConditioner>,
    star_coherence: Option<&StarCoherenceScorer>,
    n_chains: usize,
    temperature: f64,
    seed: u64,
    coherence_ball_w: Option<f64>,
    coherence_star_w: Option<f64>,
) -> Result<JackpotResult> {
    let cw = coherence_ball_w.unwrap_or(COHERENCE_WEIGHT);
    let scw_val = coherence_star_w.unwrap_or(STAR_COHERENCE_WEIGHT);
    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;
    let uniform_pair = 1.0 / 66.0;

    let burn_in: usize = 200;
    let thin: usize = 10;
    let n_chains_eff = n_chains.max(1);
    let samples_per_chain = (count / n_chains_eff).max(1);
    let total_iter_per_chain = burn_in + samples_per_chain * thin;

    // Compute optimal starting grid: top-5 balls, top-2 stars by probability
    let mut ball_indices_sorted: Vec<usize> = (0..ball_probs.len()).collect();
    ball_indices_sorted.sort_by(|&a, &b| {
        ball_probs[b].partial_cmp(&ball_probs[a]).unwrap_or(CmpOrdering::Equal)
    });
    let mut star_indices_sorted: Vec<usize> = (0..star_probs.len()).collect();
    star_indices_sorted.sort_by(|&a, &b| {
        star_probs[b].partial_cmp(&star_probs[a]).unwrap_or(CmpOrdering::Equal)
    });

    let mut optimal_balls = [0u8; 5];
    for (i, &idx) in ball_indices_sorted.iter().take(5).enumerate() {
        optimal_balls[i] = (idx + 1) as u8;
    }
    optimal_balls.sort();

    let mut optimal_stars = [0u8; 2];
    for (i, &idx) in star_indices_sorted.iter().take(2).enumerate() {
        optimal_stars[i] = (idx + 1) as u8;
    }
    optimal_stars.sort();

    // Pre-compute log ball probabilities (ratio vs uniform)
    let log_ball_probs: Vec<f64> = (0..ball_probs.len())
        .map(|i| (ball_probs[i] / uniform_ball).max(1e-30).ln())
        .collect();

    // Collect all samples from all chains
    let mut all_samples: Vec<([u8; 5], [u8; 2], f64)> =
        Vec::with_capacity(n_chains_eff * samples_per_chain);

    for chain_idx in 0..n_chains_eff {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chain_idx as u64));

        // Initialize chain state: start from optimal, perturb for chain_idx > 0
        let mut balls = optimal_balls;
        let mut stars = optimal_stars;

        if chain_idx > 0 {
            // Perturb 1-2 balls for starting diversity
            let n_perturb = if chain_idx % 2 == 0 { 1 } else { 2 };
            for _ in 0..n_perturb {
                let slot = rng.random_range(0usize..5);
                // Pick a new ball not already in the set, weighted by ball_probs
                let mut candidate_weights: Vec<f64> = Vec::with_capacity(50);
                let mut candidate_nums: Vec<u8> = Vec::with_capacity(50);
                for num in 1u8..=50 {
                    if !balls.contains(&num) {
                        candidate_weights.push(ball_probs[(num - 1) as usize].max(1e-30));
                        candidate_nums.push(num);
                    }
                }
                if let Ok(dist) = WeightedIndex::new(&candidate_weights) {
                    let chosen = candidate_nums[dist.sample(&mut rng)];
                    balls[slot] = chosen;
                    balls.sort();
                }
            }
        }

        // Run Gibbs sampling chain
        for iter in 0..total_iter_per_chain {
            // Choose a random position: 0-4 = ball slot, 5-6 = star slot
            let pos = rng.random_range(0usize..7);

            if pos < 5 {
                // Ball slot: compute conditional for each candidate ball (1-50)
                // excluding the other 4 selected balls
                let other_balls: Vec<u8> = balls
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != pos)
                    .map(|(_, &b)| b)
                    .collect();

                let mut cand_weights: Vec<f64> = Vec::with_capacity(50);
                let mut cand_nums: Vec<u8> = Vec::with_capacity(50);

                for num in 1u8..=50 {
                    if other_balls.contains(&num) {
                        continue;
                    }

                    // Marginal probability component
                    let marginal_log = log_ball_probs[(num - 1) as usize];

                    // Joint model bonus if available
                    let joint_bonus = if let Some(jm) = joint_model {
                        let mut test_balls = [0u8; 5];
                        for (i, &b) in other_balls.iter().enumerate() {
                            test_balls[i] = b;
                        }
                        test_balls[4] = num;
                        test_balls.sort();

                        let (joint_log, joint_conf) =
                            jm.score_balls_with_confidence(&test_balls);
                        if joint_log.is_finite() && joint_log > -100.0 {
                            let joint_weight = 0.15 + 0.55 * joint_conf; // v22: [0.15, 0.70]
                            joint_weight * joint_log
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    let log_weight = marginal_log + joint_bonus;
                    // Apply temperature: weight = exp(log_weight / temperature)
                    let weight = (log_weight / temperature).exp().max(1e-30);
                    cand_weights.push(weight);
                    cand_nums.push(num);
                }

                // Sample from conditional distribution
                if let Ok(dist) = WeightedIndex::new(&cand_weights) {
                    let chosen = cand_nums[dist.sample(&mut rng)];
                    balls[pos] = chosen;
                    balls.sort();
                }
            } else {
                // Star slot (pos - 5 = 0 or 1)
                let star_slot = pos - 5;
                let other_star = stars[1 - star_slot];

                let mut cand_weights: Vec<f64> = Vec::with_capacity(12);
                let mut cand_nums: Vec<u8> = Vec::with_capacity(12);

                for num in 1u8..=12 {
                    if num == other_star {
                        continue;
                    }

                    let log_weight = if let Some(pp) = star_pair_probs {
                        // Use pair probabilities for star scoring
                        let (s1, s2) = if num < other_star {
                            (num, other_star)
                        } else {
                            (other_star, num)
                        };
                        let pidx = crate::models::star_pair::pair_index(s1, s2);
                        (pp[pidx] / uniform_pair).max(1e-30).ln()
                    } else {
                        // Marginal star probability
                        (star_probs[(num - 1) as usize] / uniform_star).max(1e-30).ln()
                    };

                    let weight = (log_weight / temperature).exp().max(1e-30);
                    cand_weights.push(weight);
                    cand_nums.push(num);
                }

                if let Ok(dist) = WeightedIndex::new(&cand_weights) {
                    let chosen = cand_nums[dist.sample(&mut rng)];
                    stars[star_slot] = chosen;
                    stars.sort();
                }
            }

            // After burn-in, collect a sample every `thin` iterations
            if iter >= burn_in && (iter - burn_in) % thin == 0 {
                // Apply structural filter
                if let Some(f) = filter {
                    if !f.accept_balls(&balls) {
                        continue;
                    }
                }

                // Score the current state using the full scoring formula
                let score = gibbs_score_grid(
                    &balls,
                    &stars,
                    &log_ball_probs,
                    uniform_star,
                    uniform_pair,
                    coherence,
                    joint_model,
                    star_pair_probs,
                    conditioner,
                    star_coherence,
                    star_probs,
                    cw,
                    scw_val,
                );

                all_samples.push((balls, stars, score));
            }
        }
    }

    // Deduplicate: same balls+stars -> keep highest score
    let mut dedup_map: HashMap<([u8; 5], [u8; 2]), f64> = HashMap::new();
    for &(balls, stars, score) in &all_samples {
        let entry = dedup_map.entry((balls, stars)).or_insert(f64::NEG_INFINITY);
        if score > *entry {
            *entry = score;
        }
    }

    // Convert to suggestions (score stored as log, convert to linear for output)
    let mut suggestions: Vec<Suggestion> = dedup_map
        .into_iter()
        .map(|((balls, stars), log_score)| Suggestion {
            balls,
            stars,
            score: log_score.exp(),
        })
        .collect();

    // Sort by score descending, take top count
    suggestions.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(CmpOrdering::Equal)
    });
    suggestions.truncate(count);

    // Recalculer le score bayésien pur (sans coherence bonus) pour les métriques
    for s in &mut suggestions {
        s.score = compute_bayesian_score(&s.balls, &s.stars, ball_probs, star_probs);
    }

    // Compute JackpotResult stats (same formula as generate_suggestions_jackpot)
    let n_sugg = suggestions.len().max(1) as f64;
    let mean_score = suggestions.iter().map(|s| s.score).sum::<f64>() / n_sugg;
    let total_prob = mean_score * n_sugg / 139_838_160.0;
    let improvement = mean_score;

    let enumeration_size = (n_chains_eff * total_iter_per_chain) as u64;
    let filtered_size = all_samples.len() as u64;

    // v19 D2: Overlap-adjusted improvement factor
    let adjusted = compute_overlap_adjusted_factor(&suggestions, improvement);

    Ok(JackpotResult {
        suggestions,
        total_jackpot_probability: total_prob,
        enumeration_size,
        filtered_size,
        improvement_factor: improvement,
        adjusted_improvement_factor: adjusted,
    })
}

/// Score a grid using the same formula as `generate_suggestions_jackpot`.
/// Returns log-space score: log_ball_score + log_star_score + coherence_bonus + star_coherence_bonus.
#[allow(clippy::too_many_arguments)]
fn gibbs_score_grid(
    balls: &[u8; 5],
    stars: &[u8; 2],
    log_ball_probs: &[f64],
    uniform_star: f64,
    uniform_pair: f64,
    coherence: Option<&CoherenceScorer>,
    joint_model: Option<&crate::models::joint::JointConditionalModel>,
    star_pair_probs: Option<&[f64; 66]>,
    conditioner: Option<&BallStarConditioner>,
    star_coherence: Option<&StarCoherenceScorer>,
    star_probs: &[f64],
    cw: f64,
    scw: f64,
) -> f64 {
    // Ball score: sum of log(prob/uniform) for each ball
    let marginal_log_ball_score: f64 = balls
        .iter()
        .map(|&b| log_ball_probs[(b - 1) as usize])
        .sum();

    // Joint model blend (same as generate_suggestions_jackpot)
    let log_ball_score = if let Some(jm) = joint_model {
        let (joint_log, joint_conf) = jm.score_balls_with_confidence(balls);
        if joint_log.is_finite() && joint_log > -100.0 {
            let joint_weight = 0.15 + 0.55 * joint_conf; // v22: [0.15, 0.70]
            let marginal_weight = 1.0 - joint_weight;
            marginal_weight * marginal_log_ball_score + joint_weight * joint_log
        } else {
            marginal_log_ball_score
        }
    } else {
        marginal_log_ball_score
    };

    // Star score
    let base_star_score = if let Some(pp) = star_pair_probs {
        let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
        pp[pidx] / uniform_pair
    } else {
        (star_probs[(stars[0] - 1) as usize] / uniform_star)
            * (star_probs[(stars[1] - 1) as usize] / uniform_star)
    };

    let log_star_score = if let Some(cond) = conditioner {
        let cond_probs = cond.conditioned_pair_probs(balls);
        let cond_blend = cond.adaptive_blend(balls);
        let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
        let conditioned = cond_probs[pidx] / uniform_pair;
        let blended = cond_blend * conditioned + (1.0 - cond_blend) * base_star_score;
        blended.max(1e-30).ln()
    } else {
        base_star_score.max(1e-30).ln()
    };

    // Coherence bonus (balls only)
    let coherence_bonus = if let Some(cs) = coherence {
        let c = cs.score_balls(balls);
        cw * (c - 0.5).clamp(-0.5, 0.5)
    } else {
        0.0
    };

    // Star coherence bonus
    let star_coherence_bonus = if let Some(scs) = star_coherence {
        let sc = scs.score_star_pair(stars);
        scw * (sc - 0.5).clamp(-0.5, 0.5)
    } else {
        0.0
    };

    log_ball_score + log_star_score + coherence_bonus + star_coherence_bonus
}

// ═══════════════════════════════════════════════════════════════
// Diversification par profil mod-4 pour les 3 grilles finales
// ═══════════════════════════════════════════════════════════════

use crate::models::mod4_profile::{enumerate_profiles, extract_profile};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StarStrategy {
    Concentrated, // Même top-2 étoiles sur toutes les grilles (défaut)
    Triangular,   // 3 paires parmi top-3 étoiles
    Disjoint,     // Paires disjointes
}

impl StarStrategy {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "triangular" | "triangulaire" => Self::Triangular,
            "disjoint" | "disjointe" => Self::Disjoint,
            _ => Self::Concentrated,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Concentrated => "concentrée",
            Self::Triangular => "triangulaire",
            Self::Disjoint => "disjointe",
        }
    }
}

/// Résultat de la diversification : 3 grilles avec profils mod-4 distincts et étoiles diversifiées.
#[derive(Debug, Clone)]
pub struct DiverseGrids {
    pub grids: Vec<Suggestion>,
    pub profiles: Vec<[u8; 4]>,
    pub star_pairs: Vec<[u8; 2]>,
    pub star_strategy: &'static str,
}

/// Génère N grilles diversifiées par profil mod-4.
///
/// 1. Prédit les top-N profils mod-4 les plus probables (via la distribution de profils).
/// 2. Pour chaque profil, génère la meilleure grille contrainte à ce profil.
/// 3. Les N grilles ont naturellement des structures mod-4 différentes.
pub fn generate_diverse_grids(
    ball_probs: &[f64],
    star_probs: &[f64],
    draws: &[Draw],
    n_grids: usize,
    seed: u64,
    popularity: Option<&PopularityModel>,
) -> DiverseGrids {
    generate_diverse_grids_with_strategy(ball_probs, star_probs, draws, n_grids, seed, popularity, StarStrategy::Triangular)
}

pub fn generate_diverse_grids_with_strategy(
    ball_probs: &[f64],
    star_probs: &[f64],
    draws: &[Draw],
    n_grids: usize,
    seed: u64,
    _popularity: Option<&PopularityModel>,
    star_strategy: StarStrategy,
) -> DiverseGrids {
    // Scoring de cohérence historique (paires, triplets, sum, spread)
    let coherence = CoherenceScorer::from_history(draws, Pool::Balls);
    let profiles = enumerate_profiles(5); // 56 profils pour les boules

    // Scorer chaque profil : somme des probabilités des meilleures boules par classe mod-4
    let mut profile_scores: Vec<(usize, f64, [u8; 4])> = profiles
        .iter()
        .enumerate()
        .map(|(idx, profile)| {
            let score = score_profile_against_probs(profile, ball_probs);
            (idx, score, *profile)
        })
        .collect();

    // Bonus pour les profils historiquement fréquents (si assez de tirages)
    if draws.len() >= 20 {
        let mut profile_freq = HashMap::new();
        for d in draws {
            let p = extract_profile(&d.balls);
            *profile_freq.entry(p).or_insert(0usize) += 1;
        }
        let total = draws.len() as f64;
        for entry in &mut profile_scores {
            let freq = *profile_freq.get(&entry.2).unwrap_or(&0) as f64 / total;
            // Pondérer : 70% score probabiliste, 30% fréquence historique
            entry.1 = 0.7 * entry.1 + 0.3 * freq;
        }
    }

    profile_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Sélectionner les top-N profils DISTINCTS
    let mut selected_profiles: Vec<[u8; 4]> = Vec::new();
    for &(_, _, profile) in &profile_scores {
        if selected_profiles.len() >= n_grids {
            break;
        }
        // S'assurer que le profil est suffisamment distinct des déjà sélectionnés
        let distinct = selected_profiles.iter().all(|sp| {
            profile.iter().zip(sp.iter()).map(|(&a, &b)| (a as i8 - b as i8).unsigned_abs()).sum::<u8>() >= 2
        });
        if distinct {
            selected_profiles.push(profile);
        }
    }

    // Si pas assez de profils distincts, prendre les meilleurs restants
    if selected_profiles.len() < n_grids {
        for &(_, _, profile) in &profile_scores {
            if selected_profiles.len() >= n_grids {
                break;
            }
            if !selected_profiles.contains(&profile) {
                selected_profiles.push(profile);
            }
        }
    }

    // Sélectionner N paires d'étoiles selon la stratégie
    let star_pairs = select_star_pairs(star_probs, n_grids, star_strategy);

    let star_strategy_label = star_strategy.label();

    // Pour chaque profil, générer K=50 candidats (en variant les boules choisies par classe)
    // puis sélectionner gloutonement les n_grids maximisant les boules distinctes couvertes.
    let mut rng = StdRng::seed_from_u64(seed);
    let k_candidates = 200;

    // Générer les candidats par profil, triés par score combiné (prob × cohérence)
    let mut all_candidates: Vec<(usize, Suggestion)> = Vec::new(); // (profile_idx, grid)
    for (pi, profile) in selected_profiles.iter().enumerate() {
        let assigned_stars = star_pairs[pi % star_pairs.len()];
        let mut candidates = generate_profile_candidates(profile, ball_probs, star_probs, assigned_stars, k_candidates);
        // Rescorer les candidats avec la cohérence
        for c in &mut candidates {
            let coh = coherence.score_balls(&c.balls);
            c.score *= 0.5 + coh;
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        for c in candidates {
            all_candidates.push((pi, c));
        }
    }

    // Sélection exhaustive : 95% log-probabilité + 5% couverture
    // La diversité structurelle est déjà assurée par les profils modulaires
    // Grouper les candidats par profil
    let n_profiles = selected_profiles.len();
    let mut candidates_by_profile: Vec<Vec<&Suggestion>> = vec![Vec::new(); n_profiles];
    for (pi, candidate) in &all_candidates {
        candidates_by_profile[*pi].push(candidate);
    }

    let grids = if n_grids <= 3 && n_profiles >= n_grids {
        // Évaluation exhaustive de tous les triplets de profils × candidats
        let mut best_composite = f64::NEG_INFINITY;
        let mut best_combo: Vec<Suggestion> = Vec::new();

        // Générer toutes les combinaisons de n_grids profils distincts
        let profile_combos = combinations_indices(n_profiles, n_grids);

        for profile_combo in &profile_combos {
            let candidate_lists: Vec<&Vec<&Suggestion>> = profile_combo
                .iter()
                .map(|&pi| &candidates_by_profile[pi])
                .collect();

            let mut indices = vec![0usize; n_grids];
            let sizes: Vec<usize> = candidate_lists.iter().map(|c| c.len()).collect();

            if sizes.iter().any(|&s| s == 0) {
                continue;
            }

            loop {
                // Collecter les grilles de cette combinaison
                let combo: Vec<&Suggestion> = indices.iter().enumerate()
                    .map(|(g, &idx)| candidate_lists[g][idx])
                    .collect();

                // Score = 100% log-probabilité (maximiser P(5+2) par grille)
                // La diversité structurelle est assurée par les profils mod-4
                // (L1 distance >= 2 entre profils sélectionnés)
                let composite = {
                    let sum_log: f64 = combo.iter()
                        .map(|s| s.score.max(1e-30).ln())
                        .sum();
                    let avg_log = sum_log / combo.len() as f64;
                    let log_score = ((avg_log + 30.0) / 30.0).clamp(0.0, 1.0);
                    // Bonus couverture : combien de boules distinctes sur les N grilles
                    let distinct_balls: std::collections::HashSet<u8> = combo.iter()
                        .flat_map(|s| s.balls.iter().copied()).collect();
                    let coverage = distinct_balls.len() as f64 / (5.0 * combo.len() as f64);
                    0.95 * log_score + 0.05 * coverage
                };

                if composite > best_composite {
                    best_composite = composite;
                    best_combo = combo.iter().map(|s| (*s).clone()).collect();
                }

                // Incrémenter le compteur (produit cartésien)
                let mut carry = true;
                for g in (0..n_grids).rev() {
                    if carry {
                        indices[g] += 1;
                        if indices[g] >= sizes[g] {
                            indices[g] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
                if carry {
                    break;
                }
            }
        }

        if best_combo.is_empty() {
            // Fallback : meilleur candidat par profil
            selected_profiles.iter().enumerate()
                .take(n_grids)
                .map(|(pi, profile)| {
                    let assigned_stars = star_pairs[pi % star_pairs.len()];
                    best_grid_for_profile(profile, ball_probs, star_probs, assigned_stars, &mut rng)
                })
                .collect()
        } else {
            best_combo
        }
    } else {
        // Fallback pour n_grids > 3 : sélection gloutonne par score
        let mut grids = Vec::with_capacity(n_grids);
        let mut used_profiles: Vec<bool> = vec![false; n_profiles];
        for _ in 0..n_grids {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_grid = None;
            let mut best_pi = 0;
            for (pi, candidates) in candidates_by_profile.iter().enumerate() {
                if used_profiles[pi] { continue; }
                for c in candidates {
                    if c.score > best_score {
                        best_score = c.score;
                        best_grid = Some((*c).clone());
                        best_pi = pi;
                    }
                }
            }
            if let Some(grid) = best_grid {
                grids.push(grid);
                used_profiles[best_pi] = true;
            }
        }
        grids
    };

    DiverseGrids {
        grids,
        profiles: selected_profiles,
        star_pairs: star_pairs.clone(),
        star_strategy: star_strategy_label,
    }
}

/// Génère toutes les combinaisons de `k` indices parmi `n` (C(n,k)).
fn combinations_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    fn recurse(start: usize, n: usize, k: usize, combo: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if combo.len() == k {
            result.push(combo.clone());
            return;
        }
        for i in start..n {
            combo.push(i);
            recurse(i + 1, n, k, combo, result);
            combo.pop();
        }
    }
    recurse(0, n, k, &mut combo, &mut result);
    result
}

/// Score un profil mod-4 contre les probabilités marginales.
/// Pour chaque classe de résidu, prend les top-n_r boules par probabilité.
fn score_profile_against_probs(profile: &[u8; 4], ball_probs: &[f64]) -> f64 {
    let mut score = 1.0f64;
    let uniform = 1.0 / ball_probs.len() as f64;

    for (r, &n_r) in profile.iter().enumerate() {
        if n_r == 0 {
            continue;
        }
        // Boules de la classe r : indices k tels que k % 4 == r
        let mut class_probs: Vec<(usize, f64)> = ball_probs
            .iter()
            .enumerate()
            .filter(|(k, _)| k % 4 == r)
            .map(|(k, &p)| (k, p))
            .collect();
        class_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Prendre les n_r meilleures
        for &(_, p) in class_probs.iter().take(n_r as usize) {
            score *= p / uniform;
        }
    }

    score
}

/// Génère la meilleure grille pour un profil mod-4 donné, avec une paire d'étoiles imposée.
fn best_grid_for_profile(
    profile: &[u8; 4],
    ball_probs: &[f64],
    star_probs: &[f64],
    assigned_stars: [u8; 2],
    _rng: &mut StdRng,
) -> Suggestion {
    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    // Pour chaque classe mod-4, prendre les top n_r boules
    let mut balls = Vec::with_capacity(5);
    for (r, &n_r) in profile.iter().enumerate() {
        if n_r == 0 {
            continue;
        }
        let mut class_probs: Vec<(u8, f64)> = ball_probs
            .iter()
            .enumerate()
            .filter(|(k, _)| k % 4 == r)
            .map(|(k, &p)| ((k + 1) as u8, p))
            .collect();
        class_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(num, _) in class_probs.iter().take(n_r as usize) {
            balls.push(num);
        }
    }

    balls.sort();
    let mut balls_arr = [0u8; 5];
    for (i, &b) in balls.iter().take(5).enumerate() {
        balls_arr[i] = b;
    }

    let score: f64 = balls_arr.iter().map(|&b| ball_probs[(b - 1) as usize] / uniform_ball).product::<f64>()
        * assigned_stars.iter().map(|&s| star_probs[(s - 1) as usize] / uniform_star).product::<f64>();

    Suggestion {
        balls: balls_arr,
        stars: assigned_stars,
        score,
    }
}

/// Génère K candidats pour un profil mod-4 en variant les boules sélectionnées.
/// Pour chaque classe mod-4, au lieu de prendre uniquement le top-n_r, on explore aussi
/// les substitutions du 2ème ou 3ème meilleur candidat dans chaque classe.
fn generate_profile_candidates(
    profile: &[u8; 4],
    ball_probs: &[f64],
    star_probs: &[f64],
    assigned_stars: [u8; 2],
    k: usize,
) -> Vec<Suggestion> {
    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    // Pour chaque classe mod-4, lister les boules triées par prob
    let mut class_sorted: Vec<Vec<(u8, f64)>> = Vec::new();
    for (r, &n_r) in profile.iter().enumerate() {
        if n_r == 0 {
            class_sorted.push(Vec::new());
            continue;
        }
        let mut probs: Vec<(u8, f64)> = ball_probs
            .iter()
            .enumerate()
            .filter(|(k, _)| k % 4 == r)
            .map(|(k, &p)| ((k + 1) as u8, p))
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        class_sorted.push(probs);
    }

    let mut candidates = Vec::new();

    // Candidat 0 : le meilleur (baseline)
    let base = make_grid_from_classes(profile, &class_sorted, &[0; 4], ball_probs, star_probs, assigned_stars, uniform_ball, uniform_star);
    candidates.push(base);

    // Variantes : pour chaque classe non-vide, substituer le dernier élément par le suivant
    for r in 0..4 {
        if profile[r] == 0 || class_sorted[r].len() <= profile[r] as usize {
            continue;
        }
        let mut offsets = [0usize; 4];
        offsets[r] = 1; // décaler de 1 dans la classe r
        let variant = make_grid_from_classes(profile, &class_sorted, &offsets, ball_probs, star_probs, assigned_stars, uniform_ball, uniform_star);
        if !candidates.iter().any(|c: &Suggestion| c.balls == variant.balls) {
            candidates.push(variant);
        }
        if candidates.len() >= k { break; }
    }

    // Variantes combinées (2 classes décalées)
    if candidates.len() < k {
        for r1 in 0..4 {
            for r2 in (r1 + 1)..4 {
                if profile[r1] == 0 || profile[r2] == 0 { continue; }
                if class_sorted[r1].len() <= profile[r1] as usize { continue; }
                if class_sorted[r2].len() <= profile[r2] as usize { continue; }
                let mut offsets = [0usize; 4];
                offsets[r1] = 1;
                offsets[r2] = 1;
                let variant = make_grid_from_classes(profile, &class_sorted, &offsets, ball_probs, star_probs, assigned_stars, uniform_ball, uniform_star);
                if !candidates.iter().any(|c: &Suggestion| c.balls == variant.balls) {
                    candidates.push(variant);
                }
                if candidates.len() >= k { break; }
            }
            if candidates.len() >= k { break; }
        }
    }

    candidates.truncate(k);
    candidates
}

/// Construit une grille à partir des classes mod-4 triées avec des offsets.
#[allow(clippy::too_many_arguments)]
fn make_grid_from_classes(
    profile: &[u8; 4],
    class_sorted: &[Vec<(u8, f64)>],
    offsets: &[usize; 4],
    ball_probs: &[f64],
    star_probs: &[f64],
    assigned_stars: [u8; 2],
    uniform_ball: f64,
    uniform_star: f64,
) -> Suggestion {
    let mut balls = Vec::with_capacity(5);
    for (r, &n_r) in profile.iter().enumerate() {
        if n_r == 0 { continue; }
        let offset = offsets[r];
        // Prendre n_r boules en commençant par le top, mais le dernier est décalé de offset
        let available = &class_sorted[r];
        for rank in 0..(n_r as usize) {
            let actual_rank = if rank == (n_r as usize - 1) {
                // Dernier élément de la classe : décaler par offset
                (rank + offset).min(available.len() - 1)
            } else {
                rank
            };
            if actual_rank < available.len() {
                balls.push(available[actual_rank].0);
            }
        }
    }

    balls.sort();
    balls.dedup();
    // S'assurer d'avoir 5 boules
    while balls.len() < 5 {
        // Ajouter la prochaine meilleure boule pas encore prise
        for class in class_sorted {
            for &(b, _) in class {
                if !balls.contains(&b) {
                    balls.push(b);
                    if balls.len() >= 5 { break; }
                }
            }
            if balls.len() >= 5 { break; }
        }
        if balls.len() < 5 { break; } // sécurité
    }
    balls.sort();

    let mut balls_arr = [0u8; 5];
    for (i, &b) in balls.iter().take(5).enumerate() {
        balls_arr[i] = b;
    }

    let score: f64 = balls_arr.iter().map(|&b| ball_probs[(b - 1) as usize] / uniform_ball).product::<f64>()
        * assigned_stars.iter().map(|&s| star_probs[(s - 1) as usize] / uniform_star).product::<f64>();

    Suggestion { balls: balls_arr, stars: assigned_stars, score }
}

/// Sélectionne N paires d'étoiles selon la stratégie choisie.
fn select_star_pairs(star_probs: &[f64], n_pairs: usize, strategy: StarStrategy) -> Vec<[u8; 2]> {
    // Trier les étoiles par probabilité décroissante
    let mut star_ranked: Vec<(u8, f64)> = star_probs
        .iter()
        .enumerate()
        .map(|(i, &p)| ((i + 1) as u8, p))
        .collect();
    star_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if n_pairs <= 1 {
        let mut pair = [star_ranked[0].0, star_ranked[1].0];
        pair.sort();
        return vec![pair];
    }

    match strategy {
        StarStrategy::Concentrated => {
            // Même paire top-2 sur toutes les grilles
            let mut pair = [star_ranked[0].0, star_ranked[1].0];
            pair.sort();
            vec![pair; n_pairs]
        }
        StarStrategy::Disjoint => {
            // Paires disjointes : {s1,s2}, {s3,s4}, {s5,s6}
            let mut pairs = Vec::new();
            for i in 0..n_pairs.min(star_ranked.len() / 2) {
                let mut pair = [star_ranked[2 * i].0, star_ranked[2 * i + 1].0];
                pair.sort();
                pairs.push(pair);
            }
            pairs
        }
        StarStrategy::Triangular => {
            // Triangulaire : {s1,s2}, {s1,s3}, {s2,s3}
            if n_pairs >= 3 && star_ranked.len() >= 3 {
                let top3 = [star_ranked[0].0, star_ranked[1].0, star_ranked[2].0];
                let mut pairs = Vec::new();
                for i in 0..3 {
                    for j in (i + 1)..3 {
                        if pairs.len() < n_pairs {
                            let mut pair = [top3[i], top3[j]];
                            pair.sort();
                            pairs.push(pair);
                        }
                    }
                }
                pairs
            } else {
                // Fallback vers concentrated si pas assez d'étoiles
                let mut pair = [star_ranked[0].0, star_ranked[1].0];
                pair.sort();
                vec![pair; n_pairs]
            }
        }
    }
}

// ── Conviction scoring ──

#[derive(Debug, Clone)]
pub struct ConvictionScore {
    pub ball_entropy: f64,
    pub star_entropy: f64,
    pub ball_concentration: f64,
    pub star_concentration: f64,
    pub ball_agreement: f64,
    pub star_agreement: f64,
    pub overall: f64,
    pub verdict: ConvictionVerdict,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConvictionVerdict {
    HighConviction,
    MediumConviction,
    LowConviction,
}

impl std::fmt::Display for ConvictionVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighConviction => write!(f, "HAUTE"),
            Self::MediumConviction => write!(f, "MOYENNE"),
            Self::LowConviction => write!(f, "BASSE"),
        }
    }
}

fn shannon_entropy(probs: &[f64]) -> f64 {
    probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

pub fn compute_conviction(
    ball_probs: &[f64],
    star_probs: &[f64],
    ball_spread: &[f64],
    star_spread: &[f64],
) -> ConvictionScore {
    let ball_entropy = shannon_entropy(ball_probs);
    let star_entropy = shannon_entropy(star_probs);

    let h_max_balls = (ball_probs.len() as f64).ln();
    let h_max_stars = (star_probs.len() as f64).ln();

    let ball_concentration = 1.0 - ball_entropy / h_max_balls;
    let star_concentration = 1.0 - star_entropy / h_max_stars;

    let mean_ball_spread: f64 = ball_spread.iter().sum::<f64>() / ball_spread.len() as f64;
    let mean_ball_prob: f64 = ball_probs.iter().sum::<f64>() / ball_probs.len() as f64;
    let cv_balls = mean_ball_spread / mean_ball_prob.max(1e-15);
    let ball_agreement = (-2.0 * cv_balls).exp();

    let mean_star_spread: f64 = star_spread.iter().sum::<f64>() / star_spread.len() as f64;
    let mean_star_prob: f64 = star_probs.iter().sum::<f64>() / star_probs.len() as f64;
    let cv_stars = mean_star_spread / mean_star_prob.max(1e-15);
    let star_agreement = (-2.0 * cv_stars).exp();

    let overall = 0.4 * ball_concentration + 0.1 * star_concentration
        + 0.35 * ball_agreement + 0.15 * star_agreement;

    let verdict = if overall >= 0.55 {
        ConvictionVerdict::HighConviction
    } else if overall >= 0.30 {
        ConvictionVerdict::MediumConviction
    } else {
        ConvictionVerdict::LowConviction
    };

    ConvictionScore {
        ball_entropy,
        star_entropy,
        ball_concentration,
        star_concentration,
        ball_agreement,
        star_agreement,
        overall,
        verdict,
    }
}

// ═══════════════════════════════════════════════════════════════════
// v19 D3: Abstention sélective (Conformal Prediction Set)
// ═══════════════════════════════════════════════════════════════════

/// Abstention recommendation based on prediction set size.
#[derive(Debug, Clone, PartialEq)]
pub enum AbstentionAdvice {
    /// Model is confident — play
    Play,
    /// Medium confidence — play with caution
    Cautious,
    /// Low confidence — consider skipping this draw
    Skip,
}

/// Compute the conformal prediction set size for the current prediction.
/// The prediction set contains the smallest number of items that cover
/// a probability mass of `alpha` (default 0.90).
///
/// Returns (ball_set_size, star_set_size, advice).
pub fn compute_abstention(
    ball_probs: &[f64],
    star_probs: &[f64],
    alpha: f64,
) -> (usize, usize, AbstentionAdvice) {
    let ball_set = prediction_set_size(ball_probs, alpha);
    let star_set = prediction_set_size(star_probs, alpha);

    let advice = if ball_set <= 35 && star_set <= 8 {
        AbstentionAdvice::Play
    } else if ball_set <= 42 && star_set <= 10 {
        AbstentionAdvice::Cautious
    } else {
        AbstentionAdvice::Skip
    };

    (ball_set, star_set, advice)
}

/// Smallest set of items covering `alpha` probability mass.
fn prediction_set_size(probs: &[f64], alpha: f64) -> usize {
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumul = 0.0;
    let mut count = 0;
    for &(_, p) in &indexed {
        cumul += p;
        count += 1;
        if cumul >= alpha {
            break;
        }
    }
    count
}

/// RQA-Adaptive Temperature adjustment (v7, Phase 3.1).
/// Quand DET élevé (système déterministe) : sharpen agressif justifié.
/// Quand DET bas : relâcher. Retourne un facteur multiplicatif sur T.
/// T_adjusted = T_base × rqa_factor
pub fn rqa_temperature_factor(draws: &[Draw]) -> f64 {
    if draws.len() < 60 {
        return 1.0; // pas assez de données
    }
    // Build ball sum time series (chronological)
    let ball_sums: Vec<f64> = draws.iter().rev()
        .map(|d| d.balls.iter().map(|&b| b as f64).sum())
        .collect();

    // Lightweight RQA: Takens embedding (dim=3, delay=1), compute DET
    let dim = 3usize;
    let delay = 1usize;
    let n = ball_sums.len();
    let required = (dim - 1) * delay + 1;
    if n < required + 10 {
        return 1.0;
    }

    let m = n - (dim - 1) * delay;
    let embedded: Vec<[f64; 3]> = (0..m)
        .map(|i| [ball_sums[i], ball_sums[i + delay], ball_sums[i + 2 * delay]])
        .collect();

    // Compute std for epsilon
    let mean = ball_sums.iter().sum::<f64>() / n as f64;
    let variance = ball_sums.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let epsilon = 0.2 * std_dev;
    if epsilon < 1e-15 {
        return 1.0;
    }
    let eps_sq = epsilon * epsilon;

    // Count recurrence and diagonal lines for DET
    let n_emb = embedded.len();
    let mut total_recurrence = 0u64;
    let mut diag_points = 0u64;

    // Only check diagonals k=1..min(n_emb, 50) for efficiency
    let max_diag = n_emb.min(50);
    for k in 1..max_diag {
        let mut current_line = 0u32;
        for i in 0..(n_emb - k) {
            let dist_sq: f64 = (0..3).map(|d| (embedded[i][d] - embedded[i + k][d]).powi(2)).sum();
            if dist_sq < eps_sq {
                total_recurrence += 1;
                current_line += 1;
            } else {
                if current_line >= 2 {
                    diag_points += current_line as u64;
                }
                current_line = 0;
            }
        }
        if current_line >= 2 {
            diag_points += current_line as u64;
        }
    }

    let det = if total_recurrence > 0 {
        diag_points as f64 / total_recurrence as f64
    } else {
        0.0
    };

    // DET baseline for random: ~0.3-0.4. Map to temperature factor.
    // DET > 0.5 → system is predictable → sharpen (factor < 1)
    // DET < 0.3 → system is chaotic → relax (factor > 1)
    let det_baseline = 0.35;
    let det_std = 0.10;
    let predictability_score = (det - det_baseline) / det_std;

    // T_adjusted = T_base × exp(-0.1 × predictability_score)
    // Clamped to [0.7, 1.3] to avoid extreme adjustments
    let factor = (-0.1 * predictability_score).exp().clamp(0.70, 1.30);
    factor
}

/// Température forcée pour le mode few-grid (3-10 grilles).
/// Avec peu de grilles, on concentre agressivement sur nos meilleurs paris.
/// Retourne (T_balls, T_stars).
pub fn few_grid_temperature(n_grids: usize) -> (f64, f64) {
    match n_grids {
        0..=3 => (0.55, 0.12),  // v23b: star T maximally aggressive
        4..=6 => (0.60, 0.16),  // v23b
        7..=10 => (0.65, 0.20), // v23b
        _ => (0.70, 0.25),      // >10: more conservative
    }
}

/// Sélection optimale de N grilles maximisant P(au moins un 5+2).
/// v7: structured overlap — favorise 1-2 boules communes (structuré) et
/// pénalise 0 (trop dispersé) ou ≥3 (trop similaire).
/// Basé sur Liu, Liu, Teo (2024): structurer le chevauchement entre tickets
/// améliore P(gain) vs couverture pure ou diversité pure.
pub fn select_optimal_n_grids(
    candidates: &[Suggestion],
    n_grids: usize,
    max_common_balls: usize,
    max_common_stars: usize,
) -> Vec<Suggestion> {
    if candidates.is_empty() || n_grids == 0 {
        return vec![];
    }

    // Candidates should already be sorted by score descending
    let mut selected: Vec<Suggestion> = Vec::with_capacity(n_grids);

    // Grille 1 = plus haute P(5+2)
    selected.push(candidates[0].clone());

    // Grilles suivantes avec scoring overlap-aware (v7)
    // overlap_bonus favorise 1-2 boules communes (structuré)
    // et pénalise 0 (trop dispersé) ou ≥3 (trop similaire)
    for _ in 1..n_grids {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = None;

        for (ci, candidate) in candidates.iter().enumerate() {
            // Skip si déjà sélectionné
            if selected.iter().any(|s| s.balls == candidate.balls && s.stars == candidate.stars) {
                continue;
            }

            // Vérifier la contrainte hard sur étoiles et boules max
            let hard_ok = selected.iter().all(|s| {
                let common_b = candidate.balls.iter().filter(|b| s.balls.contains(b)).count();
                let common_s = candidate.stars.iter().filter(|st| s.stars.contains(st)).count();
                common_b <= max_common_balls && common_s <= max_common_stars
            });
            if !hard_ok {
                continue;
            }

            // v19 F6: Marginal contribution to P(at least 1 win) with overlap bonus.
            // P(no win before) = Π(1 - p_i) for selected grids
            // Marginal contribution = P(no win before) × p_candidate
            let p_no_win_before: f64 = selected.iter()
                .map(|s| 1.0 - s.score / 139_838_160.0)
                .product::<f64>()
                .max(1e-30);
            let marginal = p_no_win_before * (candidate.score / 139_838_160.0);

            // v23: Structured overlap (Liu-Teo 2024) — minimize overlap VARIANCE
            // instead of using discrete bonuses. Overlaps should be UNIFORMLY distributed.
            let overlaps: Vec<usize> = selected.iter().map(|s| {
                candidate.balls.iter().filter(|b| s.balls.contains(b)).count()
            }).collect();
            let mean_overlap = overlaps.iter().sum::<usize>() as f64 / overlaps.len().max(1) as f64;
            let overlap_variance = if overlaps.len() > 1 {
                overlaps.iter().map(|&o| (o as f64 - mean_overlap).powi(2)).sum::<f64>() / overlaps.len() as f64
            } else { 0.0 };

            // Target: mean overlap ≈ 1.0-1.5, variance ≈ 0.0
            let mean_penalty = (-(mean_overlap - 1.2).powi(2) / 2.0).exp(); // peak at 1.2 common
            let variance_penalty = (-overlap_variance / 0.5).exp(); // penalize high variance
            let overlap_bonus = mean_penalty * variance_penalty;

            let score = marginal * overlap_bonus;
            if score > best_score {
                best_score = score;
                best_idx = Some(ci);
            }
        }

        if let Some(idx) = best_idx {
            selected.push(candidates[idx].clone());
        } else {
            // Fallback: plus de candidats valides, prendre le meilleur restant sans contrainte
            for candidate in candidates {
                if selected.len() >= n_grids {
                    break;
                }
                if !selected.iter().any(|s| s.balls == candidate.balls && s.stars == candidate.stars) {
                    selected.push(candidate.clone());
                }
            }
            break;
        }
    }

    selected
}

/// v21: Sélection exacte pour N≤5 grilles.
/// Énumère C(top_k, N) combinaisons et retourne celle maximisant P(≥1 hit).
pub fn select_optimal_n_grids_exact(
    candidates: &[Suggestion],
    n_grids: usize,
    max_common_balls: usize,
) -> Vec<Suggestion> {
    let top_k = candidates.len().min(100); // cap for tractability
    let top = &candidates[..top_k];

    if n_grids > 5 || top_k < n_grids || n_grids < 2 {
        // Fall back to greedy for large N
        return select_optimal_n_grids(candidates, n_grids, 4, 1);
    }

    let total = 139_838_160.0_f64;

    // Check overlap constraint between two grids
    let check_overlap = |a: &Suggestion, b: &Suggestion| -> bool {
        let common = a.balls.iter().filter(|x| b.balls.contains(x)).count();
        common <= max_common_balls
    };

    let mut best_p_any: f64 = 0.0;
    let mut best_indices: Vec<usize> = Vec::new();

    match n_grids {
        2 => {
            for i in 0..top_k {
                for j in (i + 1)..top_k {
                    if !check_overlap(&top[i], &top[j]) { continue; }
                    let p_none = (1.0 - top[i].score / total) * (1.0 - top[j].score / total);
                    let p_any = 1.0 - p_none;
                    if p_any > best_p_any {
                        best_p_any = p_any;
                        best_indices = vec![i, j];
                    }
                }
            }
        },
        3 => {
            for i in 0..top_k {
                for j in (i + 1)..top_k {
                    if !check_overlap(&top[i], &top[j]) { continue; }
                    for k in (j + 1)..top_k {
                        if !check_overlap(&top[i], &top[k]) { continue; }
                        if !check_overlap(&top[j], &top[k]) { continue; }
                        let p_none = (1.0 - top[i].score / total)
                            * (1.0 - top[j].score / total)
                            * (1.0 - top[k].score / total);
                        let p_any = 1.0 - p_none;
                        if p_any > best_p_any {
                            best_p_any = p_any;
                            best_indices = vec![i, j, k];
                        }
                    }
                }
            }
        },
        4 => {
            for i in 0..top_k {
                for j in (i + 1)..top_k {
                    if !check_overlap(&top[i], &top[j]) { continue; }
                    for k in (j + 1)..top_k {
                        if !check_overlap(&top[i], &top[k]) { continue; }
                        if !check_overlap(&top[j], &top[k]) { continue; }
                        for l in (k + 1)..top_k {
                            if !check_overlap(&top[i], &top[l]) { continue; }
                            if !check_overlap(&top[j], &top[l]) { continue; }
                            if !check_overlap(&top[k], &top[l]) { continue; }
                            let p_none = (1.0 - top[i].score / total)
                                * (1.0 - top[j].score / total)
                                * (1.0 - top[k].score / total)
                                * (1.0 - top[l].score / total);
                            let p_any = 1.0 - p_none;
                            if p_any > best_p_any {
                                best_p_any = p_any;
                                best_indices = vec![i, j, k, l];
                            }
                        }
                    }
                }
            }
        },
        5 => {
            for i in 0..top_k {
                for j in (i + 1)..top_k {
                    if !check_overlap(&top[i], &top[j]) { continue; }
                    for k in (j + 1)..top_k {
                        if !check_overlap(&top[i], &top[k]) { continue; }
                        if !check_overlap(&top[j], &top[k]) { continue; }
                        for l in (k + 1)..top_k {
                            if !check_overlap(&top[i], &top[l]) { continue; }
                            if !check_overlap(&top[j], &top[l]) { continue; }
                            if !check_overlap(&top[k], &top[l]) { continue; }
                            for m in (l + 1)..top_k {
                                if !check_overlap(&top[i], &top[m]) { continue; }
                                if !check_overlap(&top[j], &top[m]) { continue; }
                                if !check_overlap(&top[k], &top[m]) { continue; }
                                if !check_overlap(&top[l], &top[m]) { continue; }
                                let p_none = (1.0 - top[i].score / total)
                                    * (1.0 - top[j].score / total)
                                    * (1.0 - top[k].score / total)
                                    * (1.0 - top[l].score / total)
                                    * (1.0 - top[m].score / total);
                                let p_any = 1.0 - p_none;
                                if p_any > best_p_any {
                                    best_p_any = p_any;
                                    best_indices = vec![i, j, k, l, m];
                                }
                            }
                        }
                    }
                }
            }
        },
        _ => unreachable!(),
    }

    if best_indices.is_empty() {
        // No valid combination found with overlap constraint, relax and use greedy
        return select_optimal_n_grids(candidates, n_grids, 4, 1);
    }

    best_indices.iter().map(|&i| top[i].clone()).collect()
}

/// G7: Minimax grid selection — maximize worst-case P(5+2) under uncertainty.
/// epsilon controls the perturbation budget (proportional to model disagreement).
pub fn select_minimax_grids(
    candidates: &[Suggestion],
    n_grids: usize,
    epsilon: f64,
    max_common_balls: usize,
    max_common_stars: usize,
) -> Vec<Suggestion> {
    if candidates.is_empty() || n_grids == 0 {
        return Vec::new();
    }

    // Worst-case score: reduce each candidate's score proportionally to epsilon
    // Grids with more diverse number spread are more robust to perturbations
    let mut scored: Vec<(usize, f64)> = candidates.iter().enumerate().map(|(i, s)| {
        // Diversity proxy: spread of balls (max - min) normalized
        let ball_spread = (s.balls[4] - s.balls[0]) as f64 / 49.0;
        // More spread = more robust to local probability shifts
        let robustness = 0.5 + 0.5 * ball_spread;
        let worst_case = s.score * (1.0 - epsilon * (1.0 - robustness));
        (i, worst_case)
    }).collect();

    // Sort by worst-case score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy selection with diversity constraints (same as select_optimal_n_grids)
    let mut selected: Vec<Suggestion> = Vec::with_capacity(n_grids);
    for (idx, _worst_case_score) in scored {
        let cand = &candidates[idx];
        let diverse = selected.iter().all(|s| {
            let common_balls = cand.balls.iter().filter(|b| s.balls.contains(b)).count();
            let common_stars = cand.stars.iter().filter(|st| s.stars.contains(st)).count();
            common_balls <= max_common_balls && common_stars <= max_common_stars
        });
        if diverse {
            selected.push(cand.clone());
            if selected.len() >= n_grids {
                break;
            }
        }
    }

    selected
}

/// v17: Simulated annealing refinement for grid selection.
///
/// Starts from the greedy selection and improves via random swaps.
/// Objective: maximize P(at least one 5+2) = 1 - Π(1 - score_i/139_838_160)
pub fn select_optimal_n_grids_sa(
    candidates: &[Suggestion],
    n_grids: usize,
    max_common_balls: usize,
    max_common_stars: usize,
    seed: u64,
) -> Vec<Suggestion> {
    if candidates.len() <= n_grids || n_grids <= 1 {
        return select_optimal_n_grids(candidates, n_grids, max_common_balls, max_common_stars);
    }

    // Start with greedy solution
    let mut best = select_optimal_n_grids(candidates, n_grids, max_common_balls, max_common_stars);
    if best.is_empty() {
        return best;
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Objective: P(at least one hit) = 1 - Π(1 - p_i)
    // where p_i = score_i / 139_838_160
    let objective = |selection: &[Suggestion]| -> f64 {
        let log_miss: f64 = selection
            .iter()
            .map(|s| (1.0 - s.score / 139_838_160.0).max(1e-30).ln())
            .sum();
        1.0 - log_miss.exp()
    };

    // Overlap constraint check: does candidates[new_idx] fit in selection at position `slot`?
    let check_constraints =
        |selection: &[Suggestion], new_idx: usize, slot: usize| -> bool {
            let candidate = &candidates[new_idx];
            for (j, s) in selection.iter().enumerate() {
                if j == slot {
                    continue;
                }
                let common_b = candidate
                    .balls
                    .iter()
                    .filter(|b| s.balls.contains(b))
                    .count();
                let common_s = candidate
                    .stars
                    .iter()
                    .filter(|st| s.stars.contains(st))
                    .count();
                if common_b > max_common_balls || common_s > max_common_stars {
                    return false;
                }
            }
            true
        };

    let mut current = best.clone();
    let mut current_obj = objective(&current);
    let mut best_obj = current_obj;

    // SA parameters — temperatures are very small because P(5+2) per grid is ~10^-8
    let t_max: f64 = 1e-10;
    let t_min: f64 = 1e-14;
    let n_iterations: usize = 5000;
    let cooling_rate: f64 = (t_min / t_max).powf(1.0 / n_iterations as f64);

    let mut temp = t_max;

    for _ in 0..n_iterations {
        // Choose a random slot to swap
        let slot = rng.random_range(0..n_grids);

        // Choose a random candidate to swap in
        let new_idx = rng.random_range(0..candidates.len());

        // Skip if same grid already in selection
        if current
            .iter()
            .any(|s| s.balls == candidates[new_idx].balls && s.stars == candidates[new_idx].stars)
        {
            temp *= cooling_rate;
            continue;
        }

        // Check overlap constraints
        if !check_constraints(&current, new_idx, slot) {
            temp *= cooling_rate;
            continue;
        }

        // Try the swap
        let old = current[slot].clone();
        current[slot] = candidates[new_idx].clone();
        let new_obj = objective(&current);

        let delta = new_obj - current_obj;

        if delta > 0.0 || rng.random::<f64>() < (delta / temp).exp() {
            // Accept
            current_obj = new_obj;
            if current_obj > best_obj {
                best = current.clone();
                best_obj = current_obj;
            }
        } else {
            // Reject — restore
            current[slot] = old;
        }

        temp *= cooling_rate;
    }

    best
}

/// - LowConviction → T=1.0 (pas de sharpening)
pub fn conviction_temperature(verdict: &ConvictionVerdict) -> f64 {
    match verdict {
        ConvictionVerdict::HighConviction => 0.60,
        ConvictionVerdict::MediumConviction => 0.85,
        ConvictionVerdict::LowConviction => 1.0,
    }
}

/// Skill-based temperature: T = exp(-skill / σ)
/// With σ=0.035: skill=0.01 → T≈0.75, skill=0.02 → T≈0.56, skill=0.05 → T≈0.24
/// More aggressive than the previous linear formula for typical skill levels.
const SKILL_TEMP_SIGMA: f64 = 0.035;

/// Compute temperature from calibrated skill level.
/// skill = average bits above uniform for this pool.
/// v13: soft clamp with floor at 0.08 (was hard clamp at 0.12).
/// The formula naturally asymptotes to `floor` without a hard clamp.
pub fn skill_temperature(skill: f64) -> f64 {
    if skill <= 0.0 {
        1.0 // no skill → no sharpening
    } else {
        let base = (-skill / SKILL_TEMP_SIGMA).exp();
        let floor = 0.08;
        floor + (1.0 - floor) * base
    }
}

/// Températures séparées balls/stars basées sur:
/// 1. La conviction de l'ensemble (concentration + agreement)
/// 2. Le skill calibré optionnel (si fourni, override la conviction)
///
/// ball_skill / star_skill = total weighted skill from calibration.
pub fn conviction_temperature_split(conviction: &ConvictionScore) -> (f64, f64) {
    conviction_temperature_split_with_skill(conviction, None, None)
}

/// Version avec skill calibré pour un sharpening informé par la calibration.
pub fn conviction_temperature_split_with_skill(
    conviction: &ConvictionScore,
    ball_skill: Option<f64>,
    star_skill: Option<f64>,
) -> (f64, f64) {
    let ball_temp = match ball_skill {
        Some(s) if s > 0.0 => skill_temperature(s),
        _ => {
            // v13: Smooth sigmoid interpolation (no step discontinuities).
            // ball_score ≈ 0 → T ≈ 0.12, ball_score = 0.3 → T ≈ 0.085, ball_score ≈ 1 → T ≈ 0.05
            let ball_score = 0.7 * conviction.ball_concentration + 0.3 * conviction.ball_agreement;
            let sigmoid = 1.0 / (1.0 + (-8.0 * (ball_score - 0.3)).exp());
            (0.12 - 0.07 * sigmoid).clamp(0.05, 0.15)
        }
    };

    let star_temp = match star_skill {
        Some(s) if s > 0.0 => skill_temperature(s).min(0.35),
        _ => {
            // v23b: Maximum star sharpening — floor at 0.08 for high conviction.
            // star_score ≈ 0 → T ≈ 0.28, star_score = 0.3 → T ≈ 0.16, star_score ≈ 1 → T ≈ 0.08
            let star_score = 0.7 * conviction.star_concentration + 0.3 * conviction.star_agreement;
            let sigmoid = 1.0 / (1.0 + (-12.0 * (star_score - 0.25)).exp());
            (0.28 - 0.20 * sigmoid).clamp(0.08, 0.30)
        }
    };

    (ball_temp, star_temp)
}

/// Conformal prediction set for abstention recommendation
pub struct ConformalPrediction {
    pub ball_set_size: usize,
    pub star_set_size: usize,
    pub ball_entropy: f64,
    pub star_entropy: f64,
    pub recommendation: &'static str,
}

/// Compute conformal prediction sets from ensemble probabilities.
/// prediction set = numbers needed to cover (1-alpha) probability mass.
pub fn conformal_prediction(ball_probs: &[f64], star_probs: &[f64], alpha: f64) -> ConformalPrediction {
    // Reuse existing prediction_set_size (takes coverage target directly)
    let coverage = 1.0 - alpha;
    let ball_set_size = prediction_set_size(ball_probs, coverage);
    let star_set_size = prediction_set_size(star_probs, coverage);

    let ball_entropy = -ball_probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f64>();
    let star_entropy = -star_probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f64>();

    let recommendation = if ball_set_size < 35 {
        "JOUER"
    } else if ball_set_size <= 42 {
        "PRUDENT"
    } else {
        "PASSER"
    };

    ConformalPrediction {
        ball_set_size,
        star_set_size,
        ball_entropy,
        star_entropy,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conviction_uniform() {
        let ball_probs = vec![1.0 / 50.0; 50];
        let star_probs = vec![1.0 / 12.0; 12];
        let ball_spread = vec![0.001; 50];
        let star_spread = vec![0.001; 12];

        let conv = compute_conviction(&ball_probs, &star_probs, &ball_spread, &star_spread);
        assert!(conv.ball_concentration.abs() < 0.01,
            "uniform balls should have ~0 concentration, got {}", conv.ball_concentration);
        assert!(conv.star_concentration.abs() < 0.01,
            "uniform stars should have ~0 concentration, got {}", conv.star_concentration);
    }

    #[test]
    fn test_conviction_concentrated() {
        let mut ball_probs = vec![0.005; 50];
        ball_probs[0] = 0.5;
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();

        let star_probs = vec![1.0 / 12.0; 12];
        let ball_spread = vec![0.001; 50];
        let star_spread = vec![0.001; 12];

        let conv = compute_conviction(&ball_probs, &star_probs, &ball_spread, &star_spread);
        assert!(conv.ball_concentration > 0.2,
            "concentrated balls should have high concentration, got {}", conv.ball_concentration);
    }

    #[test]
    fn test_conviction_verdict_thresholds() {
        let ball_probs = vec![1.0 / 50.0; 50];
        let star_probs = vec![1.0 / 12.0; 12];
        // High spread → low agreement → low conviction
        let ball_spread = vec![0.05; 50];
        let star_spread = vec![0.1; 12];

        let conv = compute_conviction(&ball_probs, &star_probs, &ball_spread, &star_spread);
        assert_eq!(conv.verdict, ConvictionVerdict::LowConviction,
            "uniform probs + high spread → low conviction, got {:?} (overall={})", conv.verdict, conv.overall);
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        let probs = vec![0.25; 4];
        let h = shannon_entropy(&probs);
        let expected = (4.0_f64).ln();
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let mut probs = vec![0.0; 4];
        probs[0] = 1.0;
        let h = shannon_entropy(&probs);
        assert!(h.abs() < 1e-10, "deterministic should have 0 entropy, got {h}");
    }

    #[test]
    fn test_date_seed_format() {
        let seed = date_seed();
        assert!(seed >= 20_000_000, "seed trop petit: {seed}");
        assert!(seed <= 99_991_231, "seed trop grand: {seed}");
        // Vérifier que c'est bien 8 chiffres
        let s = seed.to_string();
        assert_eq!(s.len(), 8, "seed devrait avoir 8 chiffres: {s}");
    }

    #[test]
    fn test_date_seed_deterministic() {
        let s1 = date_seed();
        let s2 = date_seed();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_ball_distance_identical() {
        let a = [1, 2, 3, 4, 5];
        let b = [1, 2, 3, 4, 5];
        assert_eq!(ball_distance(&a, &b), 0);
    }

    #[test]
    fn test_ball_distance_completely_different() {
        let a = [1, 2, 3, 4, 5];
        let b = [6, 7, 8, 9, 10];
        assert_eq!(ball_distance(&a, &b), 5);
    }

    #[test]
    fn test_ball_distance_partial() {
        let a = [1, 2, 3, 4, 5];
        let b = [1, 2, 3, 8, 9];
        assert_eq!(ball_distance(&a, &b), 2);
    }

    #[test]
    fn test_oversampling_improves_score() {
        let n = 50;
        let ball_probs: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.02).collect();
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();

        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];

        // Sans oversampling (oversample=1)
        let no_over = generate_suggestions_from_probs(&ball_probs, &star_probs, 5, 42, 1, 0).unwrap();
        // Avec oversampling (oversample=20)
        let with_over = generate_suggestions_from_probs(&ball_probs, &star_probs, 5, 42, 20, 0).unwrap();

        // Le meilleur score avec oversampling devrait être >= sans
        assert!(with_over[0].score >= no_over[0].score,
            "oversampling devrait améliorer le meilleur score: {} vs {}", with_over[0].score, no_over[0].score);
    }

    #[test]
    fn test_diversity_enforced() {
        let n = 50;
        let ball_probs: Vec<f64> = vec![1.0 / n as f64; n];
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];
        let min_diff = 2;

        let suggestions = generate_suggestions_from_probs(&ball_probs, &star_probs, 5, 42, 20, min_diff).unwrap();

        for i in 0..suggestions.len() {
            for j in (i + 1)..suggestions.len() {
                let dist = ball_distance(&suggestions[i].balls, &suggestions[j].balls);
                assert!(dist >= min_diff,
                    "Grilles {} et {} trop similaires (distance={}, min={}): {:?} vs {:?}",
                    i, j, dist, min_diff, suggestions[i].balls, suggestions[j].balls);
            }
        }
    }

    #[test]
    fn test_exact_count_returned() {
        let ball_probs: Vec<f64> = vec![1.0 / 50.0; 50];
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];

        for count in [1, 3, 5, 10] {
            let suggestions = generate_suggestions_from_probs(&ball_probs, &star_probs, count, 42, 10, 2).unwrap();
            assert_eq!(suggestions.len(), count, "devrait retourner exactement {count} suggestions");
        }
    }

    #[test]
    fn test_optimal_grid_picks_highest_probs() {
        let mut ball_probs = vec![0.01; 50];
        // Les 5 plus hautes probas aux indices 9,19,29,39,49 (numéros 10,20,30,40,50)
        for &i in &[9, 19, 29, 39, 49] {
            ball_probs[i] = 0.10;
        }
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();

        let star_probs = vec![1.0 / 12.0; 12];

        let grid = optimal_grid(&ball_probs, &star_probs);
        assert_eq!(grid.balls, [10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_optimal_grid_sorted() {
        // Probas décroissantes : indice 49 le plus haut, 0 le plus bas
        let ball_probs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();

        let star_probs: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let total: f64 = star_probs.iter().sum();
        let star_probs: Vec<f64> = star_probs.iter().map(|p| p / total).collect();

        let grid = optimal_grid(&ball_probs, &star_probs);
        // Boules triées
        assert!(grid.balls.windows(2).all(|w| w[0] <= w[1]));
        // Étoiles triées
        assert!(grid.stars[0] <= grid.stars[1]);
        // Top 5 boules = 46,47,48,49,50
        assert_eq!(grid.balls, [46, 47, 48, 49, 50]);
        // Top 2 étoiles = 11,12
        assert_eq!(grid.stars, [11, 12]);
    }

    #[test]
    fn test_compute_bayesian_score_uniform() {
        let ball_probs = vec![1.0 / 50.0; 50];
        let star_probs = vec![1.0 / 12.0; 12];
        let balls = [1, 2, 3, 4, 5];
        let stars = [1, 2];
        let score = compute_bayesian_score(&balls, &stars, &ball_probs, &star_probs);
        assert!((score - 1.0).abs() < 1e-10, "score uniforme devrait être 1.0, got {score}");
    }

    #[test]
    fn test_compute_bayesian_score_peaked() {
        let mut ball_probs = vec![0.01; 50];
        for &i in &[0, 1, 2, 3, 4] {
            ball_probs[i] = 0.10;
        }
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();
        let star_probs = vec![1.0 / 12.0; 12];

        let balls_good = [1, 2, 3, 4, 5];
        let balls_bad = [46, 47, 48, 49, 50];
        let stars = [1, 2];

        let score_good = compute_bayesian_score(&balls_good, &stars, &ball_probs, &star_probs);
        let score_bad = compute_bayesian_score(&balls_bad, &stars, &ball_probs, &star_probs);
        assert!(score_good > score_bad, "bons numéros devraient scorer plus haut");
    }

    #[test]
    fn test_structural_filter_accept() {
        let filter = StructuralFilter {
            sum_range: (50, 200),
            max_consecutive: 3,
            odd_range: (1, 4),
            spread_range: (5, 49),
        };
        // sum=100, odd=2 (10,20 pairs + 5,15,25 impairs → 3 impairs) → passe
        assert!(filter.accept_balls(&[5, 10, 20, 30, 35]));
        // sum=240 > 200 → rejeté
        assert!(!filter.accept_balls(&[46, 47, 48, 49, 50]));
        // sum=15 < 50 → rejeté
        assert!(!filter.accept_balls(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_structural_filter_consecutive() {
        let filter = StructuralFilter {
            sum_range: (10, 250),
            max_consecutive: 2,
            odd_range: (0, 5),
            spread_range: (0, 49),
        };
        // 3 consécutifs : 1,2,3 → rejeté (max_consecutive=2)
        assert!(!filter.accept_balls(&[1, 2, 3, 10, 20]));
        // 2 consécutifs max : ok
        assert!(filter.accept_balls(&[1, 2, 10, 20, 30]));
    }

    #[test]
    fn test_structural_filter_odd_range() {
        let filter = StructuralFilter {
            sum_range: (10, 250),
            max_consecutive: 5,
            odd_range: (1, 4),
            spread_range: (0, 49),
        };
        // 5 impairs → hors range (0, 4)
        assert!(!filter.accept_balls(&[1, 3, 5, 7, 9]));
        // 0 impairs → hors range (1, 4)
        assert!(!filter.accept_balls(&[2, 4, 6, 8, 10]));
        // 3 impairs → ok
        assert!(filter.accept_balls(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_coherence_scorer_basic() {
        let draws = crate::models::make_test_draws(50);
        let scorer = CoherenceScorer::from_history(&draws, Pool::Balls);
        assert!(scorer.mean_sum > 0.0);
        assert!(scorer.std_sum > 0.0);
        assert!(scorer.mean_spread > 0.0);
    }

    #[test]
    fn test_coherence_score_range() {
        let draws = crate::models::make_test_draws(50);
        let scorer = CoherenceScorer::from_history(&draws, Pool::Balls);
        let score = scorer.score_balls(&[10, 20, 30, 40, 50]);
        assert!(score >= 0.0, "score devrait être >= 0, got {score}");
        assert!(score <= 2.0, "score devrait être raisonnable, got {score}");
    }

    #[test]
    fn test_generate_suggestions_joint() {
        let draws = crate::models::make_test_draws(50);
        let ball_probs: Vec<f64> = vec![1.0 / 50.0; 50];
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];
        let suggestions =
            generate_suggestions_joint(&ball_probs, &star_probs, &draws, 5, 42, 10, 2, None)
                .unwrap();
        assert_eq!(suggestions.len(), 5);
        for s in &suggestions {
            assert!(s.score > 0.0);
        }
    }

    #[test]
    fn test_seed_determinism() {
        let ball_probs: Vec<f64> = vec![1.0 / 50.0; 50];
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];

        let s1 = generate_suggestions_from_probs(&ball_probs, &star_probs, 5, 123, 10, 2).unwrap();
        let s2 = generate_suggestions_from_probs(&ball_probs, &star_probs, 5, 123, 10, 2).unwrap();

        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.balls, b.balls);
            assert_eq!(a.stars, b.stars);
            assert_eq!(a.score, b.score);
        }
    }

    // ── Tests mode Jackpot ──

    #[test]
    fn test_comb() {
        assert_eq!(comb(50, 5), 2_118_760);
        assert_eq!(comb(12, 2), 66);
        assert_eq!(comb(5, 0), 1);
        assert_eq!(comb(5, 5), 1);
        assert_eq!(comb(3, 5), 0);
        assert_eq!(comb(10, 3), 120);
    }

    #[test]
    fn test_compute_adaptive_k() {
        let (kb, ks) = compute_adaptive_k(5);
        assert!(comb(kb, 5) * comb(ks, 2) >= 15, "K trop petit pour 5 suggestions");
        assert!(kb <= 50 && ks <= 12);

        let (kb, ks) = compute_adaptive_k(5000);
        assert!(comb(kb, 5) * comb(ks, 2) >= 15000, "K trop petit pour 5000 suggestions");

        let (kb, ks) = compute_adaptive_k(50000);
        assert!(comb(kb, 5) * comb(ks, 2) >= 150000, "K trop petit pour 50000 suggestions");
    }

    #[test]
    fn test_jackpot_basic() {
        let ball_probs: Vec<f64> = vec![1.0 / 50.0; 50];
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 5, None, None, None, None, None, None, None, None, None, None, None, None, 0.0).unwrap();
        assert_eq!(result.suggestions.len(), 5);
        assert!(result.total_jackpot_probability > 0.0);
        assert!(result.enumeration_size > 0);
        // Avec probas uniformes, improvement_factor dépend du sous-ensemble K adaptatif
        // On vérifie simplement que c'est positif et fini
        assert!(result.improvement_factor > 0.0 && result.improvement_factor.is_finite(),
            "improvement_factor devrait être positif et fini, got {}", result.improvement_factor);
    }

    #[test]
    fn test_jackpot_peaked_better_than_uniform() {
        // Probas concentrées sur les 10 premières boules
        let mut ball_probs = vec![0.005; 50];
        for i in 0..10 {
            ball_probs[i] = 0.075;
        }
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();

        let mut star_probs = vec![0.05; 12];
        star_probs[0] = 0.2;
        star_probs[1] = 0.15;
        let total: f64 = star_probs.iter().sum();
        let star_probs: Vec<f64> = star_probs.iter().map(|p| p / total).collect();

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 100, None, None, None, None, None, None, None, None, None, None, None, None, 0.0).unwrap();
        assert_eq!(result.suggestions.len(), 100);
        assert!(result.improvement_factor > 1.0,
            "Des probas concentrées devraient donner un facteur > 1, got {}", result.improvement_factor);
    }

    #[test]
    fn test_jackpot_sorted_descending() {
        let ball_probs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 20, None, None, None, None, None, None, None, None, None, None, None, None, 0.0).unwrap();
        for w in result.suggestions.windows(2) {
            assert!(w[0].score >= w[1].score,
                "Suggestions non triées : {} < {}", w[0].score, w[1].score);
        }
    }

    #[test]
    fn test_jackpot_deterministic() {
        let ball_probs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let total: f64 = ball_probs.iter().sum();
        let ball_probs: Vec<f64> = ball_probs.iter().map(|p| p / total).collect();
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];

        let r1 = generate_suggestions_jackpot(&ball_probs, &star_probs, 10, None, None, None, None, None, None, None, None, None, None, None, None, 0.0).unwrap();
        let r2 = generate_suggestions_jackpot(&ball_probs, &star_probs, 10, None, None, None, None, None, None, None, None, None, None, None, None, 0.0).unwrap();
        for (a, b) in r1.suggestions.iter().zip(r2.suggestions.iter()) {
            assert_eq!(a.balls, b.balls);
            assert_eq!(a.stars, b.stars);
        }
    }

    #[test]
    fn test_jackpot_with_filter() {
        let ball_probs: Vec<f64> = vec![1.0 / 50.0; 50];
        let star_probs: Vec<f64> = vec![1.0 / 12.0; 12];
        let filter = StructuralFilter {
            sum_range: (80, 180),
            max_consecutive: 3,
            odd_range: (1, 4),
            spread_range: (10, 45),
        };

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 10, Some(&filter), None, None, None, None, None, None, None, None, None, None, None, 0.0).unwrap();
        // Toutes les suggestions doivent passer le filtre
        for s in &result.suggestions {
            assert!(filter.accept_balls(&s.balls),
                "Suggestion {:?} ne passe pas le filtre", s.balls);
        }
    }

    #[test]
    fn test_ball_context_bins() {
        // v21: 15 contextes (5 sum × 3 spread)
        let draws = crate::models::make_test_draws(100);
        let conditioner = BallStarConditioner::from_history(&draws);
        // Low sum + low spread
        let ctx_low = conditioner.ball_context(&[1, 3, 5, 7, 9]);
        assert!(ctx_low < 15, "Context should be valid: {}", ctx_low);
        // High sum + high spread
        let ctx_high = conditioner.ball_context(&[2, 20, 30, 40, 50]);
        assert!(ctx_high < 15, "Context should be valid: {}", ctx_high);
        // Different contexts for different balls
        assert_ne!(ctx_low, ctx_high, "Low and high contexts should differ");
    }

    #[test]
    fn test_conditioner_15_contexts_valid() {
        let draws = crate::models::make_test_draws(100);
        let conditioner = BallStarConditioner::from_history(&draws);
        // Test a variety of ball combinations
        let test_balls: Vec<[u8; 5]> = vec![
            [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10], [25, 26, 27, 28, 29], [5, 15, 25, 35, 45],
        ];
        for balls in &test_balls {
            let ctx = conditioner.ball_context(balls);
            assert!(ctx < 15, "Context {} out of range for balls {:?}", ctx, balls);
        }
    }

    #[test]
    fn test_conditioner_distributions_normalized() {
        let draws = crate::models::make_test_draws(200);
        let conditioner = BallStarConditioner::from_history(&draws);
        for ctx in 0..BallStarConditioner::N_CONTEXTS {
            let sum: f64 = conditioner.table[ctx].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Context {} should sum to 1.0, got {}", ctx, sum,
            );
        }
    }

    #[test]
    fn test_conditioner_conditioned_lookup() {
        let draws = crate::models::make_test_draws(100);
        let conditioner = BallStarConditioner::from_history(&draws);
        let balls = [5, 10, 20, 30, 40];
        let probs = conditioner.conditioned_pair_probs(&balls);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conviction_temperature_split() {
        // v13: smooth sigmoid interpolation
        let conv = ConvictionScore {
            ball_entropy: 3.0,
            star_entropy: 2.0,
            ball_concentration: 0.1,
            star_concentration: 0.6,
            ball_agreement: 0.1,
            star_agreement: 0.5,
            overall: 0.3,
            verdict: ConvictionVerdict::MediumConviction,
        };
        let (bt, st) = conviction_temperature_split(&conv);
        // ball_score = 0.7*0.1 + 0.3*0.1 = 0.10, sigmoid: near low end → T ≈ 0.12
        assert!(bt >= 0.05 && bt <= 0.15, "bt={bt}");
        // v23b: star_score = 0.7*0.6 + 0.3*0.5 = 0.57, sigmoid: near high end → T ≈ 0.08-0.12
        assert!(st >= 0.08 && st <= 0.15, "st={st}");

        // Extreme low conviction → highest T
        let conv_low = ConvictionScore {
            ball_entropy: 3.0, star_entropy: 2.0,
            ball_concentration: 0.0, star_concentration: 0.0,
            ball_agreement: 0.0, star_agreement: 0.0,
            overall: 0.0, verdict: ConvictionVerdict::LowConviction,
        };
        let (bt_low, st_low) = conviction_temperature_split(&conv_low);
        assert!(bt_low > 0.11 && bt_low < 0.13, "low conviction bt={bt_low}");
        // v23b: star T lowered further: low conviction → T ≈ 0.26-0.30
        assert!(st_low > 0.25 && st_low < 0.30, "low conviction st={st_low}");

        // Extreme high conviction → lowest T
        let conv_high = ConvictionScore {
            ball_entropy: 3.0, star_entropy: 2.0,
            ball_concentration: 1.0, star_concentration: 1.0,
            ball_agreement: 1.0, star_agreement: 1.0,
            overall: 1.0, verdict: ConvictionVerdict::HighConviction,
        };
        let (bt_high, st_high) = conviction_temperature_split(&conv_high);
        assert!(bt_high >= 0.05 && bt_high < 0.06, "high conviction bt={bt_high}");
        // v23b: star T maximally aggressive: high conviction → T ≈ 0.08
        assert!(st_high >= 0.08 && st_high < 0.10, "high conviction st={st_high}");
    }

    #[test]
    fn test_skill_temperature() {
        // No skill → T=1.0
        assert!((skill_temperature(0.0) - 1.0).abs() < 1e-10);
        assert!((skill_temperature(-0.5) - 1.0).abs() < 1e-10);

        // v13: soft clamp with floor=0.08. T = 0.08 + 0.92 * exp(-skill/0.035)
        // Small skill (0.01) → T ≈ 0.08 + 0.92*0.751 = 0.77
        let t = skill_temperature(0.01);
        assert!(t > 0.72 && t < 0.82, "skill=0.01 → T={t}");

        // Medium skill (0.03) → T ≈ 0.08 + 0.92*0.424 = 0.47
        let t = skill_temperature(0.03);
        assert!(t > 0.40 && t < 0.55, "skill=0.03 → T={t}");

        // High skill (0.05) → T ≈ 0.08 + 0.92*0.239 = 0.30
        let t = skill_temperature(0.05);
        assert!(t > 0.25 && t < 0.35, "skill=0.05 → T={t}");

        // Very high skill (0.10) → T ≈ 0.08 + 0.92*0.057 = 0.13
        let t = skill_temperature(0.10);
        assert!(t > 0.10 && t < 0.16, "skill=0.10 → T={t}");
    }

    #[test]
    fn test_skill_temperature_floor() {
        // v13: Very high skill asymptotes to floor=0.08
        let t = skill_temperature(0.50);
        assert!(t > 0.079 && t < 0.085, "High skill should approach floor 0.08, got {t}");
    }

    #[test]
    fn test_skill_based_temperature_overrides_conviction() {
        let conv = ConvictionScore {
            ball_entropy: 3.0,
            star_entropy: 2.0,
            ball_concentration: 0.1,
            star_concentration: 0.1,
            ball_agreement: 0.1,
            star_agreement: 0.1,
            overall: 0.15,
            verdict: ConvictionVerdict::LowConviction,
        };
        // With skill, should use skill-based temp (not conviction fallback)
        let (bt, st) = conviction_temperature_split_with_skill(&conv, Some(0.05), Some(0.03));
        let expected_bt = skill_temperature(0.05);
        let expected_st = skill_temperature(0.03).min(0.35); // v12: tighter cap
        assert!((bt - expected_bt).abs() < 1e-10, "bt={bt} expected {expected_bt}");
        assert!((st - expected_st).abs() < 1e-10, "st={st} expected {expected_st}");
        // v13: verify they're actually aggressive (soft clamp floor=0.08)
        assert!(expected_bt < 0.35, "bt should be aggressive with soft clamp formula");
    }

    #[test]
    fn test_optimal_subset_uniform() {
        // v12: Fixed formula — uniform distribution → moderate k (~17)
        // With correct formula: P(5_in_m) × coverage, smaller m gives denser coverage
        let probs = vec![1.0 / 50.0; 50];
        let k = optimal_subset_k(&probs, 5000);
        assert!(k >= 12 && k <= 30, "Uniform probs → moderate k: got {}", k);
    }

    #[test]
    fn test_optimal_subset_concentrated() {
        // Very concentrated: top 15 balls have ~80% of probability
        let mut probs = vec![0.004; 50]; // remaining ~20%
        for i in 0..15 {
            probs[i] = 0.05; // ~75%
        }
        let total: f64 = probs.iter().sum();
        for p in &mut probs { *p /= total; }

        let k = optimal_subset_k(&probs, 5000);
        // Concentrated → even smaller k (focus on high-prob balls)
        assert!(k < 25, "Concentrated probs → smaller k: got {}", k);
        assert!(k >= 12, "k should be at least 12: got {}", k);
    }
}
