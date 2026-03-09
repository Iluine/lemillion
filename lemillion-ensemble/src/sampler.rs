use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering as CmpOrdering;

use anyhow::Result;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use lemillion_db::models::{Draw, Pool, Suggestion};
use crate::expected_value::{PopularityModel, anti_popularity, compute_ev};
use crate::models::summary_predictor::SummaryPredictor;

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

// ════════════════════════════════════════════════════════════════
// Conditioning ball→star
// ════════════════════════════════════════════════════════════════

/// Conditionne les probabilités de paires d'étoiles sur le contexte des boules.
/// 5 sum_bins × 3 spread_bins = 15 contextes.
/// Pour chaque contexte, distribution sur les 66 paires d'étoiles (Laplace smoothed).
pub struct BallStarConditioner {
    table: Vec<[f64; 66]>,
}

impl BallStarConditioner {
    /// 9 contextes (3 sum × 3 spread) pour ~70 draws/bin.
    /// Approche hiérarchique: déviations multiplicatives par rapport à la distribution globale.
    const N_CONTEXTS: usize = 9; // 3 sum_bins × 3 spread_bins
    const LAPLACE_ALPHA: f64 = 0.3;

    pub fn from_history(draws: &[Draw]) -> Self {
        let mut counts = vec![[0.0f64; 66]; Self::N_CONTEXTS];
        let mut global_counts = [0.0f64; 66];

        for (t, draw) in draws.iter().enumerate() {
            let weight = (-0.02 * t as f64).exp();
            let mut balls = draw.balls;
            balls.sort();
            let ctx = Self::ball_context(&balls);
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

        // Approche hiérarchique: table[ctx][p] = global × ratio_ctx
        // Cela permet de profiter de la force statistique globale
        let mut table = vec![[0.0f64; 66]; Self::N_CONTEXTS];
        for ctx in 0..Self::N_CONTEXTS {
            let ctx_total: f64 = counts[ctx].iter().sum::<f64>();
            if ctx_total < 10.0 {
                // Pas assez de données: utiliser la distribution globale
                // Pour 10-30 obs, blend 50/50 avec global dans le else branch
                table[ctx] = global_probs;
            } else {
                // Ratio multiplicatif: P(pair|ctx) ∝ global × (obs/expected)
                let expected_per_pair = ctx_total / 66.0;
                for p in 0..66 {
                    let ratio = (counts[ctx][p] + Self::LAPLACE_ALPHA)
                        / (expected_per_pair + Self::LAPLACE_ALPHA);
                    table[ctx][p] = global_probs[p] * ratio;
                }
                // Renormaliser
                let total: f64 = table[ctx].iter().sum();
                if total > 0.0 {
                    for p in &mut table[ctx] {
                        *p /= total;
                    }
                }
            }
        }

        Self { table }
    }

    #[inline]
    pub fn ball_context(balls: &[u8; 5]) -> usize {
        let sum: u32 = balls.iter().map(|&b| b as u32).sum();
        let spread = balls[4] - balls[0];
        // 3 sum bins: low (<115) / medium (115-140) / high (>140)
        let sum_bin = if sum < 115 { 0 } else if sum <= 140 { 1 } else { 2 };
        let spread_bin = match spread {
            0..=19 => 0,
            20..=34 => 1,
            _ => 2,
        };
        sum_bin * 3 + spread_bin
    }

    #[inline]
    pub fn conditioned_pair_probs(&self, balls: &[u8; 5]) -> &[f64; 66] {
        &self.table[Self::ball_context(balls)]
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

        // Combinaison pondérée
        0.35 * sum_score + 0.25 * spread_score + 0.25 * pair_score + 0.15 * triplet_score
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
    uniform_prob: f64,
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
        let weights: Vec<f64> = available.iter().map(|(_, w)| *w).collect();
        let dist = WeightedIndex::new(&weights)?;
        let idx = dist.sample(rng);

        let (number, prob) = available.remove(idx);
        selected.push(number);
        score *= prob / uniform_prob;
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
    /// Facteur d'amélioration vs N tickets uniformes.
    pub improvement_factor: f64,
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
/// Énumère exhaustivement les combinaisons dans la zone haute probabilité,
/// triées par score bayésien, sans diversité.
pub fn generate_suggestions_jackpot(
    ball_probs: &[f64],
    star_probs: &[f64],
    count: usize,
    filter: Option<&StructuralFilter>,
    _coherence: Option<&CoherenceScorer>,
    _joint_model: Option<&crate::models::joint::JointConditionalModel>,
    star_pair_probs: Option<&[f64; 66]>,
    excluded_balls: Option<&[u8]>,
    conditioner: Option<&BallStarConditioner>,
    _neural_scorer: Option<&crate::models::neural_scorer::NeuralScorer>,
) -> Result<JackpotResult> {
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

    // K adaptatif basé sur l'entropie de la distribution (v7)
    // Haute entropie (flat) → plus de boules nécessaires ; basse entropie (peaked) → moins
    let (k_balls, _k_stars) = compute_adaptive_k(count);
    let h: f64 = ball_probs.iter()
        .filter(|&&p| p > 1e-30)
        .map(|&p| -p * p.ln())
        .sum();
    let entropy_ratio = h / (ball_probs.len() as f64).ln();
    let min_k = (25.0 + 20.0 * entropy_ratio).round() as usize; // 25-45 pour plus de couverture
    let k_balls = k_balls.max(min_k);
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

    // Blend ratio conditionné vs marginal pour les étoiles
    const COND_BLEND: f64 = 0.65;

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

                            let log_ball_score: f64 = [i0,i1,i2,i3,i4].iter()
                                .map(|&i| log_ball_probs[top_balls[i]])
                                .sum();

                            let cond_probs = conditioner.map(|c| c.conditioned_pair_probs(&balls));

                            for (star_idx, &(stars, base_star_score)) in star_pairs.iter().enumerate() {
                                enumeration_size += 1;
                                let log_star_score = if let Some(cp) = cond_probs {
                                    let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
                                    let conditioned = cp[pidx] / uniform_pair;
                                    let blended = COND_BLEND * conditioned + (1.0 - COND_BLEND) * base_star_score;
                                    blended.max(1e-30).ln()
                                } else {
                                    log_base_star_scores[star_idx]
                                };
                                let log_score = log_ball_score + log_star_score;

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

        // score = prod(prob/uniform) = facteur d'amélioration vs uniforme par grille
        // P(5+2) par grille sous le modèle = score / 139_838_160
        let n_sugg = suggestions.len() as f64;
        let mean_score = suggestions.iter().map(|s| s.score).sum::<f64>() / n_sugg;
        let total_prob = mean_score * n_sugg / 139_838_160.0;
        let improvement = mean_score;

        Ok(JackpotResult {
            suggestions,
            total_jackpot_probability: total_prob,
            enumeration_size,
            filtered_size,
            improvement_factor: improvement,
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

                            let log_ball_score: f64 = [i0,i1,i2,i3,i4].iter()
                                .map(|&i| log_ball_probs[top_balls[i]])
                                .sum();

                            let cond_probs = conditioner.map(|c| c.conditioned_pair_probs(&balls));

                            for (star_idx, &(stars, base_star_score)) in star_pairs.iter().enumerate() {
                                enumeration_size += 1;
                                if passes_filter {
                                    filtered_size += 1;
                                    let log_star_score = if let Some(cp) = cond_probs {
                                        let pidx = crate::models::star_pair::pair_index(stars[0], stars[1]);
                                        let conditioned = cp[pidx] / uniform_pair;
                                        let blended = COND_BLEND * conditioned + (1.0 - COND_BLEND) * base_star_score;
                                        blended.max(1e-30).ln()
                                    } else {
                                        log_base_star_scores[star_idx]
                                    };
                                    let score = (log_ball_score + log_star_score).exp();
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

        // score = prod(prob/uniform) = facteur d'amélioration vs uniforme par grille
        // P(5+2) par grille sous le modèle = score / 139_838_160
        let n_sugg = all.len() as f64;
        let mean_score = all.iter().map(|s| s.score).sum::<f64>() / n_sugg;
        let total_prob = mean_score * n_sugg / 139_838_160.0;
        let improvement = mean_score;

        Ok(JackpotResult {
            suggestions: all,
            total_jackpot_probability: total_prob,
            enumeration_size,
            filtered_size,
            improvement_factor: improvement,
        })
    }
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
        0..=3 => (0.55, 0.25),
        4..=6 => (0.60, 0.30),
        7..=10 => (0.65, 0.30),
        _ => (0.70, 0.35), // >10 grilles : plus conservateur
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

            // Score = P(5+2) × overlap_bonus moyen sur toutes les grilles sélectionnées
            let overlap_bonus: f64 = selected.iter().map(|s| {
                let common_b = candidate.balls.iter().filter(|b| s.balls.contains(b)).count();
                // Structured overlap bonus:
                // 0 common = 0.7 (trop dispersé, pas de couverture partagée)
                // 1 common = 1.2 (optimal: un pivot commun)
                // 2 common = 1.0 (bon: overlap modéré)
                // 3 common = 0.5 (trop similaire)
                // 4+ common = 0.2 (quasi-doublon)
                match common_b {
                    0 => 0.70,
                    1 => 1.20,
                    2 => 1.00,
                    3 => 0.50,
                    _ => 0.20,
                }
            }).sum::<f64>() / selected.len() as f64;

            let score = candidate.score * overlap_bonus;
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

/// - LowConviction → T=1.0 (pas de sharpening)
pub fn conviction_temperature(verdict: &ConvictionVerdict) -> f64 {
    match verdict {
        ConvictionVerdict::HighConviction => 0.60,
        ConvictionVerdict::MediumConviction => 0.85,
        ConvictionVerdict::LowConviction => 1.0,
    }
}

/// Skill-based temperature: T = 1 / (1 + C × total_skill)
/// With typical skill ~0.01 bits/draw: T≈0.91 (très léger)
/// With skill ~0.05: T≈0.67 (modéré)
/// With skill ~0.10: T≈0.50 (agressif)
/// C is the sensitivity constant.
const SKILL_TEMP_C: f64 = 10.0;

/// Compute temperature from calibrated skill level.
/// skill = average bits above uniform for this pool.
pub fn skill_temperature(skill: f64) -> f64 {
    if skill <= 0.0 {
        1.0 // no skill → no sharpening
    } else {
        1.0 / (1.0 + SKILL_TEMP_C * skill)
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
            // Fallback: conviction-based (relative, not absolute thresholds)
            let ball_score = 0.7 * conviction.ball_concentration + 0.3 * conviction.ball_agreement;
            if ball_score >= 0.5 {
                0.35
            } else if ball_score >= 0.2 {
                0.55
            } else {
                0.80
            }
        }
    };

    let star_temp = match star_skill {
        Some(s) if s > 0.0 => skill_temperature(s).min(0.40), // stars always get some sharpening
        _ => {
            let star_score = 0.7 * conviction.star_concentration + 0.3 * conviction.star_agreement;
            if star_score >= 0.5 {
                0.20
            } else if star_score >= 0.2 {
                0.35
            } else {
                0.40
            }
        }
    };

    (ball_temp, star_temp)
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

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 5, None, None, None, None, None, None, None).unwrap();
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

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 100, None, None, None, None, None, None, None).unwrap();
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

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 20, None, None, None, None, None, None, None).unwrap();
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

        let r1 = generate_suggestions_jackpot(&ball_probs, &star_probs, 10, None, None, None, None, None, None, None).unwrap();
        let r2 = generate_suggestions_jackpot(&ball_probs, &star_probs, 10, None, None, None, None, None, None, None).unwrap();
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

        let result = generate_suggestions_jackpot(&ball_probs, &star_probs, 10, Some(&filter), None, None, None, None, None, None).unwrap();
        // Toutes les suggestions doivent passer le filtre
        for s in &result.suggestions {
            assert!(filter.accept_balls(&s.balls),
                "Suggestion {:?} ne passe pas le filtre", s.balls);
        }
    }

    #[test]
    fn test_ball_context_bins() {
        // Low sum (15 < 115), low spread (4 < 20): bin = 0*3 + 0 = 0
        assert_eq!(BallStarConditioner::ball_context(&[1, 2, 3, 4, 5]), 0 * 3 + 0);
        // High sum (145 > 140), high spread (45 >= 35): bin = 2*3 + 2 = 8
        assert_eq!(BallStarConditioner::ball_context(&[5, 20, 30, 40, 50]), 2 * 3 + 2);
        // Mid sum (125, 115<=125<=140), mid spread (30 in 20-34): bin = 1*3 + 1 = 4
        assert_eq!(BallStarConditioner::ball_context(&[10, 20, 25, 30, 40]), 1 * 3 + 1);
    }

    #[test]
    fn test_conditioner_distributions_normalized() {
        let draws = crate::models::make_test_draws(100);
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
        // ball_score = 0.7*0.1 + 0.3*0.1 = 0.10 < 0.2 → T=0.80
        assert!((bt - 0.80).abs() < 1e-10);
        // star_score = 0.7*0.6 + 0.3*0.5 = 0.57 >= 0.5 → T=0.20
        assert!((st - 0.20).abs() < 1e-10);
    }

    #[test]
    fn test_skill_temperature() {
        // No skill → T=1.0
        assert!((skill_temperature(0.0) - 1.0).abs() < 1e-10);
        assert!((skill_temperature(-0.5) - 1.0).abs() < 1e-10);

        // Small skill (0.01) → T ≈ 0.91
        let t = skill_temperature(0.01);
        assert!(t > 0.85 && t < 0.95, "skill=0.01 → T={t}");

        // Medium skill (0.05) → T ≈ 0.67
        let t = skill_temperature(0.05);
        assert!(t > 0.60 && t < 0.75, "skill=0.05 → T={t}");

        // High skill (0.10) → T ≈ 0.50
        let t = skill_temperature(0.10);
        assert!(t > 0.45 && t < 0.55, "skill=0.10 → T={t}");
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
        let expected_st = skill_temperature(0.03).min(0.40);
        assert!((bt - expected_bt).abs() < 1e-10, "bt={bt} expected {expected_bt}");
        assert!((st - expected_st).abs() < 1e-10, "st={st} expected {expected_st}");
    }
}
