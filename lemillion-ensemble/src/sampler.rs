use std::collections::HashMap;

use anyhow::Result;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use lemillion_db::models::{Draw, Pool, Suggestion};

/// Filtre structurel rejetant les candidats implausibles avant scoring.
pub struct StructuralFilter {
    pub sum_range: (u16, u16),
    pub max_consecutive: u8,
    pub odd_range: (u8, u8),
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
            },
        }
    }

    fn from_ball_history(draws: &[Draw]) -> Self {
        if draws.is_empty() {
            return Self {
                sum_range: (50, 200),
                max_consecutive: 3,
                odd_range: (0, 5),
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

        let p2 = |v: &[u16]| v[v.len() * 2 / 100];
        let p98 = |v: &[u16]| v[v.len() * 98 / 100];
        let p2_u8 = |v: &[u8]| v[v.len() * 2 / 100];
        let p98_u8 = |v: &[u8]| v[v.len() * 98 / 100];

        Self {
            sum_range: (p2(&sums), p98(&sums)),
            max_consecutive: p98_u8(&max_consecs),
            odd_range: (p2_u8(&odd_counts), p98_u8(&odd_counts)),
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

/// Sélection gloutonne : meilleur score + diversité minimum entre paires.
fn select_diverse(candidates: &[Suggestion], count: usize, min_ball_diff: usize) -> Vec<Suggestion> {
    // candidates doit déjà être trié par score décroissant
    let mut selected: Vec<Suggestion> = Vec::with_capacity(count);

    for candidate in candidates {
        if selected.len() >= count {
            break;
        }
        let dominated = selected.iter().any(|s| ball_distance(&candidate.balls, &s.balls) < min_ball_diff);
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
        if let Some(f) = filter {
            if !f.accept_balls(&balls_arr) {
                continue;
            }
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
// Scoring de cohérence conjointe
// ════════════════════════════════════════════════════════════════

/// Statistiques historiques pour le scoring de cohérence des grilles.
pub struct CoherenceScorer {
    pub mean_sum: f64,
    pub std_sum: f64,
    pub mean_spread: f64,
    pub std_spread: f64,
    pub pair_freq: HashMap<(u8, u8), f64>,
}

impl CoherenceScorer {
    /// Construit un scorer depuis l'historique des tirages.
    pub fn from_history(draws: &[Draw], pool: Pool) -> Self {
        let (sums, spreads) = Self::compute_sum_spread_stats(draws, pool);
        let pair_freq = Self::compute_pair_frequencies(draws, pool);

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
            // Normaliser : fréquence moyenne des paires vs fréquence attendue
            // C(50,5) tirages, chaque paire apparaît ~pick_count*(pick_count-1)/(size*(size-1))
            let expected_pair_freq = (5.0 * 4.0) / (50.0 * 49.0);
            let avg_pair = pair_total / pair_count as f64;
            (avg_pair / expected_pair_freq).min(3.0) / 3.0
        } else {
            0.5
        };

        // Combinaison pondérée
        0.4 * sum_score + 0.3 * spread_score + 0.3 * pair_score
    }
}

/// Génère des suggestions avec scoring conjoint (cohérence historique).
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

        if let Some(f) = filter {
            if !f.accept_balls(&balls_arr) {
                continue;
            }
        }

        let mut stars_arr = [0u8; 2];
        for (i, &s) in stars.iter().enumerate() {
            stars_arr[i] = s;
        }
        stars_arr.sort();

        let bayesian_score = ball_score * star_score;
        let coherence_score = coherence.score_balls(&balls_arr);
        let score = bayesian_score * (0.5 + coherence_score);

        candidates.push(Suggestion {
            balls: balls_arr,
            stars: stars_arr,
            score,
        });
    }

    // Phase 2 : Candidats par recombinaison de templates historiques (50%)
    let n_templates = (count * oversample).saturating_sub(candidates.len());
    if !draws.is_empty() && n_templates > 0 {
        let template_candidates =
            generate_template_candidates(draws, ball_probs, star_probs, &coherence, n_templates, &mut rng, filter);
        candidates.extend(template_candidates);
    }

    // Trier par score décroissant
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(select_diverse(&candidates, count, min_ball_diff))
}

/// Génère des candidats par recombinaison de templates historiques.
fn generate_template_candidates(
    draws: &[Draw],
    ball_probs: &[f64],
    star_probs: &[f64],
    coherence: &CoherenceScorer,
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
                if new_ball >= 1 && new_ball <= 50 && !balls.contains(&new_ball) {
                    balls[replace_pos] = new_ball;
                    break;
                }
            }
        }
        balls.sort();

        if let Some(f) = filter {
            if !f.accept_balls(&balls) {
                continue;
            }
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
        let coherence_score = coherence.score_balls(&balls);
        let score = bayesian_score * (0.5 + coherence_score);

        candidates.push(Suggestion {
            balls,
            stars,
            score,
        });
    }

    candidates
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
