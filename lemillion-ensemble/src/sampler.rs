use anyhow::Result;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use lemillion_db::models::Suggestion;

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
    let mut rng = StdRng::seed_from_u64(seed);

    let uniform_ball = 1.0 / ball_probs.len() as f64;
    let uniform_star = 1.0 / star_probs.len() as f64;

    let n_candidates = count * oversample;
    let mut candidates = Vec::with_capacity(n_candidates);

    for _ in 0..n_candidates {
        let (balls, ball_score) = sample_without_replacement(ball_probs, 5, uniform_ball, &mut rng)?;
        let (stars, star_score) = sample_without_replacement(star_probs, 2, uniform_star, &mut rng)?;

        let mut balls_arr = [0u8; 5];
        for (i, &b) in balls.iter().enumerate() {
            balls_arr[i] = b;
        }
        balls_arr.sort();

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
