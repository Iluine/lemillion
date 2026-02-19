use anyhow::Result;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use lemillion_db::models::{NumberProbability, Suggestion};

pub fn generate_suggestions(
    ball_probs: &[NumberProbability],
    star_probs: &[NumberProbability],
    count: usize,
    seed: Option<u64>,
) -> Result<Vec<Suggestion>> {
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut rand::rng()),
    };

    let uniform_ball = 1.0 / 50.0f64;
    let uniform_star = 1.0 / 12.0f64;

    let mut suggestions = Vec::with_capacity(count);

    for _ in 0..count {
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

        suggestions.push(Suggestion {
            balls: balls_arr,
            stars: stars_arr,
            score,
        });
    }

    Ok(suggestions)
}

fn sample_without_replacement(
    probs: &[NumberProbability],
    count: usize,
    uniform_prob: f64,
    rng: &mut StdRng,
) -> Result<(Vec<u8>, f64)> {
    let mut available: Vec<(u8, f64)> = probs.iter().map(|p| (p.number, p.probability)).collect();
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
