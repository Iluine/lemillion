pub mod dirichlet;
pub mod ewma;
pub mod sampler;

use lemillion_db::models::{NumberProbability, NumberStats, ProbabilityTag};

pub fn compute_stats(
    draws: &[([u8; 5], [u8; 2])],
    pool_size: u8,
    is_stars: bool,
) -> Vec<NumberStats> {
    let mut stats: Vec<NumberStats> = (1..=pool_size)
        .map(|n| NumberStats {
            number: n,
            frequency: 0,
            gap: 0,
        })
        .collect();

    for (i, (balls, stars)) in draws.iter().enumerate() {
        let numbers = if is_stars {
            stars.as_slice()
        } else {
            balls.as_slice()
        };
        for &n in numbers {
            let idx = (n - 1) as usize;
            if idx < stats.len() {
                stats[idx].frequency += 1;
                if stats[idx].gap == 0 {
                    stats[idx].gap = i as u32;
                }
            }
        }
    }

    for stat in &mut stats {
        if stat.frequency == 0 {
            stat.gap = draws.len() as u32;
        }
    }

    stats
}

pub fn tag_probabilities(probs: &mut [NumberProbability], pool_size: u8) {
    let uniform = 1.0 / pool_size as f64;
    let threshold = 0.3;

    for p in probs.iter_mut() {
        let deviation = (p.probability - uniform) / uniform;
        if deviation > threshold {
            p.tag = ProbabilityTag::Hot;
        } else if deviation < -threshold {
            p.tag = ProbabilityTag::Cold;
        } else {
            p.tag = ProbabilityTag::Normal;
        }
    }
}
