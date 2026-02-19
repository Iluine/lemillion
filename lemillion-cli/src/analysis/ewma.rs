use lemillion_db::models::{NumberProbability, ProbabilityTag};

pub fn ewma_probabilities(
    draws: &[([u8; 5], [u8; 2])],
    pool_size: u8,
    alpha: f64,
    is_stars: bool,
) -> Vec<NumberProbability> {
    let mut scores = vec![0.0f64; pool_size as usize];

    let floor = alpha.powi(draws.len() as i32 + 1);

    for (t, (balls, stars)) in draws.iter().enumerate() {
        let weight = alpha.powi(t as i32);
        let numbers = if is_stars { stars.as_slice() } else { balls.as_slice() };
        for &n in numbers {
            let idx = (n - 1) as usize;
            if idx < scores.len() {
                scores[idx] += weight;
            }
        }
    }

    for score in &mut scores {
        if *score < floor {
            *score = floor;
        }
    }

    let total: f64 = scores.iter().sum();

    (1..=pool_size)
        .map(|n| {
            let probability = if total > 0.0 {
                scores[(n - 1) as usize] / total
            } else {
                1.0 / pool_size as f64
            };
            NumberProbability {
                number: n,
                probability,
                tag: ProbabilityTag::Normal,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_sums_to_one() {
        let draws = vec![
            ([1, 2, 3, 4, 5], [1, 2]),
            ([6, 7, 8, 9, 10], [3, 4]),
            ([11, 12, 13, 14, 15], [5, 6]),
        ];
        let probs = ewma_probabilities(&draws, 50, 0.9, false);
        let sum: f64 = probs.iter().map(|p| p.probability).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum = {}", sum);
    }

    #[test]
    fn test_ewma_recent_higher() {
        let draws = vec![
            ([1, 2, 3, 4, 5], [1, 2]),
            ([6, 7, 8, 9, 10], [3, 4]),
        ];
        let probs = ewma_probabilities(&draws, 50, 0.9, false);

        let p1 = probs.iter().find(|p| p.number == 1).unwrap().probability;
        let p6 = probs.iter().find(|p| p.number == 6).unwrap().probability;
        assert!(p1 > p6, "P(1)={} devrait être > P(6)={}", p1, p6);
    }

    #[test]
    fn test_ewma_stars_sums_to_one() {
        let draws = vec![
            ([1, 2, 3, 4, 5], [1, 2]),
            ([6, 7, 8, 9, 10], [3, 4]),
        ];
        let probs = ewma_probabilities(&draws, 12, 0.9, true);
        let sum: f64 = probs.iter().map(|p| p.probability).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum = {}", sum);
    }
}
