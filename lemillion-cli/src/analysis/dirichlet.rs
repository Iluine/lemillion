use lemillion_db::models::{NumberProbability, ProbabilityTag};

pub fn dirichlet_probabilities(
    draws: &[([u8; 5], [u8; 2])],
    pool_size: u8,
    alpha: f64,
    is_stars: bool,
) -> Vec<NumberProbability> {
    let mut counts = vec![0u32; pool_size as usize];

    for (balls, stars) in draws {
        let numbers = if is_stars { stars.as_slice() } else { balls.as_slice() };
        for &n in numbers {
            let idx = (n - 1) as usize;
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }
    }

    let total: u32 = counts.iter().sum();
    let denominator = pool_size as f64 * alpha + total as f64;

    (1..=pool_size)
        .map(|n| {
            let count = counts[(n - 1) as usize];
            let probability = (alpha + count as f64) / denominator;
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
    fn test_dirichlet_sums_to_one() {
        let draws = vec![
            ([1, 2, 3, 4, 5], [1, 2]),
            ([6, 7, 8, 9, 10], [3, 4]),
            ([1, 3, 5, 7, 9], [1, 5]),
        ];
        let probs = dirichlet_probabilities(&draws, 50, 1.0, false);
        let sum: f64 = probs.iter().map(|p| p.probability).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum = {}", sum);
    }

    #[test]
    fn test_dirichlet_stars_sums_to_one() {
        let draws = vec![
            ([1, 2, 3, 4, 5], [1, 2]),
            ([6, 7, 8, 9, 10], [3, 4]),
        ];
        let probs = dirichlet_probabilities(&draws, 12, 1.0, true);
        let sum: f64 = probs.iter().map(|p| p.probability).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum = {}", sum);
    }

    #[test]
    fn test_dirichlet_uniform_with_no_draws() {
        let draws: Vec<([u8; 5], [u8; 2])> = vec![];
        let probs = dirichlet_probabilities(&draws, 50, 1.0, false);
        let expected = 1.0 / 50.0;
        for p in &probs {
            assert!((p.probability - expected).abs() < 1e-10);
        }
    }
}
