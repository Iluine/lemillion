use lemillion_db::models::{Draw, Pool};
use ndarray::Array1;

use crate::config::Encoding;

/// Encode a draw as an input vector.
/// - OneHot: 62-dim, indices (ball-1) and 50+(star-1) set to 1.0
/// - Normalized: 7-dim, [b1/50, ..., b5/50, s1/12, s2/12] (sorted ascending)
pub fn encode_draw(draw: &Draw, encoding: Encoding) -> Array1<f64> {
    match encoding {
        Encoding::OneHot => {
            let mut v = Array1::zeros(62);
            for &b in &draw.balls {
                v[(b - 1) as usize] = 1.0;
            }
            for &s in &draw.stars {
                v[50 + (s - 1) as usize] = 1.0;
            }
            v
        }
        Encoding::Normalized => {
            let mut balls = draw.balls;
            let mut stars = draw.stars;
            balls.sort_unstable();
            stars.sort_unstable();
            let mut v = Array1::zeros(7);
            for (i, &b) in balls.iter().enumerate() {
                v[i] = b as f64 / Pool::Balls.size() as f64;
            }
            for (i, &s) in stars.iter().enumerate() {
                v[5 + i] = s as f64 / Pool::Stars.size() as f64;
            }
            v
        }
    }
}

/// Encode a draw's target as multi-hot vector for a given pool.
/// - Balls: 50-dim, Stars: 12-dim
pub fn encode_target_onehot(draw: &Draw, pool: Pool) -> Array1<f64> {
    let size = pool.size();
    let mut v = Array1::zeros(size);
    for &n in pool.numbers_from(draw) {
        v[(n - 1) as usize] = 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_draw() -> Draw {
        Draw {
            draw_id: "001".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-01".to_string(),
            balls: [3, 15, 27, 38, 44],
            stars: [2, 9],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
        }
    }

    #[test]
    fn test_onehot_dimension() {
        let v = encode_draw(&test_draw(), Encoding::OneHot);
        assert_eq!(v.len(), 62);
    }

    #[test]
    fn test_onehot_sum() {
        let v = encode_draw(&test_draw(), Encoding::OneHot);
        let sum: f64 = v.sum();
        assert!((sum - 7.0).abs() < 1e-10, "sum={sum}, expected 7");
    }

    #[test]
    fn test_onehot_correct_indices() {
        let draw = test_draw();
        let v = encode_draw(&draw, Encoding::OneHot);
        // balls: 3,15,27,38,44 -> indices 2,14,26,37,43
        assert_eq!(v[2], 1.0);
        assert_eq!(v[14], 1.0);
        assert_eq!(v[26], 1.0);
        assert_eq!(v[37], 1.0);
        assert_eq!(v[43], 1.0);
        // stars: 2,9 -> indices 51,58
        assert_eq!(v[51], 1.0);
        assert_eq!(v[58], 1.0);
        // a zero
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn test_normalized_dimension() {
        let v = encode_draw(&test_draw(), Encoding::Normalized);
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_normalized_range() {
        let v = encode_draw(&test_draw(), Encoding::Normalized);
        for &val in v.iter() {
            assert!(val >= 0.0 && val <= 1.0, "val={val} out of [0,1]");
        }
    }

    #[test]
    fn test_normalized_sorted_ascending() {
        let v = encode_draw(&test_draw(), Encoding::Normalized);
        // balls part should be sorted
        for i in 0..4 {
            assert!(v[i] <= v[i + 1], "balls not sorted at {i}");
        }
        // stars part
        assert!(v[5] <= v[6], "stars not sorted");
    }

    #[test]
    fn test_normalized_values() {
        let v = encode_draw(&test_draw(), Encoding::Normalized);
        // balls [3,15,27,38,44] sorted: [3,15,27,38,44]
        assert!((v[0] - 3.0 / 50.0).abs() < 1e-10);
        assert!((v[4] - 44.0 / 50.0).abs() < 1e-10);
        // stars [2,9] sorted: [2,9]
        assert!((v[5] - 2.0 / 12.0).abs() < 1e-10);
        assert!((v[6] - 9.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_target_balls() {
        let draw = test_draw();
        let t = encode_target_onehot(&draw, Pool::Balls);
        assert_eq!(t.len(), 50);
        let sum: f64 = t.sum();
        assert!((sum - 5.0).abs() < 1e-10);
        assert_eq!(t[2], 1.0);  // ball 3
        assert_eq!(t[14], 1.0); // ball 15
    }

    #[test]
    fn test_target_stars() {
        let draw = test_draw();
        let t = encode_target_onehot(&draw, Pool::Stars);
        assert_eq!(t.len(), 12);
        let sum: f64 = t.sum();
        assert!((sum - 2.0).abs() < 1e-10);
        assert_eq!(t[1], 1.0); // star 2
        assert_eq!(t[8], 1.0); // star 9
    }
}
