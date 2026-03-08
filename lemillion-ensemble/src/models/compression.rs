use std::collections::HashMap;
use std::io::Write;

use flate2::Compression;
use flate2::write::DeflateEncoder;
use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// Compression — prediction via compression ratio change.
///
/// Exploits the signal found by research: the compression ratio of EuroMillions
/// draw sequences is below the random baseline, meaning there IS some
/// predictable structure captured by deflate.
///
/// For each candidate number k, we measure how much the compression ratio
/// changes when k is appended to the recent draw sequence. Numbers that
/// reduce the compression ratio (make the sequence more compressible) are
/// deemed more "predictable" by the compressor — i.e., more consistent with
/// the patterns already present — and receive higher probability.
///
/// This is related to Normalized Compression Distance (NCD) and Kolmogorov
/// complexity estimation via real-world compressors.
pub struct CompressionModel {
    smoothing: f64,
    window: usize,
    min_draws: usize,
}

impl Default for CompressionModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            window: 50,
            min_draws: 30,
        }
    }
}

/// Compress data using deflate and return the compressed size in bytes.
fn compress_deflate(data: &[u8]) -> usize {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap_or(());
    let compressed = encoder.finish().unwrap_or_default();
    compressed.len()
}

/// Encode the recent draws as a byte sequence for the given pool.
/// For Balls: each draw contributes its 5 sorted ball values as bytes.
/// For Stars: each draw contributes its 2 sorted star values as bytes.
/// Draws are encoded oldest-first (chronological order) since compression
/// algorithms detect sequential patterns better in temporal order.
fn encode_draws(draws: &[Draw], pool: Pool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(draws.len() * pool.pick_count());
    // draws[0] = most recent, so reverse for chronological order
    for draw in draws.iter().rev() {
        let numbers = pool.numbers_from(draw);
        for &n in numbers {
            bytes.push(n);
        }
    }
    bytes
}

impl ForecastModel for CompressionModel {
    fn name(&self) -> &str {
        "Compression"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Take the last `window` draws (or fewer if not enough)
        let window = self.window.min(draws.len());
        let recent = &draws[..window];

        // Encode the base sequence
        let base_bytes = encode_draws(recent, pool);
        let base_compressed = compress_deflate(&base_bytes);

        // For each candidate number k, append it to the sequence and measure
        // the compression ratio change.
        // A number that makes the sequence MORE compressible (lower ratio)
        // is more "expected" by the compressor = higher score.
        let mut scores = vec![0.0f64; size];

        for k in 1..=size {
            let mut extended = base_bytes.clone();
            extended.push(k as u8);

            let extended_compressed = compress_deflate(&extended);

            // Marginal compression cost: how many extra compressed bytes
            // does adding this number require?
            // Lower marginal cost = number fits better with existing patterns.
            let marginal_cost = extended_compressed as f64 - base_compressed as f64;

            // Convert to score: negate so that lower cost = higher score.
            // We use the raw difference; the softmax below handles scaling.
            scores[k - 1] = -marginal_cost;
        }

        // Softmax to convert scores to probabilities
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = probs.iter().sum();
        if sum_exp > 0.0 {
            for p in &mut probs {
                *p /= sum_exp;
            }
        } else {
            return uniform;
        }

        // Smooth towards uniform
        let uniform_val = 1.0 / size as f64;
        for p in probs.iter_mut() {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = if pool == Pool::Balls {
            PROB_FLOOR_BALLS
        } else {
            PROB_FLOOR_STARS
        };
        floor_only(&mut probs, floor);
        probs
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("window".into(), self.window as f64),
            ("min_draws".into(), self.min_draws as f64),
        ])
    }

    fn sampling_strategy(&self) -> SamplingStrategy {
        SamplingStrategy::Sparse { span_multiplier: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_compression_valid_distribution() {
        let draws = make_test_draws(100);
        let model = CompressionModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_compression_stars() {
        let draws = make_test_draws(100);
        let model = CompressionModel::default();
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_compression_deterministic() {
        let draws = make_test_draws(50);
        let model = CompressionModel::default();
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "Compression should be deterministic");
        }
    }

    #[test]
    fn test_compression_few_draws_returns_uniform() {
        let draws = make_test_draws(5);
        let model = CompressionModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        let uniform = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - uniform).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compression_no_negative() {
        let draws = make_test_draws(100);
        let model = CompressionModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_compression_empty_draws() {
        let model = CompressionModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compression_large_draws() {
        let draws = make_test_draws(200);
        let model = CompressionModel::default();
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_compress_deflate_basic() {
        let data = vec![1u8, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        let compressed_len = compress_deflate(&data);
        assert!(compressed_len > 0);
    }

    #[test]
    fn test_encode_draws_chronological() {
        // draws[0] = most recent, encode should reverse to chronological
        let draws = vec![
            Draw {
                draw_id: "2".to_string(), day: "MARDI".to_string(),
                date: "2024-01-02".to_string(),
                balls: [10, 20, 30, 40, 50], stars: [5, 10],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
            Draw {
                draw_id: "1".to_string(), day: "MARDI".to_string(),
                date: "2024-01-01".to_string(),
                balls: [1, 2, 3, 4, 5], stars: [1, 2],
                winner_count: 0, winner_prize: 0.0, my_million: String::new(),
            },
        ];

        let ball_bytes = encode_draws(&draws, Pool::Balls);
        // Oldest first: [1,2,3,4,5, 10,20,30,40,50]
        assert_eq!(ball_bytes, vec![1, 2, 3, 4, 5, 10, 20, 30, 40, 50]);

        let star_bytes = encode_draws(&draws, Pool::Stars);
        // Oldest first: [1,2, 5,10]
        assert_eq!(star_bytes, vec![1, 2, 5, 10]);
    }

    #[test]
    fn test_compression_sampling_strategy() {
        let model = CompressionModel::default();
        assert_eq!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        );
    }
}
