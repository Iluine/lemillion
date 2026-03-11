use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::{floor_only, ForecastModel, SamplingStrategy, PROB_FLOOR_BALLS, PROB_FLOOR_STARS};

/// MapperTDA — Mapper algorithm for topological cluster discovery in draw space.
///
/// Creates a topological summary (simplicial complex → graph) of the draw space.
/// Encodes draws as 7D vectors [b1/50, b2/50, b3/50, b4/50, b5/50, s1/12, s2/12],
/// applies a filter function (distance to centroid), divides the filter range into
/// overlapping intervals, clusters within each interval via k-means, and builds a
/// Mapper graph where nodes are clusters and edges connect clusters sharing draws.
///
/// Prediction: frequency of each number within the cluster closest to the most
/// recent draw, weighted by local graph connectivity (node degree).
pub struct MapperTdaModel {
    smoothing: f64,
    n_intervals: usize,
    overlap_frac: f64,
    k_clusters: usize,
    min_draws: usize,
    kmeans_max_iter: usize,
}

impl Default for MapperTdaModel {
    fn default() -> Self {
        Self {
            smoothing: 0.25,
            n_intervals: 10,
            overlap_frac: 0.50,
            k_clusters: 3,
            min_draws: 50,
            kmeans_max_iter: 20,
        }
    }
}

/// Encode a draw as a 7D vector: [ball1/50, ..., ball5/50, star1/12, star2/12].
fn encode_draw_7d(draw: &Draw) -> [f64; 7] {
    [
        draw.balls[0] as f64 / 50.0,
        draw.balls[1] as f64 / 50.0,
        draw.balls[2] as f64 / 50.0,
        draw.balls[3] as f64 / 50.0,
        draw.balls[4] as f64 / 50.0,
        draw.stars[0] as f64 / 12.0,
        draw.stars[1] as f64 / 12.0,
    ]
}

/// Euclidean distance in 7D.
fn dist_7d(a: &[f64; 7], b: &[f64; 7]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Component-wise centroid of a set of 7D points.
fn centroid_7d(points: &[[f64; 7]]) -> [f64; 7] {
    let n = points.len() as f64;
    if n == 0.0 {
        return [0.0; 7];
    }
    let mut c = [0.0f64; 7];
    for p in points {
        for (i, &v) in p.iter().enumerate() {
            c[i] += v;
        }
    }
    for v in &mut c {
        *v /= n;
    }
    c
}

/// Simple k-means clustering on 7D points. Returns cluster assignments (0..k).
/// Uses deterministic initialization: evenly-spaced picks from sorted filter values.
fn kmeans_7d(
    points: &[[f64; 7]],
    indices: &[usize],
    k: usize,
    max_iter: usize,
) -> Vec<usize> {
    let n = indices.len();
    if n == 0 {
        return vec![];
    }
    let actual_k = k.min(n);
    if actual_k <= 1 {
        return vec![0; n];
    }

    // Deterministic init: pick evenly-spaced points
    let mut centers: Vec<[f64; 7]> = (0..actual_k)
        .map(|i| {
            let idx = i * n / actual_k;
            points[indices[idx]]
        })
        .collect();

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        let mut changed = false;

        // Assign each point to nearest center
        for (i, &global_idx) in indices.iter().enumerate() {
            let pt = &points[global_idx];
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for (c, center) in centers.iter().enumerate() {
                let d = dist_7d(pt, center);
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centers
        let mut sums = vec![[0.0f64; 7]; actual_k];
        let mut counts = vec![0usize; actual_k];
        for (i, &global_idx) in indices.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (d, &v) in sums[c].iter_mut().zip(points[global_idx].iter()) {
                *d += v;
            }
        }
        for (c, center) in centers.iter_mut().enumerate() {
            if counts[c] > 0 {
                let cnt = counts[c] as f64;
                for (d, &s) in center.iter_mut().zip(sums[c].iter()) {
                    *d = s / cnt;
                }
            }
        }
    }

    assignments
}

/// A node in the Mapper graph: holds the global draw indices belonging to this cluster.
struct MapperNode {
    draw_indices: Vec<usize>,
}

/// Build the Mapper graph. Returns nodes and adjacency (node degree per node).
fn build_mapper_graph(
    points: &[[f64; 7]],
    n_intervals: usize,
    overlap_frac: f64,
    k_clusters: usize,
    kmeans_max_iter: usize,
) -> (Vec<MapperNode>, Vec<usize>) {
    let n = points.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Filter function: distance to global centroid
    let global_centroid = centroid_7d(points);
    let filter_values: Vec<f64> = points.iter().map(|p| dist_7d(p, &global_centroid)).collect();

    let f_min = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = filter_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let f_range = f_max - f_min;

    if f_range < 1e-15 {
        // All points at same distance — single cluster
        let node = MapperNode {
            draw_indices: (0..n).collect(),
        };
        return (vec![node], vec![0]);
    }

    // Divide filter range into overlapping intervals
    let step = f_range / n_intervals as f64;
    let half_width = step * (1.0 + overlap_frac) / 2.0;

    // For each point, track which mapper nodes it belongs to (for edge detection)
    let mut point_to_nodes: Vec<Vec<usize>> = vec![vec![]; n];
    let mut nodes: Vec<MapperNode> = Vec::new();

    for interval_idx in 0..n_intervals {
        let center = f_min + step * (interval_idx as f64 + 0.5);
        let lo = center - half_width;
        let hi = center + half_width;

        // Collect points in this interval
        let in_interval: Vec<usize> = (0..n)
            .filter(|&i| filter_values[i] >= lo && filter_values[i] <= hi)
            .collect();

        if in_interval.is_empty() {
            continue;
        }

        // Cluster within interval
        let assignments = kmeans_7d(points, &in_interval, k_clusters, kmeans_max_iter);
        let actual_k = assignments.iter().cloned().max().unwrap_or(0) + 1;

        for cluster_id in 0..actual_k {
            let members: Vec<usize> = in_interval
                .iter()
                .zip(assignments.iter())
                .filter(|&(_, &a)| a == cluster_id)
                .map(|(&idx, _)| idx)
                .collect();

            if members.is_empty() {
                continue;
            }

            let node_id = nodes.len();
            for &m in &members {
                point_to_nodes[m].push(node_id);
            }
            nodes.push(MapperNode {
                draw_indices: members,
            });
        }
    }

    // Compute node degrees from edges (shared draws between nodes)
    let n_nodes = nodes.len();
    let mut degree = vec![0usize; n_nodes];

    // Track edges to avoid double-counting
    let mut seen_edges: Vec<(usize, usize)> = Vec::new();
    for node_list in &point_to_nodes {
        for (i, &a) in node_list.iter().enumerate() {
            for &b in node_list.iter().skip(i + 1) {
                let edge = if a < b { (a, b) } else { (b, a) };
                // Simple dedup (acceptable for small graphs)
                if !seen_edges.contains(&edge) {
                    seen_edges.push(edge);
                    degree[a] += 1;
                    degree[b] += 1;
                }
            }
        }
    }

    (nodes, degree)
}

impl MapperTdaModel {
    fn predict_pool(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform_val = 1.0 / size as f64;
        let uniform = vec![uniform_val; size];

        if draws.len() < self.min_draws {
            return uniform;
        }

        // Encode all draws in 7D (chronological order for Mapper, but we reverse
        // since draws[0] = most recent)
        let encoded: Vec<[f64; 7]> = draws.iter().rev().map(|d| encode_draw_7d(d)).collect();
        let n = encoded.len();

        // Build Mapper graph
        let (nodes, degree) = build_mapper_graph(
            &encoded,
            self.n_intervals,
            self.overlap_frac,
            self.k_clusters,
            self.kmeans_max_iter,
        );

        if nodes.is_empty() {
            return uniform;
        }

        // Find which node contains the last draw (chronologically last = index n-1)
        let last_idx = n - 1;
        let mut best_node: Option<usize> = None;
        let mut best_dist = f64::MAX;

        for (node_id, node) in nodes.iter().enumerate() {
            if node.draw_indices.contains(&last_idx) {
                // Prefer the node with highest degree (most connected = richer context)
                let d = degree[node_id];
                // Use negative degree as "distance" so higher degree wins
                let neg_degree = -(d as f64);
                if neg_degree < best_dist {
                    best_dist = neg_degree;
                    best_node = Some(node_id);
                }
            }
        }

        // If last draw not in any node, find nearest node by centroid distance
        let best_node = best_node.unwrap_or_else(|| {
            let last_pt = &encoded[last_idx];
            let mut nearest = 0;
            let mut nearest_dist = f64::MAX;
            for (node_id, node) in nodes.iter().enumerate() {
                let node_points: Vec<[f64; 7]> =
                    node.draw_indices.iter().map(|&i| encoded[i]).collect();
                let c = centroid_7d(&node_points);
                let d = dist_7d(last_pt, &c);
                if d < nearest_dist {
                    nearest_dist = d;
                    nearest = node_id;
                }
            }
            nearest
        });

        // Compute number frequencies within the best node's cluster
        let node = &nodes[best_node];
        let node_degree = degree[best_node];
        // Chronological draws corresponding to this node
        let chrono_draws: Vec<&Draw> = draws.iter().rev().collect();

        let mut freqs = vec![0.0f64; size];
        let mut total = 0.0;

        for &draw_idx in &node.draw_indices {
            let draw = chrono_draws[draw_idx];
            let numbers = pool.numbers_from(draw);
            for &num in numbers {
                freqs[(num - 1) as usize] += 1.0;
                total += 1.0;
            }
        }

        if total == 0.0 {
            return uniform;
        }

        // Normalize to frequency distribution
        for f in &mut freqs {
            *f /= total;
        }

        // Degree-based weight: higher connectivity → more trust in cluster signal
        // degree_weight in [0.3, 0.8]
        let max_degree = degree.iter().cloned().max().unwrap_or(1).max(1) as f64;
        let degree_weight = 0.3 + 0.5 * (node_degree as f64 / max_degree);

        // Blend cluster frequency with uniform, weighted by connectivity
        let mut probs = vec![0.0f64; size];
        for k in 0..size {
            probs[k] = degree_weight * freqs[k] + (1.0 - degree_weight) * uniform_val;
        }

        // Apply smoothing
        for p in &mut probs {
            *p = (1.0 - self.smoothing) * *p + self.smoothing * uniform_val;
        }

        let floor = match pool {
            Pool::Balls => PROB_FLOOR_BALLS,
            Pool::Stars => PROB_FLOOR_STARS,
        };
        floor_only(&mut probs, floor);
        probs
    }
}

impl ForecastModel for MapperTdaModel {
    fn name(&self) -> &str {
        "MapperTDA"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        self.predict_pool(draws, pool)
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("smoothing".into(), self.smoothing),
            ("n_intervals".into(), self.n_intervals as f64),
            ("overlap_frac".into(), self.overlap_frac),
            ("k_clusters".into(), self.k_clusters as f64),
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
    fn test_mapper_balls_valid_distribution() {
        let model = MapperTdaModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_mapper_stars_valid_distribution() {
        let model = MapperTdaModel::default();
        let draws = make_test_draws(100);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_mapper_no_negative_probabilities() {
        let model = MapperTdaModel::default();
        let draws = make_test_draws(100);
        for pool in [Pool::Balls, Pool::Stars] {
            let dist = model.predict(&draws, pool);
            for (i, &p) in dist.iter().enumerate() {
                assert!(p >= 0.0, "Negative prob at index {}: {}", i, p);
            }
        }
    }

    #[test]
    fn test_mapper_few_draws_returns_uniform() {
        let model = MapperTdaModel::default();
        let draws = make_test_draws(20);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!(
                (p - expected).abs() < 1e-10,
                "Too few draws should return uniform"
            );
        }
    }

    #[test]
    fn test_mapper_empty_draws_returns_uniform() {
        let model = MapperTdaModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mapper_deterministic() {
        let model = MapperTdaModel::default();
        let draws = make_test_draws(100);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "MapperTDA should be deterministic");
        }
    }

    #[test]
    fn test_mapper_sparse_strategy() {
        let model = MapperTdaModel::default();
        assert!(matches!(
            model.sampling_strategy(),
            SamplingStrategy::Sparse { span_multiplier: 3 }
        ));
    }

    #[test]
    fn test_mapper_name() {
        let model = MapperTdaModel::default();
        assert_eq!(model.name(), "MapperTDA");
    }

    #[test]
    fn test_kmeans_single_point() {
        let points = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]];
        let indices = vec![0];
        let assignments = kmeans_7d(&points, &indices, 3, 20);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0], 0);
    }

    #[test]
    fn test_kmeans_two_clusters() {
        let points = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        ];
        let indices: Vec<usize> = (0..4).collect();
        let assignments = kmeans_7d(&points, &indices, 2, 20);
        assert_eq!(assignments.len(), 4);
        // Points 0,1 should be in same cluster; points 2,3 in same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_build_mapper_graph_creates_nodes() {
        let draws = make_test_draws(80);
        let encoded: Vec<[f64; 7]> = draws.iter().map(|d| encode_draw_7d(d)).collect();
        let (nodes, degree) = build_mapper_graph(&encoded, 10, 0.50, 3, 20);
        assert!(!nodes.is_empty(), "Mapper graph should have nodes");
        assert_eq!(nodes.len(), degree.len());
        // Every draw should appear in at least one node
        let mut covered = vec![false; encoded.len()];
        for node in &nodes {
            for &idx in &node.draw_indices {
                covered[idx] = true;
            }
        }
        let coverage = covered.iter().filter(|&&c| c).count();
        assert!(
            coverage > encoded.len() / 2,
            "At least half the draws should be covered: {}/{}",
            coverage,
            encoded.len()
        );
    }

    #[test]
    fn test_encode_draw_7d_range() {
        let draw = Draw {
            draw_id: "1".to_string(),
            day: "MARDI".to_string(),
            date: "2024-01-01".to_string(),
            balls: [1, 25, 30, 40, 50],
            stars: [1, 12],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        };
        let enc = encode_draw_7d(&draw);
        for &v in &enc {
            assert!(
                (0.0..=1.0).contains(&v),
                "Encoded value out of [0,1]: {}",
                v
            );
        }
        // Check specific values
        assert!((enc[0] - 1.0 / 50.0).abs() < 1e-10); // ball 1
        assert!((enc[4] - 1.0).abs() < 1e-10); // ball 50
        assert!((enc[6] - 1.0).abs() < 1e-10); // star 12
    }
}
