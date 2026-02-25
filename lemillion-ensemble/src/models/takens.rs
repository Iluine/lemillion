use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// Modèle basé sur le théorème de Takens : reconstruction d'espace de phase + K-NN.
pub struct TakensKnnModel {
    k: usize,
    tau: usize,
    embedding_dim: usize,
}

impl TakensKnnModel {
    pub fn new(k: usize, tau: usize, embedding_dim: usize) -> Self {
        Self {
            k,
            tau,
            embedding_dim,
        }
    }
}

impl Default for TakensKnnModel {
    fn default() -> Self {
        Self {
            k: 5,
            tau: 1,
            embedding_dim: 3,
        }
    }
}

/// Encode un tirage en vecteur scalaire pour l'embedding.
fn encode_draw(draw: &Draw, pool: Pool) -> Vec<f64> {
    match pool {
        Pool::Balls => {
            let sorted = {
                let mut b = draw.balls;
                b.sort();
                b
            };
            let sum = sorted.iter().map(|&b| b as f64).sum::<f64>();
            let spread = (sorted[4] - sorted[0]) as f64;
            let odd = sorted.iter().filter(|&&b| b % 2 == 1).count() as f64;
            let centroid = sum / 5.0;
            vec![sum / 250.0, spread / 49.0, odd / 5.0, centroid / 50.0]
        }
        Pool::Stars => {
            let sorted = {
                let mut s = draw.stars;
                s.sort();
                s
            };
            let sum = sorted.iter().map(|&s| s as f64).sum::<f64>();
            let spread = (sorted[1] - sorted[0]) as f64;
            vec![sum / 24.0, spread / 11.0]
        }
    }
}

/// Construit les vecteurs d'embedding de Takens à partir d'une série multivariée.
/// Retourne (embedded_vectors, indices_dans_la_série_originale).
fn build_embedding(
    series: &[Vec<f64>],
    dim: usize,
    tau: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let offset = (dim - 1) * tau;
    if offset >= series.len() {
        return (vec![], vec![]);
    }

    let feat_dim = series[0].len();
    let mut vectors = Vec::new();
    let mut indices = Vec::new();

    for t in offset..series.len() {
        let mut v = Vec::with_capacity(dim * feat_dim);
        for d in 0..dim {
            v.extend_from_slice(&series[t - d * tau]);
        }
        vectors.push(v);
        indices.push(t);
    }

    (vectors, indices)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Trouve les K plus proches voisins du point query.
/// Retourne les indices et distances, triés par distance croissante.
fn find_k_nearest(
    embedded: &[Vec<f64>],
    query: &[f64],
    k: usize,
    exclude_last: usize,
) -> Vec<(usize, f64)> {
    let search_end = embedded.len().saturating_sub(exclude_last);
    let mut distances: Vec<(usize, f64)> = embedded[..search_end]
        .iter()
        .enumerate()
        .map(|(i, v)| (i, euclidean_distance(v, query)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);
    distances
}

impl ForecastModel for TakensKnnModel {
    fn name(&self) -> &str {
        "TakensKNN"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        // Besoin d'assez de tirages pour l'embedding
        let min_draws = (self.embedding_dim - 1) * self.tau + self.k + 2;
        if draws.len() < min_draws {
            return uniform;
        }

        // Ordre chronologique (draws[0] = le plus récent → on inverse)
        let chrono_draws: Vec<&Draw> = draws.iter().rev().collect();

        // Encoder chaque tirage
        let series: Vec<Vec<f64>> = chrono_draws.iter().map(|d| encode_draw(d, pool)).collect();

        // Construire l'embedding
        let (embedded, embed_indices) =
            build_embedding(&series, self.embedding_dim, self.tau);

        if embedded.len() < self.k + 1 {
            return uniform;
        }

        // Le dernier point de l'embedding est notre query
        let query = embedded.last().unwrap();

        // Trouver les K plus proches voisins (exclure le dernier point lui-même)
        let neighbors = find_k_nearest(&embedded, query, self.k, 1);

        if neighbors.is_empty() {
            return uniform;
        }

        // Pour chaque voisin, regarder le tirage qui a SUIVI
        let mut weights = vec![0.0f64; size];
        let mut total_weight = 0.0f64;
        let epsilon = 1e-8;

        for &(embed_idx, dist) in &neighbors {
            let orig_idx = embed_indices[embed_idx];
            // Le successeur dans la série chronologique
            if orig_idx + 1 >= chrono_draws.len() {
                continue;
            }
            let successor = chrono_draws[orig_idx + 1];
            let w = 1.0 / (dist + epsilon);
            total_weight += w;

            for &num in pool.numbers_from(successor) {
                let idx = (num - 1) as usize;
                if idx < size {
                    weights[idx] += w;
                }
            }
        }

        if total_weight == 0.0 {
            return uniform;
        }

        // Normaliser
        let raw_sum: f64 = weights.iter().sum();
        if raw_sum == 0.0 {
            return uniform;
        }

        // Lisser avec la distribution uniforme (mix 70/30 pour éviter les zéros)
        let alpha = 0.7;
        let uniform_val = 1.0 / size as f64;
        let mut dist: Vec<f64> = weights
            .iter()
            .map(|&w| alpha * (w / raw_sum) + (1.0 - alpha) * uniform_val)
            .collect();

        // Renormaliser
        let sum: f64 = dist.iter().sum();
        for p in &mut dist {
            *p /= sum;
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("k".into(), self.k as f64),
            ("tau".into(), self.tau as f64),
            ("embedding_dim".into(), self.embedding_dim as f64),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_encode_draw_balls() {
        let draw = Draw {
            draw_id: "001".into(),
            day: "MARDI".into(),
            date: "2024-01-01".into(),
            balls: [10, 20, 30, 40, 50],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: String::new(),
        };
        let enc = encode_draw(&draw, Pool::Balls);
        assert_eq!(enc.len(), 4);
        assert!((enc[0] - 150.0 / 250.0).abs() < 1e-10); // sum=150
        assert!((enc[1] - 40.0 / 49.0).abs() < 1e-10); // spread=40
    }

    #[test]
    fn test_build_embedding() {
        let series: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let (embedded, indices) = build_embedding(&series, 3, 1);
        assert_eq!(embedded.len(), 8); // 10 - 2*1 = 8
        assert_eq!(embedded[0].len(), 6); // 3 * 2 features
        assert_eq!(indices[0], 2);
    }

    #[test]
    fn test_takens_predict_valid_distribution() {
        let model = TakensKnnModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls));
    }

    #[test]
    fn test_takens_predict_stars() {
        let model = TakensKnnModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars));
    }

    #[test]
    fn test_takens_few_draws_uniform() {
        let model = TakensKnnModel::default();
        let draws = make_test_draws(3);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_find_k_nearest() {
        let embedded = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
        ];
        let query = vec![0.5, 0.5];
        let neighbors = find_k_nearest(&embedded, &query, 2, 0);
        assert_eq!(neighbors.len(), 2);
        // Les deux plus proches devraient être les 3 premiers points
    }
}
