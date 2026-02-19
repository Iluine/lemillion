use std::collections::HashMap;
use lemillion_db::models::{Draw, Pool};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::prelude::*;
use super::ForecastModel;
use crate::features;

pub struct RandomForestModel {
    n_trees: usize,
    max_depth: usize,
    window: usize,
}

impl RandomForestModel {
    pub fn new(n_trees: usize, max_depth: usize, window: usize) -> Self {
        Self { n_trees, max_depth, window }
    }
}

impl ForecastModel for RandomForestModel {
    fn name(&self) -> &str {
        "RandomForest"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        if draws.len() < 3 {
            return vec![1.0 / size as f64; size];
        }

        let effective_window = self.window.min(draws.len().saturating_sub(1));
        if effective_window < 2 {
            return vec![1.0 / size as f64; size];
        }

        // Collecter les données d'entraînement
        let mut all_features = Vec::new();
        let mut all_labels = Vec::new();

        for t in 1..effective_window {
            let rows = features::extract_features_for_draw(draws, pool, t);
            for row in rows {
                all_features.push(row.features);
                all_labels.push(row.label);
            }
        }

        if all_features.is_empty() {
            return vec![1.0 / size as f64; size];
        }

        let n_features = features::FEATURE_NAMES.len();
        let features_per_split = (n_features as f64).sqrt().ceil() as usize;

        // Entraîner la forêt
        let mut rng = StdRng::seed_from_u64(42);
        let mut forest = Vec::with_capacity(self.n_trees);

        for _ in 0..self.n_trees {
            // Bootstrap sampling
            let n_samples = all_features.len();
            let indices: Vec<usize> = (0..n_samples).map(|_| rng.random_range(0..n_samples)).collect();

            let boot_features: Vec<&Vec<f64>> = indices.iter().map(|&i| &all_features[i]).collect();
            let boot_labels: Vec<f64> = indices.iter().map(|&i| all_labels[i]).collect();

            let tree = build_tree(&boot_features, &boot_labels, self.max_depth, features_per_split, &mut rng);
            forest.push(tree);
        }

        // Prédire pour le tirage actuel (index 0)
        let current_rows = features::extract_features_for_draw(draws, pool, 0);
        let mut scores: Vec<f64> = current_rows
            .iter()
            .map(|row| {
                let sum: f64 = forest.iter().map(|tree| predict_tree(tree, &row.features)).sum();
                sum / self.n_trees as f64
            })
            .collect();

        // Normaliser
        // Appliquer un plancher pour éviter les zéros
        let floor = 1e-6;
        for s in &mut scores {
            if *s < floor {
                *s = floor;
            }
        }

        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in &mut scores {
                *s /= total;
            }
        } else {
            scores = vec![1.0 / size as f64; size];
        }

        scores
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("n_trees".to_string(), self.n_trees as f64),
            ("max_depth".to_string(), self.max_depth as f64),
            ("window".to_string(), self.window as f64),
        ])
    }
}

#[derive(Debug)]
enum TreeNode {
    Leaf { value: f64 },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

fn build_tree(
    features: &[&Vec<f64>],
    labels: &[f64],
    max_depth: usize,
    features_per_split: usize,
    rng: &mut StdRng,
) -> TreeNode {
    if max_depth == 0 || labels.len() < 4 {
        return TreeNode::Leaf {
            value: labels.iter().sum::<f64>() / labels.len().max(1) as f64,
        };
    }

    // Vérifier si toutes les étiquettes sont identiques
    let first = labels[0];
    if labels.iter().all(|&l| (l - first).abs() < 1e-10) {
        return TreeNode::Leaf { value: first };
    }

    let n_features = features[0].len();
    // Sélectionner des features aléatoires
    let mut feature_indices: Vec<usize> = (0..n_features).collect();
    feature_indices.shuffle(rng);
    feature_indices.truncate(features_per_split);

    let mut best_gini = f64::MAX;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for &feat_idx in &feature_indices {
        // Collecte des valeurs uniques triées pour cette feature
        let mut values: Vec<f64> = features.iter().map(|f| f[feat_idx]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();

        if values.len() < 2 {
            continue;
        }

        // Tester les seuils entre valeurs consécutives (échantillonnage pour performance)
        let step = (values.len() / 10).max(1);
        for i in (0..values.len() - 1).step_by(step) {
            let threshold = (values[i] + values[i + 1]) / 2.0;
            let gini = split_gini(features, labels, feat_idx, threshold);

            if gini < best_gini {
                best_gini = gini;
                best_feature = feat_idx;
                best_threshold = threshold;
            }
        }
    }

    if best_gini >= gini_impurity(labels) {
        return TreeNode::Leaf {
            value: labels.iter().sum::<f64>() / labels.len() as f64,
        };
    }

    // Séparer
    let mut left_features = Vec::new();
    let mut left_labels = Vec::new();
    let mut right_features = Vec::new();
    let mut right_labels = Vec::new();

    for (i, feat) in features.iter().enumerate() {
        if feat[best_feature] <= best_threshold {
            left_features.push(*feat);
            left_labels.push(labels[i]);
        } else {
            right_features.push(*feat);
            right_labels.push(labels[i]);
        }
    }

    if left_features.is_empty() || right_features.is_empty() {
        return TreeNode::Leaf {
            value: labels.iter().sum::<f64>() / labels.len() as f64,
        };
    }

    TreeNode::Split {
        feature_idx: best_feature,
        threshold: best_threshold,
        left: Box::new(build_tree(&left_features, &left_labels, max_depth - 1, features_per_split, rng)),
        right: Box::new(build_tree(&right_features, &right_labels, max_depth - 1, features_per_split, rng)),
    }
}

fn gini_impurity(labels: &[f64]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let n = labels.len() as f64;
    let p = labels.iter().sum::<f64>() / n;
    2.0 * p * (1.0 - p)
}

fn split_gini(features: &[&Vec<f64>], labels: &[f64], feature_idx: usize, threshold: f64) -> f64 {
    let mut left_labels = Vec::new();
    let mut right_labels = Vec::new();

    for (i, feat) in features.iter().enumerate() {
        if feat[feature_idx] <= threshold {
            left_labels.push(labels[i]);
        } else {
            right_labels.push(labels[i]);
        }
    }

    let n = labels.len() as f64;
    let n_left = left_labels.len() as f64;
    let n_right = right_labels.len() as f64;

    if n_left == 0.0 || n_right == 0.0 {
        return f64::MAX;
    }

    (n_left / n) * gini_impurity(&left_labels) + (n_right / n) * gini_impurity(&right_labels)
}

fn predict_tree(node: &TreeNode, features: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf { value } => *value,
        TreeNode::Split { feature_idx, threshold, left, right } => {
            if features[*feature_idx] <= *threshold {
                predict_tree(left, features)
            } else {
                predict_tree(right, features)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_random_forest_balls_sums_to_one() {
        let model = RandomForestModel::new(20, 3, 50);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_random_forest_stars_sums_to_one() {
        let model = RandomForestModel::new(20, 3, 50);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}", dist.iter().sum::<f64>(), dist.len());
    }

    #[test]
    fn test_random_forest_no_negative() {
        let model = RandomForestModel::new(20, 3, 50);
        let draws = make_test_draws(30);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_random_forest_empty_draws() {
        let model = RandomForestModel::new(20, 3, 50);
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        assert_eq!(dist.len(), 50);
    }
}
