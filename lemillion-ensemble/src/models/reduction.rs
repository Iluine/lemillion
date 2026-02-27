use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Stratégie de réduction de dimensionnalité pour les meta-features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReductionStrategy {
    None,
    Pca {
        n_components: usize,
        mean: Vec<f64>,
        components: Vec<Vec<f64>>,
        explained_variance_ratio: f64,
    },
}

impl Default for ReductionStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Transformation PCA calculée à partir de données d'entraînement.
pub struct PcaTransform {
    pub n_components: usize,
    pub mean: Vec<f64>,
    pub components: Vec<Vec<f64>>,
    pub explained_variance_ratio: f64,
}

impl PcaTransform {
    /// Ajuste la PCA sur les features avec un nombre fixe de composantes.
    pub fn fit(features: &[Vec<f64>], n_components: usize) -> Result<Self> {
        let (components, _eigenvalues, mean) =
            lemillion_esn::linalg::pca_svd(features, n_components)?;

        // Compute total variance for explained ratio
        let (_, all_ev, _) =
            lemillion_esn::linalg::pca_svd(features, n_components.min(features.len().min(features[0].len())))?;
        let total_var: f64 = all_ev.iter().sum();
        let explained: f64 = _eigenvalues.iter().sum();
        let ratio = if total_var > 0.0 { explained / total_var } else { 0.0 };

        let comp_vec: Vec<Vec<f64>> = (0..n_components)
            .map(|i| components.row(i).to_vec())
            .collect();

        Ok(Self {
            n_components,
            mean,
            components: comp_vec,
            explained_variance_ratio: ratio,
        })
    }

    /// Ajuste la PCA avec sélection automatique du nombre de composantes.
    pub fn fit_auto(features: &[Vec<f64>], target_variance: f64) -> Result<Self> {
        let (components, _eigenvalues, mean, ratio) =
            lemillion_esn::linalg::pca_svd_auto(features, target_variance)?;

        let k = components.nrows();
        let comp_vec: Vec<Vec<f64>> = (0..k)
            .map(|i| components.row(i).to_vec())
            .collect();

        Ok(Self {
            n_components: k,
            mean,
            components: comp_vec,
            explained_variance_ratio: ratio,
        })
    }

    /// Projette un vecteur dans l'espace PCA.
    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.n_components];
        for (i, comp) in self.components.iter().enumerate() {
            let mut dot = 0.0;
            for (j, &c) in comp.iter().enumerate() {
                dot += (x[j] - self.mean[j]) * c;
            }
            result[i] = dot;
        }
        result
    }

    /// Projette un batch de vecteurs.
    pub fn transform_batch(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features.iter().map(|x| self.transform(x)).collect()
    }

    /// Convertit en ReductionStrategy pour sérialisation.
    pub fn to_strategy(&self) -> ReductionStrategy {
        ReductionStrategy::Pca {
            n_components: self.n_components,
            mean: self.mean.clone(),
            components: self.components.clone(),
            explained_variance_ratio: self.explained_variance_ratio,
        }
    }
}

/// Standardisation z-score des features (mean=0, std=1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStandardizer {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl FeatureStandardizer {
    /// Calcule mean et std par feature.
    pub fn fit(features: &[Vec<f64>]) -> Self {
        let n = features.len() as f64;
        let d = if features.is_empty() { 0 } else { features[0].len() };

        let mut mean = vec![0.0; d];
        for row in features {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut var = vec![0.0; d];
        for row in features {
            for (j, &v) in row.iter().enumerate() {
                let diff = v - mean[j];
                var[j] += diff * diff;
            }
        }

        let std: Vec<f64> = var
            .iter()
            .map(|&v| {
                let s = (v / (n - 1.0).max(1.0)).sqrt();
                if s < 1e-12 { 1.0 } else { s }
            })
            .collect();

        Self { mean, std }
    }

    /// Standardise un vecteur : (x - mean) / std.
    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(j, &v)| (v - self.mean[j]) / self.std[j])
            .collect()
    }

    /// Standardise un batch.
    pub fn transform_batch(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features.iter().map(|x| self.transform(x)).collect()
    }
}

/// Applique une ReductionStrategy sérialisée à un vecteur (pour l'inférence).
pub fn apply_reduction(x: &[f64], strategy: &ReductionStrategy) -> Vec<f64> {
    match strategy {
        ReductionStrategy::None => x.to_vec(),
        ReductionStrategy::Pca { n_components, mean, components, .. } => {
            let mut result = vec![0.0; *n_components];
            for (i, comp) in components.iter().enumerate() {
                let mut dot = 0.0;
                for (j, &c) in comp.iter().enumerate() {
                    dot += (x[j] - mean[j]) * c;
                }
                result[i] = dot;
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_standardizer() {
        let features = vec![
            vec![1.0, 10.0, 100.0],
            vec![3.0, 30.0, 300.0],
            vec![5.0, 50.0, 500.0],
        ];
        let std = FeatureStandardizer::fit(&features);
        assert!((std.mean[0] - 3.0).abs() < 1e-10);
        assert!((std.mean[1] - 30.0).abs() < 1e-10);

        let transformed = std.transform(&features[0]);
        // (1 - 3) / std(1,3,5) = -2/2 = -1
        assert!((transformed[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pca_transform_roundtrip() {
        let features = vec![
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.1, 0.0],
            vec![3.0, 0.0, 0.1],
            vec![4.0, 0.1, 0.1],
            vec![5.0, 0.0, 0.0],
        ];
        let pca = PcaTransform::fit_auto(&features, 0.95).unwrap();
        assert!(pca.n_components >= 1);
        assert!(pca.explained_variance_ratio >= 0.95);

        let projected = pca.transform(&features[0]);
        assert_eq!(projected.len(), pca.n_components);

        // Test strategy roundtrip
        let strategy = pca.to_strategy();
        let applied = apply_reduction(&features[0], &strategy);
        for (a, b) in projected.iter().zip(applied.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_reduction_none() {
        let x = vec![1.0, 2.0, 3.0];
        let result = apply_reduction(&x, &ReductionStrategy::None);
        assert_eq!(result, x);
    }
}
