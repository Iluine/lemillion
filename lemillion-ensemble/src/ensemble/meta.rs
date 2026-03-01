use lemillion_db::models::Draw;

/// Features du contexte courant pour le méta-prédicteur.
#[derive(Debug, Clone)]
pub struct RegimeFeatures {
    /// Somme des boules du dernier tirage (normalisée 0-1)
    pub sum_norm: f64,
    /// Spread (max - min) du dernier tirage (normalisé 0-1)
    pub spread_norm: f64,
    /// Cosine mod4 entre les 2 derniers tirages
    pub mod4_cosine: f64,
    /// Entropie des fréquences récentes (fenêtre 10)
    pub recent_entropy: f64,
}

impl RegimeFeatures {
    /// Extrait les features du contexte à partir des N derniers tirages.
    pub fn from_draws(draws: &[Draw]) -> Self {
        if draws.is_empty() {
            return Self {
                sum_norm: 0.5,
                spread_norm: 0.5,
                mod4_cosine: 0.5,
                recent_entropy: 1.0,
            };
        }

        let last = &draws[0];

        // Sum normalisée : min=5, max=250, mais réaliste ~60-200
        let sum: f64 = last.balls.iter().map(|&b| b as f64).sum();
        let sum_norm = (sum - 15.0) / (240.0 - 15.0);

        // Spread normalisé
        let max_b = *last.balls.iter().max().unwrap() as f64;
        let min_b = *last.balls.iter().min().unwrap() as f64;
        let spread_norm = (max_b - min_b) / 49.0;

        // Cosine mod4 entre les 2 derniers tirages
        let mod4_cosine = if draws.len() >= 2 {
            let m4_a = mod4_vec(&draws[0].balls);
            let m4_b = mod4_vec(&draws[1].balls);
            cosine_sim(&m4_a, &m4_b)
        } else {
            0.5
        };

        // Entropie des fréquences récentes (fenêtre 10)
        let window = draws.len().min(10);
        let recent_entropy = if window >= 3 {
            let mut counts = [0u32; 50];
            for d in &draws[..window] {
                for &b in &d.balls {
                    counts[(b - 1) as usize] += 1;
                }
            }
            let total = (window * 5) as f64;
            let mut h = 0.0f64;
            for &c in &counts {
                if c > 0 {
                    let p = c as f64 / total;
                    h -= p * p.ln();
                }
            }
            // Normaliser par H_max = ln(50)
            h / (50.0f64).ln()
        } else {
            1.0
        };

        Self {
            sum_norm: sum_norm.clamp(0.0, 1.0),
            spread_norm: spread_norm.clamp(0.0, 1.0),
            mod4_cosine: mod4_cosine.clamp(0.0, 1.0),
            recent_entropy: recent_entropy.clamp(0.0, 1.0),
        }
    }

    /// Retourne le vecteur de features (4D).
    pub fn as_vec(&self) -> [f64; 4] {
        [self.sum_norm, self.spread_norm, self.mod4_cosine, self.recent_entropy]
    }
}

/// Méta-prédicteur : ajuste les poids des modèles en fonction du régime courant.
///
/// Entraîné par ridge regression sur les LL historiques.
/// Pour chaque modèle m, apprend w_m = β_m · features + β_m0.
#[derive(Debug, Clone)]
pub struct MetaPredictor {
    /// Coefficients par modèle : (intercept, [coeff; 4])
    pub coefficients: Vec<(f64, [f64; 4])>,
    pub model_names: Vec<String>,
}

impl MetaPredictor {
    /// Entraîne le méta-prédicteur à partir des LL détaillés et de l'historique.
    ///
    /// Pour chaque modèle, effectue une ridge regression:
    ///   LL_m(t) ~ β_m · features(t)
    /// puis utilise les coefficients pour ajuster les poids dans le contexte courant.
    pub fn train(
        draws: &[Draw],
        detailed_ll: &[(String, Vec<f64>)],
        ridge_lambda: f64,
    ) -> Option<Self> {
        if detailed_ll.is_empty() || draws.len() < 20 {
            return None;
        }

        let n_models = detailed_ll.len();
        let model_names: Vec<String> = detailed_ll.iter().map(|(n, _)| n.clone()).collect();

        // Construire la matrice de features pour les points de test
        // Les LL détaillés correspondent à ~100 points de test uniformément espacés
        let n_ll = detailed_ll[0].1.len();
        if n_ll < 5 {
            return None;
        }

        let stride = (draws.len() / n_ll).max(1);
        let mut features: Vec<[f64; 4]> = Vec::new();
        for i in 0..n_ll {
            let t = (i * stride).min(draws.len() - 1);
            if t + 1 < draws.len() {
                let ctx = &draws[t + 1..]; // contexte avant le tirage test
                let rf = RegimeFeatures::from_draws(ctx);
                features.push(rf.as_vec());
            }
        }

        let n_samples = features.len().min(n_ll);
        if n_samples < 5 {
            return None;
        }

        // Ridge regression pour chaque modèle
        let mut coefficients = Vec::with_capacity(n_models);
        for (_, lls) in detailed_ll {
            let n = n_samples.min(lls.len());
            let coeff = ridge_regression_4d(&features[..n], &lls[..n], ridge_lambda);
            coefficients.push(coeff);
        }

        Some(Self { coefficients, model_names })
    }

    /// Calcule les ajustements de poids pour le contexte courant.
    /// Retourne un multiplicateur par modèle (centré autour de 1.0).
    pub fn weight_adjustments(&self, features: &RegimeFeatures) -> Vec<(String, f64)> {
        let x = features.as_vec();
        let mut adjustments: Vec<f64> = self.coefficients.iter()
            .map(|(intercept, coeff)| {
                let pred = intercept + coeff.iter().zip(x.iter()).map(|(c, f)| c * f).sum::<f64>();
                // Convertir en multiplicateur via softmax-like
                pred.exp()
            })
            .collect();

        // Normaliser pour que la moyenne soit 1.0
        let mean: f64 = adjustments.iter().sum::<f64>() / adjustments.len() as f64;
        if mean > 0.0 {
            for a in &mut adjustments {
                *a /= mean;
            }
        }

        self.model_names.iter()
            .zip(adjustments)
            .map(|(name, adj)| (name.clone(), adj))
            .collect()
    }
}

// ─── Utilitaires ──────────────────────────────────────────────────────────

fn mod4_vec(balls: &[u8; 5]) -> [f64; 4] {
    let mut v = [0.0f64; 4];
    for &b in balls {
        v[((b - 1) % 4) as usize] += 1.0;
    }
    v
}

fn cosine_sim(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-15 || norm_b < 1e-15 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Ridge regression 4D : y = β0 + β1*x1 + β2*x2 + β3*x3 + β4*x4
/// Résolution analytique : β = (X^T X + λI)^{-1} X^T y
fn ridge_regression_4d(features: &[[f64; 4]], targets: &[f64], lambda: f64) -> (f64, [f64; 4]) {
    let n = features.len();
    if n == 0 {
        return (0.0, [0.0; 4]);
    }

    // Centrer les targets
    let mean_y: f64 = targets.iter().sum::<f64>() / n as f64;

    // X^T X (4x4) + lambda * I
    let mut xtx = [[0.0f64; 4]; 4];
    let mut xty = [0.0f64; 4];

    for (feat, &target) in features.iter().zip(targets.iter()) {
        let y = target - mean_y;
        for (j, &fj) in feat.iter().enumerate() {
            xty[j] += fj * y;
            for (k, &fk) in feat.iter().enumerate() {
                xtx[j][k] += fj * fk;
            }
        }
    }

    // Ajouter régularisation
    for (j, row) in xtx.iter_mut().enumerate() {
        row[j] += lambda * n as f64;
    }

    // Résoudre par Gauss-Jordan 4x4
    let beta = solve_4x4(xtx, xty);

    // Intercept = mean_y - beta . mean_x
    let mut intercept = mean_y;
    for j in 0..4 {
        let mean_x: f64 = features.iter().map(|f| f[j]).sum::<f64>() / n as f64;
        intercept -= beta[j] * mean_x;
    }

    (intercept, beta)
}

/// Résolution d'un système 4x4 par élimination de Gauss avec pivot partiel.
#[allow(clippy::needless_range_loop)]
fn solve_4x4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    // Forward elimination
    for col in 0..4 {
        // Pivot partiel
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            if a_row[col].abs() > max_val {
                max_val = a_row[col].abs();
                max_row = row;
            }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        if a[col][col].abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..4 {
            let factor = a[row][col] / a[col][col];
            for k in col..4 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = [0.0f64; 4];
    for col in (0..4).rev() {
        if a[col][col].abs() < 1e-15 {
            continue;
        }
        x[col] = b[col];
        for k in (col + 1)..4 {
            x[col] -= a[col][k] * x[k];
        }
        x[col] /= a[col][col];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_regime_features_empty() {
        let f = RegimeFeatures::from_draws(&[]);
        assert!((f.sum_norm - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_regime_features_valid_range() {
        let draws = make_test_draws(20);
        let f = RegimeFeatures::from_draws(&draws);
        assert!(f.sum_norm >= 0.0 && f.sum_norm <= 1.0);
        assert!(f.spread_norm >= 0.0 && f.spread_norm <= 1.0);
        assert!(f.mod4_cosine >= 0.0 && f.mod4_cosine <= 1.0);
        assert!(f.recent_entropy >= 0.0 && f.recent_entropy <= 1.0);
    }

    #[test]
    fn test_ridge_regression_4d_constant() {
        // Si tous les targets sont identiques, les coefficients doivent être ~0
        let features: Vec<[f64; 4]> = (0..20)
            .map(|i| [i as f64 / 20.0, 0.5, 0.3, 0.7])
            .collect();
        let targets = vec![1.0; 20];
        let (intercept, coeff) = ridge_regression_4d(&features, &targets, 1.0);
        assert!((intercept - 1.0).abs() < 0.5, "Intercept should be near 1.0, got {}", intercept);
        for &c in &coeff {
            assert!(c.abs() < 1.0, "Coefficient should be small, got {}", c);
        }
    }

    #[test]
    fn test_meta_predictor_no_data() {
        let draws = make_test_draws(5);
        let result = MetaPredictor::train(&draws, &[], 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_cosine_sim_identical() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_adjustments_positive() {
        let mp = MetaPredictor {
            coefficients: vec![(0.0, [0.1, 0.2, 0.3, 0.4]), (0.0, [-0.1, 0.0, 0.1, 0.2])],
            model_names: vec!["A".into(), "B".into()],
        };
        let f = RegimeFeatures { sum_norm: 0.5, spread_norm: 0.5, mod4_cosine: 0.5, recent_entropy: 0.5 };
        let adj = mp.weight_adjustments(&f);
        assert_eq!(adj.len(), 2);
        for (_, v) in &adj {
            assert!(*v > 0.0, "Weight adjustment must be positive");
        }
    }
}
