use lemillion_db::models::Draw;

const N_FEATURES: usize = 7;

/// Features du contexte courant pour le méta-prédicteur (7D).
#[derive(Debug, Clone)]
pub struct RegimeFeatures {
    /// Somme des boules du dernier tirage (normalisée 0-1)
    pub sum_norm: f64,
    /// Spread (max - min) du dernier tirage (normalisé 0-1)
    pub spread_norm: f64,
    /// Cosine mod4 entre les 2 derniers tirages
    pub mod4_cosine: f64,
    /// Entropie des fréquences récentes (fenêtre 20)
    pub recent_entropy: f64,
    /// Jour de la semaine (MARDI=0, VENDREDI=1)
    pub day_of_week: f64,
    /// Ratio de compression des gaps courants vs géométrique attendu
    pub gap_compression: f64,
    /// Hurst exponent (DFA) moyen sur les 50 séries de présence des boules.
    /// H > 0.5 = persistence (momentum), H < 0.5 = mean-reversion, H = 0.5 = random.
    pub hurst_exponent: f64,
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
                day_of_week: 0.5,
                gap_compression: 1.0,
                hurst_exponent: 0.5,
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

        // Entropie des fréquences récentes (fenêtre 20)
        let window = draws.len().min(20);
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

        // Jour de la semaine : MARDI=0, VENDREDI=1 (basé sur le jour dans la date)
        let day_of_week = if last.day.to_uppercase().contains("VENDREDI")
            || last.day.to_uppercase().contains("FRI")
        {
            1.0
        } else {
            0.0
        };

        // Gap compression : ratio moyen des gaps courants vs espérance géométrique
        let gap_compression = compute_gap_compression(draws);

        // Hurst exponent via DFA sur les séries de présence des 50 boules
        let hurst_exponent = compute_hurst_dfa(draws);

        Self {
            sum_norm: sum_norm.clamp(0.0, 1.0),
            spread_norm: spread_norm.clamp(0.0, 1.0),
            mod4_cosine: mod4_cosine.clamp(0.0, 1.0),
            recent_entropy: recent_entropy.clamp(0.0, 1.0),
            day_of_week,
            gap_compression: gap_compression.clamp(0.0, 2.0),
            hurst_exponent: hurst_exponent.clamp(0.0, 1.0),
        }
    }

    /// Retourne le vecteur de features (7D).
    pub fn as_vec(&self) -> [f64; N_FEATURES] {
        [self.sum_norm, self.spread_norm, self.mod4_cosine, self.recent_entropy, self.day_of_week, self.gap_compression, self.hurst_exponent]
    }
}

/// Méta-prédicteur : ajuste les poids des modèles en fonction du régime courant.
///
/// Entraîné par ridge regression sur les LL historiques.
/// Pour chaque modèle m, apprend w_m = β_m · features + β_m0.
#[derive(Debug, Clone)]
pub struct MetaPredictor {
    /// Coefficients par modèle : (intercept, [coeff; N_FEATURES])
    pub coefficients: Vec<(f64, [f64; N_FEATURES])>,
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
        let n_ll = detailed_ll[0].1.len();
        if n_ll < 10 {
            return None;
        }

        // Split: utiliser seulement la 2e moitié des test points pour le meta-predictor
        // afin d'éviter la fuite de données (la 1ère moitié sert à la calibration)
        let meta_start = n_ll / 2;
        let meta_n = n_ll - meta_start;

        let stride = (draws.len() / n_ll).max(1);
        let mut features: Vec<[f64; N_FEATURES]> = Vec::new();
        for i in meta_start..n_ll {
            let t = (i * stride).min(draws.len() - 1);
            if t + 1 < draws.len() {
                let ctx = &draws[t + 1..]; // contexte avant le tirage test
                let rf = RegimeFeatures::from_draws(ctx);
                features.push(rf.as_vec());
            }
        }

        let n_samples = features.len().min(meta_n);
        if n_samples < 5 {
            return None;
        }

        // Ridge regression avec lambda fort pour éviter l'overfitting
        let mut coefficients = Vec::with_capacity(n_models);
        for (_, lls) in detailed_ll {
            if lls.len() <= meta_start {
                // Pas assez de données pour ce modèle — coefficients nuls
                coefficients.push((0.0, [0.0; N_FEATURES]));
                continue;
            }
            let available = lls.len() - meta_start;
            let lls_meta = &lls[meta_start..meta_start + n_samples.min(available)];
            let n = n_samples.min(lls_meta.len());
            let coeff = ridge_regression_nd(&features[..n], &lls_meta[..n], ridge_lambda);
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

/// Calcule l'exposant de Hurst moyen via DFA (Detrended Fluctuation Analysis)
/// sur les 50 séries binaires de présence de chaque boule.
/// H > 0.5 = persistence (momentum), H < 0.5 = mean-reversion, H = 0.5 = random walk.
fn compute_hurst_dfa(draws: &[Draw]) -> f64 {
    let n_draws = draws.len().min(200);
    if n_draws < 32 {
        return 0.5; // pas assez de données
    }

    let window_sizes: &[usize] = &[8, 16, 32, 64, 128];
    // Filtrer les fenêtres trop grandes pour la série
    let valid_windows: Vec<usize> = window_sizes
        .iter()
        .copied()
        .filter(|&w| w <= n_draws / 2)
        .collect();
    if valid_windows.len() < 2 {
        return 0.5;
    }

    let mut hurst_sum = 0.0f64;
    let mut hurst_count = 0u32;

    for ball in 1..=50u8 {
        // Construire la série binaire de présence (ordre chronologique : index 0 = plus ancien)
        let series: Vec<f64> = draws[..n_draws]
            .iter()
            .rev()
            .map(|d| if d.balls.contains(&ball) { 1.0 } else { 0.0 })
            .collect();

        let mean: f64 = series.iter().sum::<f64>() / series.len() as f64;

        // Série de déviation cumulative
        let mut cumdev = Vec::with_capacity(series.len());
        let mut running = 0.0f64;
        for &val in &series {
            running += val - mean;
            cumdev.push(running);
        }

        // Pour chaque taille de fenêtre, calculer la fluctuation RMS
        let mut log_n = Vec::with_capacity(valid_windows.len());
        let mut log_f = Vec::with_capacity(valid_windows.len());

        for &w in &valid_windows {
            let n_segments = cumdev.len() / w;
            if n_segments == 0 {
                continue;
            }

            let mut f2_sum = 0.0f64;
            let mut seg_count = 0usize;

            for seg in 0..n_segments {
                let start = seg * w;
                let end = start + w;
                let segment = &cumdev[start..end];

                // Régression linéaire (détrending) sur le segment
                let mut sx = 0.0f64;
                let mut sy = 0.0f64;
                let mut sxx = 0.0f64;
                let mut sxy = 0.0f64;
                let nf = w as f64;
                for (i, &y) in segment.iter().enumerate() {
                    let x = i as f64;
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    sxy += x * y;
                }
                let denom = nf * sxx - sx * sx;
                let (a, b) = if denom.abs() > 1e-15 {
                    let slope = (nf * sxy - sx * sy) / denom;
                    let intercept = (sy - slope * sx) / nf;
                    (intercept, slope)
                } else {
                    (sy / nf, 0.0)
                };

                // Variance résiduelle
                let mut var = 0.0f64;
                for (i, &y) in segment.iter().enumerate() {
                    let trend = a + b * (i as f64);
                    let resid = y - trend;
                    var += resid * resid;
                }
                f2_sum += var / nf;
                seg_count += 1;
            }

            if seg_count > 0 {
                let f_val = (f2_sum / seg_count as f64).sqrt();
                if f_val > 1e-15 {
                    log_n.push((w as f64).ln());
                    log_f.push(f_val.ln());
                }
            }
        }

        // Régression linéaire log(F) vs log(n) pour obtenir H
        if log_n.len() >= 2 {
            let n_pts = log_n.len() as f64;
            let sx: f64 = log_n.iter().sum();
            let sy: f64 = log_f.iter().sum();
            let sxx: f64 = log_n.iter().map(|x| x * x).sum();
            let sxy: f64 = log_n.iter().zip(log_f.iter()).map(|(x, y)| x * y).sum();
            let denom = n_pts * sxx - sx * sx;
            if denom.abs() > 1e-15 {
                let h = (n_pts * sxy - sx * sy) / denom;
                hurst_sum += h;
                hurst_count += 1;
            }
        }
    }

    if hurst_count > 0 {
        hurst_sum / hurst_count as f64
    } else {
        0.5
    }
}

/// Calcule le ratio de compression des gaps actuels vs l'espérance géométrique.
/// Ratio < 1 = gaps compressés (numéros reviennent plus vite qu'attendu).
fn compute_gap_compression(draws: &[Draw]) -> f64 {
    if draws.len() < 10 {
        return 1.0;
    }
    let n = 50usize;
    let k = 5usize;
    let p_appear = k as f64 / n as f64;
    let expected_gap = 1.0 / p_appear; // = 10.0

    // Gap courant pour chaque numéro (distance depuis dernière apparition)
    let mut last_seen = vec![draws.len(); n];
    for (t, d) in draws.iter().enumerate() {
        for &b in &d.balls {
            let idx = (b - 1) as usize;
            if last_seen[idx] == draws.len() {
                last_seen[idx] = t;
            }
        }
    }

    let mut sum_ratio = 0.0f64;
    let mut count = 0u32;
    for &ls in &last_seen {
        if ls < draws.len() {
            let current_gap = ls as f64;
            sum_ratio += current_gap / expected_gap;
            count += 1;
        }
    }

    if count > 0 {
        sum_ratio / count as f64
    } else {
        1.0
    }
}

/// Ridge regression ND : y = β0 + β · x
/// Résolution analytique : β = (X^T X + λI)^{-1} X^T y
fn ridge_regression_nd(features: &[[f64; N_FEATURES]], targets: &[f64], lambda: f64) -> (f64, [f64; N_FEATURES]) {
    let n = features.len();
    if n == 0 {
        return (0.0, [0.0; N_FEATURES]);
    }

    // Centrer les targets
    let mean_y: f64 = targets.iter().sum::<f64>() / n as f64;

    // X^T X (DxD) + lambda * I
    let mut xtx = [[0.0f64; N_FEATURES]; N_FEATURES];
    let mut xty = [0.0f64; N_FEATURES];

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

    // Résoudre par Gauss-Jordan NxN
    let beta = solve_nxn(xtx, xty);

    // Intercept = mean_y - beta . mean_x
    let mut intercept = mean_y;
    for j in 0..N_FEATURES {
        let mean_x: f64 = features.iter().map(|f| f[j]).sum::<f64>() / n as f64;
        intercept -= beta[j] * mean_x;
    }

    (intercept, beta)
}

/// Résolution d'un système NxN par élimination de Gauss avec pivot partiel.
#[allow(clippy::needless_range_loop)]
fn solve_nxn(mut a: [[f64; N_FEATURES]; N_FEATURES], mut b: [f64; N_FEATURES]) -> [f64; N_FEATURES] {
    // Forward elimination
    for col in 0..N_FEATURES {
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

        for row in (col + 1)..N_FEATURES {
            let factor = a[row][col] / a[col][col];
            for k in col..N_FEATURES {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = [0.0f64; N_FEATURES];
    for col in (0..N_FEATURES).rev() {
        if a[col][col].abs() < 1e-15 {
            continue;
        }
        x[col] = b[col];
        for k in (col + 1)..N_FEATURES {
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
        assert!(f.day_of_week >= 0.0 && f.day_of_week <= 1.0);
        assert!(f.gap_compression >= 0.0 && f.gap_compression <= 2.0);
        assert!(f.hurst_exponent >= 0.0 && f.hurst_exponent <= 1.0);
    }

    #[test]
    fn test_regime_features_7d() {
        let draws = make_test_draws(20);
        let f = RegimeFeatures::from_draws(&draws);
        let v = f.as_vec();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_ridge_regression_nd_constant() {
        // Si tous les targets sont identiques, les coefficients doivent être ~0
        let features: Vec<[f64; N_FEATURES]> = (0..20)
            .map(|i| [i as f64 / 20.0, 0.5, 0.3, 0.7, 0.0, 1.0, 0.5])
            .collect();
        let targets = vec![1.0; 20];
        let (intercept, coeff) = ridge_regression_nd(&features, &targets, 1.0);
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
            coefficients: vec![
                (0.0, [0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.05]),
                (0.0, [-0.1, 0.0, 0.1, 0.2, 0.0, -0.1, 0.0]),
            ],
            model_names: vec!["A".into(), "B".into()],
        };
        let f = RegimeFeatures {
            sum_norm: 0.5, spread_norm: 0.5, mod4_cosine: 0.5,
            recent_entropy: 0.5, day_of_week: 0.0, gap_compression: 1.0,
            hurst_exponent: 0.5,
        };
        let adj = mp.weight_adjustments(&f);
        assert_eq!(adj.len(), 2);
        for (_, v) in &adj {
            assert!(*v > 0.0, "Weight adjustment must be positive");
        }
    }
}
