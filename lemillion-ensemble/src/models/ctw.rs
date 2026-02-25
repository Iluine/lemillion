use std::collections::HashMap;

use lemillion_db::models::{Draw, Pool};

use super::ForecastModel;

/// Context Tree Weighting (CTW) — prédicteur universel bayésien.
///
/// Maintient un arbre de contextes de profondeur D. Chaque noeud utilise
/// l'estimateur Krichevsky-Trofimov (KT). Le mélange bayésien de tous les
/// modèles Markov d'ordre 0 à D est calculé exactement en O(n × D).
///
/// Théorème : CTW converge vers le taux d'entropie de toute source ergodique
/// stationnaire à mémoire finie en O(log(n)/n).
pub struct CtwModel {
    depth: usize,
    smoothing: f64,
}

impl CtwModel {
    pub fn new(depth: usize, smoothing: f64) -> Self {
        Self { depth, smoothing }
    }
}

impl Default for CtwModel {
    fn default() -> Self {
        Self {
            depth: 6,
            smoothing: 0.5,
        }
    }
}

/// Noeud de l'arbre de contextes.
/// Stocke les compteurs KT (a = nb de 0, b = nb de 1) et le log-poids bayésien.
struct CtNode {
    a: f64,
    b: f64,
    /// log P_e (estimated probability at this node via KT)
    log_pe: f64,
    /// log P_w (weighted mixture probability)
    log_pw: f64,
    children: [Option<Box<CtNode>>; 2],
}

impl CtNode {
    fn new() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            log_pe: 0.0,
            log_pw: 0.0,
            children: [None, None],
        }
    }

    /// KT predictive probability P(next=1 | a zeros, b ones)
    fn kt_prob_one(&self) -> f64 {
        (self.b + 0.5) / (self.a + self.b + 1.0)
    }
}

/// Traite la séquence entière et retourne la probabilité prédictive pour le prochain symbole = 1.
///
/// Optimisation : au lieu de cloner l'arbre entier pour prédire, on calcule
/// la probabilité prédictive analytiquement en traversant seulement le chemin
/// du contexte (O(D) au lieu de O(taille_arbre)).
fn process_and_predict(depth: usize, binary_series: &[u8]) -> f64 {
    let mut root = CtNode::new();

    // Traiter chaque symbole de la série
    for t in 0..binary_series.len() {
        let ctx_start = t.saturating_sub(depth);
        let context: Vec<u8> = binary_series[ctx_start..t].iter().copied().rev().collect();

        let symbol = binary_series[t];
        update_recursive(&mut root, &context, symbol, 0, depth);
    }

    // Calculer P(next=1) analytiquement en traversant le chemin du contexte.
    //
    // À chaque noeud s, le poids CTW entre le modèle KT local et le sous-arbre est :
    //   weight_e(s) = 0.5 * exp(log_pe(s) - log_pw(s))
    //   weight_tree(s) = 1 - weight_e(s)
    //
    // La prédiction récursive :
    //   P_w(next=1 | leaf) = KT_prob_one(leaf)
    //   P_w(next=1 | s) = weight_e(s) * KT_prob_one(s) + weight_tree(s) * P_w(next=1 | child)
    let ctx_start = binary_series.len().saturating_sub(depth);
    let context: Vec<u8> = binary_series[ctx_start..]
        .iter()
        .copied()
        .rev()
        .collect();

    predict_from_tree(&root, &context, 0, depth)
}

/// Calcule la probabilité prédictive CTW pour next=1 en traversant le chemin du contexte.
fn predict_from_tree(node: &CtNode, context: &[u8], current_depth: usize, max_depth: usize) -> f64 {
    let kt_p1 = node.kt_prob_one();

    if current_depth >= max_depth || context.is_empty() {
        return kt_p1;
    }

    let child_idx = context[0] as usize;
    let child_p1 = match &node.children[child_idx] {
        Some(child) => predict_from_tree(child, &context[1..], current_depth + 1, max_depth),
        None => 0.5, // pas de données → prior uniforme
    };

    // Poids du modèle KT local vs sous-arbre
    let log_ratio = node.log_pe - node.log_pw;
    let weight_e = if log_ratio.is_finite() {
        (0.5 * log_ratio.exp()).min(1.0).max(0.0)
    } else {
        0.5
    };
    let weight_tree = 1.0 - weight_e;

    weight_e * kt_p1 + weight_tree * child_p1
}

/// Mise à jour récursive d'un noeud de l'arbre.
fn update_recursive(
    node: &mut CtNode,
    context: &[u8],
    symbol: u8,
    current_depth: usize,
    max_depth: usize,
) {
    // Mettre à jour les compteurs KT
    if symbol == 1 {
        node.log_pe += ((node.b + 0.5) / (node.a + node.b + 1.0)).ln();
        node.b += 1.0;
    } else {
        node.log_pe += ((node.a + 0.5) / (node.a + node.b + 1.0)).ln();
        node.a += 1.0;
    }

    if current_depth >= max_depth || context.is_empty() {
        node.log_pw = node.log_pe;
    } else {
        let child_idx = context[0] as usize;

        if node.children[child_idx].is_none() {
            node.children[child_idx] = Some(Box::new(CtNode::new()));
        }

        let child = node.children[child_idx].as_mut().unwrap();
        update_recursive(child, &context[1..], symbol, current_depth + 1, max_depth);

        // P_w = 0.5 * P_e + 0.5 * prod(P_w(children))
        let log_children: f64 = node
            .children
            .iter()
            .filter_map(|c| c.as_ref().map(|c| c.log_pw))
            .sum();

        let log_pe = node.log_pe;
        let max_log = log_pe.max(log_children);

        if max_log.is_finite() {
            node.log_pw = max_log
                + (0.5 * (log_pe - max_log).exp() + 0.5 * (log_children - max_log).exp()).ln();
        } else {
            node.log_pw = node.log_pe;
        }
    }
}

impl ForecastModel for CtwModel {
    fn name(&self) -> &str {
        "CTW"
    }

    fn predict(&self, draws: &[Draw], pool: Pool) -> Vec<f64> {
        let size = pool.size();
        let uniform = vec![1.0 / size as f64; size];

        if draws.len() < 10 {
            return uniform;
        }

        let mut raw_probs = Vec::with_capacity(size);

        for num in 1..=size as u8 {
            // Série binaire en ordre chronologique (draws[0] = plus récent → inverser)
            let binary: Vec<u8> = draws
                .iter()
                .rev()
                .map(|d| {
                    if pool.numbers_from(d).contains(&num) {
                        1
                    } else {
                        0
                    }
                })
                .collect();

            let pred = process_and_predict(self.depth, &binary);
            raw_probs.push(pred);
        }

        // Lisser avec la distribution uniforme
        let uniform_val = 1.0 / size as f64;
        let mut dist: Vec<f64> = raw_probs
            .iter()
            .map(|&p| self.smoothing * p + (1.0 - self.smoothing) * uniform_val)
            .collect();

        // Normaliser
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for p in &mut dist {
                *p /= sum;
            }
        } else {
            return uniform;
        }

        dist
    }

    fn params(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("depth".into(), self.depth as f64),
            ("smoothing".into(), self.smoothing),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{make_test_draws, validate_distribution};

    #[test]
    fn test_ctw_balls_sums_to_one() {
        let model = CtwModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        assert!(
            validate_distribution(&dist, Pool::Balls),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_ctw_stars_sums_to_one() {
        let model = CtwModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Stars);
        assert!(
            validate_distribution(&dist, Pool::Stars),
            "Sum = {}, len = {}",
            dist.iter().sum::<f64>(),
            dist.len()
        );
    }

    #[test]
    fn test_ctw_no_negative() {
        let model = CtwModel::default();
        let draws = make_test_draws(50);
        let dist = model.predict(&draws, Pool::Balls);
        for &p in &dist {
            assert!(p >= 0.0, "Negative probability: {}", p);
        }
    }

    #[test]
    fn test_ctw_empty_draws() {
        let model = CtwModel::default();
        let draws: Vec<Draw> = vec![];
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ctw_few_draws() {
        let model = CtwModel::default();
        let draws = make_test_draws(5);
        let dist = model.predict(&draws, Pool::Balls);
        let expected = 1.0 / 50.0;
        for &p in &dist {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_kt_estimator_basic() {
        let node = CtNode::new();
        // Avec a=0, b=0 : KT P(1) = (b + 0.5) / (a + b + 1.0) = 0.5
        assert!((node.kt_prob_one() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ctw_deterministic() {
        let model = CtwModel::default();
        let draws = make_test_draws(50);
        let dist1 = model.predict(&draws, Pool::Balls);
        let dist2 = model.predict(&draws, Pool::Balls);
        for (a, b) in dist1.iter().zip(dist2.iter()) {
            assert!((a - b).abs() < 1e-15, "CTW should be deterministic");
        }
    }

    #[test]
    fn test_context_tree_predict_biased() {
        // Séquence très biaisée vers 1 → la prédiction devrait être > 0.5
        let biased: Vec<u8> = (0..100).map(|i| if i % 3 == 0 { 0 } else { 1 }).collect();
        let pred = process_and_predict(4, &biased);
        assert!(
            pred > 0.5,
            "Biased sequence should predict > 0.5, got {pred}"
        );
    }
}
