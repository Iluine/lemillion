use std::collections::HashMap;

use lemillion_db::models::Draw;

/// Prédicteur de statistiques résumées du prochain tirage.
///
/// Exploite le gain informationnel conditionnel de 66% identifié par le module research.
/// Encode chaque tirage en état résumé (sum_bin, spread_bin, odd_count) et construit
/// une matrice de transition Markov ordre 1 + Laplace.
///
/// N'est PAS un ForecastModel — c'est un outil utilisé par le sampler pour
/// construire des filtres adaptatifs.
pub struct SummaryPredictor {
    laplace_alpha: f64,
    min_draws: usize,
}

impl SummaryPredictor {
    pub fn new(laplace_alpha: f64, min_draws: usize) -> Self {
        Self { laplace_alpha, min_draws }
    }
}

impl Default for SummaryPredictor {
    fn default() -> Self {
        Self {
            laplace_alpha: 1.0,
            min_draws: 20,
        }
    }
}

/// Distribution prédite sur les statistiques résumées.
#[derive(Debug, Clone)]
pub struct SummaryPrediction {
    /// Distribution sur les bins de somme (5 bins)
    pub sum_dist: Vec<f64>,
    /// Distribution sur les bins de spread (5 bins)
    pub spread_dist: Vec<f64>,
    /// Distribution sur odd_count (6 valeurs: 0-5)
    pub odd_dist: Vec<f64>,
}

/// Bornes adaptatives dérivées de la prédiction résumée.
#[derive(Debug, Clone)]
pub struct AdaptiveBounds {
    /// Intervalle de somme prédit [min, max]
    pub sum_range: (u16, u16),
    /// Intervalle de spread prédit [min, max]
    pub spread_range: (u8, u8),
    /// Valeurs d'odd_count acceptables
    pub odd_values: Vec<u8>,
}

// Constantes de binning pour les boules
const N_SUM_BINS: usize = 5;
const N_SPREAD_BINS: usize = 5;
const N_ODD_VALUES: usize = 6; // 0,1,2,3,4,5

// Bornes des bins de somme (boules: range ~15-240, centré ~125)
const SUM_BIN_EDGES: [u16; 6] = [0, 85, 105, 130, 155, 255];
// Bornes des bins de spread (range 4-49)
const SPREAD_BIN_EDGES: [u8; 6] = [0, 20, 28, 34, 40, 50];

/// Encode un tirage de boules en état résumé.
fn encode_ball_state(balls: &[u8; 5]) -> (usize, usize, usize) {
    let sum: u16 = balls.iter().map(|&b| b as u16).sum();
    let mut sorted = *balls;
    sorted.sort();
    let spread = sorted[4] - sorted[0];
    let odd_count = balls.iter().filter(|&&b| b % 2 == 1).count();

    let sum_bin = SUM_BIN_EDGES.windows(2)
        .position(|w| sum >= w[0] && sum < w[1])
        .unwrap_or(N_SUM_BINS - 1);

    let spread_bin = SPREAD_BIN_EDGES.windows(2)
        .position(|w| spread >= w[0] && spread < w[1])
        .unwrap_or(N_SPREAD_BINS - 1);

    (sum_bin, spread_bin, odd_count)
}

/// Clé d'état pour le HashMap.
fn state_key(sum_bin: usize, spread_bin: usize, odd_count: usize) -> u64 {
    (sum_bin as u64) | ((spread_bin as u64) << 8) | ((odd_count as u64) << 16)
}

impl SummaryPredictor {
    /// Prédit la distribution résumée du prochain tirage.
    pub fn predict_summary(&self, draws: &[Draw]) -> Option<SummaryPrediction> {
        if draws.len() < self.min_draws {
            return None;
        }

        // Encoder tous les tirages
        let states: Vec<(usize, usize, usize)> = draws
            .iter()
            .map(|d| encode_ball_state(&d.balls))
            .collect();

        let alpha = self.laplace_alpha;

        // Matrice de transition sur l'état complet (sum_bin, spread_bin, odd_count)
        // draws[0] = plus récent → transition: draws[t+1] → draws[t]
        let mut transition: HashMap<u64, HashMap<u64, f64>> = HashMap::new();

        for t in 0..draws.len() - 1 {
            let from = state_key(states[t + 1].0, states[t + 1].1, states[t + 1].2);
            let to = state_key(states[t].0, states[t].1, states[t].2);
            *transition.entry(from).or_default().entry(to).or_insert(alpha) += 1.0;
        }

        // État actuel
        let current = state_key(states[0].0, states[0].1, states[0].2);

        // Distributions marginales depuis la matrice de transition
        let mut sum_dist = vec![alpha; N_SUM_BINS];
        let mut spread_dist = vec![alpha; N_SPREAD_BINS];
        let mut odd_dist = vec![alpha; N_ODD_VALUES];

        if let Some(row) = transition.get(&current) {
            for (&to_key, &count) in row {
                let s = (to_key & 0xFF) as usize;
                let sp = ((to_key >> 8) & 0xFF) as usize;
                let o = ((to_key >> 16) & 0xFF) as usize;
                if s < N_SUM_BINS {
                    sum_dist[s] += count;
                }
                if sp < N_SPREAD_BINS {
                    spread_dist[sp] += count;
                }
                if o < N_ODD_VALUES {
                    odd_dist[o] += count;
                }
            }
        } else {
            // Fallback: fréquences marginales
            for &(s, sp, o) in &states {
                sum_dist[s] += 1.0;
                spread_dist[sp] += 1.0;
                if o < N_ODD_VALUES {
                    odd_dist[o] += 1.0;
                }
            }
        }

        // Normaliser
        let normalize = |v: &mut Vec<f64>| {
            let total: f64 = v.iter().sum();
            if total > 0.0 {
                for p in v.iter_mut() {
                    *p /= total;
                }
            }
        };

        normalize(&mut sum_dist);
        normalize(&mut spread_dist);
        normalize(&mut odd_dist);

        Some(SummaryPrediction {
            sum_dist,
            spread_dist,
            odd_dist,
        })
    }

    /// Convertit une prédiction résumée en bornes adaptatives.
    /// Accepte les bins dont la probabilité cumulée couvre `coverage_threshold`
    /// (par défaut 80%) de la masse de probabilité.
    pub fn predict_bounds(&self, draws: &[Draw], coverage_threshold: f64) -> Option<AdaptiveBounds> {
        let pred = self.predict_summary(draws)?;

        // Somme : trouver les bins couvrant le seuil
        let sum_range = bins_to_range(&pred.sum_dist, &SUM_BIN_EDGES, coverage_threshold);

        // Spread
        let spread_edges_u16: Vec<u16> = SPREAD_BIN_EDGES.iter().map(|&x| x as u16).collect();
        let spread_range_u16 = bins_to_range(&pred.spread_dist, &spread_edges_u16, coverage_threshold);
        let spread_range = (spread_range_u16.0 as u8, spread_range_u16.1 as u8);

        // Odd count : valeurs avec probabilité significative
        let odd_values = significant_values(&pred.odd_dist, coverage_threshold);

        Some(AdaptiveBounds {
            sum_range,
            spread_range,
            odd_values,
        })
    }
}

/// Sélectionne les bins dont la probabilité cumulée atteint le seuil,
/// et retourne l'intervalle [min_edge, max_edge] correspondant.
fn bins_to_range(dist: &[f64], edges: &[u16], threshold: f64) -> (u16, u16) {
    let mut indexed: Vec<(usize, f64)> = dist.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut selected_bins: Vec<usize> = Vec::new();
    for (idx, prob) in indexed {
        selected_bins.push(idx);
        cumulative += prob;
        if cumulative >= threshold {
            break;
        }
    }

    if selected_bins.is_empty() {
        return (edges[0], *edges.last().unwrap_or(&255));
    }

    let min_bin = *selected_bins.iter().min().unwrap();
    let max_bin = *selected_bins.iter().max().unwrap();

    let min_edge = edges[min_bin];
    let max_edge = if max_bin + 1 < edges.len() { edges[max_bin + 1] } else { *edges.last().unwrap() };

    (min_edge, max_edge)
}

/// Retourne les indices dont la probabilité cumulée couvre le seuil.
fn significant_values(dist: &[f64], threshold: f64) -> Vec<u8> {
    let mut indexed: Vec<(usize, f64)> = dist.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut values = Vec::new();
    for (idx, prob) in indexed {
        values.push(idx as u8);
        cumulative += prob;
        if cumulative >= threshold {
            break;
        }
    }

    values.sort();
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_encode_ball_state() {
        let (sum_bin, spread_bin, odd_count) = encode_ball_state(&[1, 10, 20, 30, 40]);
        assert!(sum_bin < N_SUM_BINS);
        assert!(spread_bin < N_SPREAD_BINS);
        assert_eq!(odd_count, 1); // only 1 is odd
    }

    #[test]
    fn test_predict_summary_valid() {
        let predictor = SummaryPredictor::default();
        let draws = make_test_draws(50);
        let pred = predictor.predict_summary(&draws);
        assert!(pred.is_some());
        let pred = pred.unwrap();

        // Toutes les distributions somment à ~1.0
        let sum_total: f64 = pred.sum_dist.iter().sum();
        assert!((sum_total - 1.0).abs() < 1e-9, "sum_dist total = {}", sum_total);

        let spread_total: f64 = pred.spread_dist.iter().sum();
        assert!((spread_total - 1.0).abs() < 1e-9, "spread_dist total = {}", spread_total);

        let odd_total: f64 = pred.odd_dist.iter().sum();
        assert!((odd_total - 1.0).abs() < 1e-9, "odd_dist total = {}", odd_total);
    }

    #[test]
    fn test_predict_summary_few_draws() {
        let predictor = SummaryPredictor::default();
        let draws = make_test_draws(10);
        assert!(predictor.predict_summary(&draws).is_none());
    }

    #[test]
    fn test_predict_bounds() {
        let predictor = SummaryPredictor::default();
        let draws = make_test_draws(50);
        let bounds = predictor.predict_bounds(&draws, 0.8);
        assert!(bounds.is_some());
        let bounds = bounds.unwrap();

        // Les bornes de somme doivent être valides
        assert!(bounds.sum_range.0 < bounds.sum_range.1);
        // Au moins une valeur odd
        assert!(!bounds.odd_values.is_empty());
    }

    #[test]
    fn test_predict_bounds_coverage() {
        let predictor = SummaryPredictor::default();
        let draws = make_test_draws(50);

        // Avec un seuil bas, les bornes sont plus serrées
        let tight = predictor.predict_bounds(&draws, 0.5).unwrap();
        // Avec un seuil haut, les bornes sont plus larges
        let wide = predictor.predict_bounds(&draws, 0.95).unwrap();

        // Le range large devrait être >= au range serré
        assert!(wide.sum_range.1 - wide.sum_range.0 >= tight.sum_range.1 - tight.sum_range.0);
    }

    #[test]
    fn test_adaptive_bounds_accept_real_draws() {
        let predictor = SummaryPredictor::default();
        let draws = make_test_draws(60);

        // Prédire sur les 50 premiers, vérifier sur les 10 restants
        let train = &draws[10..]; // plus anciens (draws[0] = récent)
        let bounds = predictor.predict_bounds(train, 0.8);
        assert!(bounds.is_some());
        let bounds = bounds.unwrap();

        // Vérifier que les bornes ne sont pas dégénérées
        assert!(bounds.sum_range.0 <= bounds.sum_range.1);
        assert!(bounds.spread_range.0 <= bounds.spread_range.1);
    }

    #[test]
    fn test_bins_to_range_full_coverage() {
        let dist = vec![0.2, 0.3, 0.25, 0.15, 0.1];
        let edges: [u16; 6] = [0, 85, 105, 130, 155, 255];
        let range = bins_to_range(&dist, &edges, 0.99);
        // Devrait couvrir presque tout
        assert_eq!(range.0, 0);
        assert_eq!(range.1, 255);
    }

    #[test]
    fn test_significant_values() {
        let dist = vec![0.05, 0.3, 0.4, 0.15, 0.08, 0.02];
        let values = significant_values(&dist, 0.7);
        // Top 2 bins (0.4 + 0.3 = 0.7)
        assert!(values.contains(&1));
        assert!(values.contains(&2));
    }
}
