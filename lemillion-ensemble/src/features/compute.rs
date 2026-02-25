use lemillion_db::models::{Draw, Pool};
use super::FeatureRow;

/// Extrait les features pour chaque numéro du pool, en utilisant les tirages
/// strictement APRÈS target_draw_idx comme historique d'entraînement.
/// draws[0] = le plus récent. Le label est 1.0 si le numéro est dans le tirage target, 0.0 sinon.
pub fn extract_features_for_draw(draws: &[Draw], pool: Pool, target_draw_idx: usize) -> Vec<FeatureRow> {
    let size = pool.size();
    let target = &draws[target_draw_idx];
    let target_numbers: Vec<u8> = pool.numbers_from(target).to_vec();

    // Historique = tirages strictement après target (plus anciens)
    let history = if target_draw_idx + 1 < draws.len() {
        &draws[target_draw_idx + 1..]
    } else {
        &[]
    };

    (1..=size as u8)
        .map(|number| {
            let features = compute_features_for_number(number, history, pool, target);
            let label = if target_numbers.contains(&number) { 1.0 } else { 0.0 };
            FeatureRow { number, features, label }
        })
        .collect()
}

fn compute_features_for_number(number: u8, history: &[Draw], pool: Pool, target_draw: &Draw) -> Vec<f64> {
    let size = pool.size();

    // freq_5, freq_10, freq_20 : fréquence sur les N derniers tirages
    let freq_5 = frequency_in_window(number, history, pool, 5);
    let freq_10 = frequency_in_window(number, history, pool, 10);
    let freq_20 = frequency_in_window(number, history, pool, 20);

    // retard : nombre de tirages depuis la dernière apparition
    let gap = current_gap(number, history, pool);

    // retard_norm : gap / mean_gap (normalisé)
    let (mean_gap, std_gap) = gap_statistics(number, history, pool);
    let retard_norm = if mean_gap > 0.0 { gap as f64 / mean_gap } else { 1.0 };

    // trend : freq_5 - freq_20 (momentum court vs long terme)
    let trend = freq_5 - freq_20;

    // is_odd : 1.0 si impair, 0.0 si pair
    let is_odd = if number % 2 == 1 { 1.0 } else { 0.0 };

    // decade : dans quelle décade (0-4 pour boules, 0-2 pour étoiles)
    let decade = ((number - 1) as f64 / 10.0).floor();

    // decade_density : proportion de numéros de la même décade présents dans les 10 derniers tirages
    let decade_density = compute_decade_density(number, history, pool, 10);

    // day_of_week : MARDI=0.0, VENDREDI=1.0
    let day_of_week = if target_draw.day.contains("VENDREDI") || target_draw.day.contains("vendredi") {
        1.0
    } else {
        0.0
    };

    // recent_sum_norm : somme normalisée des numéros tirés récemment (sur 5 tirages)
    let recent_sum_norm = compute_recent_sum_norm(history, pool, 5, size);

    // recent_even_count : nombre de numéros pairs parmi les pick_count derniers tirés (sur 5 tirages)
    let recent_even_count = compute_recent_even_count(history, pool, 5);

    // freq_3 : fréquence sur les 3 derniers tirages (très court terme)
    let freq_3 = frequency_in_window(number, history, pool, 3);

    // pair_freq : fréquence d'apparition simultanée avec les numéros du tirage précédent
    let pair_freq = compute_pair_freq(number, history, pool);

    // gap_acceleration : (gap actuel - gap précédent) / mean_gap
    let gap_acceleration = compute_gap_acceleration(number, history, pool, mean_gap);

    // low_half : 1.0 si numéro dans la moitié basse du pool
    let low_half = match pool {
        Pool::Balls => if number <= 25 { 1.0 } else { 0.0 },
        Pool::Stars => if number <= 6 { 1.0 } else { 0.0 },
    };

    vec![
        freq_5,             // 0
        freq_10,            // 1
        freq_20,            // 2
        gap as f64,         // 3
        retard_norm,        // 4
        trend,              // 5
        mean_gap,           // 6
        std_gap,            // 7
        is_odd,             // 8
        decade,             // 9
        decade_density,     // 10
        day_of_week,        // 11
        recent_sum_norm,    // 12
        recent_even_count,  // 13
        freq_3,             // 14
        pair_freq,          // 15
        gap_acceleration,   // 16
        low_half,           // 17
    ]
}

fn frequency_in_window(number: u8, history: &[Draw], pool: Pool, window: usize) -> f64 {
    let w = window.min(history.len());
    if w == 0 {
        return 0.0;
    }
    let count = history[..w]
        .iter()
        .filter(|d| pool.numbers_from(d).contains(&number))
        .count();
    count as f64 / w as f64
}

fn current_gap(number: u8, history: &[Draw], pool: Pool) -> usize {
    for (i, draw) in history.iter().enumerate() {
        if pool.numbers_from(draw).contains(&number) {
            return i;
        }
    }
    history.len()
}

fn gap_statistics(number: u8, history: &[Draw], pool: Pool) -> (f64, f64) {
    let mut gaps = Vec::new();
    let mut last_seen: Option<usize> = None;

    for (i, draw) in history.iter().enumerate() {
        if pool.numbers_from(draw).contains(&number) {
            if let Some(prev) = last_seen {
                gaps.push((i - prev) as f64);
            }
            last_seen = Some(i);
        }
    }

    if gaps.is_empty() {
        return (history.len() as f64 / 2.0, 0.0);
    }

    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let variance = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gaps.len() as f64;
    (mean, variance.sqrt())
}

fn compute_decade_density(number: u8, history: &[Draw], pool: Pool, window: usize) -> f64 {
    let decade_start = ((number - 1) / 10) * 10 + 1;
    let decade_end = (decade_start + 9).min(pool.size() as u8);
    let decade_size = (decade_end - decade_start + 1) as f64;

    let w = window.min(history.len());
    if w == 0 || decade_size == 0.0 {
        return 0.0;
    }

    let mut count = 0.0;
    for draw in &history[..w] {
        for &n in pool.numbers_from(draw) {
            if n >= decade_start && n <= decade_end {
                count += 1.0;
            }
        }
    }

    count / (w as f64 * decade_size)
}

/// Fréquence d'apparition simultanée avec les numéros du tirage précédent (history[0]).
fn compute_pair_freq(number: u8, history: &[Draw], pool: Pool) -> f64 {
    if history.is_empty() {
        return 0.0;
    }
    let prev_numbers: Vec<u8> = pool.numbers_from(&history[0]).to_vec();
    if history.len() < 2 {
        return 0.0;
    }

    // Pour chaque numéro du tirage précédent, compter combien de fois `number` et ce numéro
    // apparaissent ensemble dans l'historique
    let mut pair_count = 0u32;
    let mut total = 0u32;
    for draw in &history[1..] {
        let nums: Vec<u8> = pool.numbers_from(draw).to_vec();
        let has_number = nums.contains(&number);
        for &prev in &prev_numbers {
            if nums.contains(&prev) {
                total += 1;
                if has_number {
                    pair_count += 1;
                }
            }
        }
    }

    if total == 0 { 0.0 } else { pair_count as f64 / total as f64 }
}

/// Accélération du gap : (gap actuel - gap précédent) / mean_gap.
fn compute_gap_acceleration(number: u8, history: &[Draw], pool: Pool, mean_gap: f64) -> f64 {
    // Trouver les deux dernières apparitions pour calculer le gap précédent
    let mut appearances = Vec::new();
    for (i, draw) in history.iter().enumerate() {
        if pool.numbers_from(draw).contains(&number) {
            appearances.push(i);
            if appearances.len() >= 3 {
                break;
            }
        }
    }

    let current_gap = if appearances.is_empty() {
        history.len() as f64
    } else {
        appearances[0] as f64
    };

    let previous_gap = if appearances.len() >= 2 {
        (appearances[1] - appearances[0]) as f64
    } else if appearances.len() == 1 {
        (history.len() - appearances[0]) as f64
    } else {
        current_gap
    };

    if mean_gap > 0.0 {
        (current_gap - previous_gap) / mean_gap
    } else {
        0.0
    }
}

fn compute_recent_sum_norm(history: &[Draw], pool: Pool, window: usize, pool_size: usize) -> f64 {
    let w = window.min(history.len());
    if w == 0 {
        return 0.5;
    }

    let total: f64 = history[..w]
        .iter()
        .flat_map(|d| pool.numbers_from(d).iter().map(|&n| n as f64))
        .sum();

    let pick_count = pool.pick_count() as f64;
    let max_possible = w as f64 * pick_count * pool_size as f64;
    if max_possible > 0.0 { total / max_possible } else { 0.5 }
}

fn compute_recent_even_count(history: &[Draw], pool: Pool, window: usize) -> f64 {
    let w = window.min(history.len());
    if w == 0 {
        return 0.0;
    }

    let total_numbers: f64 = history[..w]
        .iter()
        .flat_map(|d| pool.numbers_from(d).iter())
        .filter(|&&n| n % 2 == 0)
        .count() as f64;

    let total_picks = w as f64 * pool.pick_count() as f64;
    if total_picks > 0.0 { total_numbers / total_picks } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_feature_count() {
        let draws = make_test_draws(30);
        let features = extract_features_for_draw(&draws, Pool::Balls, 0);
        assert_eq!(features.len(), 50);
        assert_eq!(features[0].features.len(), 18);
    }

    #[test]
    fn test_feature_labels_sum() {
        let draws = make_test_draws(30);
        let features = extract_features_for_draw(&draws, Pool::Balls, 0);
        let label_sum: f64 = features.iter().map(|f| f.label).sum();
        assert_eq!(label_sum, 5.0); // 5 boules par tirage
    }

    #[test]
    fn test_feature_labels_stars() {
        let draws = make_test_draws(30);
        let features = extract_features_for_draw(&draws, Pool::Stars, 0);
        let label_sum: f64 = features.iter().map(|f| f.label).sum();
        assert_eq!(label_sum, 2.0); // 2 étoiles par tirage
    }

    #[test]
    fn test_features_no_nan() {
        let draws = make_test_draws(30);
        let features = extract_features_for_draw(&draws, Pool::Balls, 5);
        for row in &features {
            for &f in &row.features {
                assert!(!f.is_nan(), "NaN found in features for number {}", row.number);
                assert!(!f.is_infinite(), "Inf found in features for number {}", row.number);
            }
        }
    }
}
