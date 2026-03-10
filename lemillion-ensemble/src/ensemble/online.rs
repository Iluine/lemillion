use lemillion_db::models::{Draw, Pool};

/// Mélange online/offline pour adaptation rapide (v15).
///
/// L'expert "offline" = distribution de l'ensemble calibré (stable, mois de données).
/// L'expert "online" = EWMA sur les N derniers tirages (réactif, jours de données).
///
/// Le meta-weight est basé sur la divergence KL entre les deux :
/// - Forte divergence → le marché a changé → plus de poids online
/// - Faible divergence → stable → garder offline
pub fn online_offline_blend(
    offline_dist: &[f64],
    draws: &[Draw],
    pool: Pool,
    window: usize,
) -> Vec<f64> {
    let size = pool.size();

    if draws.is_empty() || size == 0 {
        return offline_dist.to_vec();
    }

    let ewma_alpha = 0.15;
    let actual_window = window.min(draws.len());

    // EWMA sur les tirages récents (itération chronologique = inversé)
    let recent = &draws[..actual_window];
    let mut online = vec![1.0 / size as f64; size];

    for draw in recent.iter().rev() {
        let nums = pool.numbers_from(draw);
        for idx in 0..size {
            let num = (idx + 1) as u8;
            let present = if nums.contains(&num) { 1.0 } else { 0.0 };
            online[idx] = ewma_alpha * present + (1.0 - ewma_alpha) * online[idx];
        }
    }

    // Normaliser
    let sum: f64 = online.iter().sum();
    if sum > 0.0 {
        for p in &mut online {
            *p /= sum;
        }
    }

    // KL divergence D(online || offline) pour mesurer la divergence
    let kl: f64 = online.iter().zip(offline_dist.iter())
        .map(|(&o, &f)| {
            if o > 1e-15 && f > 1e-15 {
                o * (o / f).ln()
            } else {
                0.0
            }
        })
        .sum();

    // meta_weight ∈ [0.05, 0.30] — plus KL est grand, plus on pondère online
    let meta_weight = 0.05 + 0.25 * (1.0 - (-kl * 10.0).exp());

    // Blend
    let mut blended = vec![0.0f64; size];
    for i in 0..size {
        blended[i] = meta_weight * online[i] + (1.0 - meta_weight) * offline_dist[i];
    }

    // Normaliser
    let sum: f64 = blended.iter().sum();
    if sum > 0.0 {
        for p in &mut blended {
            *p /= sum;
        }
    }

    blended
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::make_test_draws;

    #[test]
    fn test_online_offline_blend_uniform() {
        let draws = make_test_draws(50);
        let uniform = vec![1.0 / 50.0; 50];
        let blended = online_offline_blend(&uniform, &draws, Pool::Balls, 10);
        assert_eq!(blended.len(), 50);
        let sum: f64 = blended.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
    }

    #[test]
    fn test_online_offline_blend_stars() {
        let draws = make_test_draws(50);
        let uniform = vec![1.0 / 12.0; 12];
        let blended = online_offline_blend(&uniform, &draws, Pool::Stars, 10);
        assert_eq!(blended.len(), 12);
        let sum: f64 = blended.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", sum);
    }

    #[test]
    fn test_online_offline_blend_preserves_shape() {
        let draws = make_test_draws(50);
        // Peaked distribution
        let mut offline = vec![0.005; 50];
        offline[0] = 0.2;
        offline[1] = 0.15;
        let total: f64 = offline.iter().sum();
        let offline: Vec<f64> = offline.iter().map(|p| p / total).collect();

        let blended = online_offline_blend(&offline, &draws, Pool::Balls, 10);
        // Top values should still be top (online slightly perturbs but doesn't dominate)
        assert!(blended[0] > blended[49] || blended[1] > blended[49]);
    }

    #[test]
    fn test_online_offline_blend_empty_draws() {
        let offline = vec![1.0 / 50.0; 50];
        let blended = online_offline_blend(&offline, &[], Pool::Balls, 10);
        assert_eq!(blended, offline);
    }
}
