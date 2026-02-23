use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use lemillion_db::models::Draw;

use crate::config::{Encoding, EsnConfig, EsnResult, GridSearchResults};
use crate::training::train_and_evaluate;

/// Generate the full Cartesian product grid of hyperparameters.
/// 4 * 6 * 3 * 5 * 4 * 4 * 2 * 3 = 17280 configurations.
pub fn generate_grid() -> Vec<EsnConfig> {
    let reservoir_sizes = [200, 500, 800, 1000];
    let spectral_radii = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1];
    let sparsities = [0.95, 0.9, 0.8]; // fraction of zeros
    let leaking_rates = [0.1, 0.3, 0.5, 0.8, 1.0];
    let ridge_lambdas = [1e-6, 1e-4, 1e-2, 1e-1];
    let input_scalings = [0.01, 0.1, 0.5, 1.0];
    let encodings = [Encoding::OneHot, Encoding::Normalized];
    let washouts = [20, 50, 100];

    let mut configs = Vec::with_capacity(17280);

    for &rs in &reservoir_sizes {
        for &sr in &spectral_radii {
            for &sp in &sparsities {
                for &lr in &leaking_rates {
                    for &rl in &ridge_lambdas {
                        for &is in &input_scalings {
                            for &enc in &encodings {
                                for &wo in &washouts {
                                    configs.push(EsnConfig {
                                        reservoir_size: rs,
                                        spectral_radius: sr,
                                        sparsity: sp,
                                        leaking_rate: lr,
                                        ridge_lambda: rl,
                                        input_scaling: is,
                                        encoding: enc,
                                        washout: wo,
                                        noise_amplitude: 1e-4,
                                        seed: 42,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    configs
}

/// Score for ranking: weighted combination of ball and star hit rates.
fn score(result: &EsnResult) -> f64 {
    result.val_ball_hit_rate * 5.0 + result.val_star_hit_rate * 2.0
}

/// Run grid search in parallel using Rayon.
pub fn run_grid_search(
    draws: &[Draw],
    configs: &[EsnConfig],
    output_path: &str,
) -> Result<GridSearchResults> {
    let pb = ProgressBar::new(configs.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    let start = std::time::Instant::now();

    let results: Vec<EsnResult> = configs
        .par_iter()
        .filter_map(|config| {
            let result = train_and_evaluate(draws, config);
            pb.inc(1);
            match result {
                Ok((_, esn_result)) => Some(esn_result),
                Err(e) => {
                    log::warn!("Config failed: {:?}: {}", config, e);
                    None
                }
            }
        })
        .collect();

    let elapsed = start.elapsed();
    pb.finish_and_clear();

    let total_secs = elapsed.as_secs();
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    let ok = results.len();
    let failed = configs.len() - ok;
    println!(
        "Grid search terminee : {ok}/{} configs en {mins}m{secs:02}s ({failed} echecs)",
        configs.len(),
    );

    if results.is_empty() {
        anyhow::bail!("All configurations failed");
    }

    // Sort by score descending
    let mut sorted = results;
    sorted.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best_config = sorted[0].config.clone();

    let gs_results = GridSearchResults {
        results: sorted,
        best_config,
    };

    // Compare with existing results before saving
    let new_score = score(&gs_results.results[0]);
    let should_save = match std::fs::read_to_string(output_path) {
        Ok(existing_json) => match serde_json::from_str::<GridSearchResults>(&existing_json) {
            Ok(old) if !old.results.is_empty() => {
                let old_score = score(&old.results[0]);
                if new_score >= old_score {
                    println!(
                        "Resultats sauvegardes (score: {new_score:.4}, ancien: {old_score:.4})"
                    );
                    true
                } else {
                    println!(
                        "Resultats non sauvegardes : score {new_score:.4} < ancien {old_score:.4}"
                    );
                    false
                }
            }
            _ => true,
        },
        Err(_) => true,
    };

    if should_save {
        let json = serde_json::to_string_pretty(&gs_results)?;
        std::fs::write(output_path, json)?;
        log::info!("Results saved to {output_path}");
    }

    Ok(gs_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_size() {
        let grid = generate_grid();
        // 4 * 6 * 3 * 5 * 4 * 4 * 2 * 3 = 34560
        assert_eq!(grid.len(), 34560, "grid size should be 34560, got {}", grid.len());
    }

    #[test]
    fn test_grid_all_seed_42() {
        let grid = generate_grid();
        assert!(grid.iter().all(|c| c.seed == 42));
    }

    #[test]
    fn test_mini_grid_search() {
        use lemillion_db::models::Draw;

        let draws: Vec<Draw> = (0..100)
            .map(|i| {
                let base = (i % 10) as u8;
                Draw {
                    draw_id: format!("{:03}", i),
                    day: "MARDI".to_string(),
                    date: format!("2024-01-{:02}", (i % 28) + 1),
                    balls: [
                        (base * 5 + 1).min(50).max(1),
                        (base * 5 + 2).min(50).max(1),
                        (base * 5 + 3).min(50).max(1),
                        (base * 5 + 4).min(50).max(1),
                        (base * 5 + 5).min(50).max(1),
                    ],
                    stars: [(base % 12 + 1), ((base + 1) % 12 + 1)],
                    winner_count: 0,
                    winner_prize: 0.0,
                    my_million: String::new(),
                }
            })
            .collect();

        let configs = vec![
            EsnConfig {
                reservoir_size: 20,
                spectral_radius: 0.9,
                sparsity: 0.8,
                leaking_rate: 0.3,
                ridge_lambda: 1e-2,
                input_scaling: 0.1,
                encoding: Encoding::OneHot,
                washout: 5,
                noise_amplitude: 0.0,
                seed: 42,
            },
            EsnConfig {
                reservoir_size: 30,
                spectral_radius: 0.95,
                sparsity: 0.9,
                leaking_rate: 0.5,
                ridge_lambda: 1e-4,
                input_scaling: 0.1,
                encoding: Encoding::Normalized,
                washout: 5,
                noise_amplitude: 0.0,
                seed: 42,
            },
        ];

        let tmp = std::env::temp_dir().join("esn_test_gs.json");
        let results = run_grid_search(&draws, &configs, tmp.to_str().unwrap()).unwrap();
        assert_eq!(results.results.len(), 2);
        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }
}
