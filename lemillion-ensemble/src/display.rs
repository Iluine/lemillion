use comfy_table::{Table, ContentArrangement, presets::UTF8_FULL, Cell, Color};
use textplots::Plot;

use lemillion_db::models::{Pool, Suggestion};
use crate::analysis::{AnalysisResult, Verdict};
use crate::ensemble::EnsemblePrediction;
use crate::ensemble::calibration::{EnsembleWeights, ModelCalibration};
use crate::ensemble::consensus::{ConsensusCategory, ConsensusEntry};

pub fn display_calibration_results(calibrations: &[ModelCalibration], windows: &[usize]) {
    println!("\n== Résultats de calibration ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);

    // Check if any model uses sparse
    let has_sparse = calibrations.iter().any(|c| c.results.iter().any(|r| r.sparse));

    let mut header = vec!["Modèle".to_string()];
    for w in windows {
        header.push(format!("w={}", w));
        if has_sparse {
            header.push(format!("w={}(S)", w));
        }
    }
    header.push("Best".to_string());
    table.set_header(&header);

    for cal in calibrations {
        let mut row: Vec<String> = vec![cal.model_name.clone()];
        for w in windows {
            // Consecutive result
            let ll = cal.results.iter()
                .find(|r| r.window == *w && !r.sparse)
                .map(|r| format!("{:.3}", r.log_likelihood))
                .unwrap_or_else(|| "—".to_string());
            row.push(ll);

            if has_sparse {
                // Sparse result
                let ll_sparse = cal.results.iter()
                    .find(|r| r.window == *w && r.sparse)
                    .map(|r| format!("{:.3}", r.log_likelihood))
                    .unwrap_or_else(|| "—".to_string());
                row.push(ll_sparse);
            }
        }
        let sparse_tag = if cal.best_sparse { "(S)" } else { "" };
        row.push(format!("w={}{} ({:.3})", cal.best_window, sparse_tag, cal.best_ll));
        table.add_row(row);
    }

    println!("{table}");
}

pub fn display_calibration_chart(calibrations: &[ModelCalibration], windows: &[usize]) {
    println!("\n== Skill par fenêtre (log-likelihood) ==\n");

    // Graphique ASCII simple avec textplots
    let x_min = *windows.iter().min().unwrap_or(&20) as f32;
    let x_max = *windows.iter().max().unwrap_or(&100) as f32;

    let mut all_lls: Vec<f64> = calibrations.iter()
        .flat_map(|c| c.results.iter().filter(|r| !r.sparse).map(|r| r.log_likelihood))
        .filter(|ll| ll.is_finite())
        .collect();
    if all_lls.is_empty() {
        println!("  (Pas de données à afficher)");
        return;
    }
    all_lls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let y_min = all_lls[0] as f32 - 0.5;
    let y_max = all_lls[all_lls.len() - 1] as f32 + 0.5;

    for cal in calibrations {
        let points: Vec<(f32, f32)> = cal.results.iter()
            .filter(|r| !r.sparse && r.log_likelihood.is_finite())
            .map(|r| (r.window as f32, r.log_likelihood as f32))
            .collect();

        if points.is_empty() {
            continue;
        }

        println!("  {} :", cal.model_name);
        let shape = textplots::Shape::Points(&points);
        let mut chart = textplots::Chart::new_with_y_range(120, 40, x_min, x_max, y_min, y_max);
        println!("{}", chart.lineplot(&shape));
    }
}

pub fn display_weights(weights: &EnsembleWeights) {
    println!("\n== Poids de l'ensemble ==\n");

    println!("── Boules ──");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Modèle", "Poids", "Contribution"]);

    for (name, weight) in &weights.ball_weights {
        let bar = "█".repeat((weight * 30.0).round() as usize);
        table.add_row(vec![name.as_str(), &format!("{:.4}", weight), &bar]);
    }
    println!("{table}");

    println!("\n── Étoiles ──");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Modèle", "Poids", "Contribution"]);

    for (name, weight) in &weights.star_weights {
        let bar = "█".repeat((weight * 30.0).round() as usize);
        table.add_row(vec![name.as_str(), &format!("{:.4}", weight), &bar]);
    }
    println!("{table}");
}

pub fn display_forecast(prediction: &EnsemblePrediction, pool: Pool) {
    let (top_n, label) = match pool {
        Pool::Balls => (15, "Boules"),
        Pool::Stars => (6, "Étoiles"),
    };

    println!("\n── Top {} {} ──", top_n, label);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);

    let mut header = vec!["#".to_string(), "Ensemble".to_string()];
    for (name, _) in &prediction.model_distributions {
        header.push(name.clone());
    }
    header.push("Spread".to_string());
    table.set_header(&header);

    // Trier par probabilité d'ensemble décroissante
    let mut indices: Vec<usize> = (0..prediction.distribution.len()).collect();
    indices.sort_by(|&a, &b| prediction.distribution[b].partial_cmp(&prediction.distribution[a]).unwrap_or(std::cmp::Ordering::Equal));

    for &idx in indices.iter().take(top_n) {
        let number = idx + 1;
        let mut row: Vec<String> = vec![
            format!("{:2}", number),
            format!("{:.4}", prediction.distribution[idx]),
        ];
        for (_, dist) in &prediction.model_distributions {
            row.push(format!("{:.4}", dist[idx]));
        }
        row.push(format!("{:.4}", prediction.spread[idx]));
        table.add_row(row);
    }

    println!("{table}");
}

pub fn display_consensus(entries: &[ConsensusEntry], pool: Pool) {
    let label = match pool {
        Pool::Balls => "Boules",
        Pool::Stars => "Étoiles",
    };

    println!("\n== Consensus Map ({}) ==\n", label);

    let categories = [
        (ConsensusCategory::StrongPick, "STRONG PICK", Color::Green),
        (ConsensusCategory::DivisivePick, "DIVISIVE", Color::Yellow),
        (ConsensusCategory::StrongAvoid, "AVOID", Color::Red),
        (ConsensusCategory::Uncertain, "UNCERTAIN", Color::White),
    ];

    for (cat, label, color) in &categories {
        let numbers: Vec<String> = entries
            .iter()
            .filter(|e| e.category == *cat)
            .map(|e| format!("{:2}", e.number))
            .collect();

        if !numbers.is_empty() {
            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_header(vec![
                    Cell::new(format!("{} ({})", label, numbers.len())).fg(*color),
                ]);
            table.add_row(vec![numbers.join("  ")]);
            println!("{table}");
        }
    }
}

pub fn display_optimal_grid(grid: &Suggestion, consensus: i32) {
    println!("\n== Grille optimale (consensus max) ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Boules", "Étoiles", "Score", "Consensus"]);

    let balls_str = grid.balls
        .iter()
        .map(|b| format!("{:2}", b))
        .collect::<Vec<_>>()
        .join(" - ");

    let stars_str = grid.stars
        .iter()
        .map(|s| format!("{:2}", s))
        .collect::<Vec<_>>()
        .join(" - ");

    table.add_row(vec![
        Cell::new(&balls_str).fg(Color::Green),
        Cell::new(&stars_str).fg(Color::Yellow),
        Cell::new(format!("{:.4}", grid.score)),
        Cell::new(format!("{:+}", consensus)),
    ]);
    println!("{table}");
}

pub fn display_suggestions(suggestions: &[Suggestion], consensus_scores: &[i32]) {
    println!("\n== Suggestions de grilles ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "Boules", "Étoiles", "Score", "Consensus"]);

    for (i, sug) in suggestions.iter().enumerate() {
        let balls_str = sug.balls
            .iter()
            .map(|b| format!("{:2}", b))
            .collect::<Vec<_>>()
            .join(" - ");

        let stars_str = sug.stars
            .iter()
            .map(|s| format!("{:2}", s))
            .collect::<Vec<_>>()
            .join(" - ");

        let cs = consensus_scores.get(i).copied().unwrap_or(0);

        if i == 0 {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)).fg(Color::Green),
                Cell::new(&balls_str).fg(Color::Green),
                Cell::new(&stars_str).fg(Color::Green),
                Cell::new(format!("{:.4}", sug.score)).fg(Color::Green),
                Cell::new(format!("{:+}", cs)).fg(Color::Green),
            ]);
        } else {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)),
                Cell::new(&balls_str),
                Cell::new(&stars_str),
                Cell::new(format!("{:.4}", sug.score)),
                Cell::new(format!("{:+}", cs)),
            ]);
        }
    }
    println!("{table}");
}

pub fn display_compare(
    balls: &[u8; 5],
    stars: &[u8; 2],
    ball_pred: &EnsemblePrediction,
    star_pred: &EnsemblePrediction,
) {
    println!("\n== Analyse de la grille ==\n");

    println!("── Boules ──");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);

    let mut header = vec!["Boule".to_string(), "Ensemble".to_string()];
    for (name, _) in &ball_pred.model_distributions {
        header.push(name.clone());
    }
    header.push("Rang".to_string());
    table.set_header(&header);

    for &b in balls {
        let idx = (b - 1) as usize;
        let mut row = vec![
            format!("{:2}", b),
            format!("{:.4}", ball_pred.distribution[idx]),
        ];
        for (_, dist) in &ball_pred.model_distributions {
            row.push(format!("{:.4}", dist[idx]));
        }
        // Calculer le rang
        let rank = ball_pred.distribution.iter()
            .filter(|&&p| p > ball_pred.distribution[idx])
            .count() + 1;
        row.push(format!("{}/50", rank));
        table.add_row(row);
    }
    println!("{table}");

    println!("\n── Étoiles ──");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);

    let mut header = vec!["Étoile".to_string(), "Ensemble".to_string()];
    for (name, _) in &star_pred.model_distributions {
        header.push(name.clone());
    }
    header.push("Rang".to_string());
    table.set_header(&header);

    for &s in stars {
        let idx = (s - 1) as usize;
        let mut row = vec![
            format!("{:2}", s),
            format!("{:.4}", star_pred.distribution[idx]),
        ];
        for (_, dist) in &star_pred.model_distributions {
            row.push(format!("{:.4}", dist[idx]));
        }
        let rank = star_pred.distribution.iter()
            .filter(|&&p| p > star_pred.distribution[idx])
            .count() + 1;
        row.push(format!("{}/12", rank));
        table.add_row(row);
    }
    println!("{table}");

    // Score bayésien
    let ball_score: f64 = balls.iter()
        .map(|&b| ball_pred.distribution[(b - 1) as usize] / (1.0 / 50.0))
        .product();
    let star_score: f64 = stars.iter()
        .map(|&s| star_pred.distribution[(s - 1) as usize] / (1.0 / 12.0))
        .product();
    println!("\nScore bayésien : {:.4} (boules: {:.4}, étoiles: {:.4})",
        ball_score * star_score, ball_score, star_score);
}

/// Affiche les résultats de l'analyse de non-aléatoire.
pub fn display_analysis(results: &[AnalysisResult], n_draws: usize) {
    println!("\n== Analyse de non-aléatoire ({} tirages) ==\n", n_draws);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Test", "Valeur", "Attendu (aléatoire)", "Verdict", "Détails"]);

    for r in results {
        let verdict_color = match r.verdict {
            Verdict::Signal => Color::Green,
            Verdict::Neutral => Color::Yellow,
            Verdict::Random => Color::Red,
        };

        let value_str = if r.value.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.4}", r.value)
        };

        let expected_str = if r.expected_random.is_infinite() {
            "inf".to_string()
        } else {
            format!("{:.4}", r.expected_random)
        };

        table.add_row(vec![
            Cell::new(&r.test_name),
            Cell::new(&value_str),
            Cell::new(&expected_str),
            Cell::new(format!("{}", r.verdict)).fg(verdict_color),
            Cell::new(&r.detail),
        ]);
    }
    println!("{table}");

    // Résumé
    let signals = results.iter().filter(|r| r.verdict == Verdict::Signal).count();
    let randoms = results.iter().filter(|r| r.verdict == Verdict::Random).count();
    let neutrals = results.iter().filter(|r| r.verdict == Verdict::Neutral).count();

    println!("\n── Verdict global ──");
    println!("  SIGNAL: {}  NEUTRE: {}  ALÉATOIRE: {}", signals, neutrals, randoms);

    if signals >= 3 {
        println!("  --> Structure potentielle détectée. Les modèles ont une chance de faire mieux que l'uniforme.");
    } else if randoms >= 3 {
        println!("  --> Aucune structure détectée. La séquence est indiscernable d'un processus aléatoire.");
    } else {
        println!("  --> Résultats mitigés. Signal faible ou insuffisant.");
    }
}

/// Résultat d'un backtest pour un tirage.
pub struct BacktestRow {
    pub date: String,
    pub actual_balls: [u8; 5],
    pub actual_stars: [u8; 2],
    pub score: f64,
    pub percentile: f64,
    pub optimal_ball_match: u8,
    pub optimal_star_match: u8,
    pub consensus: i32,
    pub bits_info: f64,
}

pub fn display_backtest_results(rows: &[BacktestRow]) {
    println!("\n== Backtest ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "Date", "Tirage réel", "Score", "Percentile", "Bits", "Optimal", "Consensus",
        ]);

    for row in rows {
        let balls_str = row
            .actual_balls
            .iter()
            .map(|b| format!("{:2}", b))
            .collect::<Vec<_>>()
            .join("-");
        let stars_str = row
            .actual_stars
            .iter()
            .map(|s| format!("{:2}", s))
            .collect::<Vec<_>>()
            .join("-");
        let draw_str = format!("{} + {}", balls_str, stars_str);

        let match_str = format!(
            "{}/5b {}/2s",
            row.optimal_ball_match, row.optimal_star_match
        );

        let pct_color = if row.percentile >= 70.0 {
            Color::Green
        } else if row.percentile >= 50.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let bits_color = if row.bits_info > 0.5 {
            Color::Green
        } else if row.bits_info > -0.5 {
            Color::Yellow
        } else {
            Color::Red
        };

        table.add_row(vec![
            Cell::new(&row.date),
            Cell::new(&draw_str),
            Cell::new(format!("{:.4}", row.score)),
            Cell::new(format!("{:.1}%", row.percentile)).fg(pct_color),
            Cell::new(format!("{:+.2}", row.bits_info)).fg(bits_color),
            Cell::new(&match_str),
            Cell::new(format!("{:+}", row.consensus)),
        ]);
    }
    println!("{table}");

    // Résumé amélioré
    if !rows.is_empty() {
        let n = rows.len() as f64;
        let avg_score = rows.iter().map(|r| r.score).sum::<f64>() / n;
        let avg_pct = rows.iter().map(|r| r.percentile).sum::<f64>() / n;
        let best_pct = rows
            .iter()
            .map(|r| r.percentile)
            .fold(0.0f64, f64::max);
        let worst_pct = rows
            .iter()
            .map(|r| r.percentile)
            .fold(100.0f64, f64::min);
        let best_score = rows.iter().map(|r| r.score).fold(0.0f64, f64::max);
        let worst_score = rows.iter().map(|r| r.score).fold(f64::MAX, f64::min);
        let avg_bits = rows.iter().map(|r| r.bits_info).sum::<f64>() / n;

        // Ball hit rate
        let avg_ball_hit =
            rows.iter().map(|r| r.optimal_ball_match as f64).sum::<f64>() / n;
        let avg_star_hit =
            rows.iter().map(|r| r.optimal_star_match as f64).sum::<f64>() / n;

        println!("\n── Résumé ──");
        println!(
            "  Score moyen      : {:.4} (min: {:.4}, max: {:.4})",
            avg_score, worst_score, best_score
        );
        println!(
            "  Percentile moyen : {:.1}% (min: {:.1}%, max: {:.1}%)",
            avg_pct, worst_pct, best_pct
        );
        println!("  Bits d'info moy. : {:+.3}", avg_bits);
        println!(
            "  Hit rate optimal : {:.2}/5 boules, {:.2}/2 étoiles",
            avg_ball_hit, avg_star_hit
        );

        let above_50 = rows.iter().filter(|r| r.percentile >= 50.0).count();
        println!(
            "  Tirages > 50e pct: {}/{}",
            above_50,
            rows.len()
        );

        // Rang estimé
        let total_combinations: f64 = 2_118_760.0 * 66.0; // C(50,5) * C(12,2)
        let avg_rank = total_combinations * (1.0 - avg_pct / 100.0);
        println!(
            "  Rang estimé moy. : ~{:.0} / {:.0}",
            avg_rank, total_combinations
        );
    }
}
