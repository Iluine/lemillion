use comfy_table::{Table, ContentArrangement, presets::UTF8_FULL, Cell, Color};
use textplots::Plot;

use lemillion_db::models::{Pool, Suggestion};
use crate::ensemble::EnsemblePrediction;
use crate::ensemble::calibration::{EnsembleWeights, ModelCalibration};
use crate::ensemble::consensus::{ConsensusCategory, ConsensusEntry};

pub fn display_calibration_results(calibrations: &[ModelCalibration], windows: &[usize]) {
    println!("\n== Résultats de calibration ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);

    let mut header = vec!["Modèle".to_string()];
    for w in windows {
        header.push(format!("w={}", w));
    }
    header.push("Best".to_string());
    table.set_header(&header);

    for cal in calibrations {
        let mut row: Vec<String> = vec![cal.model_name.clone()];
        for w in windows {
            let ll = cal.results.iter()
                .find(|r| r.window == *w)
                .map(|r| format!("{:.3}", r.log_likelihood))
                .unwrap_or_else(|| "—".to_string());
            row.push(ll);
        }
        row.push(format!("w={} ({:.3})", cal.best_window, cal.best_ll));
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
        .flat_map(|c| c.results.iter().map(|r| r.log_likelihood))
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
            .filter(|r| r.log_likelihood.is_finite())
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
