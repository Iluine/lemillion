use comfy_table::{Table, ContentArrangement, presets::UTF8_FULL, Cell, Color};
use textplots::Plot;

use lemillion_db::models::{Pool, Suggestion};
use crate::analysis::{AnalysisResult, Verdict};
use crate::ensemble::EnsemblePrediction;
use crate::ensemble::calibration::{EnsembleWeights, ModelCalibration};
use crate::ensemble::consensus::{ConsensusCategory, ConsensusEntry};
use crate::expected_value::{PopularityModel, PRIZE_TIERS, jackpot_threshold, compute_ev};
use crate::sampler::{ConvictionScore, ConvictionVerdict, JackpotResult, ScoredSuggestion};
use crate::coverage::CoverageStats;
use crate::research::{ResearchReport, ResearchVerdict};

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
        let best_tag = if cal.best_window == 0 {
            "(FH)".to_string()
        } else {
            let sparse_tag = if cal.best_sparse { "(S)" } else { "" };
            format!("w={}{}", cal.best_window, sparse_tag)
        };
        row.push(format!("{} ({:.3})", best_tag, cal.best_ll));
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
        .flat_map(|c| c.results.iter().filter(|r| !r.sparse && r.window > 0).map(|r| r.log_likelihood))
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
            .filter(|r| !r.sparse && r.window > 0 && r.log_likelihood.is_finite())
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

pub fn display_optimal_grid(grid: &Suggestion, consensus: f64) {
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
        Cell::new(format!("{:+.2}", consensus)),
    ]);
    println!("{table}");
}

pub fn display_suggestions(suggestions: &[Suggestion], consensus_scores: &[f64]) {
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

        let cs = consensus_scores.get(i).copied().unwrap_or(0.0);

        if i == 0 {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)).fg(Color::Green),
                Cell::new(&balls_str).fg(Color::Green),
                Cell::new(&stars_str).fg(Color::Green),
                Cell::new(format!("{:.4}", sug.score)).fg(Color::Green),
                Cell::new(format!("{:+.2}", cs)).fg(Color::Green),
            ]);
        } else {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)),
                Cell::new(&balls_str),
                Cell::new(&stars_str),
                Cell::new(format!("{:.4}", sug.score)),
                Cell::new(format!("{:+.2}", cs)),
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
    pub consensus: f64,
    pub bits_info: f64,
    /// Nombre de suggestions ayant touche chaque rang de prix (13 rangs)
    pub tier_hits: [u32; 13],
    /// Meilleur rang atteint (0=jackpot, None=aucun gain)
    pub best_tier: Option<u8>,
    /// Gain total sur toutes les suggestions
    pub total_payout: f64,
    /// ROI = total_payout / (n_suggestions * TICKET_PRICE)
    pub roi: f64,
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
            Cell::new(format!("{:+.2}", row.consensus)),
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

/// Résultat d'un sweep de température.
pub struct TemperatureSweepRow {
    pub temperature: f64,
    pub avg_score: f64,
    pub avg_bits: f64,
    pub avg_percentile: f64,
}

pub fn display_temperature_sweep(rows: &[TemperatureSweepRow]) {
    println!("\n== Sweep de température ==\n");

    if rows.is_empty() {
        println!("  (Aucun résultat)");
        return;
    }

    let best_bits_idx = rows.iter().enumerate()
        .max_by(|a, b| a.1.avg_bits.partial_cmp(&b.1.avg_bits).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_score_idx = rows.iter().enumerate()
        .max_by(|a, b| a.1.avg_score.partial_cmp(&b.1.avg_score).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["T", "Score moyen", "Bits d'info moy.", "Percentile moy."]);

    for (i, row) in rows.iter().enumerate() {
        if i == best_bits_idx {
            table.add_row(vec![
                Cell::new(format!("{:.1}", row.temperature)).fg(Color::Green),
                Cell::new(format!("{:.4}", row.avg_score)).fg(Color::Green),
                Cell::new(format!("{:+.3}", row.avg_bits)).fg(Color::Green),
                Cell::new(format!("{:.1}%", row.avg_percentile)).fg(Color::Green),
            ]);
        } else {
            table.add_row(vec![
                Cell::new(format!("{:.1}", row.temperature)),
                Cell::new(format!("{:.4}", row.avg_score)),
                Cell::new(format!("{:+.3}", row.avg_bits)),
                Cell::new(format!("{:.1}%", row.avg_percentile)),
            ]);
        }
    }

    println!("{table}");

    println!("\n── Résumé ──");
    println!("  Meilleure T (bits)  : {:.1}", rows[best_bits_idx].temperature);
    println!("  Meilleure T (score) : {:.1}", rows[best_score_idx].temperature);
}

/// Affiche le resume EV : jackpot, seuil, recommendation jouer/ne pas jouer.
pub fn display_ev_summary(popularity: &PopularityModel, jackpot: f64) {
    println!("\n== Analyse Esperance de Gain ==\n");

    let threshold = jackpot_threshold();

    // EV pour une grille moyenne et une grille impopulaire
    let avg_grid_balls = [10, 20, 25, 35, 45];
    let avg_grid_stars = [4, 9];
    let ev_avg = compute_ev(&avg_grid_balls, &avg_grid_stars, popularity, jackpot);

    let unpop_grid_balls = [33, 38, 42, 47, 50];
    let unpop_grid_stars = [10, 12];
    let ev_unpop = compute_ev(&unpop_grid_balls, &unpop_grid_stars, popularity, jackpot);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metrique", "Valeur"]);

    table.add_row(vec![
        Cell::new("Jackpot actuel"),
        Cell::new(format!("{:.0} EUR", jackpot)).fg(Color::Cyan),
    ]);
    table.add_row(vec![
        Cell::new("Seuil EV positive"),
        Cell::new(format!("{:.0} EUR", threshold)),
    ]);
    table.add_row(vec![
        Cell::new("EV/EUR (grille moyenne)"),
        Cell::new(format!("{:.4}", ev_avg.ev_per_euro)).fg(
            if ev_avg.ev_per_euro >= 1.0 { Color::Green } else { Color::Red }
        ),
    ]);
    table.add_row(vec![
        Cell::new("EV/EUR (grille impopulaire)"),
        Cell::new(format!("{:.4}", ev_unpop.ev_per_euro)).fg(
            if ev_unpop.ev_per_euro >= 1.0 { Color::Green } else { Color::Red }
        ),
    ]);

    println!("{table}");

    // Recommendation
    if jackpot >= threshold * 0.8 {
        println!("\n  --> Le jackpot est proche ou au-dessus du seuil. Jouer avec des grilles IMPOPULAIRES.");
    } else {
        println!("\n  --> EV negative. Si vous jouez, privilegiez les grilles impopulaires pour minimiser les pertes.");
    }
}

/// Affiche la carte de popularite : numeros a favoriser (impopulaires) et eviter (populaires).
pub fn display_popularity_map(popularity: &PopularityModel) {
    println!("\n== Carte de popularite ==\n");

    // Boules : trier par popularite
    let mut ball_indices: Vec<(usize, f64)> = popularity.ball_popularity
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    ball_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top impopulaires (a favoriser)
    let unpopular_balls: Vec<String> = ball_indices
        .iter()
        .take(10)
        .map(|(i, p)| format!("{:2} ({:.2})", i + 1, p))
        .collect();

    // Top populaires (a eviter)
    let popular_balls: Vec<String> = ball_indices
        .iter()
        .rev()
        .take(10)
        .map(|(i, p)| format!("{:2} ({:.2})", i + 1, p))
        .collect();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("A FAVORISER (impopulaires)").fg(Color::Green),
            Cell::new("A EVITER (populaires)").fg(Color::Red),
        ]);
    table.add_row(vec![unpopular_balls.join("  "), popular_balls.join("  ")]);
    println!("── Boules ──");
    println!("{table}");

    // Etoiles
    let mut star_indices: Vec<(usize, f64)> = popularity.star_popularity
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    star_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let unpopular_stars: Vec<String> = star_indices
        .iter()
        .take(4)
        .map(|(i, p)| format!("{:2} ({:.2})", i + 1, p))
        .collect();
    let popular_stars: Vec<String> = star_indices
        .iter()
        .rev()
        .take(4)
        .map(|(i, p)| format!("{:2} ({:.2})", i + 1, p))
        .collect();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("A FAVORISER (impopulaires)").fg(Color::Green),
            Cell::new("A EVITER (populaires)").fg(Color::Red),
        ]);
    table.add_row(vec![unpopular_stars.join("  "), popular_stars.join("  ")]);
    println!("\n── Etoiles ──");
    println!("{table}");
}

/// Affiche les suggestions EV-aware.
pub fn display_suggestions_ev(suggestions: &[ScoredSuggestion], consensus_scores: &[f64]) {
    println!("\n== Suggestions (mode EV) ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "Boules", "Etoiles", "EV/EUR", "Anti-Pop", "Consensus"]);

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

        let cs = consensus_scores.get(i).copied().unwrap_or(0.0);

        let ev_color = if sug.ev_per_euro >= 1.0 { Color::Green } else { Color::Yellow };

        if i == 0 {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)).fg(Color::Green),
                Cell::new(&balls_str).fg(Color::Green),
                Cell::new(&stars_str).fg(Color::Green),
                Cell::new(format!("{:.4}", sug.ev_per_euro)).fg(Color::Green),
                Cell::new(format!("{:.2}", sug.anti_popularity)).fg(Color::Green),
                Cell::new(format!("{:+.2}", cs)).fg(Color::Green),
            ]);
        } else {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)),
                Cell::new(&balls_str),
                Cell::new(&stars_str),
                Cell::new(format!("{:.4}", sug.ev_per_euro)).fg(ev_color),
                Cell::new(format!("{:.2}", sug.anti_popularity)),
                Cell::new(format!("{:+.2}", cs)),
            ]);
        }
    }
    println!("{table}");
}

/// Affiche les resultats de backtest avec matches partiels et ROI.
pub fn display_backtest_ev_results(rows: &[BacktestRow]) {
    println!("\n== Backtest (mode EV) ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "Date", "Tirage", "Best", "2+0", "2+1", "3+0", "1+2", "Gain", "ROI",
        ]);

    for row in rows {
        let balls_str = row.actual_balls
            .iter()
            .map(|b| format!("{:2}", b))
            .collect::<Vec<_>>()
            .join("-");
        let stars_str = row.actual_stars
            .iter()
            .map(|s| format!("{:2}", s))
            .collect::<Vec<_>>()
            .join("-");
        let draw_str = format!("{} + {}", balls_str, stars_str);

        let best_str = match row.best_tier {
            Some(t) => PRIZE_TIERS[t as usize].name.to_string(),
            None => "—".to_string(),
        };

        let roi_color = if row.roi >= 1.0 {
            Color::Green
        } else if row.roi >= 0.5 {
            Color::Yellow
        } else {
            Color::Red
        };

        table.add_row(vec![
            Cell::new(&row.date),
            Cell::new(&draw_str),
            Cell::new(&best_str),
            Cell::new(format!("{}", row.tier_hits[12])), // 2+0
            Cell::new(format!("{}", row.tier_hits[11])), // 2+1
            Cell::new(format!("{}", row.tier_hits[9])),  // 3+0
            Cell::new(format!("{}", row.tier_hits[10])), // 1+2
            Cell::new(format!("{:.2}", row.total_payout)),
            Cell::new(format!("{:.3}", row.roi)).fg(roi_color),
        ]);
    }
    println!("{table}");

    // Resume
    if !rows.is_empty() {
        let n = rows.len() as f64;
        let avg_roi = rows.iter().map(|r| r.roi).sum::<f64>() / n;
        let total_payout: f64 = rows.iter().map(|r| r.total_payout).sum();
        let total_cost: f64 = rows.iter().map(|r| {
            // Estimer le nombre de suggestions a partir du ROI et du payout
            if r.roi > 0.0 { r.total_payout / r.roi } else { 0.0 }
        }).sum();

        let best_roi_row = rows.iter().max_by(|a, b| {
            a.roi.partial_cmp(&b.roi).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compter les gains par rang
        let mut total_tier_hits = [0u32; 13];
        for row in rows {
            for (i, total) in total_tier_hits.iter_mut().enumerate() {
                *total += row.tier_hits[i];
            }
        }

        let any_wins = rows.iter().filter(|r| r.best_tier.is_some()).count();

        println!("\n── Resume EV ──");
        println!("  ROI moyen         : {:.4}", avg_roi);
        println!("  Gain total        : {:.2} EUR", total_payout);
        if total_cost > 0.0 {
            println!("  Cout total        : {:.2} EUR", total_cost);
        }
        println!("  Tirages avec gain : {}/{}", any_wins, rows.len());

        if let Some(best) = best_roi_row {
            println!("  Meilleur ROI      : {:.4} ({})", best.roi, best.date);
        }

        println!("\n── Repartition des gains ──");
        let mut gain_table = Table::new();
        gain_table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Rang", "Hits", "Gain/hit"]);

        for (i, tier) in PRIZE_TIERS.iter().enumerate() {
            if total_tier_hits[i] > 0 {
                let gain_per = if tier.is_parimutuel { "variable".to_string() } else { format!("{:.0} EUR", tier.fixed_prize) };
                gain_table.add_row(vec![
                    Cell::new(tier.name),
                    Cell::new(format!("{}", total_tier_hits[i])),
                    Cell::new(gain_per),
                ]);
            }
        }
        println!("{gain_table}");
    }
}

/// Affiche les statistiques de couverture.
pub fn display_coverage_stats(stats: &CoverageStats, n_tickets: usize) {
    println!("\n== Statistiques de couverture ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metrique", "Valeur"]);

    table.add_row(vec![
        Cell::new("Tickets"),
        Cell::new(format!("{}", n_tickets)),
    ]);
    table.add_row(vec![
        Cell::new("Cout total"),
        Cell::new(format!("{:.2} EUR", stats.total_cost)),
    ]);
    table.add_row(vec![
        Cell::new("Boules couvertes"),
        Cell::new(format!("{}/50", stats.unique_balls)).fg(
            if stats.unique_balls >= 40 { Color::Green } else { Color::Yellow }
        ),
    ]);
    table.add_row(vec![
        Cell::new("Etoiles couvertes"),
        Cell::new(format!("{}/12", stats.unique_stars)).fg(
            if stats.unique_stars >= 10 { Color::Green } else { Color::Yellow }
        ),
    ]);
    table.add_row(vec![
        Cell::new("P(au moins 1 gain)"),
        Cell::new(format!("{:.2}%", stats.any_win_probability * 100.0)),
    ]);
    table.add_row(vec![
        Cell::new("EV totale"),
        Cell::new(format!("{:.4} EUR", stats.total_ev)),
    ]);
    table.add_row(vec![
        Cell::new("EV/EUR"),
        Cell::new(format!("{:.4}", stats.total_ev / stats.total_cost)).fg(
            if stats.total_ev >= stats.total_cost { Color::Green } else { Color::Red }
        ),
    ]);

    println!("{table}");

    // Probabilites par rang
    println!("\n── Probabilites par rang ({} tickets) ──", n_tickets);
    let mut tier_table = Table::new();
    tier_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Rang", "P(1 ticket)", "P({} tickets)", "Gain"]);

    for (i, tier) in PRIZE_TIERS.iter().enumerate() {
        let gain_str = if tier.is_parimutuel {
            "variable".to_string()
        } else {
            format!("{:.0} EUR", tier.fixed_prize)
        };
        tier_table.add_row(vec![
            Cell::new(tier.name),
            Cell::new(format!("1/{:.0}", 1.0 / tier.probability)),
            Cell::new(format!("{:.6}%", stats.tier_probabilities[i] * 100.0)),
            Cell::new(gain_str),
        ]);
    }
    println!("{tier_table}");
}

/// Affiche le rapport de recherche de biais.
pub fn display_research_report(report: &ResearchReport) {
    let categories = [
        ("Physical", &report.physical),
        ("Mathematical", &report.mathematical),
        ("Informational", &report.informational),
    ];

    for (cat_name, results) in &categories {
        if results.is_empty() {
            continue;
        }

        println!("\n== {} ({} tests) ==\n", cat_name, results.len());

        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Test", "Statistique", "p-value", "Effet", "Verdict", "Détails"]);

        for r in *results {
            let verdict_color = match r.verdict {
                ResearchVerdict::Significant => Color::Green,
                ResearchVerdict::Marginal => Color::Yellow,
                ResearchVerdict::NotSignificant => Color::White,
            };

            let p_str = match r.p_value {
                Some(p) => {
                    if p < 0.001 { format!("{:.2e}", p) }
                    else { format!("{:.4}", p) }
                }
                None => "—".to_string(),
            };

            let stat_str = if r.statistic.is_nan() {
                "N/A".to_string()
            } else {
                format!("{:.4}", r.statistic)
            };

            // Truncate detail to keep table readable
            let detail = if r.detail.len() > 80 {
                format!("{}…", &r.detail[..77])
            } else {
                r.detail.clone()
            };

            table.add_row(vec![
                Cell::new(&r.test_name),
                Cell::new(&stat_str),
                Cell::new(&p_str),
                Cell::new(format!("{:.4}", r.effect_size)),
                Cell::new(format!("{}", r.verdict)).fg(verdict_color),
                Cell::new(&detail),
            ]);
        }
        println!("{table}");
    }

    // Résumé global
    let all = report.all_results();
    let sig_count = all.iter().filter(|r| r.verdict == ResearchVerdict::Significant).count();
    let marg_count = all.iter().filter(|r| r.verdict == ResearchVerdict::Marginal).count();
    let ns_count = all.iter().filter(|r| r.verdict == ResearchVerdict::NotSignificant).count();

    println!("\n── Résumé global ({} tests) ──", all.len());
    println!("  SIGNIFICATIF: {}  MARGINAL: {}  NON-SIGNIFICATIF: {}", sig_count, marg_count, ns_count);

    // Per-category summary
    for (cat_name, results) in &categories {
        if results.is_empty() {
            continue;
        }
        let s = results.iter().filter(|r| r.verdict == ResearchVerdict::Significant).count();
        let m = results.iter().filter(|r| r.verdict == ResearchVerdict::Marginal).count();
        println!("  {} : {} sig, {} marg / {} total", cat_name, s, m, results.len());
    }

    if sig_count >= 5 {
        println!("\n  --> Plusieurs biais significatifs détectés. Investigation approfondie recommandée.");
    } else if sig_count + marg_count >= 5 {
        println!("\n  --> Signaux faibles détectés. Augmenter la fenêtre ou le nombre de tirages pour confirmer.");
    } else {
        println!("\n  --> Pas de biais exploitable détecté. Les tirages semblent conformes à l'uniformité.");
    }
}

// ════════════════════════════════════════════════════════════════
// Affichage mode Jackpot
// ════════════════════════════════════════════════════════════════

/// Affiche les résultats du mode jackpot.
pub fn display_conviction(conviction: &ConvictionScore) {
    println!("\n== Conviction de l'ensemble ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Métrique", "Boules", "Étoiles"]);

    table.add_row(vec![
        Cell::new("Entropie (bits)"),
        Cell::new(format!("{:.3}", conviction.ball_entropy)),
        Cell::new(format!("{:.3}", conviction.star_entropy)),
    ]);

    table.add_row(vec![
        Cell::new("Concentration"),
        Cell::new(format!("{:.1}%", conviction.ball_concentration * 100.0)),
        Cell::new(format!("{:.1}%", conviction.star_concentration * 100.0)),
    ]);

    table.add_row(vec![
        Cell::new("Accord inter-modèles"),
        Cell::new(format!("{:.1}%", conviction.ball_agreement * 100.0)),
        Cell::new(format!("{:.1}%", conviction.star_agreement * 100.0)),
    ]);

    let (verdict_str, verdict_color) = match conviction.verdict {
        ConvictionVerdict::HighConviction => ("HAUTE", Color::Green),
        ConvictionVerdict::MediumConviction => ("MOYENNE", Color::Yellow),
        ConvictionVerdict::LowConviction => ("BASSE", Color::Red),
    };

    table.add_row(vec![
        Cell::new("Score global"),
        Cell::new(format!("{:.2}", conviction.overall)).fg(verdict_color),
        Cell::new(verdict_str).fg(verdict_color),
    ]);

    println!("{table}");
}

pub fn display_jackpot_results(
    result: &JackpotResult,
    ball_consensus: &[ConsensusEntry],
    star_consensus: &[ConsensusEntry],
    conviction: &ConvictionScore,
) {
    use crate::ensemble::consensus::consensus_score;

    println!("\n== Mode Jackpot ==\n");

    // Table de résumé
    let mut summary = Table::new();
    summary
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Métrique", "Valeur"]);

    summary.add_row(vec![
        Cell::new("Combinaisons énumérées"),
        Cell::new(format_count(result.enumeration_size)),
    ]);
    summary.add_row(vec![
        Cell::new("Passant le filtre"),
        Cell::new(format_count(result.filtered_size)),
    ]);
    summary.add_row(vec![
        Cell::new("Retournées (top-N)"),
        Cell::new(format!("{}", result.suggestions.len())),
    ]);
    summary.add_row(vec![
        Cell::new("P(jackpot totale)"),
        Cell::new(format!("{:.2e}", result.total_jackpot_probability)).fg(Color::Cyan),
    ]);
    summary.add_row(vec![
        Cell::new("Facteur vs uniforme"),
        Cell::new(format!("{:.2}x", result.improvement_factor)).fg(
            if result.improvement_factor > 1.0 { Color::Green } else { Color::Yellow }
        ),
    ]);

    let equiv_uniform = result.total_jackpot_probability * 139_838_160.0;
    summary.add_row(vec![
        Cell::new("Equiv. tickets uniformes"),
        Cell::new(format!("{:.0}", equiv_uniform)),
    ]);

    println!("{summary}");

    // Conviction
    display_conviction(conviction);

    // Table des suggestions
    let n_display = result.suggestions.len().min(50);
    println!("\n── Top {} suggestions ──\n", n_display);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "Boules", "Étoiles", "Score", "P(5+2)", "Consensus"]);

    for (i, sug) in result.suggestions.iter().take(n_display).enumerate() {
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

        let cs = consensus_score(&sug.balls, &sug.stars, ball_consensus, star_consensus);

        // P(5+2) = produit des probas individuelles (pas le score bayésien)
        // On ne le recalcule pas ici, mais le score est proportionnel

        if i == 0 {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)).fg(Color::Green),
                Cell::new(&balls_str).fg(Color::Green),
                Cell::new(&stars_str).fg(Color::Green),
                Cell::new(format!("{:.4}", sug.score)).fg(Color::Green),
                Cell::new(format!("{:.2e}", sug.score / (139_838_160.0_f64))).fg(Color::Green),
                Cell::new(format!("{:+.2}", cs)).fg(Color::Green),
            ]);
        } else {
            table.add_row(vec![
                Cell::new(format!("{}", i + 1)),
                Cell::new(&balls_str),
                Cell::new(&stars_str),
                Cell::new(format!("{:.4}", sug.score)),
                Cell::new(format!("{:.2e}", sug.score / (139_838_160.0_f64))),
                Cell::new(format!("{:+.2}", cs)),
            ]);
        }
    }

    if result.suggestions.len() > n_display {
        println!("  ... et {} autres suggestions", result.suggestions.len() - n_display);
    }

    println!("{table}");
}

/// Résultat backtest mode jackpot pour un tirage.
pub struct JackpotBacktestRow {
    pub date: String,
    pub actual_balls: [u8; 5],
    pub actual_stars: [u8; 2],
    pub actual_score: f64,
    pub in_top_n: bool,
    pub jackpot_probability: f64,
    pub improvement_factor: f64,
    pub best_tier: Option<u8>,
    pub total_payout: f64,
    pub conviction: f64,
}

/// Affiche les résultats de backtest mode jackpot.
pub fn display_jackpot_backtest_results(rows: &[JackpotBacktestRow]) {
    println!("\n== Backtest (mode Jackpot) ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "Date", "Tirage réel", "Score", "Top-N?", "P(jackpot)", "Amélioration", "Conv.", "Best", "Gain",
        ]);

    for row in rows {
        let balls_str = row.actual_balls
            .iter()
            .map(|b| format!("{:2}", b))
            .collect::<Vec<_>>()
            .join("-");
        let stars_str = row.actual_stars
            .iter()
            .map(|s| format!("{:2}", s))
            .collect::<Vec<_>>()
            .join("-");
        let draw_str = format!("{} + {}", balls_str, stars_str);

        let in_top = if row.in_top_n {
            Cell::new("OUI").fg(Color::Green)
        } else {
            Cell::new("non").fg(Color::Red)
        };

        let best_str = match row.best_tier {
            Some(t) => PRIZE_TIERS[t as usize].name.to_string(),
            None => "—".to_string(),
        };

        let conv_color = if row.conviction >= 0.6 {
            Color::Green
        } else if row.conviction >= 0.3 {
            Color::Yellow
        } else {
            Color::Red
        };

        table.add_row(vec![
            Cell::new(&row.date),
            Cell::new(&draw_str),
            Cell::new(format!("{:.4}", row.actual_score)),
            in_top,
            Cell::new(format!("{:.2e}", row.jackpot_probability)),
            Cell::new(format!("{:.2}x", row.improvement_factor)),
            Cell::new(format!("{:.2}", row.conviction)).fg(conv_color),
            Cell::new(&best_str),
            Cell::new(format!("{:.2}", row.total_payout)),
        ]);
    }
    println!("{table}");

    // Résumé
    if !rows.is_empty() {
        let n = rows.len() as f64;
        let avg_improvement = rows.iter().map(|r| r.improvement_factor).sum::<f64>() / n;
        let avg_prob = rows.iter().map(|r| r.jackpot_probability).sum::<f64>() / n;
        let hits_in_top = rows.iter().filter(|r| r.in_top_n).count();
        let total_payout: f64 = rows.iter().map(|r| r.total_payout).sum();

        let avg_conviction = rows.iter().map(|r| r.conviction).sum::<f64>() / n;

        println!("\n── Résumé Jackpot ──");
        println!("  P(jackpot) moyenne   : {:.2e}", avg_prob);
        println!("  Amélioration moyenne : {:.2}x", avg_improvement);
        println!("  Conviction moyenne   : {:.2}", avg_conviction);
        println!("  Dans top-N           : {}/{}", hits_in_top, rows.len());
        println!("  Gain total           : {:.2} EUR", total_payout);
    }
}

/// Formate un grand nombre avec séparateurs de milliers.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(' ');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

