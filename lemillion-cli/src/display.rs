use comfy_table::{Table, ContentArrangement, presets::UTF8_FULL, Cell, Color};

use crate::import::ImportResult;
use lemillion_db::models::{Draw, NumberProbability, NumberStats, ProbabilityTag, Suggestion};

pub fn display_draws(draws: &[Draw]) {
    if draws.is_empty() {
        println!("Aucun tirage à afficher.");
        return;
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Date", "Jour", "Boules", "Étoiles", "Gagnants R1", "Gains R1"]);

    for draw in draws {
        let mut sorted_balls = draw.balls;
        sorted_balls.sort();
        let mut sorted_stars = draw.stars;
        sorted_stars.sort();

        let balls_str = sorted_balls
            .iter()
            .map(|b| format!("{:2}", b))
            .collect::<Vec<_>>()
            .join(" - ");

        let stars_str = sorted_stars
            .iter()
            .map(|s| format!("{:2}", s))
            .collect::<Vec<_>>()
            .join(" - ");

        let prize = if draw.winner_prize > 0.0 {
            format!("{:.2} €", draw.winner_prize)
        } else {
            "—".to_string()
        };

        table.add_row(vec![
            &draw.date,
            &draw.day,
            &balls_str,
            &stars_str,
            &draw.winner_count.to_string(),
            &prize,
        ]);
    }

    println!("{table}");
}

pub fn display_import_summary(result: &ImportResult) {
    println!("Import terminé :");
    println!("  Total lignes lues : {}", result.total_records);
    println!("  Insérés           : {}", result.inserted);
    println!("  Doublons ignorés  : {}", result.skipped);
    if result.errors > 0 {
        println!("  Erreurs           : {}", result.errors);
    }
}

pub fn display_stats(ball_stats: &[NumberStats], star_stats: &[NumberStats], window: u32) {
    println!("\n📊 Statistiques sur les {} derniers tirages\n", window);

    println!("── Boules (1-50) ──");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Numéro", "Fréquence", "Retard"]);

    let mut sorted = ball_stats.to_vec();
    sorted.sort_by(|a, b| b.frequency.cmp(&a.frequency));

    for stat in &sorted {
        table.add_row(vec![
            &format!("{:2}", stat.number),
            &stat.frequency.to_string(),
            &stat.gap.to_string(),
        ]);
    }
    println!("{table}");

    println!("\n── Étoiles (1-12) ──");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Numéro", "Fréquence", "Retard"]);

    let mut sorted = star_stats.to_vec();
    sorted.sort_by(|a, b| b.frequency.cmp(&a.frequency));

    for stat in &sorted {
        table.add_row(vec![
            &format!("{:2}", stat.number),
            &stat.frequency.to_string(),
            &stat.gap.to_string(),
        ]);
    }
    println!("{table}");
}

pub fn display_probabilities(ball_probs: &[NumberProbability], star_probs: &[NumberProbability], model_name: &str) {
    println!("\n🎯 Probabilités ({model_name})\n");

    println!("── Boules ──");
    display_prob_table(ball_probs);

    println!("\n── Étoiles ──");
    display_prob_table(star_probs);
}

fn display_prob_table(probs: &[NumberProbability]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Numéro", "Probabilité", "Tag"]);

    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal));

    for prob in &sorted {
        let color = match prob.tag {
            ProbabilityTag::Hot => Color::Green,
            ProbabilityTag::Cold => Color::Red,
            ProbabilityTag::Normal => Color::White,
        };
        table.add_row(vec![
            Cell::new(format!("{:2}", prob.number)),
            Cell::new(format!("{:.4}", prob.probability)),
            Cell::new(prob.tag.to_string()).fg(color),
        ]);
    }
    println!("{table}");
}

pub fn display_suggestions(suggestions: &[Suggestion]) {
    println!("\n🎲 Suggestions de grilles\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "Boules", "Étoiles", "Score bayésien"]);

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

        table.add_row(vec![
            &format!("{}", i + 1),
            &balls_str,
            &stars_str,
            &format!("{:.4}", sug.score),
        ]);
    }
    println!("{table}");
}
