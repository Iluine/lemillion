mod interactive;

use std::path::PathBuf;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use lemillion_db::db::{count_draws, db_path, fetch_last_draws, migrate, open_db};
use lemillion_db::models::Pool;
use lemillion_ensemble::display;
use lemillion_ensemble::ensemble::EnsembleCombiner;
use lemillion_ensemble::ensemble::calibration::{
    EnsembleWeights, calibrate_model, compute_weights, load_weights, save_weights,
};
use lemillion_ensemble::ensemble::consensus::{build_consensus_map, consensus_score};
use lemillion_ensemble::models::all_models;
use lemillion_ensemble::sampler::{generate_suggestions_from_probs, optimal_grid};

#[derive(Parser)]
#[command(name = "lemillion-ensemble", about = "EuroMillions Ensemble Forecasting")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Calibrer les modèles par walk-forward validation
    Calibrate {
        /// Fenêtres d'analyse (séparées par des virgules)
        #[arg(short, long, default_value = "20,30,40,50,60,80,100")]
        windows: String,

        /// Fichier de sortie pour les poids
        #[arg(short, long, default_value = "calibration.json")]
        output: String,
    },

    /// Afficher les poids de l'ensemble
    Weights {
        /// Fichier de calibration
        #[arg(short, long, default_value = "calibration.json")]
        calibration: String,
    },

    /// Prédire avec l'ensemble
    Predict {
        /// Fichier de calibration
        #[arg(short, long, default_value = "calibration.json")]
        calibration: String,

        /// Nombre de suggestions
        #[arg(short, long, default_value = "5")]
        suggestions: usize,

        /// Seed pour la reproductibilité (défaut: date du jour YYYYMMDD)
        #[arg(long)]
        seed: Option<u64>,

        /// Facteur de suréchantillonnage (nombre de candidats = suggestions × oversample)
        #[arg(long, default_value = "20")]
        oversample: usize,

        /// Différence minimale de boules entre deux suggestions
        #[arg(long, default_value = "2")]
        min_diff: usize,
    },

    /// Historique des derniers tirages
    History {
        /// Nombre de tirages
        #[arg(short, long, default_value = "10")]
        last: u32,
    },

    /// Comparer un tirage avec les prédictions de l'ensemble
    Compare {
        /// 5 boules + 2 étoiles (7 nombres)
        numbers: Vec<u8>,
    },

    /// Mode interactif (REPL)
    Interactive,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = db_path();
    let conn = open_db(&path)?;
    migrate(&conn)?;

    match cli.command {
        Command::Calibrate { windows, output } => cmd_calibrate(&conn, &windows, &output),
        Command::Weights { calibration } => cmd_weights(&calibration),
        Command::Predict { calibration, suggestions, seed, oversample, min_diff } => cmd_predict(&conn, &calibration, suggestions, seed, oversample, min_diff),
        Command::History { last } => cmd_history(&conn, last),
        Command::Compare { numbers } => cmd_compare(&conn, &numbers),
        Command::Interactive => interactive::run_interactive(&conn),
    }
}

pub(crate) fn cmd_calibrate(conn: &lemillion_db::rusqlite::Connection, windows_str: &str, output: &str) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let windows: Vec<usize> = windows_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .context("Format de fenêtres invalide")?;

    let draws = fetch_last_draws(conn, n)?;
    let models = all_models();

    println!("Calibration de {} modèles sur {} tirages avec {} fenêtres...",
        models.len(), draws.len(), windows.len());

    let total_steps = (models.len() * 2) as u64; // balls + stars pour chaque modèle
    let pb = ProgressBar::new(total_steps);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("=> "));

    let mut ball_calibrations = Vec::new();
    let mut star_calibrations = Vec::new();

    for model in &models {
        pb.set_message(format!("{} (boules)", model.name()));
        let ball_cal = calibrate_model(model.as_ref(), &draws, &windows, Pool::Balls);
        ball_calibrations.push(ball_cal);
        pb.inc(1);

        pb.set_message(format!("{} (étoiles)", model.name()));
        let star_cal = calibrate_model(model.as_ref(), &draws, &windows, Pool::Stars);
        star_calibrations.push(star_cal);
        pb.inc(1);
    }

    pb.finish_with_message("Calibration terminée");

    // Afficher les résultats
    println!("\n── Boules ──");
    display::display_calibration_results(&ball_calibrations, &windows);
    println!("\n── Étoiles ──");
    display::display_calibration_results(&star_calibrations, &windows);

    // Afficher le graphique
    display::display_calibration_chart(&ball_calibrations, &windows);

    // Calculer et afficher les poids
    let ball_weights = compute_weights(&ball_calibrations, Pool::Balls);
    let star_weights = compute_weights(&star_calibrations, Pool::Stars);

    let ensemble_weights = EnsembleWeights {
        ball_weights,
        star_weights,
        calibrations: ball_calibrations.into_iter().chain(star_calibrations).collect(),
    };

    display::display_weights(&ensemble_weights);

    // Sauvegarder
    let output_path = PathBuf::from(output);
    save_weights(&ensemble_weights, &output_path)?;
    println!("\nPoids sauvegardés dans : {}", output);

    Ok(())
}

pub(crate) fn cmd_weights(calibration_path: &str) -> Result<()> {
    let weights = load_weights(&PathBuf::from(calibration_path))
        .context("Impossible de charger le fichier de calibration. Lancez d'abord : lemillion-ensemble calibrate")?;
    display::display_weights(&weights);
    Ok(())
}

pub(crate) fn cmd_predict(conn: &lemillion_db::rusqlite::Connection, calibration_path: &str, n_suggestions: usize, seed: Option<u64>, oversample: usize, min_diff: usize) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;
    let models = all_models();

    // Charger les poids ou utiliser les poids uniformes
    let weights = load_weights(&PathBuf::from(calibration_path));
    let combiner = match weights {
        Ok(w) => {
            let ball_w: Vec<f64> = models.iter()
                .map(|m| w.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                .collect();
            let star_w: Vec<f64> = models.iter()
                .map(|m| w.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                .collect();
            EnsembleCombiner::with_weights(models, ball_w, star_w)
        }
        Err(_) => {
            println!("(Pas de fichier de calibration, utilisation de poids uniformes)");
            EnsembleCombiner::new(models)
        }
    };

    // Prédire
    let ball_pred = combiner.predict(&draws, Pool::Balls);
    let star_pred = combiner.predict(&draws, Pool::Stars);

    // Afficher les distributions
    display::display_forecast(&ball_pred, Pool::Balls);
    display::display_forecast(&star_pred, Pool::Stars);

    // Consensus maps
    let ball_consensus = build_consensus_map(&ball_pred, Pool::Balls);
    let star_consensus = build_consensus_map(&star_pred, Pool::Stars);
    display::display_consensus(&ball_consensus, Pool::Balls);
    display::display_consensus(&star_consensus, Pool::Stars);

    // Grille optimale
    let optimal = optimal_grid(&ball_pred.distribution, &star_pred.distribution);
    let optimal_cs = consensus_score(&optimal.balls, &optimal.stars, &ball_consensus, &star_consensus);
    display::display_optimal_grid(&optimal, optimal_cs);

    // Résolution du seed
    let effective_seed = seed.unwrap_or_else(|| {
        let ds = lemillion_ensemble::sampler::date_seed();
        println!("(Seed du jour : {ds})");
        ds
    });

    // Suggestions
    let suggestions = generate_suggestions_from_probs(
        &ball_pred.distribution,
        &star_pred.distribution,
        n_suggestions,
        effective_seed,
        oversample,
        min_diff,
    )?;

    // Trier par consensus score décroissant (à score égal, garder l'ordre par score bayésien)
    let mut scored: Vec<(usize, i32)> = suggestions.iter().enumerate()
        .map(|(i, s)| (i, consensus_score(&s.balls, &s.stars, &ball_consensus, &star_consensus)))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));

    let sorted_suggestions: Vec<_> = scored.iter().map(|(i, _)| suggestions[*i].clone()).collect();
    let consensus_scores: Vec<i32> = scored.iter().map(|(_, cs)| *cs).collect();

    display::display_suggestions(&sorted_suggestions, &consensus_scores);

    Ok(())
}

pub(crate) fn cmd_history(conn: &lemillion_db::rusqlite::Connection, last: u32) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, last)?;

    let mut table = comfy_table::Table::new();
    table
        .load_preset(comfy_table::presets::UTF8_FULL)
        .set_content_arrangement(comfy_table::ContentArrangement::Dynamic)
        .set_header(vec!["Date", "Jour", "Boules", "Étoiles"]);

    for draw in &draws {
        let mut sorted_balls = draw.balls;
        sorted_balls.sort();
        let mut sorted_stars = draw.stars;
        sorted_stars.sort();

        let balls_str = sorted_balls.iter().map(|b| format!("{:2}", b)).collect::<Vec<_>>().join(" - ");
        let stars_str = sorted_stars.iter().map(|s| format!("{:2}", s)).collect::<Vec<_>>().join(" - ");

        table.add_row(vec![&draw.date, &draw.day, &balls_str, &stars_str]);
    }

    println!("{table}");
    Ok(())
}

pub(crate) fn cmd_compare(conn: &lemillion_db::rusqlite::Connection, numbers: &[u8]) -> Result<()> {
    if numbers.len() != 7 {
        bail!("Attendu 7 nombres : 5 boules + 2 étoiles. Reçu : {}", numbers.len());
    }

    let balls: [u8; 5] = [numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]];
    let stars: [u8; 2] = [numbers[5], numbers[6]];

    lemillion_db::models::validate_draw(&balls, &stars)?;

    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;
    let models = all_models();

    // Charger les poids si disponibles
    let weights = load_weights(&PathBuf::from("calibration.json"));
    let combiner = match weights {
        Ok(w) => {
            let ball_w: Vec<f64> = models.iter()
                .map(|m| w.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                .collect();
            let star_w: Vec<f64> = models.iter()
                .map(|m| w.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                .collect();
            EnsembleCombiner::with_weights(models, ball_w, star_w)
        }
        Err(_) => {
            println!("(Pas de fichier de calibration, utilisation de poids uniformes)");
            EnsembleCombiner::new(models)
        }
    };

    let ball_pred = combiner.predict(&draws, Pool::Balls);
    let star_pred = combiner.predict(&draws, Pool::Stars);

    display::display_compare(&balls, &stars, &ball_pred, &star_pred);

    Ok(())
}
