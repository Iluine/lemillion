mod analysis;
mod display;
mod import;

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use crate::analysis::{compute_stats, tag_probabilities};
use crate::analysis::dirichlet::dirichlet_probabilities;
use crate::analysis::ewma::ewma_probabilities;
use crate::analysis::sampler::generate_suggestions;
use lemillion_db::db::{count_draws, db_path, fetch_last_draws, fetch_last_draws_numbers, insert_draw, migrate, open_db};
use lemillion_db::models::{Draw, validate_draw};
use crate::display::{
    display_draws, display_import_summary, display_probabilities, display_stats,
    display_suggestions,
};

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum PredictionModel {
    #[default]
    Dirichlet,
    Ewma,
}

#[derive(Parser)]
#[command(name = "lemillion", about = "Analyseur de probabilités EuroMillions")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Importer les tirages depuis un fichier CSV
    Import {
        /// Chemin vers le fichier CSV
        #[arg(short, long, default_value = "assets/euromillions_202002.csv")]
        file: PathBuf,
    },

    /// Afficher le chemin de la base de données
    DbPath,

    /// Lister les derniers tirages
    List {
        /// Nombre de tirages à afficher
        #[arg(short, long, default_value = "10")]
        last: u32,
    },

    /// Afficher les statistiques (fréquences et retards)
    Stats {
        /// Fenêtre d'analyse (nombre de tirages)
        #[arg(short, long, default_value = "100")]
        window: u32,
    },

    /// Prédire les prochains tirages
    Predict {
        /// Modèle de prédiction
        #[arg(short, long, default_value = "dirichlet")]
        model: PredictionModel,

        /// Paramètre alpha (Dirichlet: prior, EWMA: facteur de décroissance)
        #[arg(short, long, default_value = "1.0")]
        alpha: f64,

        /// Fenêtre d'analyse (nombre de tirages)
        #[arg(short, long, default_value = "100")]
        window: u32,

        /// Nombre de grilles à suggérer
        #[arg(short, long, default_value = "3")]
        count: usize,

        /// Seed pour la reproductibilité
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Ajouter un tirage manuellement
    Add,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = db_path();
    let conn = open_db(&path)?;
    migrate(&conn)?;

    match cli.command {
        Command::Import { file } => cmd_import(&conn, &file),
        Command::DbPath => {
            println!("{}", path.display());
            Ok(())
        }
        Command::List { last } => cmd_list(&conn, last),
        Command::Stats { window } => cmd_stats(&conn, window),
        Command::Predict {
            model,
            alpha,
            window,
            count,
            seed,
        } => cmd_predict(&conn, model, alpha, window, count, seed),
        Command::Add => cmd_add(&conn),
    }
}

fn cmd_import(conn: &lemillion_db::rusqlite::Connection, file: &PathBuf) -> Result<()> {
    let result = import::import_csv(conn, file)?;
    display_import_summary(&result);
    Ok(())
}

fn cmd_list(conn: &lemillion_db::rusqlite::Connection, last: u32) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        println!("Base vide. Lancez d'abord : lemillion import");
        return Ok(());
    }
    let draws = fetch_last_draws(conn, last)?;
    display_draws(&draws);
    Ok(())
}

fn cmd_stats(conn: &lemillion_db::rusqlite::Connection, window: u32) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        println!("Base vide. Lancez d'abord : lemillion import");
        return Ok(());
    }
    let effective_window = window.min(n);
    let draws = fetch_last_draws_numbers(conn, effective_window)?;

    let ball_stats = compute_stats(&draws, 50, false);
    let star_stats = compute_stats(&draws, 12, true);

    display_stats(&ball_stats, &star_stats, effective_window);
    Ok(())
}

fn cmd_predict(
    conn: &lemillion_db::rusqlite::Connection,
    model: PredictionModel,
    alpha: f64,
    window: u32,
    count: usize,
    seed: Option<u64>,
) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        println!("Base vide. Lancez d'abord : lemillion import");
        return Ok(());
    }
    let effective_window = window.min(n);
    let draws = fetch_last_draws_numbers(conn, effective_window)?;

    let model_name = match model {
        PredictionModel::Dirichlet => format!("Dirichlet α={}", alpha),
        PredictionModel::Ewma => format!("EWMA α={}", alpha),
    };

    let (mut ball_probs, mut star_probs) = match model {
        PredictionModel::Dirichlet => (
            dirichlet_probabilities(&draws, 50, alpha, false),
            dirichlet_probabilities(&draws, 12, alpha, true),
        ),
        PredictionModel::Ewma => (
            ewma_probabilities(&draws, 50, alpha, false),
            ewma_probabilities(&draws, 12, alpha, true),
        ),
    };

    tag_probabilities(&mut ball_probs, 50);
    tag_probabilities(&mut star_probs, 12);

    display_probabilities(&ball_probs, &star_probs, &model_name);

    let suggestions = generate_suggestions(&ball_probs, &star_probs, count, seed)?;
    display_suggestions(&suggestions);

    Ok(())
}

fn cmd_add(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    println!("Ajout d'un tirage manuellement\n");

    let draw_id = prompt("Identifiant du tirage (ex: 26014) : ")?;
    let day = prompt("Jour (ex: MARDI) : ")?;
    let raw_date = prompt("Date (JJ/MM/AAAA) : ")?;

    let date_parts: Vec<&str> = raw_date.split('/').collect();
    if date_parts.len() != 3 {
        bail!("Format de date invalide");
    }
    let date = format!("{}-{}-{}", date_parts[2], date_parts[1], date_parts[0]);

    let balls = prompt_balls()?;
    let stars = prompt_stars()?;

    validate_draw(&balls, &stars)?;

    let draw = Draw {
        draw_id,
        day,
        date,
        balls,
        stars,
        winner_count: 0,
        winner_prize: 0.0,
        my_million: String::new(),
    };

    println!("\nTirage à insérer :");
    display_draws(&[draw.clone()]);

    let confirm = prompt("\nConfirmer l'insertion ? (o/n) : ")?;
    if confirm.trim().to_lowercase() == "o" {
        let inserted = insert_draw(conn, &draw)?;
        if inserted {
            println!("Tirage inséré avec succès.");
        } else {
            println!("Ce tirage existe déjà (doublon ignoré).");
        }
    } else {
        println!("Insertion annulée.");
    }

    Ok(())
}

fn prompt(msg: &str) -> Result<String> {
    print!("{}", msg);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("Erreur de lecture")?;
    Ok(input.trim().to_string())
}

fn prompt_balls() -> Result<[u8; 5]> {
    loop {
        let input = prompt("5 boules (séparées par des espaces, 1-50) : ")?;
        let nums: Result<Vec<u8>, _> = input.split_whitespace().map(|s| s.parse::<u8>()).collect();
        match nums {
            Ok(v) if v.len() == 5 => {
                let arr = [v[0], v[1], v[2], v[3], v[4]];
                if validate_draw(&arr, &[1, 2]).is_ok() {
                    return Ok(arr);
                }
                println!("Numéros invalides (1-50, pas de doublons). Réessayez.");
            }
            _ => println!("Entrez exactement 5 numéros. Réessayez."),
        }
    }
}

fn prompt_stars() -> Result<[u8; 2]> {
    loop {
        let input = prompt("2 étoiles (séparées par un espace, 1-12) : ")?;
        let nums: Result<Vec<u8>, _> = input.split_whitespace().map(|s| s.parse::<u8>()).collect();
        match nums {
            Ok(v) if v.len() == 2 => {
                let arr = [v[0], v[1]];
                if validate_draw(&[1, 2, 3, 4, 5], &arr).is_ok() {
                    return Ok(arr);
                }
                println!("Étoiles invalides (1-12, pas de doublons). Réessayez.");
            }
            _ => println!("Entrez exactement 2 numéros. Réessayez."),
        }
    }
}
