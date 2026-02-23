use anyhow::{Context, Result};
use chrono::Datelike;
use clap::{Parser, Subcommand};

use lemillion_db::db;

use lemillion_esn::config::{Encoding, EsnConfig};
use lemillion_esn::display;
use lemillion_esn::gridsearch;
use lemillion_esn::training;

#[derive(Parser)]
#[command(name = "lemillion-esn", about = "Echo State Network pour EuroMillions")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Entrainer un ESN avec les parametres donnes
    Train {
        #[arg(long, default_value = "500")]
        reservoir_size: usize,
        #[arg(long, default_value = "0.95")]
        spectral_radius: f64,
        #[arg(long, default_value = "0.9")]
        sparsity: f64,
        #[arg(long, default_value = "0.3")]
        leaking_rate: f64,
        #[arg(long, default_value = "1e-4")]
        ridge_lambda: f64,
        #[arg(long, default_value = "0.1")]
        input_scaling: f64,
        #[arg(long, default_value = "onehot")]
        encoding: Encoding,
        #[arg(long, default_value = "50")]
        washout: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long)]
        save: Option<String>,
    },
    /// Recherche de grille sur les hyperparametres
    Gridsearch {
        #[arg(short, long, default_value = "esn_gridsearch.json")]
        output: String,
        #[arg(long, default_value = "20")]
        top: usize,
    },
    /// Predire le prochain tirage avec un ESN entraine
    Predict {
        #[arg(short, long, default_value = "esn_best.json")]
        config: String,
        #[arg(short, long, default_value = "5")]
        suggestions: usize,
        #[arg(long)]
        seed: Option<u64>,
    },
}

fn date_seed() -> u64 {
    let today = chrono::Local::now().date_naive();
    let y = today.year() as u64;
    let m = today.month() as u64;
    let d = today.day() as u64;
    y * 10_000 + m * 100 + d
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let db_path = db::db_path();
    let conn = db::open_db(&db_path)?;
    db::migrate(&conn)?;

    let draw_count = db::count_draws(&conn)?;
    if draw_count == 0 {
        anyhow::bail!("Aucun tirage en base. Importez d'abord avec lemillion-cli.");
    }

    let draws = db::fetch_last_draws(&conn, draw_count)?;
    println!("{} tirages charges", draws.len());

    match cli.command {
        Command::Train {
            reservoir_size,
            spectral_radius,
            sparsity,
            leaking_rate,
            ridge_lambda,
            input_scaling,
            encoding,
            washout,
            seed,
            save,
        } => {
            let config = EsnConfig {
                reservoir_size,
                spectral_radius,
                sparsity,
                leaking_rate,
                ridge_lambda,
                input_scaling,
                encoding,
                washout,
                noise_amplitude: 1e-4,
                seed,
            };

            println!("Entrainement ESN...");
            println!("  reservoir_size={reservoir_size}, rho={spectral_radius}, sparsity={sparsity}");
            println!("  leaking_rate={leaking_rate}, ridge_lambda={ridge_lambda}, input_scaling={input_scaling}");
            println!("  encoding={encoding:?}, washout={washout}, seed={seed}");

            let (_, result) = training::train_and_evaluate(&draws, &config)?;
            display::display_metrics(&result);

            if let Some(path) = save {
                let json = serde_json::to_string_pretty(&config)?;
                std::fs::write(&path, json)?;
                println!("\nConfiguration sauvegardee dans {path}");
            }
        }
        Command::Gridsearch { output, top } => {
            let configs = gridsearch::generate_grid();
            println!("{} configurations a evaluer", configs.len());

            let results = gridsearch::run_grid_search(&draws, &configs, &output)?;
            display::display_grid_search_top(&results, top);

            // Save best config separately
            let best_json = serde_json::to_string_pretty(&results.best_config)?;
            std::fs::write("esn_best.json", best_json)?;
            println!("\nMeilleure configuration sauvegardee dans esn_best.json");
        }
        Command::Predict {
            config: config_path,
            suggestions: _,
            seed,
        } => {
            let json = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Impossible de lire {config_path}"))?;
            let mut config: EsnConfig = serde_json::from_str(&json)
                .with_context(|| format!("JSON invalide dans {config_path}"))?;

            // Override seed if provided, else use date-based seed
            config.seed = seed.unwrap_or_else(date_seed);

            println!("Entrainement ESN avec la configuration de {config_path}...");
            let (mut esn, result) = training::train_and_evaluate(&draws, &config)?;
            display::display_metrics(&result);

            let (ball_probs, star_probs) = training::predict_next(&mut esn, &draws);
            display::display_prediction(&ball_probs, &star_probs);
        }
    }

    Ok(())
}
