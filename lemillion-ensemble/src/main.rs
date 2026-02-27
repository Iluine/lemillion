mod interactive;

use std::path::PathBuf;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use lemillion_db::db::{count_draws, db_path, delete_draw, fetch_last_draws, insert_draw, migrate, open_db};
use lemillion_db::models::{Draw, Pool};
use lemillion_ensemble::display;
use lemillion_ensemble::ensemble::EnsembleCombiner;
use lemillion_ensemble::ensemble::calibration::{
    EnsembleWeights, calibrate_model, compute_weights_with_params, load_weights, save_weights,
};
use lemillion_ensemble::ensemble::consensus::{build_consensus_map, consensus_score};
use lemillion_ensemble::models::all_models;
use lemillion_ensemble::sampler::{
    apply_temperature, compute_bayesian_score, generate_suggestions_filtered,
    generate_suggestions_ev, optimal_grid, StructuralFilter,
};
use lemillion_ensemble::expected_value::{
    PopularityModel, count_matches, match_to_tier, PRIZE_TIERS, TICKET_PRICE,
};
use lemillion_ensemble::coverage::{optimize_coverage, compute_coverage_stats};

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
        #[arg(short, long, default_value = "20,30,50,80,100,150,200,300")]
        windows: String,

        /// Fichier de sortie pour les poids
        #[arg(short, long, default_value = "calibration.json")]
        output: String,

        /// Température pour le scaling des poids (T<1 = sharpening, T>1 = flattening)
        #[arg(short = 't', long, default_value = "0.5")]
        temperature: f64,
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

        /// Température (T<1 = sharpening, T>1 = flattening)
        #[arg(short = 't', long)]
        temperature: Option<f64>,

        /// Montant du jackpot actuel en EUR
        #[arg(long, default_value = "17000000")]
        jackpot: f64,
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

    /// Ajouter un tirage manuellement: draw_id jour date b1 b2 b3 b4 b5 s1 s2
    AddDraw {
        /// Paramètres: draw_id jour date b1 b2 b3 b4 b5 s1 s2
        args: Vec<String>,
    },

    /// Corriger le tirage erroné 20260221 → 26015
    FixDraw,

    /// Reconstruire la base en ordre chronologique
    Rebuild,

    /// Backtest sur les derniers tirages
    Backtest {
        /// Nombre de tirages à tester
        #[arg(short, long, default_value = "10")]
        last: usize,

        /// Nombre de suggestions par tirage
        #[arg(short, long, default_value = "5000")]
        suggestions: usize,

        /// Facteur de suréchantillonnage
        #[arg(long, default_value = "20")]
        oversample: usize,

        /// Fichier de calibration
        #[arg(short, long, default_value = "calibration.json")]
        calibration: String,

        /// Température pour recomputer les poids (T<1 = sharpening, T>1 = flattening)
        #[arg(short = 't', long)]
        temperature: Option<f64>,

        /// Balayer plusieurs températures pour trouver la meilleure
        #[arg(long)]
        sweep_temperature: bool,
    },

    /// Analyser la non-aléatoire des tirages
    Analyze,

    /// Optimiser la couverture d'un ensemble de tickets
    Coverage {
        /// Nombre de tickets
        #[arg(short = 'n', long, default_value = "10")]
        tickets: usize,

        /// Montant du jackpot actuel en EUR
        #[arg(long, default_value = "17000000")]
        jackpot: f64,

        /// Seed pour la reproductibilité
        #[arg(long)]
        seed: Option<u64>,
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
        Command::Calibrate { windows, output, temperature } => cmd_calibrate(&conn, &windows, &output, temperature),
        Command::Weights { calibration } => cmd_weights(&calibration),
        Command::Predict { calibration, suggestions, seed, oversample, min_diff, temperature, jackpot } => cmd_predict(&conn, &calibration, suggestions, seed, oversample, min_diff, temperature, jackpot),
        Command::History { last } => cmd_history(&conn, last),
        Command::Compare { numbers } => cmd_compare(&conn, &numbers),
        Command::AddDraw { args } => cmd_add_draw(&conn, &args),
        Command::FixDraw => cmd_fix_draw(&conn),
        Command::Rebuild => cmd_rebuild(&conn),
        Command::Backtest { last, suggestions, oversample, calibration, temperature, sweep_temperature } => cmd_backtest(&conn, last, suggestions, oversample, &calibration, temperature, sweep_temperature),
        Command::Analyze => cmd_analyze(&conn),
        Command::Coverage { tickets, jackpot, seed } => cmd_coverage(&conn, tickets, jackpot, seed),
        Command::Interactive => interactive::run_interactive(&conn),
    }
}

pub(crate) fn cmd_calibrate(conn: &lemillion_db::rusqlite::Connection, windows_str: &str, output: &str, temperature: f64) -> Result<()> {
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
    println!("\nTempérature : {:.2}", temperature);
    let ball_weights = compute_weights_with_params(&ball_calibrations, Pool::Balls, temperature);
    let star_weights = compute_weights_with_params(&star_calibrations, Pool::Stars, temperature);

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_predict(conn: &lemillion_db::rusqlite::Connection, calibration_path: &str, n_suggestions: usize, seed: Option<u64>, oversample: usize, min_diff: usize, temperature: Option<f64>, jackpot: f64) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;

    // Modele de popularite
    let popularity = PopularityModel::from_history(&draws);

    // Afficher resume EV et carte de popularite
    display::display_ev_summary(&popularity, jackpot);
    display::display_popularity_map(&popularity);

    let models = all_models();
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

    // Appliquer la température si fournie
    let (ball_dist, star_dist) = match temperature {
        Some(t) => {
            if t <= 0.0 {
                bail!("La température doit être > 0 (reçu : {t})");
            }
            println!("\n(Température appliquée : {:.2})", t);
            (apply_temperature(&ball_pred.distribution, t),
             apply_temperature(&star_pred.distribution, t))
        }
        None => (ball_pred.distribution.clone(), star_pred.distribution.clone()),
    };

    // Grille optimale
    let optimal = optimal_grid(&ball_dist, &star_dist);
    let optimal_cs = consensus_score(&optimal.balls, &optimal.stars, &ball_consensus, &star_consensus);
    display::display_optimal_grid(&optimal, optimal_cs);

    // Résolution du seed
    let effective_seed = seed.unwrap_or_else(|| {
        let ds = lemillion_ensemble::sampler::date_seed();
        println!("(Seed du jour : {ds})");
        ds
    });

    // Suggestions EV-aware (anti-popularite)
    let ev_suggestions = generate_suggestions_ev(
        &ball_dist,
        &star_dist,
        &draws,
        n_suggestions,
        effective_seed,
        oversample,
        min_diff,
        &popularity,
        jackpot,
    )?;

    // Tri par EV descendant
    let mut indexed: Vec<(usize, f64)> = ev_suggestions.iter().enumerate()
        .map(|(i, s)| (i, s.ev_per_euro))
        .collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_suggestions: Vec<_> = indexed.iter().map(|(i, _)| ev_suggestions[*i].clone()).collect();
    let consensus_scores_ev: Vec<f64> = sorted_suggestions.iter()
        .map(|s| consensus_score(&s.balls, &s.stars, &ball_consensus, &star_consensus))
        .collect();

    display::display_suggestions_ev(&sorted_suggestions, &consensus_scores_ev);

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

fn cmd_add_draw(conn: &lemillion_db::rusqlite::Connection, args: &[String]) -> Result<()> {
    if args.len() != 10 {
        bail!("Attendu 10 arguments: draw_id jour date b1 b2 b3 b4 b5 s1 s2\nExemple: add-draw 26016 MARDI 2026-02-24 10 27 40 43 47 6 10");
    }
    let draw = Draw {
        draw_id: args[0].clone(),
        day: args[1].clone(),
        date: args[2].clone(),
        balls: [
            args[3].parse().context("b1 invalide")?,
            args[4].parse().context("b2 invalide")?,
            args[5].parse().context("b3 invalide")?,
            args[6].parse().context("b4 invalide")?,
            args[7].parse().context("b5 invalide")?,
        ],
        stars: [
            args[8].parse().context("s1 invalide")?,
            args[9].parse().context("s2 invalide")?,
        ],
        winner_count: 0,
        winner_prize: 0.0,
        my_million: String::new(),
    };
    lemillion_db::models::validate_draw(&draw.balls, &draw.stars)?;
    let inserted = insert_draw(conn, &draw)?;
    if inserted {
        println!("Tirage {} ({}) inséré.", draw.draw_id, draw.date);
    } else {
        println!("Tirage {} déjà présent.", draw.draw_id);
    }
    let n = count_draws(conn)?;
    println!("Total : {} tirages en base.", n);
    Ok(())
}

fn cmd_fix_draw(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    // Supprimer le tirage erroné
    let deleted = delete_draw(conn, "20260221")?;
    if deleted {
        println!("Tirage 20260221 supprimé.");
    } else {
        println!("Tirage 20260221 non trouvé (déjà supprimé ?).");
    }

    // Insérer le tirage corrigé
    let corrected = Draw {
        draw_id: "26015".to_string(),
        day: "VENDREDI".to_string(),
        date: "2026-02-20".to_string(),
        balls: [13, 24, 28, 33, 35],
        stars: [5, 9],
        winner_count: 0,
        winner_prize: 0.0,
        my_million: String::new(),
    };

    let inserted = insert_draw(conn, &corrected)?;
    if inserted {
        println!("Tirage 26015 (2026-02-20) inséré.");
    } else {
        println!("Tirage 26015 déjà présent.");
    }

    let n = count_draws(conn)?;
    println!("Total : {} tirages en base.", n);
    Ok(())
}

fn cmd_backtest(
    conn: &lemillion_db::rusqlite::Connection,
    last: usize,
    n_suggestions: usize,
    oversample: usize,
    calibration_path: &str,
    temperature: Option<f64>,
    sweep_temperature: bool,
) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;
    if draws.len() < last + 10 {
        bail!(
            "Pas assez de tirages ({}) pour backtester {} tirages avec un historique suffisant",
            draws.len(),
            last
        );
    }

    let weights = load_weights(&PathBuf::from(calibration_path));

    // Validation de la température
    if let Some(t) = temperature
        && t <= 0.0
    {
        bail!("La température doit être > 0 (reçu : {t})");
    }

    // Sweep de température
    if sweep_temperature {
        return cmd_backtest_sweep(&draws, weights.as_ref().ok(), last, n_suggestions, oversample);
    }

    // Si --temperature fourni, recomputer les poids depuis les calibrations stockées
    let weights = match (weights, temperature) {
        (Ok(mut w), Some(t)) => {
            let n_models = w.ball_weights.len();
            if w.calibrations.len() >= n_models * 2 {
                let ball_cals = &w.calibrations[..n_models];
                let star_cals = &w.calibrations[n_models..n_models * 2];
                w.ball_weights = compute_weights_with_params(ball_cals, Pool::Balls, t);
                w.star_weights = compute_weights_with_params(star_cals, Pool::Stars, t);
                println!("Température : {:.2} (poids recomputés depuis les calibrations)", t);
            } else {
                println!("Température : {:.2} (calibrations insuffisantes, poids inchangés)", t);
            }
            Ok(w)
        }
        (w, _) => w,
    };

    if let Some(t) = temperature {
        println!(
            "Backtest de {} tirages avec {} suggestions chacun (oversample={}, T={:.2})\n",
            last, n_suggestions, oversample, t
        );
    } else {
        println!(
            "Backtest de {} tirages avec {} suggestions chacun (oversample={})\n",
            last, n_suggestions, oversample
        );
    }

    let pb = ProgressBar::new(last as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    let config = BacktestConfig {
        n_suggestions,
        oversample,
        temperature,
    };

    let rows = run_backtest_inner(&draws, weights.as_ref().ok(), last, &config, Some(&pb))?;

    pb.finish_and_clear();

    display::display_backtest_results(&rows);
    display::display_backtest_ev_results(&rows);

    Ok(())
}

struct BacktestConfig {
    n_suggestions: usize,
    oversample: usize,
    temperature: Option<f64>,
}

fn run_backtest_inner(
    draws: &[Draw],
    weights: Option<&EnsembleWeights>,
    last: usize,
    config: &BacktestConfig,
    pb: Option<&ProgressBar>,
) -> Result<Vec<display::BacktestRow>> {
    let mut rows = Vec::with_capacity(last);

    for i in 0..last {
        let test_draw = &draws[i];
        let training_draws = &draws[i + 1..];

        if let Some(pb) = pb {
            pb.set_message(test_draw.date.to_string());
        }

        let models = all_models();
        let combiner = match weights {
            Some(w) => {
                let ball_w: Vec<f64> = models
                    .iter()
                    .map(|m| {
                        w.ball_weights
                            .iter()
                            .find(|(n, _)| n == m.name())
                            .map(|(_, w)| *w)
                            .unwrap_or(0.0)
                    })
                    .collect();
                let star_w: Vec<f64> = models
                    .iter()
                    .map(|m| {
                        w.star_weights
                            .iter()
                            .find(|(n, _)| n == m.name())
                            .map(|(_, w)| *w)
                            .unwrap_or(0.0)
                    })
                    .collect();
                EnsembleCombiner::with_weights(models, ball_w, star_w)
            }
            None => EnsembleCombiner::new(models),
        };

        let ball_pred = combiner.predict(training_draws, Pool::Balls);
        let star_pred = combiner.predict(training_draws, Pool::Stars);

        let ball_dist = match config.temperature {
            Some(t) => apply_temperature(&ball_pred.distribution, t),
            None => ball_pred.distribution.clone(),
        };
        let star_dist = match config.temperature {
            Some(t) => apply_temperature(&star_pred.distribution, t),
            None => star_pred.distribution.clone(),
        };

        let real_score = compute_bayesian_score(
            &test_draw.balls,
            &test_draw.stars,
            &ball_dist,
            &star_dist,
        );

        let filter = StructuralFilter::from_history(training_draws, Pool::Balls);
        let suggestions = generate_suggestions_filtered(
            &ball_dist,
            &star_dist,
            config.n_suggestions,
            42 + i as u64,
            config.oversample,
            2,
            Some(&filter),
        )?;

        let below_count = suggestions.iter().filter(|s| s.score < real_score).count();
        let percentile = 100.0 * below_count as f64 / suggestions.len() as f64;

        let optimal = optimal_grid(&ball_dist, &star_dist);
        let ball_match = test_draw
            .balls
            .iter()
            .filter(|b| optimal.balls.contains(b))
            .count() as u8;
        let star_match = test_draw
            .stars
            .iter()
            .filter(|s| optimal.stars.contains(s))
            .count() as u8;

        let ball_consensus = build_consensus_map(&ball_pred, Pool::Balls);
        let star_consensus = build_consensus_map(&star_pred, Pool::Stars);
        let cs = consensus_score(&test_draw.balls, &test_draw.stars, &ball_consensus, &star_consensus);

        let median_score = if suggestions.len() >= 2 {
            let mut scores: Vec<f64> = suggestions.iter().map(|s| s.score).collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            scores[scores.len() / 2]
        } else {
            1.0
        };
        let bits_info = if median_score > 0.0 && real_score > 0.0 {
            (real_score / median_score).log2()
        } else {
            0.0
        };

        // Matches partiels : compter les hits par rang de prix
        let mut tier_hits = [0u32; 13];
        let mut best_tier: Option<u8> = None;
        let mut total_payout = 0.0f64;

        for suggestion in &suggestions {
            let (bm, sm) = count_matches(&suggestion.balls, &suggestion.stars, test_draw);
            if let Some(tier_idx) = match_to_tier(bm, sm) {
                tier_hits[tier_idx] += 1;
                let payout = if PRIZE_TIERS[tier_idx].is_parimutuel {
                    0.0 // on ne peut pas estimer les gains parimutuels en backtest
                } else {
                    PRIZE_TIERS[tier_idx].fixed_prize
                };
                total_payout += payout;

                match best_tier {
                    None => best_tier = Some(tier_idx as u8),
                    Some(current) => {
                        if (tier_idx as u8) < current {
                            best_tier = Some(tier_idx as u8);
                        }
                    }
                }
            }
        }

        let cost = config.n_suggestions as f64 * TICKET_PRICE;
        let roi = if cost > 0.0 { total_payout / cost } else { 0.0 };

        rows.push(display::BacktestRow {
            date: test_draw.date.clone(),
            actual_balls: test_draw.balls,
            actual_stars: test_draw.stars,
            score: real_score,
            percentile,
            optimal_ball_match: ball_match,
            optimal_star_match: star_match,
            consensus: cs,
            bits_info,
            tier_hits,
            best_tier,
            total_payout,
            roi,
        });

        if let Some(pb) = pb {
            pb.inc(1);
        }
    }

    Ok(rows)
}

fn recompute_weights_at_t(w: &EnsembleWeights, t: f64) -> EnsembleWeights {
    let mut out = w.clone();
    let n_models = w.ball_weights.len();
    if w.calibrations.len() >= n_models * 2 {
        out.ball_weights = compute_weights_with_params(&w.calibrations[..n_models], Pool::Balls, t);
        out.star_weights = compute_weights_with_params(&w.calibrations[n_models..n_models * 2], Pool::Stars, t);
    }
    out
}

fn cmd_backtest_sweep(
    draws: &[Draw],
    weights: Option<&EnsembleWeights>,
    last: usize,
    n_suggestions: usize,
    oversample: usize,
) -> Result<()> {
    let temperatures = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0];

    println!(
        "Sweep de température ({} valeurs) sur {} tirages avec {} suggestions\n",
        temperatures.len(), last, n_suggestions,
    );

    let total_steps = (temperatures.len() * last) as u64;
    let pb = ProgressBar::new(total_steps);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} T={msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut sweep_rows = Vec::new();

    for &t in &temperatures {
        pb.set_message(format!("{:.1}", t));

        let recomputed = weights.map(|w| recompute_weights_at_t(w, t));

        let config = BacktestConfig {
            n_suggestions,
            oversample,
            temperature: Some(t),
        };

        let rows = run_backtest_inner(draws, recomputed.as_ref(), last, &config, Some(&pb))?;

        let n = rows.len() as f64;
        if n > 0.0 {
            let avg_score = rows.iter().map(|r| r.score).sum::<f64>() / n;
            let avg_bits = rows.iter().map(|r| r.bits_info).sum::<f64>() / n;
            let avg_pct = rows.iter().map(|r| r.percentile).sum::<f64>() / n;

            sweep_rows.push(display::TemperatureSweepRow {
                temperature: t,
                avg_score,
                avg_bits,
                avg_percentile: avg_pct,
            });
        }
    }

    pb.finish_and_clear();
    display::display_temperature_sweep(&sweep_rows);

    Ok(())
}

fn cmd_coverage(conn: &lemillion_db::rusqlite::Connection, n_tickets: usize, jackpot: f64, seed: Option<u64>) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;
    let popularity = PopularityModel::from_history(&draws);

    let effective_seed = seed.unwrap_or_else(|| {
        let ds = lemillion_ensemble::sampler::date_seed();
        println!("(Seed du jour : {ds})");
        ds
    });

    println!("Optimisation de couverture : {} tickets, jackpot = {:.0} EUR\n", n_tickets, jackpot);

    let tickets = optimize_coverage(n_tickets, &popularity, jackpot, &draws, effective_seed)?;

    // Afficher les tickets
    display::display_suggestions_ev(
        &tickets,
        &tickets.iter().map(|_| 0.0).collect::<Vec<_>>(),
    );

    // Stats de couverture
    let stats = compute_coverage_stats(&tickets, &popularity, jackpot);
    display::display_coverage_stats(&stats, n_tickets);

    Ok(())
}

fn cmd_analyze(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;
    let results = lemillion_ensemble::analysis::run_all_tests(&draws);
    display::display_analysis(&results, draws.len());

    Ok(())
}

fn cmd_rebuild(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide, rien à reconstruire.");
    }

    // Lire tous les tirages (triés par date DESC)
    let mut draws = fetch_last_draws(conn, n)?;
    // Inverser pour obtenir l'ordre chronologique (ancien → récent)
    draws.reverse();

    println!("Reconstruction de {} tirages en ordre chronologique...", draws.len());

    // Supprimer et recréer la table dans une transaction
    let tx = conn.unchecked_transaction()
        .context("Impossible de démarrer la transaction")?;

    tx.execute_batch("DROP TABLE IF EXISTS draws")
        .context("Échec de la suppression de la table")?;
    migrate(&tx)?;

    for draw in &draws {
        insert_draw(&tx, draw)?;
    }

    tx.commit().context("Échec du commit")?;

    let final_count = count_draws(conn)?;
    println!("Reconstruction terminée : {} tirages.", final_count);

    // Vérifier le premier et le dernier
    let first = fetch_last_draws(conn, final_count)?;
    if let Some(oldest) = first.last() {
        println!("Premier tirage : {} ({})", oldest.draw_id, oldest.date);
    }
    if let Some(newest) = first.first() {
        println!("Dernier tirage : {} ({})", newest.draw_id, newest.date);
    }

    Ok(())
}

