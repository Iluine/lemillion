mod interactive;

use std::path::PathBuf;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use lemillion_db::db::{count_draws, db_path, delete_draw, fetch_last_draws, fetch_draw_by_date, fetch_draws_before_date, insert_draw, migrate, open_db};
use lemillion_db::models::{Draw, Pool};
use lemillion_ensemble::display;
use lemillion_ensemble::ensemble::{EnsembleCombiner, compute_hedge_weights};
use lemillion_ensemble::ensemble::calibration::{
    EnsembleWeights, calibrate_model, collect_detailed_ll, compute_weights_with_params,
    compute_weights_with_threshold, detect_redundancy, load_weights, save_weights,
};
use lemillion_ensemble::ensemble::consensus::{build_consensus_map, consensus_score};
use lemillion_ensemble::models::all_models;
use lemillion_ensemble::sampler::{
    apply_temperature, compute_bayesian_score, compute_conviction, conviction_temperature,
    generate_suggestions_filtered, generate_diverse_grids_with_strategy,
    generate_suggestions_ev, generate_suggestions_jackpot, optimal_grid,
    ConvictionVerdict, StarStrategy, StructuralFilter,
};
use lemillion_ensemble::expected_value::{
    PopularityModel, count_matches, match_to_tier, PRIZE_TIERS, TICKET_PRICE,
};
use lemillion_ensemble::coverage::{optimize_coverage, compute_coverage_stats};
use lemillion_ensemble::research::{ResearchCategory, run_all_research};

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
        #[arg(short = 't', long, default_value = "1.0")]
        temperature: f64,

        /// Seuil de skill minimum (modèles avec skill <= seuil reçoivent poids 0)
        #[arg(long, default_value = "0.0")]
        min_skill: f64,
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

        /// Mode jackpot : maximiser P(5+2) par énumération exhaustive
        #[arg(long)]
        jackpot_mode: bool,

        /// Désactiver le filtre structurel
        #[arg(long)]
        no_filter: bool,

        /// Nombre de top modèles à conserver (0 = tous)
        #[arg(long, default_value = "0")]
        top_models: usize,

        /// Désactiver le MetaPredictor (ajustement contextuel des poids)
        #[arg(long)]
        no_meta_predictor: bool,

        /// Désactiver le Hedge (multiplicative weight update)
        #[arg(long)]
        no_hedge: bool,

        /// Force du agreement boost (0.0 = désactivé)
        #[arg(long, default_value = "0.0")]
        agreement_boost: f64,

        /// Stratégie de diversification des étoiles (concentrated/triangular/disjoint)
        #[arg(long, default_value = "triangular")]
        star_strategy: String,
    },

    /// Historique des derniers tirages
    History {
        /// Nombre de tirages
        #[arg(short, long, default_value = "10")]
        last: u32,
    },

    /// Comparer un tirage avec les prédictions de l'ensemble
    Compare {
        /// 5 boules + 2 étoiles (7 nombres). Optionnel si --date est fourni.
        numbers: Vec<u8>,

        /// Date du tirage (YYYY-MM-DD). Récupère automatiquement les numéros et n'utilise que les tirages antérieurs.
        #[arg(short, long)]
        date: Option<String>,

        /// Fichier de calibration
        #[arg(short, long, default_value = "calibration.json")]
        calibration: String,
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

        /// Mode jackpot : maximiser P(5+2) par énumération exhaustive
        #[arg(long)]
        jackpot_mode: bool,

        /// Nombre de top modèles à conserver (0 = tous)
        #[arg(long, default_value = "0")]
        top_models: usize,

        /// Mode 3 grilles : simuler le scénario réel (3 grilles, 7.50€/tirage)
        #[arg(long)]
        backtest_3_grids: bool,

        /// Stratégie étoiles pour le backtest 3 grilles (concentrated, triangular, disjoint)
        #[arg(long, default_value = "triangular")]
        star_strategy: String,
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

    /// Recherche de biais exploitables
    Research {
        /// Catégorie de tests : physical, mathematical, informational, all
        #[arg(short = 't', long, default_value = "all")]
        tests: String,

        /// Fenêtre : derniers N tirages (défaut: tous)
        #[arg(short, long)]
        window: Option<usize>,
    },

    /// Benchmark des modèles physiques par random-split
    Benchmark {
        /// Taille du jeu d'entraînement
        #[arg(long, default_value = "600")]
        train: usize,

        /// Seed pour le shuffle
        #[arg(long, default_value = "42")]
        seed: u64,
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
        Command::Calibrate { windows, output, temperature, min_skill } => cmd_calibrate(&conn, &windows, &output, temperature, min_skill),
        Command::Weights { calibration } => cmd_weights(&calibration),
        Command::Predict { calibration, suggestions, seed, oversample, min_diff, temperature, jackpot, jackpot_mode, no_filter, top_models, no_meta_predictor, no_hedge, agreement_boost, star_strategy } => cmd_predict(&conn, &calibration, suggestions, seed, oversample, min_diff, temperature, jackpot, jackpot_mode, no_filter, top_models, no_meta_predictor, no_hedge, agreement_boost, &star_strategy),
        Command::History { last } => cmd_history(&conn, last),
        Command::Compare { numbers, date, calibration } => cmd_compare(&conn, &numbers, date.as_deref(), &calibration),
        Command::AddDraw { args } => cmd_add_draw(&conn, &args),
        Command::FixDraw => cmd_fix_draw(&conn),
        Command::Rebuild => cmd_rebuild(&conn),
        Command::Backtest { last, suggestions, oversample, calibration, temperature, sweep_temperature, jackpot_mode, top_models, backtest_3_grids, star_strategy } => cmd_backtest(&conn, last, suggestions, oversample, &calibration, temperature, sweep_temperature, jackpot_mode, top_models, backtest_3_grids, &star_strategy),
        Command::Analyze => cmd_analyze(&conn),
        Command::Coverage { tickets, jackpot, seed } => cmd_coverage(&conn, tickets, jackpot, seed),
        Command::Research { tests, window } => cmd_research(&conn, &tests, window),
        Command::Benchmark { train, seed } => cmd_benchmark(&conn, train, seed),
        Command::Interactive => interactive::run_interactive(&conn),
    }
}

pub(crate) fn cmd_calibrate(conn: &lemillion_db::rusqlite::Connection, windows_str: &str, output: &str, temperature: f64, min_skill: f64) -> Result<()> {
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

    use rayon::prelude::*;

    let model_results: Vec<_> = models
        .par_iter()
        .map(|model| {
            let ball_cal = calibrate_model(model.as_ref(), &draws, &windows, Pool::Balls);
            let star_cal = calibrate_model(model.as_ref(), &draws, &windows, Pool::Stars);
            pb.inc(2);
            (ball_cal, star_cal)
        })
        .collect();

    let (ball_calibrations, star_calibrations): (Vec<_>, Vec<_>) =
        model_results.into_iter().unzip();

    pb.finish_with_message("Calibration terminée");

    // Afficher les résultats
    println!("\n── Boules ──");
    display::display_calibration_results(&ball_calibrations, &windows);
    println!("\n── Étoiles ──");
    display::display_calibration_results(&star_calibrations, &windows);

    // Afficher le graphique
    display::display_calibration_chart(&ball_calibrations, &windows);

    // Collecter les detailed_ll pour le meta-predictor (boules + étoiles)
    println!("\nCollecte des LL détaillés pour le méta-apprentissage...");
    let detailed_ll: Vec<(String, Vec<f64>)> = models
        .par_iter()
        .zip(ball_calibrations.par_iter())
        .map(|(model, ball_cal)| {
            let strategy = model.sampling_strategy();
            let lls = collect_detailed_ll(
                model.as_ref(), &draws, ball_cal.best_window, Pool::Balls, strategy,
            );
            (model.name().to_string(), lls)
        })
        .collect();

    let star_detailed_ll: Vec<(String, Vec<f64>)> = models
        .par_iter()
        .zip(star_calibrations.par_iter())
        .map(|(model, star_cal)| {
            let strategy = model.sampling_strategy();
            let lls = collect_detailed_ll(
                model.as_ref(), &draws, star_cal.best_window, Pool::Stars, strategy,
            );
            (model.name().to_string(), lls)
        })
        .collect();

    // Détection de redondance (boules)
    let redundancies = detect_redundancy(&detailed_ll, 0.90);
    if !redundancies.is_empty() {
        println!("\n── Redondance boules (corrélation LL > 0.90) ──");
        for r in &redundancies {
            let severity = if r.correlation > 0.95 { " [HALVING]" } else { "" };
            println!("  {} <-> {} : {:.3}{}", r.model_a, r.model_b, r.correlation, severity);
        }
    }

    // Détection de redondance (étoiles)
    let star_redundancies = detect_redundancy(&star_detailed_ll, 0.90);
    if !star_redundancies.is_empty() {
        println!("\n── Redondance étoiles (corrélation LL > 0.90) ──");
        for r in &star_redundancies {
            let severity = if r.correlation > 0.95 { " [HALVING]" } else { "" };
            println!("  {} <-> {} : {:.3}{}", r.model_a, r.model_b, r.correlation, severity);
        }
    }

    // Calculer et afficher les poids
    println!("\nTempérature : {:.2}, seuil skill : {:.4}", temperature, min_skill);
    let mut ball_weights = compute_weights_with_threshold(&ball_calibrations, Pool::Balls, temperature, min_skill);
    let mut star_weights = compute_weights_with_threshold(&star_calibrations, Pool::Stars, temperature, min_skill);

    // Halver le poids du modèle le plus faible pour les paires boules > 0.95
    for r in &redundancies {
        if r.correlation > 0.95 {
            let w_a = ball_weights.iter().find(|(n, _)| *n == r.model_a).map(|(_, w)| *w).unwrap_or(0.0);
            let w_b = ball_weights.iter().find(|(n, _)| *n == r.model_b).map(|(_, w)| *w).unwrap_or(0.0);
            let weaker = if w_a <= w_b { &r.model_a } else { &r.model_b };
            if let Some((_, w)) = ball_weights.iter_mut().find(|(n, _)| n == weaker) {
                *w *= 0.5;
            }
        }
    }
    // Renormaliser les poids boules après halving
    let total_bw: f64 = ball_weights.iter().map(|(_, w)| w).sum();
    if total_bw > 0.0 {
        for (_, w) in ball_weights.iter_mut() {
            *w /= total_bw;
        }
    }

    // Halver le poids du modèle le plus faible pour les paires étoiles > 0.95
    for r in &star_redundancies {
        if r.correlation > 0.95 {
            let w_a = star_weights.iter().find(|(n, _)| *n == r.model_a).map(|(_, w)| *w).unwrap_or(0.0);
            let w_b = star_weights.iter().find(|(n, _)| *n == r.model_b).map(|(_, w)| *w).unwrap_or(0.0);
            let weaker = if w_a <= w_b { &r.model_a } else { &r.model_b };
            if let Some((_, w)) = star_weights.iter_mut().find(|(n, _)| n == weaker) {
                *w *= 0.5;
            }
        }
    }
    // Renormaliser les poids étoiles après halving
    let total_sw: f64 = star_weights.iter().map(|(_, w)| w).sum();
    if total_sw > 0.0 {
        for (_, w) in star_weights.iter_mut() {
            *w /= total_sw;
        }
    }

    let ensemble_weights = EnsembleWeights {
        ball_weights,
        star_weights,
        calibrations: ball_calibrations.into_iter().chain(star_calibrations).collect(),
        detailed_ll,
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

fn filter_top_models(
    ball_weights: &mut [f64],
    star_weights: &mut [f64],
    model_names: &[String],
    top_n: usize,
) {
    // Filter ball weights: keep top N, zero the rest
    let mut ball_indexed: Vec<(usize, f64)> = ball_weights.iter().copied().enumerate().collect();
    ball_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_ball_indices: Vec<usize> = ball_indexed.iter().take(top_n).map(|(i, _)| *i).collect();
    for (i, w) in ball_weights.iter_mut().enumerate() {
        if !top_ball_indices.contains(&i) {
            *w = 0.0;
        }
    }
    let ball_sum: f64 = ball_weights.iter().sum();
    if ball_sum > 0.0 {
        for w in ball_weights.iter_mut() {
            *w /= ball_sum;
        }
    }

    // Filter star weights: keep top N, zero the rest
    let mut star_indexed: Vec<(usize, f64)> = star_weights.iter().copied().enumerate().collect();
    star_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_star_indices: Vec<usize> = star_indexed.iter().take(top_n).map(|(i, _)| *i).collect();
    for (i, w) in star_weights.iter_mut().enumerate() {
        if !top_star_indices.contains(&i) {
            *w = 0.0;
        }
    }
    let star_sum: f64 = star_weights.iter().sum();
    if star_sum > 0.0 {
        for w in star_weights.iter_mut() {
            *w /= star_sum;
        }
    }

    // Display retained models
    println!("\n── Top {} modèles retenus ──", top_n);
    println!("  Boules:");
    for &idx in &top_ball_indices {
        if idx < model_names.len() {
            println!("    {} ({:.4})", model_names[idx], ball_weights[idx]);
        }
    }
    println!("  Étoiles:");
    for &idx in &top_star_indices {
        if idx < model_names.len() {
            println!("    {} ({:.4})", model_names[idx], star_weights[idx]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_predict(conn: &lemillion_db::rusqlite::Connection, calibration_path: &str, n_suggestions: usize, seed: Option<u64>, oversample: usize, min_diff: usize, temperature: Option<f64>, jackpot: f64, jackpot_mode: bool, no_filter: bool, top_models: usize, no_meta_predictor: bool, no_hedge: bool, agreement_boost_strength: f64, star_strategy_str: &str) -> Result<()> {
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
    let mut combiner = match weights {
        Ok(ref w) => {
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

    // Détecter si les poids boules ont du signal (non-uniformes)
    let balls_have_signal = {
        let max_w = combiner.ball_weights.iter().cloned().fold(0.0_f64, f64::max);
        let min_w = combiner.ball_weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::MAX, f64::min);
        (max_w - min_w) > 1e-6
    };

    // MetaPredictor : ajuster les poids boules selon le régime courant (activé par défaut)
    // Seulement si les boules ont du signal (sinon detailed_ll = bruit)
    if !no_meta_predictor
        && balls_have_signal
        && let Ok(ref w) = weights
        && !w.detailed_ll.is_empty()
    {
        use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
        let features = RegimeFeatures::from_draws(&draws);
        if let Some(meta) = MetaPredictor::train(&draws, &w.detailed_ll, 1e-3) {
            let adjustments = meta.weight_adjustments(&features);
            let mut n_adjusted = 0;
            for (name, adj) in &adjustments {
                if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                    && idx < combiner.ball_weights.len()
                {
                    combiner.ball_weights[idx] *= adj;
                    n_adjusted += 1;
                }
            }
            // Renormaliser les boules
            let total: f64 = combiner.ball_weights.iter().sum();
            if total > 0.0 {
                for w in combiner.ball_weights.iter_mut() {
                    *w /= total;
                }
            }
            println!("(MetaPredictor : {} poids boules ajustés pour le régime courant)", n_adjusted);
            println!("  Régime: sum={:.2} spread={:.2} mod4_cos={:.2} entropy={:.2}",
                features.sum_norm, features.spread_norm, features.mod4_cosine, features.recent_entropy);
        }
    }

    // Hedge weights : ajustement multiplicatif basé sur les 20 derniers tirages (activé par défaut)
    // Boules seulement si signal, étoiles toujours
    if !no_hedge {
        let (hedged_ball, hedged_star) = compute_hedge_weights(
            &combiner.models, &draws,
            &combiner.ball_weights, &combiner.star_weights,
            20,    // n_recent : 20 derniers tirages
            0.05,  // eta : learning rate conservateur
        );
        if balls_have_signal {
            combiner.ball_weights = hedged_ball;
        }
        combiner.star_weights = hedged_star;
        println!("(Hedge weights appliqués sur les 20 derniers tirages{})",
            if !balls_have_signal { " [étoiles seulement]" } else { "" });
    }

    // Filtrer les top modèles si demandé
    if top_models > 0 {
        let model_names: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
        filter_top_models(&mut combiner.ball_weights, &mut combiner.star_weights, &model_names, top_models);
    }

    let (ball_pred, star_pred) = if agreement_boost_strength > 0.0 {
        println!("(Agreement boost : {:.2})", agreement_boost_strength);
        (combiner.predict_with_agreement_boost(&draws, Pool::Balls, agreement_boost_strength),
         combiner.predict_with_agreement_boost(&draws, Pool::Stars, agreement_boost_strength))
    } else {
        (combiner.predict(&draws, Pool::Balls),
         combiner.predict(&draws, Pool::Stars))
    };

    // Afficher les distributions
    display::display_forecast(&ball_pred, Pool::Balls);
    display::display_forecast(&star_pred, Pool::Stars);

    // Consensus maps
    let ball_consensus = build_consensus_map(&ball_pred, Pool::Balls);
    let star_consensus = build_consensus_map(&star_pred, Pool::Stars);
    display::display_consensus(&ball_consensus, Pool::Balls);
    display::display_consensus(&star_consensus, Pool::Stars);

    if let Some(t) = temperature
        && t <= 0.0
    {
        bail!("La température doit être > 0 (reçu : {t})");
    }

    // Calculer la conviction AVANT d'appliquer la température (sur la distribution brute)
    let conviction = compute_conviction(
        &ball_pred.distribution,
        &star_pred.distribution,
        &ball_pred.spread,
        &star_pred.spread,
    );

    // Température adaptative : utiliser la conviction en mode jackpot ET en mode EV
    let effective_temp = if let Some(t) = temperature {
        t
    } else if jackpot_mode {
        let ct = conviction_temperature(&conviction.verdict);
        println!("\n(Température adaptative par conviction [{:.0}%] : {:.2})", conviction.overall * 100.0, ct);
        ct
    } else {
        // Mode EV : sharpening modéré basé sur la conviction
        let ct: f64 = match conviction.verdict {
            ConvictionVerdict::HighConviction => 0.5,
            ConvictionVerdict::MediumConviction => 0.7,
            ConvictionVerdict::LowConviction => 1.0,
        };
        if (ct - 1.0).abs() > 1e-9 {
            println!("\n(Température EV par conviction [{:.0}%] : {:.2})", conviction.overall * 100.0, ct);
        }
        ct
    };

    let (ball_dist, star_dist) = if (effective_temp - 1.0).abs() > 1e-9 {
        println!("(Température appliquée : {:.2})", effective_temp);
        (apply_temperature(&ball_pred.distribution, effective_temp),
         apply_temperature(&star_pred.distribution, effective_temp))
    } else {
        (ball_pred.distribution.clone(), star_pred.distribution.clone())
    };

    // Grille optimale
    let optimal = optimal_grid(&ball_dist, &star_dist);
    let optimal_cs = consensus_score(&optimal.balls, &optimal.stars, &ball_consensus, &star_consensus);
    display::display_optimal_grid(&optimal, optimal_cs);

    // Grilles diversifiées par profil mod-4
    let effective_seed = seed.unwrap_or_else(lemillion_ensemble::sampler::date_seed);
    let star_strat = StarStrategy::from_str_loose(star_strategy_str);
    let diverse = generate_diverse_grids_with_strategy(&ball_dist, &star_dist, &draws, 3, effective_seed, Some(&popularity), star_strat);
    display::display_diverse_grids(&diverse, &ball_consensus, &star_consensus);

    // Verdict Jackpot — P(5+2) par grille et combiné
    if !diverse.grids.is_empty() {
        const TOTAL_COMBINATIONS: f64 = 139_838_160.0; // C(50,5) * C(12,2)

        let grid_probs: Vec<f64> = diverse.grids.iter().map(|g| {
            let p_balls: f64 = g.balls.iter()
                .map(|&b| ball_dist[(b as usize) - 1])
                .product();
            let p_stars: f64 = g.stars.iter()
                .map(|&s| star_dist[(s as usize) - 1])
                .product();
            // P(5+2) = produit des probas individuelles × nombre de combinaisons
            // car prob[i] = fréquence estimée du numéro i, pas la proba de le tirer
            // score = prod(prob/uniform) et P(5+2) = score / TOTAL_COMBINATIONS
            p_balls * p_stars * TOTAL_COMBINATIONS
        }).collect();

        // P(au moins 1 grille matche 5+2) = 1 - produit(1 - P_i)
        let p_jackpot_3 = 1.0 - grid_probs.iter().map(|&p| 1.0 - p).product::<f64>();
        let p_random_3 = 3.0 / TOTAL_COMBINATIONS;
        let factor = p_jackpot_3 / p_random_3;

        println!("\n── Verdict Jackpot ──");
        for (i, &p) in grid_probs.iter().enumerate() {
            println!("  P(5+2) grille {} : {:.2e}", i + 1, p);
        }
        println!("  P(5+2) combiné  : {:.2e}", p_jackpot_3);
        println!("  Facteur vs aléatoire : {:.1}x", factor);
    }

    if jackpot_mode {
        // Mode Jackpot : énumération exhaustive
        let filter = if no_filter {
            None
        } else {
            Some(StructuralFilter::from_history(&draws, Pool::Balls))
        };

        let result = generate_suggestions_jackpot(
            &ball_dist,
            &star_dist,
            n_suggestions,
            filter.as_ref(),
        )?;

        display::display_jackpot_results(&result, &ball_consensus, &star_consensus, &conviction);
    } else {
        // Mode EV (défaut)
        let effective_seed = seed.unwrap_or_else(|| {
            let ds = lemillion_ensemble::sampler::date_seed();
            println!("(Seed du jour : {ds})");
            ds
        });

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
    }

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

pub(crate) fn cmd_compare(conn: &lemillion_db::rusqlite::Connection, numbers: &[u8], date: Option<&str>, calibration_path: &str) -> Result<()> {
    // Déterminer les numéros à comparer et les tirages pour la prédiction
    let (balls, stars, draws) = if let Some(date) = date {
        // Mode date : récupérer le tirage et n'utiliser que les tirages antérieurs
        let draw = fetch_draw_by_date(conn, date)?
            .ok_or_else(|| anyhow::anyhow!("Aucun tirage trouvé pour la date {date}"))?;
        let balls = draw.balls;
        let stars = draw.stars;
        let draws = fetch_draws_before_date(conn, date)?;
        if draws.is_empty() {
            bail!("Aucun tirage antérieur au {date} dans la base");
        }
        println!("Tirage du {date} : {:?} | {:?}", balls, stars);
        println!("Prédiction basée sur {} tirages antérieurs au {date}\n", draws.len());
        (balls, stars, draws)
    } else {
        // Mode classique : numéros fournis en arguments
        if numbers.len() != 7 {
            bail!("Attendu 7 nombres : 5 boules + 2 étoiles (ou --date YYYY-MM-DD). Reçu : {}", numbers.len());
        }
        let balls: [u8; 5] = [numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]];
        let stars: [u8; 2] = [numbers[5], numbers[6]];
        lemillion_db::models::validate_draw(&balls, &stars)?;
        let n = count_draws(conn)?;
        if n == 0 {
            bail!("Base vide. Lancez d'abord : lemillion-cli import");
        }
        let draws = fetch_last_draws(conn, n)?;
        (balls, stars, draws)
    };
    let models = all_models();

    // Charger les poids si disponibles
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
        Err(e) => {
            println!("(Pas de fichier de calibration, utilisation de poids uniformes: {e})");
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

#[allow(clippy::too_many_arguments)]
fn cmd_backtest(
    conn: &lemillion_db::rusqlite::Connection,
    last: usize,
    n_suggestions: usize,
    oversample: usize,
    calibration_path: &str,
    temperature: Option<f64>,
    sweep_temperature: bool,
    jackpot_mode: bool,
    top_models: usize,
    backtest_3_grids: bool,
    star_strategy_str: &str,
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

    // Mode jackpot backtest
    if jackpot_mode {
        return cmd_backtest_jackpot(&draws, weights.as_ref().ok(), last, n_suggestions, temperature, top_models);
    }

    // Mode backtest 3 grilles réalistes
    if backtest_3_grids {
        let star_strat = StarStrategy::from_str_loose(star_strategy_str);
        return cmd_backtest_3grids(&draws, weights.as_ref().ok(), last, temperature, star_strat);
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
    // (date, tier_name, rank, total, score)
    let mut winning_ranks: Vec<(String, &str, usize, usize, f64)> = Vec::new();

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
        // Also track rank of winning grids (suggestions are already sorted by score desc)
        let mut tier_hits = [0u32; 13];
        let mut best_tier: Option<u8> = None;
        let mut total_payout = 0.0f64;

        for (rank, suggestion) in suggestions.iter().enumerate() {
            let (bm, sm) = count_matches(&suggestion.balls, &suggestion.stars, test_draw);
            if let Some(tier_idx) = match_to_tier(bm, sm) {
                tier_hits[tier_idx] += 1;
                let payout = if PRIZE_TIERS[tier_idx].is_parimutuel {
                    0.0 // on ne peut pas estimer les gains parimutuels en backtest
                } else {
                    PRIZE_TIERS[tier_idx].fixed_prize
                };
                total_payout += payout;

                // Track rank for winning grids (tier <= 9 = 3+0 or better)
                if tier_idx <= 9 {
                    winning_ranks.push((test_draw.date.clone(), PRIZE_TIERS[tier_idx].name, rank + 1, config.n_suggestions, suggestion.score));
                }

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

    // Display winning grid rank analysis
    if !winning_ranks.is_empty() {
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        println!("\n== Rang des grilles gagnantes (3+0 ou mieux) ==\n");

        // Group by tier
        let mut by_tier: std::collections::BTreeMap<&str, Vec<(usize, usize, f64)>> = std::collections::BTreeMap::new();
        for (_, tier, rank, total, score) in &winning_ranks {
            by_tier.entry(tier).or_default().push((*rank, *total, *score));
        }

        // Show detail for high tiers (4+1, 3+2, 4+0, 3+1, 3+0)
        use comfy_table::{Table, Cell, Color};
        let mut table = Table::new();
        table.load_preset(comfy_table::presets::UTF8_FULL);
        table.set_header(vec!["Date", "Rang prix", "Position", "Percentile", "Score"]);

        for (date, tier, rank, total, score) in &winning_ranks {
            let pct = 100.0 * *rank as f64 / *total as f64;
            let pct_str = format!("{:.1}%", pct);
            let pos_str = format!("{}/{}", rank, total);
            let color = if pct <= 25.0 { Color::Green } else if pct <= 50.0 { Color::Yellow } else { Color::Red };
            table.add_row(vec![
                Cell::new(date),
                Cell::new(tier),
                Cell::new(&pos_str),
                Cell::new(&pct_str).fg(color),
                Cell::new(format!("{:.4}", score)),
            ]);
        }
        println!("{table}");

        // Summary per tier
        println!("\n── Résumé par rang ──");
        for (tier, ranks) in &by_tier {
            let percentiles: Vec<f64> = ranks.iter().map(|(r, t, _)| 100.0 * *r as f64 / *t as f64).collect();
            let avg_pct = percentiles.iter().sum::<f64>() / percentiles.len() as f64;
            let median_pct = {
                let mut sorted = percentiles.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted[sorted.len() / 2]
            };
            let in_top_quarter = percentiles.iter().filter(|&&p| p <= 25.0).count();
            let in_top_half = percentiles.iter().filter(|&&p| p <= 50.0).count();
            println!("  {} : {} grilles, percentile moyen {:.1}%, médian {:.1}%, top-25%: {}/{}, top-50%: {}/{}",
                tier, ranks.len(), avg_pct, median_pct,
                in_top_quarter, ranks.len(), in_top_half, ranks.len());
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

fn cmd_backtest_jackpot(
    draws: &[Draw],
    weights: Option<&EnsembleWeights>,
    last: usize,
    n_suggestions: usize,
    temperature: Option<f64>,
    top_models: usize,
) -> Result<()> {
    let temp_label = match temperature {
        Some(t) => format!("T={:.2}", t),
        None => "T=adaptive".to_string(),
    };

    println!(
        "Backtest Jackpot de {} tirages avec {} suggestions chacun ({}{})\n",
        last, n_suggestions, temp_label,
        if top_models > 0 { format!(", top-{}", top_models) } else { String::new() },
    );

    let pb = ProgressBar::new(last as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut rows = Vec::with_capacity(last);

    for i in 0..last {
        let test_draw = &draws[i];
        let training_draws = &draws[i + 1..];

        pb.set_message(test_draw.date.to_string());

        let models = all_models();
        let mut combiner = match weights {
            Some(w) => {
                let ball_w: Vec<f64> = models
                    .iter()
                    .map(|m| w.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                let star_w: Vec<f64> = models
                    .iter()
                    .map(|m| w.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                EnsembleCombiner::with_weights(models, ball_w, star_w)
            }
            None => EnsembleCombiner::new(models),
        };

        // Filtrer les top modèles si demandé (silencieux en backtest)
        if top_models > 0 {
            let model_names: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
            // Filter without printing (backtest loop)
            let mut ball_indexed: Vec<(usize, f64)> = combiner.ball_weights.iter().copied().enumerate().collect();
            ball_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_ball: Vec<usize> = ball_indexed.iter().take(top_models).map(|(i, _)| *i).collect();
            for (i, w) in combiner.ball_weights.iter_mut().enumerate() {
                if !top_ball.contains(&i) { *w = 0.0; }
            }
            let bs: f64 = combiner.ball_weights.iter().sum();
            if bs > 0.0 { for w in combiner.ball_weights.iter_mut() { *w /= bs; } }

            let mut star_indexed: Vec<(usize, f64)> = combiner.star_weights.iter().copied().enumerate().collect();
            star_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_star: Vec<usize> = star_indexed.iter().take(top_models).map(|(i, _)| *i).collect();
            for (i, w) in combiner.star_weights.iter_mut().enumerate() {
                if !top_star.contains(&i) { *w = 0.0; }
            }
            let ss: f64 = combiner.star_weights.iter().sum();
            if ss > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= ss; } }
            let _ = model_names; // suppress unused warning
        }

        let ball_pred = combiner.predict(training_draws, Pool::Balls);
        let star_pred = combiner.predict(training_draws, Pool::Stars);

        // Conviction calculée sur la distribution BRUTE (avant température)
        let conviction = compute_conviction(
            &ball_pred.distribution,
            &star_pred.distribution,
            &ball_pred.spread,
            &star_pred.spread,
        );

        // Température adaptative par conviction si pas de --temperature explicite
        let effective_temp = temperature.unwrap_or_else(|| conviction_temperature(&conviction.verdict));

        let ball_dist = if (effective_temp - 1.0).abs() > 1e-9 {
            apply_temperature(&ball_pred.distribution, effective_temp)
        } else {
            ball_pred.distribution.clone()
        };
        let star_dist = if (effective_temp - 1.0).abs() > 1e-9 {
            apply_temperature(&star_pred.distribution, effective_temp)
        } else {
            star_pred.distribution.clone()
        };

        let actual_score = compute_bayesian_score(
            &test_draw.balls, &test_draw.stars, &ball_dist, &star_dist,
        );

        let filter = StructuralFilter::from_history(training_draws, Pool::Balls);
        let result = generate_suggestions_jackpot(
            &ball_dist, &star_dist, n_suggestions, Some(&filter),
        )?;

        // Vérifier si le tirage réel est dans le top-N
        let in_top_n = result.suggestions.iter().any(|s| {
            let mut sb = s.balls;
            let mut tb = test_draw.balls;
            sb.sort();
            tb.sort();
            let mut ss = s.stars;
            let mut ts = test_draw.stars;
            ss.sort();
            ts.sort();
            sb == tb && ss == ts
        });

        // Matches partiels
        let mut best_tier: Option<u8> = None;
        let mut total_payout = 0.0f64;
        for suggestion in &result.suggestions {
            let (bm, sm) = count_matches(&suggestion.balls, &suggestion.stars, test_draw);
            if let Some(tier_idx) = match_to_tier(bm, sm) {
                let payout = if PRIZE_TIERS[tier_idx].is_parimutuel {
                    0.0
                } else {
                    PRIZE_TIERS[tier_idx].fixed_prize
                };
                total_payout += payout;
                match best_tier {
                    None => best_tier = Some(tier_idx as u8),
                    Some(current) if (tier_idx as u8) < current => best_tier = Some(tier_idx as u8),
                    _ => {}
                }
            }
        }

        rows.push(display::JackpotBacktestRow {
            date: test_draw.date.clone(),
            actual_balls: test_draw.balls,
            actual_stars: test_draw.stars,
            actual_score,
            in_top_n,
            jackpot_probability: result.total_jackpot_probability,
            improvement_factor: result.improvement_factor,
            best_tier,
            total_payout,
            conviction: conviction.overall,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();
    display::display_jackpot_backtest_results(&rows);

    Ok(())
}

fn proximity_score(balls: u8, stars: u8) -> f64 {
    match (balls, stars) {
        (5, 2) => 1000.0,
        (5, 1) => 100.0,
        (5, 0) => 50.0,
        (4, 2) => 30.0,
        (4, 1) => 10.0,
        (4, 0) => 5.0,
        (3, 2) => 3.0,
        (3, 1) => 1.0,
        (3, 0) => 0.5,
        (2, 2) => 0.3,
        _ => 0.0,
    }
}

/// Probabilité qu'une grille aléatoire matche exactement (b, s)
fn random_match_prob(b: u8, s: u8) -> f64 {
    fn comb(n: u64, k: u64) -> u64 {
        if k > n { return 0; }
        let k = k.min(n - k);
        (1..=k).fold(1u64, |acc, i| acc * (n - k + i) / i)
    }
    let p_balls = comb(5, b as u64) as f64 * comb(45, 5 - b as u64) as f64 / comb(50, 5) as f64;
    let p_stars = comb(2, s as u64) as f64 * comb(10, 2 - s as u64) as f64 / comb(12, 2) as f64;
    p_balls * p_stars
}

fn cmd_backtest_3grids(
    draws: &[Draw],
    weights: Option<&EnsembleWeights>,
    last: usize,
    temperature: Option<f64>,
    star_strategy: StarStrategy,
) -> Result<()> {
    println!(
        "Backtest jackpot — 3 grilles sur {} tirages (étoiles : {})\n",
        last, star_strategy.label(),
    );

    let pb = ProgressBar::new(last as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    struct DrawResult {
        date: String,
        best_balls: u8,
        best_stars: u8,
        proximity: f64,
    }

    let mut results: Vec<DrawResult> = Vec::new();
    // Matrice B×S : match_matrix[b][s] = nombre de tirages où le meilleur match est (b, s)
    let mut match_matrix = [[0u32; 3]; 6]; // [0..5 balls][0..2 stars]
    let mut total_proximity = 0.0f64;

    for i in 0..last {
        let test_draw = &draws[i];
        let training_draws = &draws[i + 1..];

        pb.set_message(test_draw.date.to_string());

        let models = all_models();
        let mut combiner = match weights {
            Some(w) => {
                let ball_w: Vec<f64> = models
                    .iter()
                    .map(|m| w.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                let star_w: Vec<f64> = models
                    .iter()
                    .map(|m| w.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                EnsembleCombiner::with_weights(models, ball_w, star_w)
            }
            None => EnsembleCombiner::new(models),
        };

        // Détecter si les poids boules ont du signal (non-uniformes)
        let balls_have_signal = {
            let max_w = combiner.ball_weights.iter().cloned().fold(0.0_f64, f64::max);
            let min_w = combiner.ball_weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::MAX, f64::min);
            (max_w - min_w) > 1e-6
        };

        // MetaPredictor : ajuster les poids boules selon le régime courant
        // Seulement si les boules ont du signal (sinon detailed_ll = bruit)
        if balls_have_signal {
            if let Some(w) = weights {
                if !w.detailed_ll.is_empty() {
                    use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
                    let features = RegimeFeatures::from_draws(training_draws);
                    if let Some(meta) = MetaPredictor::train(training_draws, &w.detailed_ll, 1e-3) {
                        let adjustments = meta.weight_adjustments(&features);
                        for (name, adj) in &adjustments {
                            if let Some(idx) = combiner.models.iter().position(|m| m.name() == name) {
                                if idx < combiner.ball_weights.len() {
                                    combiner.ball_weights[idx] *= adj;
                                }
                            }
                        }
                        let total_b: f64 = combiner.ball_weights.iter().sum();
                        if total_b > 0.0 { combiner.ball_weights.iter_mut().for_each(|w| *w /= total_b); }
                    }
                }
            }
        }

        // Hedge : ajustement multiplicatif basé sur les 20 derniers tirages
        // Boules seulement si signal, étoiles toujours
        let (hedged_ball, hedged_star) = compute_hedge_weights(
            &combiner.models, training_draws,
            &combiner.ball_weights, &combiner.star_weights,
            20, 0.05,
        );
        if balls_have_signal {
            combiner.ball_weights = hedged_ball;
        }
        combiner.star_weights = hedged_star;

        let ball_pred = combiner.predict(training_draws, Pool::Balls);
        let star_pred = combiner.predict(training_draws, Pool::Stars);

        let effective_temp = match temperature {
            Some(t) => t,
            None => {
                // Aligner le backtest avec le comportement production (jackpot mode) :
                // température adaptative basée sur la conviction des distributions
                let conviction = compute_conviction(
                    &ball_pred.distribution,
                    &star_pred.distribution,
                    &ball_pred.spread,
                    &star_pred.spread,
                );
                conviction_temperature(&conviction.verdict)
            }
        };
        let ball_dist = apply_temperature(&ball_pred.distribution, effective_temp);
        let star_dist = apply_temperature(&star_pred.distribution, effective_temp);

        let popularity = PopularityModel::from_history(training_draws);
        let diverse = generate_diverse_grids_with_strategy(
            &ball_dist, &star_dist, training_draws, 3, 42 + i as u64, Some(&popularity), star_strategy,
        );

        // Trouver le meilleur match parmi les 3 grilles (par score de proximité)
        let mut best_balls = 0u8;
        let mut best_stars = 0u8;
        let mut best_prox = 0.0f64;

        for grid in &diverse.grids {
            let (bm, sm) = count_matches(&grid.balls, &grid.stars, test_draw);
            let prox = proximity_score(bm, sm);
            if prox > best_prox || (prox == best_prox && bm + sm > best_balls + best_stars) {
                best_balls = bm;
                best_stars = sm;
                best_prox = prox;
            }
        }

        match_matrix[best_balls as usize][best_stars as usize] += 1;
        total_proximity += best_prox;

        results.push(DrawResult {
            date: test_draw.date.clone(),
            best_balls,
            best_stars,
            proximity: best_prox,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Tableau Date / Meilleur match / Score proximité
    use comfy_table::{Table, Cell, Color};
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_FULL);
    table.set_header(vec!["Date", "Meilleur match", "Proximité"]);

    for r in &results {
        let match_str = format!("{}B+{}S", r.best_balls, r.best_stars);
        let prox_str = format!("{:.1}", r.proximity);
        let color = if r.proximity >= 30.0 {
            Color::Green
        } else if r.proximity >= 1.0 {
            Color::Yellow
        } else if r.proximity > 0.0 {
            Color::Cyan
        } else {
            Color::DarkGrey
        };
        table.add_row(vec![
            Cell::new(&r.date),
            Cell::new(&match_str).fg(color),
            Cell::new(&prox_str).fg(color),
        ]);
    }
    println!("{table}");

    // Matrice B×S : observé vs attendu pour 3 grilles aléatoires
    // Calculer les probas pour le meilleur match parmi 3 grilles aléatoires
    // Pour chaque (b,s), on a besoin de P(best parmi 3 est exactement (b,s))
    // Approche : trier les (b,s) par proximité décroissante, calculer cumulativement

    // Collecter toutes les cellules (b,s) avec leur proba single-grid et proximité
    struct MatchCell { b: u8, s: u8, prob: f64, prox: f64 }
    let mut cells: Vec<MatchCell> = Vec::new();
    for b in 0..=5u8 {
        for s in 0..=2u8 {
            cells.push(MatchCell {
                b, s,
                prob: random_match_prob(b, s),
                prox: proximity_score(b, s),
            });
        }
    }
    // Trier par proximité décroissante, puis b+s décroissant (tiebreaker du backtest)
    cells.sort_by(|a, b| {
        b.prox.partial_cmp(&a.prox).unwrap_or(std::cmp::Ordering::Equal)
            .then((b.b + b.s).cmp(&(a.b + a.s)))
    });

    // Grouper les cellules avec même (proximity, b+s) — vrais ties
    // P(best parmi 3 ≥ groupe) = 1 - (1 - cum_prob)^3
    // P(best = groupe) = P(≥ groupe) - P(> groupe)
    // Au sein d'un groupe, distribuer proportionnellement à single_prob
    let mut expected_matrix = [[0.0f64; 3]; 6];
    let mut cum_prob = 0.0f64;
    let mut prev_p_at_least = 0.0f64;
    let mut ci = 0;
    while ci < cells.len() {
        let prox = cells[ci].prox;
        let bs = cells[ci].b + cells[ci].s;
        let start = ci;
        while ci < cells.len()
            && (cells[ci].prox - prox).abs() < 1e-10
            && cells[ci].b + cells[ci].s == bs
        {
            ci += 1;
        }
        let group = &cells[start..ci];
        let group_single_prob: f64 = group.iter().map(|c| c.prob).sum();
        let p_at_least_group = 1.0 - (1.0 - cum_prob - group_single_prob).powi(3);
        let p_exactly_group = (p_at_least_group - prev_p_at_least).max(0.0);
        if group_single_prob > 0.0 {
            for cell in group {
                let fraction = cell.prob / group_single_prob;
                expected_matrix[cell.b as usize][cell.s as usize] =
                    (p_exactly_group * fraction * last as f64).max(0.0);
            }
        }
        cum_prob += group_single_prob;
        prev_p_at_least = p_at_least_group;
    }

    println!("\n── Matrice Boules × Étoiles (meilleur match parmi 3 grilles) ──");
    println!("         0★         1★         2★");
    for b in (0..=5u8).rev() {
        let mut row = format!("{}B :", b);
        for s in 0..=2u8 {
            let obs = match_matrix[b as usize][s as usize];
            let exp = expected_matrix[b as usize][s as usize];
            row.push_str(&format!("  {:3} ({:5.1})", obs, exp));
        }
        println!("{row}");
    }
    println!("  (observé (attendu aléatoire))");

    // Score de proximité agrégé
    let expected_proximity: f64 = cells.iter().map(|c| {
        let p_exactly = expected_matrix[c.b as usize][c.s as usize] / last as f64;
        p_exactly * c.prox
    }).sum::<f64>() * last as f64;
    let avg_proximity = total_proximity / last as f64;
    let avg_expected = expected_proximity / last as f64;
    let prox_ratio = if avg_expected > 0.0 { avg_proximity / avg_expected } else { 0.0 };

    println!("\n── Score de proximité jackpot ──");
    println!("  Proximité totale      : {:.1}", total_proximity);
    println!("  Proximité moyenne     : {:.2} (attendu aléatoire : {:.2})", avg_proximity, avg_expected);
    println!("  Ratio vs aléatoire    : {:.2}x", prox_ratio);

    // Highlights
    let draws_4plus = results.iter().filter(|r| r.best_balls >= 4).count();
    let draws_3plus2 = results.iter().filter(|r| r.best_balls >= 3 && r.best_stars >= 2).count();
    let draws_2stars = results.iter().filter(|r| r.best_stars >= 2).count();

    println!("\n── Highlights ──");
    println!("  4+ boules matchées    : {}/{}", draws_4plus, last);
    println!("  3+ boules + 2 étoiles : {}/{}", draws_3plus2, last);
    println!("  2 étoiles matchées    : {}/{} ({:.1}%)", draws_2stars, last, 100.0 * draws_2stars as f64 / last as f64);

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

pub(crate) fn cmd_research(conn: &lemillion_db::rusqlite::Connection, tests: &str, window: Option<usize>) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;

    let category = match tests.to_lowercase().as_str() {
        "physical" | "physique" => ResearchCategory::Physical,
        "mathematical" | "mathematique" | "math" => ResearchCategory::Mathematical,
        "informational" | "informationnelle" | "info" => ResearchCategory::Informational,
        "all" | "tout" | "tous" => ResearchCategory::All,
        _ => bail!("Catégorie invalide : '{}'. Valeurs acceptées : physical, mathematical, informational, all", tests),
    };

    let window_desc = match window {
        Some(w) => format!(" (fenêtre: {} derniers tirages)", w),
        None => format!(" ({} tirages)", draws.len()),
    };
    println!("Recherche de biais — catégorie: {}{}", tests, window_desc);

    let report = run_all_research(&draws, category, window);
    display::display_research_report(&report);

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

fn cmd_benchmark(conn: &lemillion_db::rusqlite::Connection, train_size: usize, seed: u64) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let all_draws = fetch_last_draws(conn, n)?;
    let total = all_draws.len();
    if train_size >= total {
        bail!("train_size ({}) >= total draws ({})", train_size, total);
    }

    // Fisher-Yates shuffle with seed
    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng = seed.wrapping_add(1).max(1);
    for i in (1..total).rev() {
        // xorshift64
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        indices.swap(i, j);
    }

    let train_draws: Vec<lemillion_db::models::Draw> = indices[..train_size]
        .iter()
        .map(|&i| all_draws[i].clone())
        .collect();
    let test_draws: Vec<lemillion_db::models::Draw> = indices[train_size..]
        .iter()
        .map(|&i| all_draws[i].clone())
        .collect();

    println!("Benchmark: {} train, {} test (seed={})", train_draws.len(), test_draws.len(), seed);
    println!("{:<20} {:>12} {:>12} {:>12}", "Model", "Ball LL", "Star LL", "Total LL");
    println!("{}", "-".repeat(58));

    let uniform_ball_ll = (1.0f64 / 50.0).ln() * 5.0;
    let uniform_star_ll = (1.0f64 / 12.0).ln() * 2.0;

    // Benchmark physics-based models
    use lemillion_ensemble::models::{ForecastModel, stresa};
    use lemillion_ensemble::models::{mod4, physics};
    let models: Vec<Box<dyn ForecastModel>> = vec![
        Box::new(stresa::StresaChaosModel::default()),
        Box::new(stresa::StresaSgdModel::default()),
        Box::new(stresa::StresaSmcModel::default()),
        Box::new(mod4::Mod4TransitionModel::default()),
        Box::new(physics::PhysicsModel::default()),
    ];

    for model in &models {
        let ball_dist = model.predict(&train_draws, Pool::Balls);
        let star_dist = model.predict(&train_draws, Pool::Stars);

        let ball_ll: f64 = test_draws.iter()
            .map(|d| d.balls.iter()
                .map(|&b| ball_dist[(b - 1) as usize].max(1e-15).ln())
                .sum::<f64>())
            .sum::<f64>() / test_draws.len() as f64;

        let star_ll: f64 = test_draws.iter()
            .map(|d| d.stars.iter()
                .map(|&s| star_dist[(s - 1) as usize].max(1e-15).ln())
                .sum::<f64>())
            .sum::<f64>() / test_draws.len() as f64;

        let total_ll = ball_ll + star_ll;
        println!("{:<20} {:>12.4} {:>12.4} {:>12.4}", model.name(), ball_ll, star_ll, total_ll);
    }

    println!("{}", "-".repeat(58));
    println!("{:<20} {:>12.4} {:>12.4} {:>12.4}",
        "Uniform", uniform_ball_ll, uniform_star_ll, uniform_ball_ll + uniform_star_ll);

    Ok(())
}

