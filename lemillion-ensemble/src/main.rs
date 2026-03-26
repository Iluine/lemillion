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
    EnsembleWeights, STAR_DEFAULT_TEMPERATURE, apply_decorrelation_penalty, apply_family_cap, apply_family_cap_vecs,
    calibrate_model, collect_detailed_ll, compute_correlation_matrix, compute_weights_with_params,
    compute_weights_with_threshold, detect_redundancy, load_weights, save_weights,
};
use lemillion_ensemble::ensemble::consensus::{build_consensus_map, compute_exclusion_set, consensus_score};
use lemillion_ensemble::models::all_models;
use lemillion_ensemble::sampler::{
    apply_temperature, compute_bayesian_score, compute_conviction,
    conviction_temperature_split, conviction_temperature_split_with_skill,
    few_grid_temperature, rqa_temperature_factor, select_optimal_n_grids,
    select_optimal_n_grids_exact, select_optimal_n_grids_sa, generate_suggestions_filtered,
    generate_diverse_grids_with_strategy, generate_suggestions_ev,
    generate_suggestions_jackpot, generate_suggestions_gibbs,
    optimal_grid, BallStarConditioner, CoherenceScorer,
    StarStrategy, StarCoherenceScorer, StructuralFilter,
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
        /// Fenêtres d'analyse pour les boules (séparées par des virgules)
        #[arg(short, long, default_value = "20,30,50,80,100,150,200,300")]
        windows: String,

        /// Fenêtres d'analyse pour les étoiles (séparées par des virgules)
        #[arg(long)]
        star_windows: Option<String>,

        /// Fichier de sortie pour les poids
        #[arg(short, long, default_value = "calibration.json")]
        output: String,

        /// Température pour le scaling des poids (T<1 = sharpening, T>1 = flattening)
        #[arg(short = 't', long, default_value = "1.0")]
        temperature: f64,

        /// Seuil de skill minimum (modèles avec skill <= seuil reçoivent poids 0)
        #[arg(long, default_value = "0.0")]
        min_skill: f64,

        /// Pool à calibrer (balls, stars, both)
        #[arg(long, default_value = "both")]
        pool: String,

        /// Nombre de tirages récents exclus de la calibration pour validation out-of-time
        #[arg(long, default_value = "0")]
        holdout: usize,

        /// Blend recall@K dans le calcul des poids (0.0=LL pur, 1.0=recall pur, 0.3=hybride)
        #[arg(long, default_value = "0.3")]
        recall_blend: f64,
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

        /// Désactiver le stacking (blending level-2)
        #[arg(long)]
        no_stacking: bool,

        /// Activer le neural reranking en mode jackpot
        #[arg(long)]
        neural_rerank: bool,

        /// Stratégie de diversification des étoiles (concentrated/triangular/disjoint)
        #[arg(long, default_value = "triangular")]
        star_strategy: String,

        /// Nombre de grilles à jouer (3-10). Active le mode few-grid avec température forcée.
        #[arg(long)]
        n_grids: Option<usize>,
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

        /// Mode réaliste : 3 ou 10 grilles selon conviction, suivi financier
        #[arg(long)]
        realistic: bool,

        /// Stratégie étoiles pour le backtest 3 grilles (concentrated, triangular, disjoint)
        #[arg(long, default_value = "triangular")]
        star_strategy: String,

        /// Nombre de grilles few-grid pour le backtest jackpot
        #[arg(long)]
        n_grids: Option<usize>,
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

    /// Optimiser les hyperparamètres via BayesOpt
    Optimize {
        /// Nombre de grilles à optimiser
        #[arg(long, default_value = "3")]
        n_grids: usize,

        /// Nombre de tirages récents pour le backtest
        #[arg(short, long, default_value = "20")]
        last: usize,

        /// Nombre de suggestions par tirage
        #[arg(short, long, default_value = "5000")]
        suggestions: usize,

        /// Nombre d'itérations BayesOpt
        #[arg(short, long, default_value = "50")]
        iterations: usize,

        /// Fichier de calibration
        #[arg(short, long, default_value = "calibration.json")]
        calibration: String,

        /// Fichier de sortie pour les hyperparamètres optimisés
        #[arg(short, long, default_value = "hyperparams.json")]
        output: String,

        /// Seed pour la reproductibilité
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Mode interactif (REPL)
    Interactive,

    /// Évaluation honnête sur holdout (tirages jamais vus pendant la calibration)
    HoldoutEval {
        /// Fichier de calibration
        #[arg(short, long, default_value = "calibration.json")]
        calibration: String,

        /// Nombre de tirages holdout (les plus récents)
        #[arg(short = 'n', long, default_value = "50")]
        holdout: usize,

        /// Nombre de suggestions par tirage
        #[arg(short, long, default_value = "5000")]
        suggestions: usize,

        /// Nombre de resamples bootstrap pour les intervalles de confiance
        #[arg(long, default_value = "1000")]
        bootstrap: usize,

        /// Mode few-grids (N grilles)
        #[arg(long)]
        n_grids: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = db_path();
    let conn = open_db(&path)?;
    migrate(&conn)?;

    match cli.command {
        Command::Calibrate { windows, star_windows, output, temperature, min_skill, pool, holdout, recall_blend } => cmd_calibrate(&conn, &windows, star_windows.as_deref(), &output, temperature, min_skill, &pool, holdout, recall_blend),
        Command::Weights { calibration } => cmd_weights(&calibration),
        Command::Predict { calibration, suggestions, seed, oversample, min_diff, temperature, jackpot, jackpot_mode, no_filter, top_models, no_meta_predictor, no_hedge, agreement_boost, no_stacking, neural_rerank, star_strategy, n_grids } => cmd_predict(&conn, &calibration, suggestions, seed, oversample, min_diff, temperature, jackpot, jackpot_mode, no_filter, top_models, no_meta_predictor, no_hedge, agreement_boost, no_stacking, neural_rerank, &star_strategy, n_grids),
        Command::History { last } => cmd_history(&conn, last),
        Command::Compare { numbers, date, calibration } => cmd_compare(&conn, &numbers, date.as_deref(), &calibration),
        Command::AddDraw { args } => cmd_add_draw(&conn, &args),
        Command::FixDraw => cmd_fix_draw(&conn),
        Command::Rebuild => cmd_rebuild(&conn),
        Command::Backtest { last, suggestions, oversample, calibration, temperature, sweep_temperature, jackpot_mode, top_models, backtest_3_grids, realistic, star_strategy, n_grids } => cmd_backtest(&conn, last, suggestions, oversample, &calibration, temperature, sweep_temperature, jackpot_mode, top_models, backtest_3_grids, realistic, &star_strategy, n_grids),
        Command::Analyze => cmd_analyze(&conn),
        Command::Coverage { tickets, jackpot, seed } => cmd_coverage(&conn, tickets, jackpot, seed),
        Command::Research { tests, window } => cmd_research(&conn, &tests, window),
        Command::Benchmark { train, seed } => cmd_benchmark(&conn, train, seed),
        Command::Optimize { n_grids, last, suggestions, iterations, calibration, output, seed } => cmd_optimize(&conn, n_grids, last, suggestions, iterations, &calibration, &output, seed),
        Command::Interactive => interactive::run_interactive(&conn),
        Command::HoldoutEval { calibration, holdout, suggestions, bootstrap, n_grids } => cmd_holdout_eval(&conn, &calibration, holdout, suggestions, bootstrap, n_grids),
    }
}

pub(crate) fn cmd_calibrate(conn: &lemillion_db::rusqlite::Connection, windows_str: &str, star_windows_str: Option<&str>, output: &str, temperature: f64, min_skill: f64, pool_str: &str, holdout: usize, recall_blend: f64) -> Result<()> {
    use lemillion_ensemble::ensemble::calibration::DEFAULT_STAR_WINDOWS;

    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let ball_windows: Vec<usize> = windows_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .context("Format de fenêtres boules invalide")?;

    let star_windows: Vec<usize> = if let Some(sw) = star_windows_str {
        sw.split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<_, _>>()
            .context("Format de fenêtres étoiles invalide")?
    } else {
        DEFAULT_STAR_WINDOWS.to_vec()
    };

    let calibrate_balls = pool_str == "both" || pool_str == "balls";
    let calibrate_stars = pool_str == "both" || pool_str == "stars";

    let all_draws = fetch_last_draws(conn, n)?;

    // Holdout: exclure les N tirages les plus récents de la calibration
    let (holdout_draws, draws) = if holdout > 0 && holdout < all_draws.len() {
        println!("Holdout: {} tirages les plus récents exclus de la calibration", holdout);
        let (ho, cal) = all_draws.split_at(holdout);
        (ho.to_vec(), cal.to_vec())
    } else {
        (vec![], all_draws)
    };
    let models = all_models();

    // Load existing weights if doing partial calibration
    let existing_weights = if !calibrate_balls || !calibrate_stars {
        load_weights(&PathBuf::from(output)).ok()
    } else {
        None
    };

    let pool_label = match (calibrate_balls, calibrate_stars) {
        (true, true) => "boules+étoiles",
        (true, false) => "boules uniquement",
        (false, true) => "étoiles uniquement",
        _ => bail!("--pool doit être balls, stars ou both"),
    };
    println!("Calibration de {} modèles sur {} tirages [{}]",
        models.len(), draws.len(), pool_label);
    if calibrate_balls {
        println!("  Fenêtres boules : {:?}", ball_windows);
    }
    if calibrate_stars {
        println!("  Fenêtres étoiles : {:?}", star_windows);
    }

    use rayon::prelude::*;

    let total_steps = models.len() as u64 * (calibrate_balls as u64 + calibrate_stars as u64);
    let pb = ProgressBar::new(total_steps);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("=> "));

    let ball_calibrations: Vec<_> = if calibrate_balls {
        let cals: Vec<_> = models.par_iter().map(|model| {
            let cal = calibrate_model(model.as_ref(), &draws, &ball_windows, Pool::Balls);
            pb.inc(1);
            cal
        }).collect();
        cals
    } else {
        // Build from existing weights
        models.iter().map(|model| {
            let existing = existing_weights.as_ref()
                .and_then(|w| w.calibrations.iter().find(|c| c.model_name == model.name()));
            lemillion_ensemble::ensemble::calibration::ModelCalibration {
                model_name: model.name().to_string(),
                results: vec![],
                best_window: 100,
                best_sparse: false,
                best_ll: existing.map(|c| c.best_ll).unwrap_or(f64::NEG_INFINITY),
                best_n_tests: existing.map(|c| c.best_n_tests).unwrap_or(0),
                best_recall: existing.map(|c| c.best_recall).unwrap_or(0.0),
            }
        }).collect()
    };

    let star_calibrations: Vec<_> = if calibrate_stars {
        let cals: Vec<_> = models.par_iter().map(|model| {
            let cal = calibrate_model(model.as_ref(), &draws, &star_windows, Pool::Stars);
            pb.inc(1);
            cal
        }).collect();
        cals
    } else {
        models.iter().map(|model| {
            let existing = existing_weights.as_ref()
                .and_then(|w| w.calibrations.iter().find(|c| c.model_name == model.name()));
            lemillion_ensemble::ensemble::calibration::ModelCalibration {
                model_name: model.name().to_string(),
                results: vec![],
                best_window: 100,
                best_sparse: false,
                best_ll: existing.map(|c| c.best_ll).unwrap_or(f64::NEG_INFINITY),
                best_n_tests: existing.map(|c| c.best_n_tests).unwrap_or(0),
                best_recall: existing.map(|c| c.best_recall).unwrap_or(0.0),
            }
        }).collect()
    };

    pb.finish_with_message("Calibration terminée");

    // Afficher les résultats
    if calibrate_balls {
        println!("\n── Boules ──");
        display::display_calibration_results(&ball_calibrations, &ball_windows);
    }
    if calibrate_stars {
        println!("\n── Étoiles ──");
        display::display_calibration_results(&star_calibrations, &star_windows);
    }

    if calibrate_balls {
        display::display_calibration_chart(&ball_calibrations, &ball_windows);
    }

    // Collecter les detailed_ll pour le meta-predictor
    println!("\nCollecte des LL détaillés pour le méta-apprentissage...");
    let detailed_ll: Vec<(String, Vec<f64>)> = if calibrate_balls {
        models.par_iter()
            .zip(ball_calibrations.par_iter())
            .map(|(model, ball_cal)| {
                let strategy = model.sampling_strategy();
                let lls = collect_detailed_ll(
                    model.as_ref(), &draws, ball_cal.best_window, Pool::Balls, strategy,
                );
                (model.name().to_string(), lls)
            })
            .collect()
    } else {
        existing_weights.as_ref().map(|w| w.detailed_ll.clone()).unwrap_or_default()
    };

    let star_detailed_ll: Vec<(String, Vec<f64>)> = if calibrate_stars {
        models.par_iter()
            .zip(star_calibrations.par_iter())
            .map(|(model, star_cal)| {
                let strategy = model.sampling_strategy();
                let lls = collect_detailed_ll(
                    model.as_ref(), &draws, star_cal.best_window, Pool::Stars, strategy,
                );
                (model.name().to_string(), lls)
            })
            .collect()
    } else {
        vec![]
    };

    // Détection de redondance (boules)
    let redundancies = if calibrate_balls {
        let r = detect_redundancy(&detailed_ll, 0.75);
        if !r.is_empty() {
            println!("\n── Redondance boules (corrélation LL > 0.75) ──");
            for rd in &r {
                let penalty_pct = if rd.correlation > 0.80 {
                    let p = (1.0 - 0.5 * (rd.correlation - 0.80) / 0.20).max(0.30);
                    format!(" [×{:.2}]", p)
                } else {
                    String::new()
                };
                println!("  {} <-> {} : {:.3}{}", rd.model_a, rd.model_b, rd.correlation, penalty_pct);
            }
        }
        r
    } else {
        vec![]
    };

    // Détection de redondance (étoiles)
    let star_redundancies = if calibrate_stars {
        let r = detect_redundancy(&star_detailed_ll, 0.75);
        if !r.is_empty() {
            println!("\n── Redondance étoiles (corrélation LL > 0.75) ──");
            for rd in &r {
                let penalty_pct = if rd.correlation > 0.80 {
                    let p = (1.0 - 0.5 * (rd.correlation - 0.80) / 0.20).max(0.30);
                    format!(" [×{:.2}]", p)
                } else {
                    String::new()
                };
                println!("  {} <-> {} : {:.3}{}", rd.model_a, rd.model_b, rd.correlation, penalty_pct);
            }
        }
        r
    } else {
        vec![]
    };

    // Calculer et afficher les poids
    let star_temp = if (temperature - 1.0).abs() < 1e-9 {
        STAR_DEFAULT_TEMPERATURE
    } else {
        temperature
    };
    println!("\nTempérature boules : {:.2}, étoiles : {:.2}, seuil skill : {:.4}", temperature, star_temp, min_skill);

    let mut ball_weights = if calibrate_balls {
        if recall_blend > 0.001 {
            use lemillion_ensemble::ensemble::calibration::compute_weights_with_recall;
            println!("  → Mode recall-blend: {:.0}% recall + {:.0}% LL", recall_blend * 100.0, (1.0 - recall_blend) * 100.0);
            compute_weights_with_recall(&ball_calibrations, Pool::Balls, temperature, recall_blend)
        } else {
            compute_weights_with_threshold(&ball_calibrations, Pool::Balls, temperature, min_skill)
        }
    } else {
        existing_weights.as_ref().map(|w| w.ball_weights.clone()).unwrap_or_else(|| {
            models.iter().map(|m| (m.name().to_string(), 1.0 / models.len() as f64)).collect()
        })
    };

    let mut star_weights = if calibrate_stars {
        if recall_blend > 0.001 {
            use lemillion_ensemble::ensemble::calibration::compute_weights_with_recall;
            compute_weights_with_recall(&star_calibrations, Pool::Stars, star_temp, recall_blend)
        } else {
            compute_weights_with_threshold(&star_calibrations, Pool::Stars, star_temp, min_skill)
        }
    } else {
        existing_weights.as_ref().map(|w| w.star_weights.clone()).unwrap_or_else(|| {
            models.iter().map(|m| (m.name().to_string(), 1.0 / models.len() as f64)).collect()
        })
    };

    // Décorrélation continue : pénaliser les modèles corrélés
    apply_decorrelation_penalty(&mut ball_weights, &redundancies, 0.60, 0.6, 0.10);
    apply_decorrelation_penalty(&mut star_weights, &star_redundancies, 0.60, 0.6, 0.10);

    // Cap familial : éviter la monopolisation par groupes corrélés
    let te_family: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
    let stresa_family: &[&str] = &["StresaSGD", "StresaChaos"];
    let families: Vec<(&[&str], f64)> = vec![
        (te_family, 0.20),
        (stresa_family, 0.15),
    ];
    apply_family_cap(&mut ball_weights, &families);
    apply_family_cap(&mut star_weights, &families);

    // v23b: Markowitz mean-variance optimization for ball weights
    if calibrate_balls && !detailed_ll.is_empty() {
        use lemillion_ensemble::ensemble::calibration::compute_markowitz_weights;
        let model_names: Vec<String> = models.iter().map(|m| m.name().to_string()).collect();
        let base_w: Vec<f64> = ball_weights.iter().map(|(_, w)| *w).collect();
        let markowitz_w = compute_markowitz_weights(&base_w, &model_names, &detailed_ll, 5.0);
        println!("\n  Markowitz portfolio optimization appliqué (risk_aversion=5.0)");
        for (i, (name, w)) in ball_weights.iter_mut().enumerate() {
            *w = markowitz_w[i];
        }
        // Renormalize
        let total: f64 = ball_weights.iter().map(|(_, w)| *w).sum();
        if total > 0.0 { for (_, w) in ball_weights.iter_mut() { *w /= total; } }
    }

    // Stacking : collecter données + entraîner
    println!("\nEntraînement du stacking...");
    let stacking_balls = if calibrate_balls {
        use lemillion_ensemble::ensemble::stacking::{collect_stacking_data, train_stacking};
        let best_window = ball_calibrations.iter()
            .max_by(|a, b| a.best_ll.partial_cmp(&b.best_ll).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.best_window)
            .unwrap_or(100);
        let data = collect_stacking_data(&models, &draws, Pool::Balls, best_window);
        let result = train_stacking(&data, Pool::Balls, 0.01);
        if result.is_some() {
            println!("  Stacking boules : entraîné ({} points)", data.len());
        } else {
            println!("  Stacking boules : pas assez de données");
        }
        result
    } else {
        existing_weights.as_ref().and_then(|w| w.stacking_balls.clone())
    };

    let stacking_stars = if calibrate_stars {
        use lemillion_ensemble::ensemble::stacking::{collect_stacking_data, train_stacking};
        let best_window = star_calibrations.iter()
            .max_by(|a, b| a.best_ll.partial_cmp(&b.best_ll).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.best_window)
            .unwrap_or(200);
        let data = collect_stacking_data(&models, &draws, Pool::Stars, best_window);
        let result = train_stacking(&data, Pool::Stars, 0.01);
        if result.is_some() {
            println!("  Stacking étoiles : entraîné ({} points)", data.len());
        } else {
            println!("  Stacking étoiles : pas assez de données");
        }
        result
    } else {
        existing_weights.as_ref().and_then(|w| w.stacking_stars.clone())
    };

    // v7: Compute full correlation matrix for decorrelation-aware pool
    let correlation_matrix = if !detailed_ll.is_empty() {
        let cm = compute_correlation_matrix(&detailed_ll, 0.3);
        if !cm.is_empty() {
            println!("\n── Matrice de corrélation boules ({} paires > 0.3) ──", cm.len());
            for (a, b, c) in cm.iter().take(10) {
                println!("  {} × {} : {:.3}", a, b, c);
            }
            if cm.len() > 10 { println!("  ... ({} paires au total)", cm.len()); }
        }
        cm
    } else {
        Vec::new()
    };
    let star_correlation_matrix = if !star_detailed_ll.is_empty() {
        compute_correlation_matrix(&star_detailed_ll, 0.3)
    } else {
        Vec::new()
    };

    // v16: Optimize beta-transform and temperature by NLL grid-search
    println!("\nOptimisation post-pooling (beta-transform + température)...");
    let (beta_balls, beta_stars, optimal_t_balls, optimal_t_stars) = {
        let opt_combiner = EnsembleCombiner::with_weights(
            all_models(),
            ball_weights.iter().map(|(_, w)| *w).collect(),
            star_weights.iter().map(|(_, w)| *w).collect(),
        );
        lemillion_ensemble::ensemble::calibration::optimize_post_pooling(
            &opt_combiner, &draws, 50,
        )
    };
    if let Some((a, b)) = beta_balls {
        println!("  Beta boules : alpha={:.2}, beta={:.2}", a, b);
    } else {
        println!("  Beta boules : identité (1.0, 1.0)");
    }
    if let Some((a, b)) = beta_stars {
        println!("  Beta étoiles : alpha={:.2}, beta={:.2}", a, b);
    } else {
        println!("  Beta étoiles : identité (1.0, 1.0)");
    }
    if let Some(t) = optimal_t_balls {
        println!("  Température boules optimale : {:.2}", t);
    }
    if let Some(t) = optimal_t_stars {
        println!("  Température étoiles optimale : {:.2}", t);
    }

    // v17: Optimize coherence weights, stacking blend, and online blend
    println!("\nOptimisation v17 (cohérence, stacking blend, online blend)...");
    let (coherence_ball_weight, coherence_star_weight, stacking_blend_balls, stacking_blend_stars, online_ewma_alpha, online_window) = {
        let opt_combiner = EnsembleCombiner::with_weights(
            all_models(),
            ball_weights.iter().map(|(_, w)| *w).collect(),
            star_weights.iter().map(|(_, w)| *w).collect(),
        );

        let (cw_b, cw_s) = lemillion_ensemble::ensemble::calibration::optimize_coherence_weights(
            &opt_combiner, &draws, 30,
        );
        println!("  Coherence balls: {:.0}, stars: {:.0}", cw_b, cw_s);

        let (sb_b, sb_s) = lemillion_ensemble::ensemble::calibration::optimize_stacking_blend(
            &opt_combiner, &draws,
            stacking_balls.as_ref(), stacking_stars.as_ref(), 30,
        );
        println!("  Stacking blend balls: {:.1}, stars: {:.1}", sb_b, sb_s);

        let (oa, ow) = lemillion_ensemble::ensemble::calibration::optimize_online_blend(
            &opt_combiner, &draws, 30,
        );
        println!("  Online blend alpha: {:.2}, window: {}", oa, ow);

        (Some(cw_b), Some(cw_s), Some(sb_b), Some(sb_s), Some(oa), Some(ow))
    };

    // H1: Compute Brier + CRPS diagnostics (purely informational, does NOT affect weights)
    println!("\nCalcul des diagnostics Brier/CRPS...");
    let diagnostics = {
        use lemillion_ensemble::ensemble::calibration::compute_all_diagnostics;
        let diag_models = all_models();
        let mut all_diags = Vec::new();
        if calibrate_balls {
            let ball_diags = compute_all_diagnostics(&diag_models, &ball_calibrations, &draws, Pool::Balls);
            all_diags.extend(ball_diags);
        }
        if calibrate_stars {
            let star_diags = compute_all_diagnostics(&diag_models, &star_calibrations, &draws, Pool::Stars);
            all_diags.extend(star_diags);
        }
        println!("  {} diagnostics calculés", all_diags.len());
        all_diags
    };

    let mut ensemble_weights = EnsembleWeights {
        ball_weights,
        star_weights,
        calibrations: ball_calibrations.into_iter().chain(star_calibrations).collect(),
        detailed_ll,
        star_detailed_ll,
        stacking_balls,
        stacking_stars,
        correlation_matrix,
        star_correlation_matrix,
        beta_balls,
        beta_stars,
        optimal_t_balls,
        optimal_t_stars,
        coherence_ball_weight,
        coherence_star_weight,
        stacking_blend_balls,
        stacking_blend_stars,
        online_ewma_alpha,
        online_window,
        diagnostics,
        conformal_ball_max_ranks: Vec::new(),
        conformal_star_max_ranks: Vec::new(),
        coherence_stats: None,
    };

    display::display_weights(&ensemble_weights);

    // Validation out-of-time sur les tirages holdout
    if !holdout_draws.is_empty() {
        println!("\n── Validation Out-of-Time ({} tirages holdout) ──", holdout_draws.len());
        let uniform_ball_ll = lemillion_ensemble::ensemble::calibration::uniform_log_likelihood(Pool::Balls);
        let uniform_star_ll = lemillion_ensemble::ensemble::calibration::uniform_log_likelihood(Pool::Stars);

        let combiner = EnsembleCombiner::with_weights(
            all_models(),
            ensemble_weights.ball_weights.iter().map(|(_, w)| *w).collect(),
            ensemble_weights.star_weights.iter().map(|(_, w)| *w).collect(),
        );

        let mut ball_ll_sum = 0.0f64;
        let mut star_ll_sum = 0.0f64;
        let mut n_valid = 0;

        for (i, ho_draw) in holdout_draws.iter().enumerate() {
            // Train on everything after this holdout draw
            let mut context_draws = holdout_draws[i + 1..].to_vec();
            context_draws.extend_from_slice(&draws);

            if context_draws.len() < 30 { continue; }

            let ball_dist = combiner.predict(&context_draws, Pool::Balls).distribution;
            let star_dist = combiner.predict(&context_draws, Pool::Stars).distribution;

            let ball_ll: f64 = ho_draw.balls.iter()
                .map(|&b| ball_dist[(b - 1) as usize].max(1e-15).ln())
                .sum();
            let star_ll: f64 = ho_draw.stars.iter()
                .map(|&s| star_dist[(s - 1) as usize].max(1e-15).ln())
                .sum();

            ball_ll_sum += ball_ll;
            star_ll_sum += star_ll;
            n_valid += 1;
        }

        if n_valid > 0 {
            let avg_ball_ll = ball_ll_sum / n_valid as f64;
            let avg_star_ll = star_ll_sum / n_valid as f64;
            let ball_skill = avg_ball_ll - uniform_ball_ll;
            let star_skill = avg_star_ll - uniform_star_ll;
            println!("  Boules: LL={:.4} (skill vs uniforme: {:+.4} nats)", avg_ball_ll, ball_skill);
            println!("  Étoiles: LL={:.4} (skill vs uniforme: {:+.4} nats)", avg_star_ll, star_skill);
            if ball_skill > 0.0 { println!("  ✓ Boules: signal positif confirmé hors échantillon"); }
            else { println!("  ✗ Boules: pas de signal hors échantillon"); }
            if star_skill > 0.0 { println!("  ✓ Étoiles: signal positif confirmé hors échantillon"); }
            else { println!("  ✗ Étoiles: pas de signal hors échantillon"); }
        }
    }

    // Diagnostic entropie pour modèles à 0% poids boules
    {
        let test_draws = &draws[..draws.len().min(200)];
        println!("\n--- Diagnostic entropie (modèles 0% boules) ---");
        for model in &models {
            let ball_w = ensemble_weights.ball_weights.iter()
                .find(|(n, _)| n == model.name())
                .map(|(_, w)| *w)
                .unwrap_or(0.0);
            if ball_w > 0.001 { continue; }
            let dist = model.predict(test_draws, Pool::Balls);
            let h: f64 = dist.iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();
            let h_max = (50.0_f64).ln();
            let h_ratio = h / h_max;
            if h_ratio > 0.95 {
                println!("  {} : H/H_max = {:.4} (quasi-uniforme, smoothing trop élevé ?)", model.name(), h_ratio);
            }
        }
    }

    // v22: Collect conformal max-rank scores for K selection
    {
        println!("\nCollecte des rangs conformes pour K selection...");
        let combiner = EnsembleCombiner::with_weights(
            all_models(),
            ensemble_weights.ball_weights.iter().map(|(_, w)| *w).collect(),
            ensemble_weights.star_weights.iter().map(|(_, w)| *w).collect(),
        );

        let n_test = draws.len().min(300).saturating_sub(30);
        let stride = (n_test / 200).max(1);
        let mut ball_max_ranks = Vec::new();
        let mut star_max_ranks = Vec::new();

        for i in (0..n_test).step_by(stride) {
            let test_draw = &draws[i];
            let training = &draws[i + 1..];
            if training.len() < 30 { continue; }

            let ball_dist = combiner.predict(training, Pool::Balls).distribution;
            let star_dist = combiner.predict(training, Pool::Stars).distribution;

            // Rank balls by probability (descending)
            let mut ball_indexed: Vec<(usize, f64)> = ball_dist.iter().copied().enumerate().collect();
            ball_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let max_ball_rank = test_draw.balls.iter().map(|&b| {
                ball_indexed.iter().position(|(idx, _)| *idx == (b as usize - 1)).unwrap_or(49) + 1
            }).max().unwrap_or(50);
            ball_max_ranks.push(max_ball_rank);

            // Rank stars
            let mut star_indexed: Vec<(usize, f64)> = star_dist.iter().copied().enumerate().collect();
            star_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let max_star_rank = test_draw.stars.iter().map(|&s| {
                star_indexed.iter().position(|(idx, _)| *idx == (s as usize - 1)).unwrap_or(11) + 1
            }).max().unwrap_or(12);
            star_max_ranks.push(max_star_rank);
        }

        if !ball_max_ranks.is_empty() {
            let mut sorted = ball_max_ranks.clone();
            sorted.sort();
            let p95 = sorted[(0.95 * sorted.len() as f64) as usize];
            let p50 = sorted[sorted.len() / 2];
            println!("  Rangs max boules: P50={}, P95={} (sur {} test points)", p50, p95, ball_max_ranks.len());
            println!("  → K conformal recommandé (95%): {}", p95);
        }

        ensemble_weights.conformal_ball_max_ranks = ball_max_ranks;
        ensemble_weights.conformal_star_max_ranks = star_max_ranks;
    }

    // v23: Pre-compute and store coherence statistics on full history
    {
        println!("Pré-calcul des statistiques de cohérence (stabilisation v23)...");
        let coherence = CoherenceScorer::from_history(&draws, Pool::Balls);
        let stats = coherence.to_stats();
        println!("  Paires: {}, Triplets: {}", stats.pair_freq.len(), stats.triplet_freq.len());
        println!("  Sum: {:.1} ± {:.1}, Spread: {:.1} ± {:.1}", stats.mean_sum, stats.std_sum, stats.mean_spread, stats.std_spread);
        ensemble_weights.coherence_stats = Some(stats);
    }

    // Sauvegarder
    let output_path = PathBuf::from(output);
    save_weights(&ensemble_weights, &output_path)?;
    println!("\nPoids sauvegardés dans : {}", output);

    // Entraîner le neural scorer
    println!("\nEntraînement du Neural Scorer...");
    let ns = lemillion_ensemble::models::neural_scorer::NeuralScorer::train(&draws, 42);
    let ns_path = std::path::PathBuf::from("neural_scorer.json");
    ns.save(&ns_path)?;
    println!("Neural Scorer sauvegardé dans : {}", ns_path.display());

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
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_predict(conn: &lemillion_db::rusqlite::Connection, calibration_path: &str, n_suggestions: usize, seed: Option<u64>, oversample: usize, min_diff: usize, temperature: Option<f64>, jackpot: f64, jackpot_mode: bool, no_filter: bool, top_models: usize, no_meta_predictor: bool, no_hedge: bool, agreement_boost_strength: f64, no_stacking: bool, neural_rerank: bool, star_strategy_str: &str, n_grids: Option<usize>) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;

    // v20: Load optimized hyperparams if available
    let hyper = lemillion_ensemble::sampler::HyperParams::load(std::path::Path::new("hyperparams.json"));
    let has_optimized = std::path::Path::new("hyperparams.json").exists();
    if has_optimized {
        println!("(Hyperparams optimisés chargés : T=({:.2},{:.2}) CW=({:.1},{:.1}) η={:.3})",
            hyper.t_balls, hyper.t_stars, hyper.coherence_weight, hyper.star_coherence_weight, hyper.hedge_eta);
    }

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

    // MetaPredictor : ajuster les poids boules selon le régime courant
    // Seulement si les boules ont du signal (sinon detailed_ll = bruit)
    if !no_meta_predictor
        && balls_have_signal
        && let Ok(ref w) = weights
        && !w.detailed_ll.is_empty()
    {
        use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
        let features = RegimeFeatures::from_draws(&draws);
        if let Some(meta) = MetaPredictor::train(&draws, &w.detailed_ll, 1.0) {
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

    // MetaPredictor ÉTOILES : ajuster les poids étoiles selon le régime courant
    if !no_meta_predictor
        && let Ok(ref w) = weights
        && !w.star_detailed_ll.is_empty()
    {
        use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
        let features = RegimeFeatures::from_draws(&draws);
        if let Some(meta) = MetaPredictor::train(&draws, &w.star_detailed_ll, 1.0) {
            let adjustments = meta.weight_adjustments(&features);
            let mut n_adjusted = 0;
            for (name, adj) in &adjustments {
                if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                    && idx < combiner.star_weights.len()
                {
                    combiner.star_weights[idx] *= adj;
                    n_adjusted += 1;
                }
            }
            let total: f64 = combiner.star_weights.iter().sum();
            if total > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= total; } }
            println!("(MetaPredictor étoiles : {} poids ajustés)", n_adjusted);
        }
    }

    // Hedge weights : ajustement multiplicatif réactif
    // Boules seulement si signal, étoiles toujours
    if !no_hedge {
        let hedge_eta = hyper.hedge_eta;
        let (hedged_ball, hedged_star) = compute_hedge_weights(
            &combiner.models, &draws,
            &combiner.ball_weights, &combiner.star_weights,
            100,       // n_recent : 100 derniers tirages
            hedge_eta, // eta : learning rate réactif (from hyperparams)
        );
        if balls_have_signal {
            combiner.ball_weights = hedged_ball;
        }
        combiner.star_weights = hedged_star;
        println!("(Hedge weights appliqués sur les 100 derniers tirages, η={:.2}{})",
            hedge_eta, if !balls_have_signal { " [étoiles seulement]" } else { "" });

        // Re-apply family cap after hedge to prevent TE/Stresa monopolization
        let te_family: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
        let stresa_family: &[&str] = &["StresaSGD", "StresaChaos"];
        let families: Vec<(&[&str], f64)> = vec![(te_family, 0.20), (stresa_family, 0.15)];
        let model_names: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
        let mut ball_w: Vec<(String, f64)> = model_names.iter().zip(combiner.ball_weights.iter()).map(|(n, w)| (n.clone(), *w)).collect();
        let mut star_w: Vec<(String, f64)> = model_names.iter().zip(combiner.star_weights.iter()).map(|(n, w)| (n.clone(), *w)).collect();
        apply_family_cap(&mut ball_w, &families);
        apply_family_cap(&mut star_w, &families);
        combiner.ball_weights = ball_w.iter().map(|(_, w)| *w).collect();
        combiner.star_weights = star_w.iter().map(|(_, w)| *w).collect();
    }

    // Filtrer les top modèles si demandé
    if top_models > 0 {
        let model_names: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
        filter_top_models(&mut combiner.ball_weights, &mut combiner.star_weights, &model_names, top_models);
    }

    let effective_boost = if agreement_boost_strength > 0.0 {
        agreement_boost_strength
    } else if jackpot_mode {
        0.20
    } else {
        0.0
    };
    // Stacking : utiliser les poids de stacking si disponibles
    let use_stacking = !no_stacking && weights.as_ref().map(|w| w.stacking_balls.is_some() || w.stacking_stars.is_some()).unwrap_or(false);

    // v7: Decorrelation-aware prediction — utiliser la matrice de corrélation si disponible
    let has_corr_matrix = weights.as_ref().map(|w| !w.correlation_matrix.is_empty()).unwrap_or(false);

    // Helper closure: predict with decorrelation when available, otherwise standard
    let predict_pool = |pool: Pool| -> lemillion_ensemble::ensemble::EnsemblePrediction {
        if has_corr_matrix {
            let w = weights.as_ref().unwrap();
            let corr = match pool {
                Pool::Balls => &w.correlation_matrix,
                Pool::Stars => if w.star_correlation_matrix.is_empty() { &w.correlation_matrix } else { &w.star_correlation_matrix },
            };
            combiner.predict_decorrelated(&draws, pool, corr, 0.60)
        } else if effective_boost > 0.0 {
            combiner.predict_with_agreement_boost(&draws, pool, effective_boost)
        } else {
            combiner.predict(&draws, pool)
        }
    };

    let (mut ball_pred, mut star_pred) = if use_stacking {
        let w = weights.as_ref().unwrap();
        let blend_b = w.stacking_blend_balls.unwrap_or(0.6);
        let bp = if let Some(ref sw) = w.stacking_balls {
            println!("(Stacking boules : blend {:.0}% stacked + {:.0}% weighted)", blend_b * 100.0, (1.0 - blend_b) * 100.0);
            combiner.predict_stacked(&draws, Pool::Balls, sw, blend_b)
        } else {
            predict_pool(Pool::Balls)
        };
        let blend_s = w.stacking_blend_stars.unwrap_or(0.6);
        let sp = if let Some(ref sw) = w.stacking_stars {
            println!("(Stacking étoiles : blend {:.0}% stacked + {:.0}% weighted)", blend_s * 100.0, (1.0 - blend_s) * 100.0);
            combiner.predict_stacked(&draws, Pool::Stars, sw, blend_s)
        } else {
            predict_pool(Pool::Stars)
        };
        (bp, sp)
    } else {
        if has_corr_matrix {
            println!("(Decorrelation-aware pool : pénalisation des modèles corrélés > 0.5)");
        }
        if effective_boost > 0.0 && !has_corr_matrix {
            println!("(Agreement boost : {:.2}{})", effective_boost, if agreement_boost_strength == 0.0 && jackpot_mode { " [auto jackpot]" } else { "" });
        }
        (predict_pool(Pool::Balls), predict_pool(Pool::Stars))
    };

    // v15/v17: Online/offline blend — adaptation rapide aux changements récents
    {
        use lemillion_ensemble::ensemble::online::online_offline_blend_with_alpha;
        let online_alpha = weights.as_ref().ok().and_then(|w| w.online_ewma_alpha).unwrap_or(0.15);
        let online_window = weights.as_ref().ok().and_then(|w| w.online_window).unwrap_or(8);
        let blended_balls = online_offline_blend_with_alpha(&ball_pred.distribution, &draws, Pool::Balls, online_window, online_alpha);
        // Stars: skip online blend (v15b — 12 values too few for stable EWMA)
        ball_pred.distribution = blended_balls;
    }

    // v16: Beta-transform post-pooling (Ranjan & Gneiting 2010)
    if let Ok(ref w) = weights {
        if let Some((alpha, beta)) = w.beta_balls {
            lemillion_ensemble::ensemble::beta_transform(&mut ball_pred.distribution, alpha, beta);
            println!("(Beta-transform boules : α={:.2}, β={:.2})", alpha, beta);
        }
        if let Some((alpha, beta)) = w.beta_stars {
            lemillion_ensemble::ensemble::beta_transform(&mut star_pred.distribution, alpha, beta);
            println!("(Beta-transform étoiles : α={:.2}, β={:.2})", alpha, beta);
        }
    }

    // Afficher les distributions
    display::display_forecast(&ball_pred, Pool::Balls);
    display::display_forecast(&star_pred, Pool::Stars);

    // Consensus maps
    let ball_consensus = build_consensus_map(&ball_pred, Pool::Balls);
    let star_consensus = build_consensus_map(&star_pred, Pool::Stars);
    display::display_consensus(&ball_consensus, Pool::Balls);
    display::display_consensus(&star_consensus, Pool::Stars);

    // E6: Conformal prediction / abstention recommendation
    let conformal = lemillion_ensemble::sampler::conformal_prediction(&ball_pred.distribution, &star_pred.distribution, 0.10);
    println!("\nConformal Prediction (alpha=0.10):");
    println!("  Ball prediction set: {}/50 numbers", conformal.ball_set_size);
    println!("  Star prediction set: {}/12 numbers", conformal.star_set_size);
    println!("  Ball entropy: {:.3} (uniform: {:.3})", conformal.ball_entropy, (50.0_f64).ln());
    println!("  Star entropy: {:.3} (uniform: {:.3})", conformal.star_entropy, (12.0_f64).ln());
    println!("  Recommendation: {}", conformal.recommendation);

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

    // Extraire le skill calibré depuis les poids de calibration
    let (calibrated_ball_skill, calibrated_star_skill) = if let Ok(ref w) = weights {
        let uniform_ball_ll = 5.0 * (1.0_f64 / 50.0).ln();
        let uniform_star_ll = 2.0 * (1.0_f64 / 12.0).ln();
        let ball_skill: f64 = w.calibrations.iter()
            .filter_map(|c| {
                let wt = w.ball_weights.iter().find(|(n,_)| *n == c.model_name).map(|(_,w)| *w).unwrap_or(0.0);
                if wt > 0.0 { Some(wt * (c.best_ll - uniform_ball_ll).max(0.0)) } else { None }
            })
            .sum();
        let star_skill: f64 = w.calibrations.iter()
            .filter_map(|c| {
                let wt = w.star_weights.iter().find(|(n,_)| *n == c.model_name).map(|(_,w)| *w).unwrap_or(0.0);
                if wt > 0.0 { Some(wt * (c.best_ll - uniform_star_ll).max(0.0)) } else { None }
            })
            .sum();
        (Some(ball_skill), Some(star_skill))
    } else {
        (None, None)
    };

    // v7: RQA-adaptive temperature factor
    let rqa_factor = rqa_temperature_factor(&draws);
    if (rqa_factor - 1.0).abs() > 0.01 {
        println!("(RQA temperature factor: {:.2} — {})", rqa_factor,
            if rqa_factor < 1.0 { "système prédictible → sharpen" } else { "système chaotique → relax" });
    }

    // v16: Extraire les températures optimales calibrées
    let (calibrated_opt_t_balls, calibrated_opt_t_stars) = if let Ok(ref w) = weights {
        (w.optimal_t_balls, w.optimal_t_stars)
    } else {
        (None, None)
    };

    // Température adaptative : few-grid override > explicit > v16 calibrated > skill-based > conviction
    // v7: RQA factor applied post-hoc (except few-grid override and calibrated)
    let (eff_bt, eff_st) = if let Some(ng) = n_grids {
        // Mode few-grid : use optimized hyperparams if available, else default
        let (bt, st) = if has_optimized {
            (hyper.t_balls, hyper.t_stars)
        } else {
            few_grid_temperature(ng)
        };
        println!("\n(Temp few-grid [{} grilles{}] — balls: {:.2}, stars: {:.2})", ng,
            if has_optimized { " optimisé" } else { "" }, bt, st);
        (bt, st)
    } else if let Some(t) = temperature {
        // RQA modulation on explicit temperature
        let bt = (t * rqa_factor).clamp(0.1, 2.0);
        let st = (t * rqa_factor).clamp(0.1, 2.0);
        (bt, st)
    } else if jackpot_mode {
        // v17: Jackpot mode skips NLL temperature (conservative) — use conviction (aggressive)
        // NLL optimizes log-loss, conviction optimizes P(5+2) concentration
        let (mut bt, st) = conviction_temperature_split_with_skill(
            &conviction, calibrated_ball_skill, calibrated_star_skill);
        // Gros jackpot (>100M) : T_balls=1.0 pour ne pas sur-concentrer les boules
        if jackpot > 100_000_000.0 { bt = 1.0; }
        // v7: RQA modulation (balls only, stars keep conviction-based)
        bt = (bt * rqa_factor).clamp(0.1, 2.0);
        let method = if calibrated_ball_skill.is_some() { "skill+RQA" } else { "conviction+RQA" };
        println!("\n(Temp split [{method}] — balls: {:.2}{}, stars: {:.2})", bt, if jackpot > 100_000_000.0 { " [no-sharpen >100M]" } else { "" }, st);
        (bt, st)
    } else if calibrated_opt_t_balls.is_some() || calibrated_opt_t_stars.is_some() {
        // v16: NLL-optimized temperatures for EV mode (calibration quality)
        let mut bt = calibrated_opt_t_balls.unwrap_or_else(|| {
            conviction_temperature_split_with_skill(
                &conviction, calibrated_ball_skill, calibrated_star_skill).0
        });
        let st = calibrated_opt_t_stars.unwrap_or_else(|| {
            conviction_temperature_split_with_skill(
                &conviction, calibrated_ball_skill, calibrated_star_skill).1
        });
        bt = (bt * rqa_factor).clamp(0.05, 2.0);
        println!("\n(Temp calibrée NLL — balls: {:.2}, stars: {:.2})", bt, st);
        (bt, st)
    } else {
        // Fallback : skill-based or conviction
        let (bt, st) = conviction_temperature_split_with_skill(
            &conviction, calibrated_ball_skill, calibrated_star_skill);
        let bt = (bt * rqa_factor).clamp(0.1, 2.0);
        let method = if calibrated_ball_skill.is_some() { "skill+RQA" } else { "conviction+RQA" };
        println!("\n(Temp split [{method}] EV — balls: {:.2}, stars: {:.2})", bt, st);
        (bt, st)
    };

    // v23: Bayesian adaptive star temperature from pair confidence
    let eff_st = {
        let star_pair_model_tmp = lemillion_ensemble::models::star_pair::StarPairModel::default();
        if let Some((_, confidence)) = star_pair_model_tmp.predict_pair_distribution_with_confidence(&draws) {
            let adapted = (eff_st * (1.5 - confidence)).clamp(0.08, 0.30);
            println!("(Star pair confidence: {:.2} → T_stars adapted: {:.2} → {:.2})", confidence, eff_st, adapted);
            adapted
        } else {
            eff_st
        }
    };

    // v23: Build coherence scorer early (needed for entropic tilt + jackpot scoring)
    let coherence = if let Ok(ref w) = weights {
        w.coherence_stats.as_ref()
            .map(|s| CoherenceScorer::from_stats(s))
            .unwrap_or_else(|| CoherenceScorer::from_history(&draws, Pool::Balls))
    } else {
        CoherenceScorer::from_history(&draws, Pool::Balls)
    };

    // v23b: Entropic tilt at low strength (0.2) — structure-aware but gentle
    let ball_tilted = lemillion_ensemble::sampler::entropic_tilt(
        &ball_pred.distribution, &coherence.pair_freq, 0.2,
    );
    let ball_dist = if (eff_bt - 1.0).abs() > 1e-9 {
        println!("(Température boules : {:.2}, entropic tilt=0.2)", eff_bt);
        apply_temperature(&ball_tilted, eff_bt)
    } else {
        ball_tilted
    };
    let star_dist = if (eff_st - 1.0).abs() > 1e-9 {
        println!("(Température étoiles : {:.2})", eff_st);
        apply_temperature(&star_pred.distribution, eff_st)
    } else {
        star_pred.distribution.clone()
    };

    // Grille optimale
    let optimal = optimal_grid(&ball_dist, &star_dist);
    let optimal_cs = consensus_score(&optimal.balls, &optimal.stars, &ball_consensus, &star_consensus);
    display::display_optimal_grid(&optimal, optimal_cs);

    if jackpot_mode || n_grids.is_some() {
        // Mode Jackpot : profils de poids → candidats → sélection optimale
        let target_grids = n_grids.unwrap_or(6);
        let filter = if no_filter {
            None
        } else {
            Some(StructuralFilter::adaptive(&draws))
        };

        // coherence already built above for entropic tilt
        let mut joint_model = lemillion_ensemble::models::joint::JointConditionalModel::default();
        joint_model.train(&draws);

        // Star pair distribution for pair-aware scoring
        let star_pair_model = lemillion_ensemble::models::star_pair::StarPairModel::default();
        let star_pair_probs = star_pair_model.predict_pair_distribution(&draws);

        // Exclusion set from consensus — désactivé pour gros jackpots (>100M)
        let excluded_ref = if jackpot > 100_000_000.0 {
            println!("K-réduction désactivée (jackpot {:.0}M > 100M)", jackpot / 1_000_000.0);
            None
        } else {
            let excl_threshold = match conviction.verdict {
                lemillion_ensemble::sampler::ConvictionVerdict::HighConviction => -0.15,
                lemillion_ensemble::sampler::ConvictionVerdict::MediumConviction => -0.25,
                lemillion_ensemble::sampler::ConvictionVerdict::LowConviction => -0.40,
            };
            let excluded = compute_exclusion_set(&ball_consensus, excl_threshold, 10);
            if excluded.is_empty() { None } else {
                println!("K-réduction (seuil {:.2}) : {} boules exclues {:?}", excl_threshold, excluded.len(), excluded);
                Some(excluded)
            }
        };
        let excluded_ref = excluded_ref.as_deref();

        // Ball→star conditioner
        let conditioner = BallStarConditioner::from_history(&draws);

        // v15: Star coherence scorer
        let star_coherence = StarCoherenceScorer::from_history(&draws);

        // Neural scorer (optional)
        let neural_scorer_opt = if neural_rerank {
            let path = std::path::PathBuf::from("neural_scorer.json");
            match lemillion_ensemble::models::neural_scorer::NeuralScorer::load(&path) {
                Ok(ns) => {
                    println!("(Neural scorer chargé depuis {})", path.display());
                    Some(ns)
                }
                Err(_) => {
                    println!("(Neural scorer : entraînement à la volée...)");
                    let ns = lemillion_ensemble::models::neural_scorer::NeuralScorer::train(&draws, 42);
                    let _ = ns.save(&path);
                    println!("(Neural scorer entraîné et sauvegardé)");
                    Some(ns)
                }
            }
        } else {
            None
        };

        // Build weight profiles (3 profils ciblés)
        let model_names: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
        let profiles = build_weight_profiles(&combiner.ball_weights, &combiner.star_weights, &model_names, eff_bt, eff_st);
        let n_profiles = profiles.len();
        let n_per_profile = (n_suggestions / n_profiles).max(500);

        // Collect all candidates from all profiles
        let mut all_candidates: Vec<(String, lemillion_db::models::Suggestion, Vec<f64>, Vec<f64>)> = Vec::new();
        let mut last_result = None;
        // v20: Use optimized coherence weights if available, else calibration, else default
        let cw_ball = if has_optimized {
            Some(hyper.coherence_weight)
        } else {
            weights.as_ref().ok().and_then(|w| w.coherence_ball_weight)
        };
        let cw_star = if has_optimized {
            Some(hyper.star_coherence_weight)
        } else {
            weights.as_ref().ok().and_then(|w| w.coherence_star_weight)
        };

        for (label, prof_bw, prof_sw, prof_bt, prof_st) in &profiles {
            let prof_models = all_models();
            let prof_combiner = EnsembleCombiner::with_weights(prof_models, prof_bw.clone(), prof_sw.clone());
            let mut bp = prof_combiner.predict(&draws, Pool::Balls);
            let sp = prof_combiner.predict(&draws, Pool::Stars);

            // v16: Beta-transform post-pooling (skip in few-grid mode — temperatures are already tuned)
            if n_grids.is_none() {
                if let Ok(ref w) = weights {
                    if let Some((alpha, beta)) = w.beta_balls {
                        lemillion_ensemble::ensemble::beta_transform(&mut bp.distribution, alpha, beta);
                    }
                }
            }

            let bd = if (*prof_bt - 1.0).abs() > 1e-9 {
                apply_temperature(&bp.distribution, *prof_bt)
            } else {
                bp.distribution.clone()
            };
            let mut star_d = sp.distribution.clone();
            if n_grids.is_none() {
                if let Ok(ref w) = weights {
                    if let Some((alpha, beta)) = w.beta_stars {
                        lemillion_ensemble::ensemble::beta_transform(&mut star_d, alpha, beta);
                    }
                }
            }
            let sd = if (*prof_st - 1.0).abs() > 1e-9 {
                apply_temperature(&star_d, *prof_st)
            } else {
                star_d
            };

            let neural_scorer_ref = if neural_rerank { neural_scorer_opt.as_ref() } else { None };

            let conformal_ranks = weights.as_ref().ok()
                .map(|w| w.conformal_ball_max_ranks.as_slice())
                .filter(|s| !s.is_empty());
            let result = generate_suggestions_jackpot(
                &bd, &sd, n_per_profile, filter.as_ref(),
                Some(&coherence), Some(&joint_model),
                star_pair_probs.as_ref(), excluded_ref,
                Some(&conditioner),
                neural_scorer_ref,
                Some(&star_coherence),
                cw_ball, cw_star,
                conformal_ranks,
            )?;

            for s in &result.suggestions {
                all_candidates.push((label.to_string(), s.clone(), bd.clone(), sd.clone()));
            }

            last_result = Some(result);
        }

        // v17: Gibbs sampling for additional candidates
        {
            let gibbs_count = n_per_profile / 2; // 50% of one profile's size
            let gibbs_seed = seed.unwrap_or_else(|| lemillion_ensemble::sampler::date_seed());
            match generate_suggestions_gibbs(
                &ball_dist, &star_dist, gibbs_count,
                filter.as_ref(), Some(&coherence), Some(&joint_model),
                star_pair_probs.as_ref(), Some(&conditioner), Some(&star_coherence),
                8, eff_bt.max(0.3), gibbs_seed,
                cw_ball, cw_star,
            ) {
                Ok(gibbs_result) => {
                    println!("(Gibbs sampling: {} candidats supplémentaires)", gibbs_result.suggestions.len());
                    for s in &gibbs_result.suggestions {
                        all_candidates.push(("Gibbs".to_string(), s.clone(), ball_dist.clone(), star_dist.clone()));
                    }
                }
                Err(_) => {} // Silently skip if Gibbs fails
            }
        }

        // Sort all candidates by score descending (P(5+2) proxy)
        all_candidates.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));

        // Use select_optimal_n_grids for final selection
        let candidate_suggestions: Vec<lemillion_db::models::Suggestion> = all_candidates.iter()
            .map(|(_, s, _, _)| s.clone())
            .collect();
        let (max_common_balls, max_common_stars) = if n_grids.is_some() {
            (3, 1) // few-grid mode: max 3 boules, 1 étoile communes
        } else {
            (2, 2) // legacy mode
        };
        // v21: Exact exhaustive selection for N≤5, SA refinement for larger N
        let optimal_selection = if target_grids <= 5 {
            select_optimal_n_grids_exact(&candidate_suggestions, target_grids, 3)
        } else {
            select_optimal_n_grids_sa(
                &candidate_suggestions, target_grids, max_common_balls, max_common_stars,
                seed.unwrap_or_else(|| lemillion_ensemble::sampler::date_seed()),
            )
        };

        // Map back to labeled grids with their distributions
        let selected_grids: Vec<(String, lemillion_db::models::Suggestion, Vec<f64>, Vec<f64>)> = optimal_selection.iter()
            .filter_map(|sel| {
                all_candidates.iter()
                    .find(|(_, s, _, _)| s.balls == sel.balls && s.stars == sel.stars)
                    .map(|(l, s, bd, sd)| (l.clone(), s.clone(), bd.clone(), sd.clone()))
            })
            .collect();

        // Display the N grids
        const TOTAL_COMBINATIONS: f64 = 139_838_160.0;
        if !selected_grids.is_empty() {
            if n_grids.is_some() {
                println!("\n== {} Grilles Optimales (few-grid mode) ==\n", selected_grids.len());
            } else {
                println!("\n== {} Grilles Jackpot Multi-Perspectives ==\n", selected_grids.len());
            }
            let mut grid_probs = Vec::new();
            for (i, (label, g, bd, sd)) in selected_grids.iter().enumerate() {
                let score = compute_bayesian_score(&g.balls, &g.stars, bd, sd);
                let p52 = score / TOTAL_COMBINATIONS;
                let cs = consensus_score(&g.balls, &g.stars, &ball_consensus, &star_consensus);
                println!(
                    "  Grille {} [{}] : {:2} - {:2} - {:2} - {:2} - {:2}  + {:2} - {:2}   P(5+2)={:.2e}  consensus={:+.2}",
                    i + 1, label, g.balls[0], g.balls[1], g.balls[2], g.balls[3], g.balls[4],
                    g.stars[0], g.stars[1], p52, cs,
                );
                grid_probs.push(p52);
            }

            let p_jackpot = 1.0 - grid_probs.iter().map(|&p| 1.0 - p).product::<f64>();
            let p_random = selected_grids.len() as f64 / TOTAL_COMBINATIONS;
            let factor = p_jackpot / p_random;

            println!("\n── Verdict Jackpot ──");
            for (i, &p) in grid_probs.iter().enumerate() {
                println!("  P(5+2) grille {} : {:.2e}", i + 1, p);
            }
            println!("  P(5+2) combiné ({} grilles) : {:.2e}", selected_grids.len(), p_jackpot);
            println!("  Facteur vs {} grilles aléatoires : {:.1}x", selected_grids.len(), factor);
            if n_grids.is_some() {
                let cost = selected_grids.len() as f64 * 2.50;
                println!("  Coût : {:.2} EUR ({} grilles × 2.50 EUR)", cost, selected_grids.len());
            }
        }

        if let Some(result) = last_result {
            display::display_jackpot_results(&result, &ball_consensus, &star_consensus, &conviction);
        }
    } else {
        // Mode EV : grilles diversifiées + suggestions EV
        let effective_seed = seed.unwrap_or_else(lemillion_ensemble::sampler::date_seed);
        let star_strat = StarStrategy::from_str_loose(star_strategy_str);
        let diverse = generate_diverse_grids_with_strategy(&ball_dist, &star_dist, &draws, 3, effective_seed, Some(&popularity), star_strat);
        display::display_diverse_grids(&diverse, &ball_consensus, &star_consensus);

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
        Err(ref e) => {
            println!("(Pas de fichier de calibration, utilisation de poids uniformes: {e})");
            EnsembleCombiner::new(models)
        }
    };

    // ── Pipeline aligné sur cmd_predict (sans température) ──

    // MetaPredictor : ajuster les poids selon le régime courant
    let balls_have_signal = {
        let max_w = combiner.ball_weights.iter().cloned().fold(0.0_f64, f64::max);
        let min_w = combiner.ball_weights.iter().filter(|&&w| w > 0.0).cloned().fold(f64::MAX, f64::min);
        (max_w - min_w) > 1e-6
    };

    if balls_have_signal
        && let Ok(ref w) = weights
        && !w.detailed_ll.is_empty()
    {
        use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
        let features = RegimeFeatures::from_draws(&draws);
        if let Some(meta) = MetaPredictor::train(&draws, &w.detailed_ll, 1.0) {
            let adjustments = meta.weight_adjustments(&features);
            for (name, adj) in &adjustments {
                if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                    && idx < combiner.ball_weights.len()
                {
                    combiner.ball_weights[idx] *= adj;
                }
            }
            let total: f64 = combiner.ball_weights.iter().sum();
            if total > 0.0 { for w in combiner.ball_weights.iter_mut() { *w /= total; } }
        }
    }

    if let Ok(ref w) = weights
        && !w.star_detailed_ll.is_empty()
    {
        use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
        let features = RegimeFeatures::from_draws(&draws);
        if let Some(meta) = MetaPredictor::train(&draws, &w.star_detailed_ll, 1.0) {
            let adjustments = meta.weight_adjustments(&features);
            for (name, adj) in &adjustments {
                if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                    && idx < combiner.star_weights.len()
                {
                    combiner.star_weights[idx] *= adj;
                }
            }
            let total: f64 = combiner.star_weights.iter().sum();
            if total > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= total; } }
        }
    }

    // Hedge weights : ajustement multiplicatif réactif
    {
        let (hedged_ball, hedged_star) = compute_hedge_weights(
            &combiner.models, &draws,
            &combiner.ball_weights, &combiner.star_weights,
            100, 0.10,
        );
        if balls_have_signal {
            combiner.ball_weights = hedged_ball;
        }
        combiner.star_weights = hedged_star;

        // Re-apply family cap after hedge
        let te_f: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
        let stresa_f: &[&str] = &["StresaSGD", "StresaChaos"];
        let fams: Vec<(&[&str], f64)> = vec![(te_f, 0.20), (stresa_f, 0.15)];
        let mn: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
        apply_family_cap_vecs(&mn, &mut combiner.ball_weights, &fams);
        apply_family_cap_vecs(&mn, &mut combiner.star_weights, &fams);
    }

    // Prédiction avec décorrélation/stacking/agreement boost (comme cmd_predict)
    let has_corr_matrix = weights.as_ref().map(|w| !w.correlation_matrix.is_empty()).unwrap_or(false);
    let use_stacking = weights.as_ref().map(|w| w.stacking_balls.is_some() || w.stacking_stars.is_some()).unwrap_or(false);

    let (ball_pred, star_pred) = if use_stacking {
        let w = weights.as_ref().unwrap();
        let blend_b = w.stacking_blend_balls.unwrap_or(0.6);
        let bp = if let Some(ref sw) = w.stacking_balls {
            combiner.predict_stacked(&draws, Pool::Balls, sw, blend_b)
        } else if has_corr_matrix {
            let corr = &w.correlation_matrix;
            combiner.predict_decorrelated(&draws, Pool::Balls, corr, 0.60)
        } else {
            combiner.predict_with_agreement_boost(&draws, Pool::Balls, 0.20)
        };
        let blend_s = w.stacking_blend_stars.unwrap_or(0.6);
        let sp = if let Some(ref sw) = w.stacking_stars {
            combiner.predict_stacked(&draws, Pool::Stars, sw, blend_s)
        } else if has_corr_matrix {
            let corr = if w.star_correlation_matrix.is_empty() { &w.correlation_matrix } else { &w.star_correlation_matrix };
            combiner.predict_decorrelated(&draws, Pool::Stars, corr, 0.60)
        } else {
            combiner.predict_with_agreement_boost(&draws, Pool::Stars, 0.20)
        };
        (bp, sp)
    } else if has_corr_matrix {
        let w = weights.as_ref().unwrap();
        let ball_corr = &w.correlation_matrix;
        let star_corr = if w.star_correlation_matrix.is_empty() { &w.correlation_matrix } else { &w.star_correlation_matrix };
        (
            combiner.predict_decorrelated(&draws, Pool::Balls, ball_corr, 0.60),
            combiner.predict_decorrelated(&draws, Pool::Stars, star_corr, 0.60),
        )
    } else {
        (
            combiner.predict_with_agreement_boost(&draws, Pool::Balls, 0.20),
            combiner.predict_with_agreement_boost(&draws, Pool::Stars, 0.20),
        )
    };

    display::display_compare(&balls, &stars, &ball_pred, &star_pred);

    Ok(())
}

fn cmd_add_draw(conn: &lemillion_db::rusqlite::Connection, args: &[String]) -> Result<()> {
    if args.len() != 10 {
        bail!("Attendu 10 arguments: draw_id jour date b1 b2 b3 b4 b5 s1 s2\nExemple: add-draw 26016 MARDI 2026-02-24 10 27 40 43 47 6 10");
    }
    let mut draw = Draw {
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
        ball_order: None,
        star_order: None,
        cycle_number: None,
    };
    draw.normalize();
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
        ball_order: None,
        star_order: None,
        cycle_number: None,
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
    realistic: bool,
    star_strategy_str: &str,
    n_grids: Option<usize>,
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

    // Mode few-grid backtest : N grilles avec température forcée
    if let Some(ng) = n_grids {
        return cmd_backtest_few_grids(&draws, weights.as_ref().ok(), last, n_suggestions, ng, top_models);
    }

    // Mode jackpot multi-perspectives : générer N grilles jackpot diversifiées
    if jackpot_mode && backtest_3_grids {
        return cmd_backtest_jackpot_top3(&draws, weights.as_ref().ok(), last, n_suggestions, temperature, top_models);
    }

    // Mode jackpot backtest
    if jackpot_mode {
        return cmd_backtest_jackpot(&draws, weights.as_ref().ok(), last, n_suggestions, temperature, top_models);
    }

    // Mode réaliste : 3 ou 10 grilles selon conviction, suivi financier
    if realistic {
        let star_strat = StarStrategy::from_str_loose(star_strategy_str);
        return cmd_backtest_realistic(&draws, weights.as_ref().ok(), last, temperature, star_strat);
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

    use rayon::prelude::*;

    let rows: Vec<display::JackpotBacktestRow> = {
        let indexed: Vec<(usize, display::JackpotBacktestRow)> = (0..last)
            .into_par_iter()
            .map(|i| -> Result<(usize, display::JackpotBacktestRow)> {
                let test_draw = &draws[i];
                let training_draws = &draws[i + 1..];

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
                }

                // MetaPredictor : ajustement contextuel des poids
                if let Some(w) = weights {
                    if !w.detailed_ll.is_empty() {
                        use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
                        let features = RegimeFeatures::from_draws(training_draws);
                        if let Some(meta) = MetaPredictor::train(training_draws, &w.detailed_ll, 1.0) {
                            let adjustments = meta.weight_adjustments(&features);
                            for (name, adj) in &adjustments {
                                if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                                    && idx < combiner.ball_weights.len()
                                {
                                    combiner.ball_weights[idx] *= adj;
                                }
                            }
                            let total: f64 = combiner.ball_weights.iter().sum();
                            if total > 0.0 { for w in combiner.ball_weights.iter_mut() { *w /= total; } }
                        }
                        if !w.star_detailed_ll.is_empty() {
                            if let Some(meta) = MetaPredictor::train(training_draws, &w.star_detailed_ll, 1.0) {
                                let adjustments = meta.weight_adjustments(&features);
                                for (name, adj) in &adjustments {
                                    if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                                        && idx < combiner.star_weights.len()
                                    {
                                        combiner.star_weights[idx] *= adj;
                                    }
                                }
                                let total: f64 = combiner.star_weights.iter().sum();
                                if total > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= total; } }
                            }
                        }
                    }
                }

                // Hedge weights : ajustement multiplicatif réactif
                {
                    let (hedged_ball, hedged_star) = compute_hedge_weights(
                        &combiner.models, training_draws,
                        &combiner.ball_weights, &combiner.star_weights,
                        100, 0.10,
                    );
                    combiner.ball_weights = hedged_ball;
                    combiner.star_weights = hedged_star;

                    // Re-apply family cap after hedge
                    let te_f: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
                    let stresa_f: &[&str] = &["StresaSGD", "StresaChaos"];
                    let fams: Vec<(&[&str], f64)> = vec![(te_f, 0.20), (stresa_f, 0.15)];
                    let mn: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
                    apply_family_cap_vecs(&mn, &mut combiner.ball_weights, &fams);
                    apply_family_cap_vecs(&mn, &mut combiner.star_weights, &fams);
                }

                // v17: Production-aligned pipeline
                // 1. Decorrelation / Stacking / Agreement boost
                let has_corr_matrix = weights.map(|w| !w.correlation_matrix.is_empty()).unwrap_or(false);
                let use_stacking = weights.map(|w| w.stacking_balls.is_some() || w.stacking_stars.is_some()).unwrap_or(false);

                let predict_pool_bt = |pool: Pool| -> lemillion_ensemble::ensemble::EnsemblePrediction {
                    if has_corr_matrix {
                        let w = weights.unwrap();
                        let corr = match pool {
                            Pool::Balls => &w.correlation_matrix,
                            Pool::Stars => if w.star_correlation_matrix.is_empty() { &w.correlation_matrix } else { &w.star_correlation_matrix },
                        };
                        combiner.predict_decorrelated(training_draws, pool, corr, 0.60)
                    } else {
                        combiner.predict_with_agreement_boost(training_draws, pool, 0.20)
                    }
                };

                let (mut ball_pred, mut star_pred) = if use_stacking {
                    let w = weights.unwrap();
                    let blend_b = w.stacking_blend_balls.unwrap_or(0.6);
                    let bp = if let Some(ref sw) = w.stacking_balls {
                        combiner.predict_stacked(training_draws, Pool::Balls, sw, blend_b)
                    } else {
                        predict_pool_bt(Pool::Balls)
                    };
                    let blend_s = w.stacking_blend_stars.unwrap_or(0.6);
                    let sp = if let Some(ref sw) = w.stacking_stars {
                        combiner.predict_stacked(training_draws, Pool::Stars, sw, blend_s)
                    } else {
                        predict_pool_bt(Pool::Stars)
                    };
                    (bp, sp)
                } else {
                    (predict_pool_bt(Pool::Balls), predict_pool_bt(Pool::Stars))
                };

                // 2. Online/offline blend (balls only, v15/v17)
                {
                    use lemillion_ensemble::ensemble::online::online_offline_blend_with_alpha;
                    let oa = weights.and_then(|w| w.online_ewma_alpha).unwrap_or(0.15);
                    let ow = weights.and_then(|w| w.online_window).unwrap_or(8);
                    let blended = online_offline_blend_with_alpha(&ball_pred.distribution, training_draws, Pool::Balls, ow, oa);
                    ball_pred.distribution = blended;
                }

                // 3. Beta-transform post-pooling (v16)
                if let Some(w) = weights {
                    if let Some((alpha, beta)) = w.beta_balls {
                        lemillion_ensemble::ensemble::beta_transform(&mut ball_pred.distribution, alpha, beta);
                    }
                    if let Some((alpha, beta)) = w.beta_stars {
                        lemillion_ensemble::ensemble::beta_transform(&mut star_pred.distribution, alpha, beta);
                    }
                }

                // 4. Conviction (on raw distributions before temperature)
                let conviction = compute_conviction(
                    &ball_pred.distribution,
                    &star_pred.distribution,
                    &ball_pred.spread,
                    &star_pred.spread,
                );

                // 5. RQA temperature factor
                let rqa_factor = rqa_temperature_factor(training_draws);

                // 6. Temperature: jackpot mode uses conviction (aggressive), skip NLL (conservative)
                let (calibrated_ball_skill, calibrated_star_skill) = if let Some(w) = weights {
                    let uniform_ball_ll = 5.0 * (1.0_f64 / 50.0).ln();
                    let uniform_star_ll = 2.0 * (1.0_f64 / 12.0).ln();
                    let ball_skill: f64 = w.calibrations.iter()
                        .filter_map(|c| {
                            let wt = w.ball_weights.iter().find(|(n,_)| *n == c.model_name).map(|(_,w)| *w).unwrap_or(0.0);
                            if wt > 0.0 { Some(wt * (c.best_ll - uniform_ball_ll).max(0.0)) } else { None }
                        })
                        .sum();
                    let star_skill: f64 = w.calibrations.iter()
                        .filter_map(|c| {
                            let wt = w.star_weights.iter().find(|(n,_)| *n == c.model_name).map(|(_,w)| *w).unwrap_or(0.0);
                            if wt > 0.0 { Some(wt * (c.best_ll - uniform_star_ll).max(0.0)) } else { None }
                        })
                        .sum();
                    (Some(ball_skill), Some(star_skill))
                } else {
                    (None, None)
                };

                // Jackpot mode: skip NLL temperature (conservative, optimized for log-loss)
                // Use conviction temperatures (aggressive, optimized for P(5+2) concentration)
                let (eff_bt, eff_st) = if let Some(t) = temperature {
                    let bt = (t * rqa_factor).clamp(0.1, 2.0);
                    (bt, bt)
                } else {
                    let (mut bt, st) = conviction_temperature_split_with_skill(
                        &conviction, calibrated_ball_skill, calibrated_star_skill);
                    bt = (bt * rqa_factor).clamp(0.05, 2.0);
                    (bt, st)
                };

                let ball_dist = if (eff_bt - 1.0).abs() > 1e-9 {
                    apply_temperature(&ball_pred.distribution, eff_bt)
                } else {
                    ball_pred.distribution.clone()
                };
                let star_dist = if (eff_st - 1.0).abs() > 1e-9 {
                    apply_temperature(&star_pred.distribution, eff_st)
                } else {
                    star_pred.distribution.clone()
                };

                let actual_score = compute_bayesian_score(
                    &test_draw.balls, &test_draw.stars, &ball_dist, &star_dist,
                );

                let filter = StructuralFilter::adaptive(training_draws);
                let coherence = CoherenceScorer::from_history(training_draws, Pool::Balls);
                let mut joint_model = lemillion_ensemble::models::joint::JointConditionalModel::default();
                joint_model.train(training_draws);
                let star_pair_model = lemillion_ensemble::models::star_pair::StarPairModel::default();
                let star_pair_probs = star_pair_model.predict_pair_distribution(training_draws);
                let conditioner = BallStarConditioner::from_history(training_draws);
                let star_coherence = StarCoherenceScorer::from_history(training_draws);

                let bt_cw_ball = weights.and_then(|w| w.coherence_ball_weight);
                let bt_cw_star = weights.and_then(|w| w.coherence_star_weight);
                let result = generate_suggestions_jackpot(
                    &ball_dist, &star_dist, n_suggestions, Some(&filter),
                    Some(&coherence), Some(&joint_model),
                    star_pair_probs.as_ref(), None,
                    Some(&conditioner),
                    None,
                    Some(&star_coherence),
                    bt_cw_ball, bt_cw_star,
                    None,
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

                pb.inc(1);

                Ok((i, display::JackpotBacktestRow {
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
                }))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut sorted = indexed;
        sorted.sort_by_key(|(i, _)| *i);
        sorted.into_iter().map(|(_, row)| row).collect()
    };

    pb.finish_and_clear();
    display::display_jackpot_backtest_results(&rows);

    Ok(())
}

/// Évaluation honnête sur holdout : reproduit exactement le pipeline production
/// sur des tirages JAMAIS vus pendant la calibration.
/// Retourne des facteurs d'amélioration avec intervalles de confiance bootstrap.
fn cmd_holdout_eval(
    conn: &lemillion_db::rusqlite::Connection,
    calibration_path: &str,
    holdout_n: usize,
    n_suggestions: usize,
    n_bootstrap: usize,
    n_grids: Option<usize>,
) -> Result<()> {
    let n = count_draws(conn)?;
    if n == 0 { bail!("Base vide."); }
    let all_draws = fetch_last_draws(conn, n)?;
    if holdout_n >= all_draws.len() {
        bail!("Holdout ({}) >= tirages disponibles ({})", holdout_n, all_draws.len());
    }

    let holdout_draws = &all_draws[..holdout_n];
    // training_draws exclut le holdout — reproduit ce que cmd_calibrate fait avec --holdout N
    let calibration_draws = &all_draws[holdout_n..];

    println!("== Holdout Evaluation ==");
    println!("  Tirages holdout    : {} (les plus récents)", holdout_n);
    println!("  Tirages calibration: {} (excluant holdout)", calibration_draws.len());
    println!("  Suggestions/tirage : {}", n_suggestions);
    if let Some(ng) = n_grids {
        println!("  Mode few-grids     : {} grilles", ng);
    }
    println!("  Bootstrap resamples: {}", n_bootstrap);
    println!();

    let weights = load_weights(&PathBuf::from(calibration_path))
        .context("Impossible de charger calibration.json — lancez d'abord `calibrate --holdout N`")?;

    let hyper = lemillion_ensemble::sampler::HyperParams::load(
        std::path::Path::new("hyperparams.json"),
    );

    let pb = ProgressBar::new(holdout_n as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    use rayon::prelude::*;

    // Pour chaque tirage holdout, reproduire le pipeline production
    let results: Vec<display::HoldoutRow> = {
        let indexed: Vec<(usize, display::HoldoutRow)> = (0..holdout_n)
            .into_par_iter()
            .map(|i| -> Result<(usize, display::HoldoutRow)> {
                let test_draw = &holdout_draws[i];
                // training = tous les tirages après ce holdout (tous les tirages de calibration + les holdout plus anciens)
                let training_draws = &all_draws[i + 1..];

                // 1. Build combiner with calibrated weights
                let models = all_models();
                let mut combiner = {
                    let ball_w: Vec<f64> = models.iter()
                        .map(|m| weights.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                        .collect();
                    let star_w: Vec<f64> = models.iter()
                        .map(|m| weights.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                        .collect();
                    EnsembleCombiner::with_weights(models, ball_w, star_w)
                };

                // 2. MetaPredictor adjustment
                if !weights.detailed_ll.is_empty() {
                    use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
                    let features = RegimeFeatures::from_draws(training_draws);
                    if let Some(meta) = MetaPredictor::train(training_draws, &weights.detailed_ll, 1.0) {
                        let adjustments = meta.weight_adjustments(&features);
                        for (name, adj) in &adjustments {
                            if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                                && idx < combiner.ball_weights.len()
                            {
                                combiner.ball_weights[idx] *= adj;
                            }
                        }
                        let total: f64 = combiner.ball_weights.iter().sum();
                        if total > 0.0 { for w in combiner.ball_weights.iter_mut() { *w /= total; } }
                    }
                    if !weights.star_detailed_ll.is_empty() {
                        if let Some(meta) = MetaPredictor::train(training_draws, &weights.star_detailed_ll, 1.0) {
                            let adjustments = meta.weight_adjustments(&features);
                            for (name, adj) in &adjustments {
                                if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                                    && idx < combiner.star_weights.len()
                                {
                                    combiner.star_weights[idx] *= adj;
                                }
                            }
                            let total: f64 = combiner.star_weights.iter().sum();
                            if total > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= total; } }
                        }
                    }
                }

                // 3. Hedge weights
                {
                    let eta = hyper.hedge_eta;
                    let (hedged_ball, hedged_star) = compute_hedge_weights(
                        &combiner.models, training_draws,
                        &combiner.ball_weights, &combiner.star_weights,
                        100, eta,
                    );
                    combiner.ball_weights = hedged_ball;
                    combiner.star_weights = hedged_star;

                    let te_f: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
                    let stresa_f: &[&str] = &["StresaSGD", "StresaChaos"];
                    let fams: Vec<(&[&str], f64)> = vec![(te_f, 0.20), (stresa_f, 0.15)];
                    let mn: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
                    apply_family_cap_vecs(&mn, &mut combiner.ball_weights, &fams);
                    apply_family_cap_vecs(&mn, &mut combiner.star_weights, &fams);
                }

                // 4. Predict (decorrelated / stacked / agreement boost)
                let has_corr_matrix = !weights.correlation_matrix.is_empty();
                let use_stacking = weights.stacking_balls.is_some() || weights.stacking_stars.is_some();

                let predict_pool = |pool: Pool| -> lemillion_ensemble::ensemble::EnsemblePrediction {
                    if has_corr_matrix {
                        let corr = match pool {
                            Pool::Balls => &weights.correlation_matrix,
                            Pool::Stars => if weights.star_correlation_matrix.is_empty() { &weights.correlation_matrix } else { &weights.star_correlation_matrix },
                        };
                        combiner.predict_decorrelated(training_draws, pool, corr, 0.60)
                    } else {
                        combiner.predict_with_agreement_boost(training_draws, pool, 0.20)
                    }
                };

                let (mut ball_pred, mut star_pred) = if use_stacking {
                    let blend_b = weights.stacking_blend_balls.unwrap_or(0.6);
                    let bp = if let Some(ref sw) = weights.stacking_balls {
                        combiner.predict_stacked(training_draws, Pool::Balls, sw, blend_b)
                    } else {
                        predict_pool(Pool::Balls)
                    };
                    let blend_s = weights.stacking_blend_stars.unwrap_or(0.6);
                    let sp = if let Some(ref sw) = weights.stacking_stars {
                        combiner.predict_stacked(training_draws, Pool::Stars, sw, blend_s)
                    } else {
                        predict_pool(Pool::Stars)
                    };
                    (bp, sp)
                } else {
                    (predict_pool(Pool::Balls), predict_pool(Pool::Stars))
                };

                // 5. Online/offline blend
                {
                    use lemillion_ensemble::ensemble::online::online_offline_blend_with_alpha;
                    let oa = weights.online_ewma_alpha.unwrap_or(0.15);
                    let ow = weights.online_window.unwrap_or(8);
                    let blended = online_offline_blend_with_alpha(&ball_pred.distribution, training_draws, Pool::Balls, ow, oa);
                    ball_pred.distribution = blended;
                }

                // 6. Beta-transform
                if let Some((alpha, beta)) = weights.beta_balls {
                    lemillion_ensemble::ensemble::beta_transform(&mut ball_pred.distribution, alpha, beta);
                }
                if let Some((alpha, beta)) = weights.beta_stars {
                    lemillion_ensemble::ensemble::beta_transform(&mut star_pred.distribution, alpha, beta);
                }

                // 7. Conviction & temperature
                let conviction = compute_conviction(
                    &ball_pred.distribution, &star_pred.distribution,
                    &ball_pred.spread, &star_pred.spread,
                );
                let rqa_factor = rqa_temperature_factor(training_draws);

                let (eff_bt, eff_st) = if let Some(_ng) = n_grids {
                    let (bt, st) = (hyper.t_balls, hyper.t_stars);
                    (bt, st)
                } else {
                    let uniform_ball_ll = 5.0 * (1.0_f64 / 50.0).ln();
                    let uniform_star_ll = 2.0 * (1.0_f64 / 12.0).ln();
                    let ball_skill: f64 = weights.calibrations.iter()
                        .filter_map(|c| {
                            let wt = weights.ball_weights.iter().find(|(n,_)| *n == c.model_name).map(|(_,w)| *w).unwrap_or(0.0);
                            if wt > 0.0 { Some(wt * (c.best_ll - uniform_ball_ll).max(0.0)) } else { None }
                        }).sum();
                    let star_skill: f64 = weights.calibrations.iter()
                        .filter_map(|c| {
                            let wt = weights.star_weights.iter().find(|(n,_)| *n == c.model_name).map(|(_,w)| *w).unwrap_or(0.0);
                            if wt > 0.0 { Some(wt * (c.best_ll - uniform_star_ll).max(0.0)) } else { None }
                        }).sum();
                    let (mut bt, st) = conviction_temperature_split_with_skill(
                        &conviction, Some(ball_skill), Some(star_skill));
                    bt = (bt * rqa_factor).clamp(0.05, 2.0);
                    (bt, st)
                };

                // v23: Bayesian adaptive star temperature from pair confidence
                let eff_st = {
                    let star_pair_model_tmp = lemillion_ensemble::models::star_pair::StarPairModel::default();
                    if let Some((_, confidence)) = star_pair_model_tmp.predict_pair_distribution_with_confidence(training_draws) {
                        // High confidence → sharper (lower T), low confidence → conservative
                        (eff_st * (1.5 - confidence)).clamp(0.08, 0.30)
                    } else {
                        eff_st
                    }
                };

                // v23: Build coherence scorer early (needed for entropic tilt)
                let coherence = if let Some(ref stats) = weights.coherence_stats {
                    CoherenceScorer::from_stats(stats)
                } else {
                    CoherenceScorer::from_history(training_draws, Pool::Balls)
                };

                // v23b: Entropic tilt at low strength
                let ball_tilted = lemillion_ensemble::sampler::entropic_tilt(
                    &ball_pred.distribution, &coherence.pair_freq, 0.2,
                );
                let ball_dist = if (eff_bt - 1.0).abs() > 1e-9 {
                    apply_temperature(&ball_tilted, eff_bt)
                } else { ball_tilted };
                let star_dist = if (eff_st - 1.0).abs() > 1e-9 {
                    apply_temperature(&star_pred.distribution, eff_st)
                } else { star_pred.distribution.clone() };

                // 8. Actual draw score (bayesian score of true draw under model)
                let actual_score = compute_bayesian_score(
                    &test_draw.balls, &test_draw.stars, &ball_dist, &star_dist,
                );

                // 9. Ball recall@K (track ranking of winning balls)
                let mut ball_probs_indexed: Vec<(usize, f64)> = ball_dist.iter().copied().enumerate().collect();
                ball_probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let ball_ranks: Vec<usize> = test_draw.balls.iter().map(|&b| {
                    ball_probs_indexed.iter().position(|(idx, _)| *idx == (b as usize - 1)).unwrap_or(49)
                }).collect();
                let max_ball_rank = *ball_ranks.iter().max().unwrap_or(&49);
                let recall_at_25 = test_draw.balls.iter().filter(|&&b| ball_ranks[test_draw.balls.iter().position(|&x| x == b).unwrap()] < 25).count() as f64 / 5.0;

                let mut star_probs_indexed: Vec<(usize, f64)> = star_dist.iter().copied().enumerate().collect();
                star_probs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let star_ranks: Vec<usize> = test_draw.stars.iter().map(|&s| {
                    star_probs_indexed.iter().position(|(idx, _)| *idx == (s as usize - 1)).unwrap_or(11)
                }).collect();
                let max_star_rank = *star_ranks.iter().max().unwrap_or(&11);

                // 10. Generate suggestions and compute improvement factor
                let filter = StructuralFilter::adaptive(training_draws);
                // coherence already built above for entropic tilt
                let mut joint_model = lemillion_ensemble::models::joint::JointConditionalModel::default();
                joint_model.train(training_draws);
                let star_pair_model = lemillion_ensemble::models::star_pair::StarPairModel::default();
                let star_pair_probs = star_pair_model.predict_pair_distribution(training_draws);
                let conditioner = BallStarConditioner::from_history(training_draws);
                let star_coherence = StarCoherenceScorer::from_history(training_draws);

                let cw_ball = weights.coherence_ball_weight;
                let cw_star = weights.coherence_star_weight;

                let conformal_ranks = if weights.conformal_ball_max_ranks.is_empty() {
                    None
                } else {
                    Some(weights.conformal_ball_max_ranks.as_slice())
                };
                let result = generate_suggestions_jackpot(
                    &ball_dist, &star_dist, n_suggestions, Some(&filter),
                    Some(&coherence), Some(&joint_model),
                    star_pair_probs.as_ref(), None,
                    Some(&conditioner), None,
                    Some(&star_coherence),
                    cw_ball, cw_star,
                    conformal_ranks,
                )?;

                // 11. Few-grids selection if requested
                let (improvement_factor, n_selected) = if let Some(ng) = n_grids {
                    let selected = if ng <= 5 {
                        select_optimal_n_grids_exact(&result.suggestions, ng, 3)
                    } else {
                        select_optimal_n_grids_sa(&result.suggestions, ng, 3, 1, 42)
                    };
                    let total_p: f64 = selected.iter().map(|s| s.score / 139_838_160.0).sum();
                    let uniform_p = ng as f64 / 139_838_160.0;
                    (total_p / uniform_p, ng)
                } else {
                    (result.improvement_factor, n_suggestions)
                };

                pb.inc(1);

                Ok((i, display::HoldoutRow {
                    date: test_draw.date.clone(),
                    actual_balls: test_draw.balls,
                    actual_stars: test_draw.stars,
                    actual_score,
                    improvement_factor,
                    conviction: conviction.overall,
                    max_ball_rank: max_ball_rank + 1, // 1-indexed
                    max_star_rank: max_star_rank + 1,
                    recall_at_25,
                    n_selected,
                }))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut sorted = indexed;
        sorted.sort_by_key(|(i, _)| *i);
        sorted.into_iter().map(|(_, r)| r).collect()
    };

    pb.finish_and_clear();

    // Display per-draw results
    display::display_holdout_results(&results);

    // Bootstrap confidence intervals
    if n_bootstrap > 0 && !results.is_empty() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let improvements: Vec<f64> = results.iter().map(|r| r.improvement_factor).collect();
        let n = improvements.len();
        let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

        for b in 0..n_bootstrap {
            // Deterministic seeded resampling
            let mut hasher = DefaultHasher::new();
            b.hash(&mut hasher);
            let seed = hasher.finish();
            let mut rng_state = seed;
            let mut sum = 0.0;
            for _ in 0..n {
                // Simple LCG
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (rng_state >> 33) as usize % n;
                sum += improvements[idx];
            }
            bootstrap_means.push(sum / n as f64);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p = |q: f64| -> f64 {
            let idx = ((q * bootstrap_means.len() as f64) as usize).min(bootstrap_means.len() - 1);
            bootstrap_means[idx]
        };

        println!("\n── Bootstrap CI ({} resamples) ──", n_bootstrap);
        println!("  P5               : {:.2}x", p(0.05));
        println!("  P25              : {:.2}x", p(0.25));
        println!("  P50 (médiane)    : {:.2}x", p(0.50));
        println!("  P75              : {:.2}x", p(0.75));
        println!("  P95              : {:.2}x", p(0.95));
        println!("  CI 90%           : [{:.2}x, {:.2}x]", p(0.05), p(0.95));

        // Recall stats
        let avg_recall = results.iter().map(|r| r.recall_at_25).sum::<f64>() / results.len() as f64;
        let max_ranks: Vec<usize> = results.iter().map(|r| r.max_ball_rank).collect();
        let mut sorted_ranks = max_ranks.clone();
        sorted_ranks.sort();
        let k95 = sorted_ranks[(0.95 * sorted_ranks.len() as f64) as usize];
        println!("\n── Recall & K Conformal ──");
        println!("  Recall@25 moyen  : {:.1}%", avg_recall * 100.0);
        println!("  Max ball rank P50: {}", sorted_ranks[sorted_ranks.len() / 2]);
        println!("  Max ball rank P95: {} (= K conformal recommandé)", k95);
        println!("  5/5 dans top-25  : {}/{}", results.iter().filter(|r| r.recall_at_25 >= 1.0).count(), results.len());
    }

    Ok(())
}

/// Construit 6 profils de poids diversifiés pour les grilles jackpot.
/// Chaque profil a ses propres poids ET ses propres températures balls/stars.
fn build_weight_profiles(
    ball_w: &[f64],
    star_w: &[f64],
    _model_names: &[String],
    base_bt: f64,
    base_st: f64,
) -> Vec<(&'static str, Vec<f64>, Vec<f64>, f64, f64)> {
    // 3 profils ciblés : Principal + Star-variant + Exploratoire
    // Chaque profil génère plus de candidats que les 6 anciens profils combinés.

    // Profil 2: Star-variant — mêmes boules, étoiles plus sharpenées
    let star_variant_st = (base_st * 0.7).max(0.15);

    // Profil 3: Exploratoire — petite perturbation des poids boules
    // (perturbation déterministe basée sur les poids eux-mêmes)
    let exploratory_bw = {
        let mut w = ball_w.to_vec();
        // Cyclic perturbation ±20% on the top-weighted models
        for (i, v) in w.iter_mut().enumerate() {
            if *v > 0.0 {
                let factor = if i % 3 == 0 { 1.20 } else if i % 3 == 1 { 0.80 } else { 1.10 };
                *v *= factor;
            }
        }
        let total: f64 = w.iter().sum();
        if total > 0.0 { for v in w.iter_mut() { *v /= total; } }
        w
    };

    vec![
        ("Principal", ball_w.to_vec(), star_w.to_vec(), base_bt, base_st),
        ("Star-variant", ball_w.to_vec(), star_w.to_vec(), base_bt, star_variant_st),
        ("Exploratoire", exploratory_bw, star_w.to_vec(), base_bt, base_st),
    ]
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

fn cmd_backtest_jackpot_top3(
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
    // Dummy call to get n_profiles for display
    let n_profiles_display = 3;
    println!(
        "Backtest jackpot multi-perspectives — {} tirages, {} suggestions/profil, {} profils ({})\n",
        last, n_suggestions / n_profiles_display, n_profiles_display, temp_label,
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
        actual_balls: [u8; 5],
        actual_stars: [u8; 2],
        grids: Vec<(String, [u8; 5], [u8; 2], u8, u8)>, // (profil, balls, stars, match_b, match_s)
        best_balls: u8,
        best_stars: u8,
        best_profile: String,
        proximity: f64,
        conviction: f64,
    }

    let mut results: Vec<DrawResult> = Vec::new();
    let mut match_matrix = [[0u32; 3]; 6];
    let mut profile_best_count: Vec<u32> = Vec::new(); // initialized per-iteration from profile count

    for i in 0..last {
        let test_draw = &draws[i];
        let training_draws = &draws[i + 1..];

        pb.set_message(test_draw.date.to_string());

        // Build base combiner to extract weights
        let models = all_models();
        let model_names: Vec<String> = models.iter().map(|m| m.name().to_string()).collect();
        let (base_ball_w, base_star_w) = match weights {
            Some(w) => {
                let bw: Vec<f64> = models.iter()
                    .map(|m| w.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                let sw: Vec<f64> = models.iter()
                    .map(|m| w.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                (bw, sw)
            }
            None => {
                let n = models.len();
                (vec![1.0 / n as f64; n], vec![1.0 / n as f64; n])
            }
        };
        drop(models);

        // Apply top_models filtering if requested
        let (mut ball_w, mut star_w) = (base_ball_w, base_star_w);
        if top_models > 0 {
            filter_top_models_silent(&mut ball_w, &mut star_w, top_models);
        }

        // Compute conviction on calibrated profile
        let cal_models = all_models();
        let cal_combiner = EnsembleCombiner::with_weights(cal_models, ball_w.clone(), star_w.clone());
        let ball_pred = cal_combiner.predict_with_agreement_boost(training_draws, Pool::Balls, 0.20);
        let star_pred = cal_combiner.predict_with_agreement_boost(training_draws, Pool::Stars, 0.20);
        let conviction = compute_conviction(
            &ball_pred.distribution, &star_pred.distribution,
            &ball_pred.spread, &star_pred.spread,
        );
        let (eff_bt, eff_st) = if let Some(t) = temperature {
            (t, t)
        } else {
            conviction_temperature_split(&conviction)
        };

        // Build shared resources once
        let filter = StructuralFilter::adaptive(training_draws);
        let coherence = CoherenceScorer::from_history(training_draws, Pool::Balls);
        let mut joint_model = lemillion_ensemble::models::joint::JointConditionalModel::default();
        joint_model.train(training_draws);

        // Star pair distribution for pair-aware scoring
        let bt_star_pair_model = lemillion_ensemble::models::star_pair::StarPairModel::default();
        let bt_star_pair_probs = bt_star_pair_model.predict_pair_distribution(training_draws);

        // Exclusion set from consensus
        let bt_ball_pred = cal_combiner.predict_with_agreement_boost(training_draws, Pool::Balls, 0.20);
        let bt_ball_consensus = build_consensus_map(&bt_ball_pred, Pool::Balls);
        let bt_excluded = compute_exclusion_set(&bt_ball_consensus, -0.3, 10);
        let bt_excluded_ref = if bt_excluded.is_empty() { None } else { Some(bt_excluded.as_slice()) };

        // Ball→star conditioner
        let bt_conditioner = BallStarConditioner::from_history(training_draws);
        let bt_star_coherence = StarCoherenceScorer::from_history(training_draws);

        // Build weight profiles with per-profile temperatures
        let profiles = build_weight_profiles(&ball_w, &star_w, &model_names, eff_bt, eff_st);
        let n_per_profile = (n_suggestions / profiles.len()).max(100);

        // Initialize profile_best_count on first iteration
        if profile_best_count.is_empty() {
            profile_best_count = vec![0u32; profiles.len()];
        }

        // Generate one grid per profile with diversity constraint
        let mut selected_grids: Vec<(String, lemillion_db::models::Suggestion)> = Vec::new();

        for (label, prof_bw, prof_sw, prof_bt, prof_st) in &profiles {
            let prof_models = all_models();
            let prof_combiner = EnsembleCombiner::with_weights(prof_models, prof_bw.clone(), prof_sw.clone());
            let mut bp = prof_combiner.predict(training_draws, Pool::Balls);
            let sp = prof_combiner.predict(training_draws, Pool::Stars);

            // v16: Beta-transform post-pooling
            if let Some(w) = weights {
                if let Some((alpha, beta)) = w.beta_balls {
                    lemillion_ensemble::ensemble::beta_transform(&mut bp.distribution, alpha, beta);
                }
            }

            let bd = if (*prof_bt - 1.0).abs() > 1e-9 {
                apply_temperature(&bp.distribution, *prof_bt)
            } else {
                bp.distribution.clone()
            };
            let mut star_dist = sp.distribution.clone();
            if let Some(w) = weights {
                if let Some((alpha, beta)) = w.beta_stars {
                    lemillion_ensemble::ensemble::beta_transform(&mut star_dist, alpha, beta);
                }
            }
            let sd = if (*prof_st - 1.0).abs() > 1e-9 {
                apply_temperature(&star_dist, *prof_st)
            } else {
                star_dist
            };

            let result = generate_suggestions_jackpot(
                &bd, &sd, n_per_profile, Some(&filter),
                Some(&coherence), Some(&joint_model),
                bt_star_pair_probs.as_ref(), bt_excluded_ref,
                Some(&bt_conditioner),
                None,
                Some(&bt_star_coherence),
                None, None,
                None,
            )?;

            // Pick best grid with max 2 common balls with any already selected grid
            let mut picked = false;
            for s in &result.suggestions {
                let dominated = selected_grids.iter().any(|(_, t)| {
                    let common = s.balls.iter().filter(|b| t.balls.contains(b)).count();
                    common >= 3
                });
                if !dominated {
                    selected_grids.push((label.to_string(), s.clone()));
                    picked = true;
                    break;
                }
            }
            // Fallback: take top-1 if nothing diverse enough
            if !picked {
                if let Some(s) = result.suggestions.first() {
                    selected_grids.push((label.to_string(), s.clone()));
                }
            }
        }

        // Evaluate the 3 grids vs actual draw
        let mut best_balls = 0u8;
        let mut best_stars = 0u8;
        let mut best_prox = 0.0f64;
        let mut best_profile = String::new();
        let mut best_profile_idx = 0usize;
        let mut grids_info = Vec::new();

        for (idx, (label, grid)) in selected_grids.iter().enumerate() {
            let (bm, sm) = count_matches(&grid.balls, &grid.stars, test_draw);
            let prox = proximity_score(bm, sm);
            grids_info.push((label.clone(), grid.balls, grid.stars, bm, sm));
            if prox > best_prox || (prox == best_prox && bm + sm > best_balls + best_stars) {
                best_balls = bm;
                best_stars = sm;
                best_prox = prox;
                best_profile = label.clone();
                best_profile_idx = idx;
            }
        }

        match_matrix[best_balls as usize][best_stars as usize] += 1;
        if best_profile_idx < profile_best_count.len() {
            profile_best_count[best_profile_idx] += 1;
        }

        results.push(DrawResult {
            date: test_draw.date.clone(),
            actual_balls: test_draw.balls,
            actual_stars: test_draw.stars,
            grids: grids_info,
            best_balls,
            best_stars,
            best_profile,
            proximity: best_prox,
            conviction: conviction.overall,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Detailed table
    use comfy_table::{Table, Cell, Color};
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_FULL);
    let n_grids = profile_best_count.len();
    let mut header: Vec<String> = vec!["Date".into(), "Tirage réel".into()];
    for i in 1..=n_grids {
        header.push(format!("G{}", i));
    }
    header.extend(["Best".into(), "Prox.".into(), "Conv.".into()]);
    table.set_header(header);

    for r in &results {
        let draw_str = format!(
            "{:2}-{:2}-{:2}-{:2}-{:2}+{:2}-{:2}",
            r.actual_balls[0], r.actual_balls[1], r.actual_balls[2],
            r.actual_balls[3], r.actual_balls[4],
            r.actual_stars[0], r.actual_stars[1],
        );

        let mut row_cells = vec![
            Cell::new(&r.date),
            Cell::new(&draw_str),
        ];

        for (label, _balls, _stars, bm, sm) in &r.grids {
            let match_str = format!("{}+{}", bm, sm);
            let is_best = label == &r.best_profile;
            let color = if *bm >= 3 && *sm >= 2 {
                Color::Green
            } else if proximity_score(*bm, *sm) >= 1.0 {
                Color::Yellow
            } else if proximity_score(*bm, *sm) > 0.0 {
                Color::Cyan
            } else {
                Color::DarkGrey
            };
            let cell_str = if is_best { format!("[{}]", match_str) } else { match_str };
            row_cells.push(Cell::new(&cell_str).fg(color));
        }
        // Pad if less than n_grids
        while row_cells.len() < 2 + n_grids {
            row_cells.push(Cell::new("-"));
        }

        let best_match = format!("{}+{}", r.best_balls, r.best_stars);
        let best_color = if r.proximity >= 30.0 {
            Color::Green
        } else if r.proximity >= 1.0 {
            Color::Yellow
        } else if r.proximity > 0.0 {
            Color::Cyan
        } else {
            Color::DarkGrey
        };
        row_cells.push(Cell::new(&best_match).fg(best_color));
        row_cells.push(Cell::new(format!("{:.1}", r.proximity)).fg(best_color));
        row_cells.push(Cell::new(format!("{:.2}", r.conviction)));

        table.add_row(row_cells);
    }
    println!("{table}");

    // B×S matrix: observed vs expected
    let mut expected_matrix = [[0.0f64; 3]; 6];
    struct MatchCell { b: u8, s: u8, prob: f64, prox: f64 }
    let mut cells: Vec<MatchCell> = Vec::new();
    for b in 0..=5u8 {
        for s in 0..=2u8 {
            cells.push(MatchCell { b, s, prob: random_match_prob(b, s), prox: proximity_score(b, s) });
        }
    }
    cells.sort_by(|a, b| {
        b.prox.partial_cmp(&a.prox).unwrap_or(std::cmp::Ordering::Equal)
            .then((b.b + b.s).cmp(&(a.b + a.s)))
    });
    let n_grids_for_expected = n_grids;
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
        let p_at_least_group = 1.0 - (1.0 - cum_prob - group_single_prob).powi(n_grids_for_expected as i32);
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

    println!("\n── Matrice Boules × Étoiles (meilleur match parmi {} perspectives) ──", n_grids);
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
    println!("  (observé (attendu aléatoire pour {} grilles))", n_grids);

    // Aggregated proximity scores
    let total_proximity: f64 = results.iter().map(|r| r.proximity).sum();
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

    let draws_3plus = results.iter().filter(|r| r.best_balls >= 3).count();
    let draws_2stars = results.iter().filter(|r| r.best_stars >= 2).count();
    let draws_3plus2 = results.iter().filter(|r| r.best_balls >= 3 && r.best_stars >= 2).count();

    println!("\n── Highlights ──");
    println!("  3+ boules matchées    : {}/{}", draws_3plus, last);
    println!("  2 étoiles matchées    : {}/{} ({:.1}%)", draws_2stars, last, 100.0 * draws_2stars as f64 / last as f64);
    println!("  3+ boules + 2 étoiles : {}/{}", draws_3plus2, last);

    // Profile contribution (dynamic labels from last iteration's profiles)
    let profile_labels = ["Principal", "Star-variant", "Exploratoire"];
    println!("\n── Contribution par profil (meilleur match) ──");
    for (idx, count) in profile_best_count.iter().enumerate() {
        let label = if idx < profile_labels.len() { profile_labels[idx] } else { "?" };
        println!("  {:12} : {:3}/{} ({:.0}%)", label, count, last,
            100.0 * *count as f64 / last as f64);
    }

    Ok(())
}

/// Silent version of filter_top_models (no println output, for backtest inner loop)
fn filter_top_models_silent(ball_weights: &mut [f64], star_weights: &mut [f64], top_n: usize) {
    let mut ball_indexed: Vec<(usize, f64)> = ball_weights.iter().copied().enumerate().collect();
    ball_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_ball: Vec<usize> = ball_indexed.iter().take(top_n).map(|(i, _)| *i).collect();
    for (i, w) in ball_weights.iter_mut().enumerate() {
        if !top_ball.contains(&i) { *w = 0.0; }
    }
    let bs: f64 = ball_weights.iter().sum();
    if bs > 0.0 { for w in ball_weights.iter_mut() { *w /= bs; } }

    let mut star_indexed: Vec<(usize, f64)> = star_weights.iter().copied().enumerate().collect();
    star_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_star: Vec<usize> = star_indexed.iter().take(top_n).map(|(i, _)| *i).collect();
    for (i, w) in star_weights.iter_mut().enumerate() {
        if !top_star.contains(&i) { *w = 0.0; }
    }
    let ss: f64 = star_weights.iter().sum();
    if ss > 0.0 { for w in star_weights.iter_mut() { *w /= ss; } }
}

fn cmd_backtest_few_grids(
    draws: &[Draw],
    weights: Option<&EnsembleWeights>,
    last: usize,
    n_suggestions: usize,
    n_grids: usize,
    top_models: usize,
) -> Result<()> {
    // v21: Load optimized hyperparams if available, else fall back to default few-grid temperature
    let hyper = lemillion_ensemble::sampler::HyperParams::load(std::path::Path::new("hyperparams.json"));
    let has_optimized = std::path::Path::new("hyperparams.json").exists();
    let (fg_bt, fg_st) = if has_optimized {
        (hyper.t_balls, hyper.t_stars)
    } else {
        few_grid_temperature(n_grids)
    };
    println!(
        "Backtest few-grid — {} tirages, {} grilles, T_balls={:.4}, T_stars={:.4}{}, {} suggestions/profil\n",
        last, n_grids, fg_bt, fg_st,
        if has_optimized { " [optimisé]" } else { "" },
        n_suggestions,
    );

    let pb = ProgressBar::new(last as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    struct FgResult {
        date: String,
        actual_balls: [u8; 5],
        actual_stars: [u8; 2],
        grids: Vec<([u8; 5], [u8; 2], u8, u8)>,
        best_balls: u8,
        best_stars: u8,
        p52_combined: f64,
    }

    let mut results: Vec<FgResult> = Vec::new();
    let mut match_matrix = [[0u32; 3]; 6];
    const TOTAL_COMBINATIONS: f64 = 139_838_160.0;

    for i in 0..last {
        let test_draw = &draws[i];
        let training_draws = &draws[i + 1..];

        pb.set_message(test_draw.date.to_string());

        let models = all_models();
        let model_names: Vec<String> = models.iter().map(|m| m.name().to_string()).collect();
        let (mut ball_w, mut star_w) = match weights {
            Some(w) => {
                let bw: Vec<f64> = models.iter()
                    .map(|m| w.ball_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                let sw: Vec<f64> = models.iter()
                    .map(|m| w.star_weights.iter().find(|(n, _)| n == m.name()).map(|(_, w)| *w).unwrap_or(0.0))
                    .collect();
                (bw, sw)
            }
            None => {
                let n = models.len();
                (vec![1.0 / n as f64; n], vec![1.0 / n as f64; n])
            }
        };

        if top_models > 0 {
            filter_top_models_silent(&mut ball_w, &mut star_w, top_models);
        }

        // v15: MetaPredictor — ajustement contextuel des poids (aligné avec production)
        if let Some(w) = weights {
            if !w.detailed_ll.is_empty() {
                use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
                let features = RegimeFeatures::from_draws(training_draws);
                if let Some(meta) = MetaPredictor::train(training_draws, &w.detailed_ll, 1.0) {
                    let adjustments = meta.weight_adjustments(&features);
                    for (name, adj) in &adjustments {
                        if let Some(idx) = model_names.iter().position(|n| n == name) {
                            if idx < ball_w.len() {
                                ball_w[idx] *= adj;
                            }
                        }
                    }
                    let total: f64 = ball_w.iter().sum();
                    if total > 0.0 { ball_w.iter_mut().for_each(|w| *w /= total); }
                }
                if !w.star_detailed_ll.is_empty() {
                    if let Some(meta) = MetaPredictor::train(training_draws, &w.star_detailed_ll, 1.0) {
                        let adjustments = meta.weight_adjustments(&features);
                        for (name, adj) in &adjustments {
                            if let Some(idx) = model_names.iter().position(|n| n == name) {
                                if idx < star_w.len() {
                                    star_w[idx] *= adj;
                                }
                            }
                        }
                        let total: f64 = star_w.iter().sum();
                        if total > 0.0 { star_w.iter_mut().for_each(|w| *w /= total); }
                    }
                }
            }
        }

        // v15: Hedge weights — ajustement multiplicatif réactif (aligné avec production)
        {
            let (hedged_ball, hedged_star) = compute_hedge_weights(
                &models, training_draws, &ball_w, &star_w, 100, 0.10,
            );
            ball_w = hedged_ball;
            star_w = hedged_star;

            // Re-apply family cap after hedge
            let te_f: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
            let stresa_f: &[&str] = &["StresaSGD", "StresaChaos"];
            let fams: Vec<(&[&str], f64)> = vec![(te_f, 0.20), (stresa_f, 0.15)];
            apply_family_cap_vecs(&model_names, &mut ball_w, &fams);
            apply_family_cap_vecs(&model_names, &mut star_w, &fams);
        }

        drop(models);

        // Build combiner and predict
        let profiles = build_weight_profiles(&ball_w, &star_w, &model_names, fg_bt, fg_st);
        let n_per_profile = (n_suggestions / profiles.len()).max(500);

        let filter = StructuralFilter::adaptive(training_draws);
        let coherence = CoherenceScorer::from_history(training_draws, Pool::Balls);
        let mut joint_model = lemillion_ensemble::models::joint::JointConditionalModel::default();
        joint_model.train(training_draws);
        let star_pair_model = lemillion_ensemble::models::star_pair::StarPairModel::default();
        let star_pair_probs = star_pair_model.predict_pair_distribution(training_draws);
        let conditioner = BallStarConditioner::from_history(training_draws);
        let star_coherence = StarCoherenceScorer::from_history(training_draws);

        // Collect candidates from all profiles
        let mut all_candidates: Vec<lemillion_db::models::Suggestion> = Vec::new();
        let mut profile_dists: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

        for (_label, prof_bw, prof_sw, prof_bt, prof_st) in &profiles {
            let prof_models = all_models();
            let prof_combiner = EnsembleCombiner::with_weights(prof_models, prof_bw.clone(), prof_sw.clone());
            // v21: use predict_decorrelated when correlation matrix available (aligned with production)
            let has_corr = weights.map_or(false, |w| !w.correlation_matrix.is_empty());
            let (mut bp, sp) = if has_corr {
                let w = weights.unwrap();
                let ball_corr = &w.correlation_matrix;
                let star_corr = if w.star_correlation_matrix.is_empty() { &w.correlation_matrix } else { &w.star_correlation_matrix };
                (
                    prof_combiner.predict_decorrelated(training_draws, Pool::Balls, ball_corr, 0.60),
                    prof_combiner.predict_decorrelated(training_draws, Pool::Stars, star_corr, 0.60),
                )
            } else {
                (
                    prof_combiner.predict_with_agreement_boost(training_draws, Pool::Balls, 0.20),
                    prof_combiner.predict_with_agreement_boost(training_draws, Pool::Stars, 0.20),
                )
            };

            // v15: online/offline blend
            {
                use lemillion_ensemble::ensemble::online::online_offline_blend;
                bp.distribution = online_offline_blend(&bp.distribution, training_draws, Pool::Balls, 8);
                // Stars: skip online blend (v15b)
            }

            // Note: skip beta-transform in few-grid mode — few_grid_temperature() already provides
            // the right sharpening, and beta-transform interferes with it.

            let bd = if (*prof_bt - 1.0).abs() > 1e-9 {
                apply_temperature(&bp.distribution, *prof_bt)
            } else {
                bp.distribution.clone()
            };
            let sd = if (*prof_st - 1.0).abs() > 1e-9 {
                apply_temperature(&sp.distribution, *prof_st)
            } else {
                sp.distribution.clone()
            };

            let result = generate_suggestions_jackpot(
                &bd, &sd, n_per_profile, Some(&filter),
                Some(&coherence), Some(&joint_model),
                star_pair_probs.as_ref(), None,
                Some(&conditioner), None,
                Some(&star_coherence),
                None, None,
                None,
            )?;

            for s in &result.suggestions {
                all_candidates.push(s.clone());
            }
            profile_dists.push((bd, sd));
        }

        // Sort by score descending, select optimal n grids
        all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        // v21: Exact exhaustive selection for N≤5, greedy for larger N
        let selected = if n_grids <= 5 {
            select_optimal_n_grids_exact(&all_candidates, n_grids, 3)
        } else {
            select_optimal_n_grids(&all_candidates, n_grids, 3, 1)
        };

        // Evaluate against actual draw
        let mut best_balls = 0u8;
        let mut best_stars = 0u8;
        let mut grids_info = Vec::new();
        let mut grid_probs = Vec::new();

        // Use the main profile dist for P(5+2) calculation
        let (ref main_bd, ref main_sd) = profile_dists[0];

        for grid in &selected {
            let (bm, sm) = count_matches(&grid.balls, &grid.stars, test_draw);
            grids_info.push((grid.balls, grid.stars, bm, sm));
            if bm > best_balls || (bm == best_balls && sm > best_stars) {
                best_balls = bm;
                best_stars = sm;
            }
            let score = compute_bayesian_score(&grid.balls, &grid.stars, main_bd, main_sd);
            grid_probs.push(score / TOTAL_COMBINATIONS);
        }

        let p52_combined = 1.0 - grid_probs.iter().map(|&p| 1.0 - p).product::<f64>();

        match_matrix[best_balls as usize][best_stars as usize] += 1;

        results.push(FgResult {
            date: test_draw.date.clone(),
            actual_balls: test_draw.balls,
            actual_stars: test_draw.stars,
            grids: grids_info,
            best_balls,
            best_stars,
            p52_combined,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Display results table
    use comfy_table::{Table, Cell, Color};
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_FULL);
    let mut header: Vec<String> = vec!["Date".into(), "Tirage".into()];
    for i in 1..=n_grids {
        header.push(format!("G{}", i));
    }
    header.extend(["Best".into(), "P(5+2)".into()]);
    table.set_header(header);

    for r in &results {
        let draw_str = format!(
            "{:2}-{:2}-{:2}-{:2}-{:2}+{:2}-{:2}",
            r.actual_balls[0], r.actual_balls[1], r.actual_balls[2],
            r.actual_balls[3], r.actual_balls[4],
            r.actual_stars[0], r.actual_stars[1],
        );

        let mut row_cells = vec![
            Cell::new(&r.date),
            Cell::new(&draw_str),
        ];

        for (_balls, _stars, bm, sm) in &r.grids {
            let match_str = format!("{}+{}", bm, sm);
            let color = if *bm >= 3 && *sm >= 2 {
                Color::Green
            } else if proximity_score(*bm, *sm) >= 1.0 {
                Color::Yellow
            } else if proximity_score(*bm, *sm) > 0.0 {
                Color::Cyan
            } else {
                Color::DarkGrey
            };
            row_cells.push(Cell::new(&match_str).fg(color));
        }
        while row_cells.len() < 2 + n_grids {
            row_cells.push(Cell::new("-"));
        }

        let best_match = format!("{}+{}", r.best_balls, r.best_stars);
        let best_color = if r.best_balls >= 3 { Color::Yellow } else { Color::DarkGrey };
        row_cells.push(Cell::new(&best_match).fg(best_color));
        row_cells.push(Cell::new(format!("{:.2e}", r.p52_combined)));

        table.add_row(row_cells);
    }
    println!("{table}");

    // Match matrix
    println!("\n── Matrice Boules × Étoiles (meilleur match parmi {} grilles) ──", n_grids);
    println!("         0★         1★         2★");
    for b in (0..=5u8).rev() {
        let mut row = format!("{}B :", b);
        for s in 0..=2u8 {
            let obs = match_matrix[b as usize][s as usize];
            let exp = random_match_prob(b, s) * n_grids as f64 * last as f64;
            row.push_str(&format!("  {:3} ({:5.1})", obs, exp));
        }
        println!("{row}");
    }

    // Summary stats
    let avg_p52: f64 = results.iter().map(|r| r.p52_combined).sum::<f64>() / last as f64;
    let p_random = n_grids as f64 / TOTAL_COMBINATIONS;
    let factor = avg_p52 / p_random;
    let draws_3plus = results.iter().filter(|r| r.best_balls >= 3).count();
    let draws_2stars = results.iter().filter(|r| r.best_stars >= 2).count();

    println!("\n── Résumé few-grid ({} grilles) ──", n_grids);
    println!("  P(5+2) moyenne      : {:.2e}", avg_p52);
    println!("  P(5+2) aléatoire    : {:.2e}", p_random);
    println!("  Facteur vs aléatoire: {:.1}x", factor);
    println!("  3+ boules matchées  : {}/{} ({:.1}%)", draws_3plus, last, 100.0 * draws_3plus as f64 / last as f64);
    println!("  2 étoiles matchées  : {}/{} ({:.1}%)", draws_2stars, last, 100.0 * draws_2stars as f64 / last as f64);
    println!("  Coût par tirage     : {:.2} EUR", n_grids as f64 * 2.50);

    Ok(())
}

fn cmd_backtest_realistic(
    draws: &[Draw],
    weights: Option<&EnsembleWeights>,
    last: usize,
    temperature: Option<f64>,
    star_strategy: StarStrategy,
) -> Result<()> {
    use lemillion_ensemble::sampler::ConvictionVerdict;

    println!(
        "Backtest réaliste — {} tirages — 3 grilles (conviction basse/moyenne) ou 10 grilles (conviction haute)\n",
        last
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
        n_grids: usize,
        conviction: f64,
        verdict: String,
        best_balls: u8,
        best_stars: u8,
        cost: f64,
        gain: f64,
        all_gains: Vec<(u8, u8, f64)>, // (balls, stars, prize) for each winning grid
    }

    let mut results: Vec<DrawResult> = Vec::new();
    let mut tier_counts = [0u32; 13]; // match counts per prize tier

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

        // MetaPredictor : ajustement contextuel des poids
        if let Some(w) = weights {
            if !w.detailed_ll.is_empty() {
                use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
                let features = RegimeFeatures::from_draws(training_draws);
                if let Some(meta) = MetaPredictor::train(training_draws, &w.detailed_ll, 1.0) {
                    let adjustments = meta.weight_adjustments(&features);
                    for (name, adj) in &adjustments {
                        if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                            && idx < combiner.ball_weights.len()
                        {
                            combiner.ball_weights[idx] *= adj;
                        }
                    }
                    let total: f64 = combiner.ball_weights.iter().sum();
                    if total > 0.0 { for w in combiner.ball_weights.iter_mut() { *w /= total; } }
                }
                if !w.star_detailed_ll.is_empty() {
                    if let Some(meta) = MetaPredictor::train(training_draws, &w.star_detailed_ll, 1.0) {
                        let adjustments = meta.weight_adjustments(&features);
                        for (name, adj) in &adjustments {
                            if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                                && idx < combiner.star_weights.len()
                            {
                                combiner.star_weights[idx] *= adj;
                            }
                        }
                        let total: f64 = combiner.star_weights.iter().sum();
                        if total > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= total; } }
                    }
                }
            }
        }

        // Hedge weights
        {
            let (hedged_ball, hedged_star) = compute_hedge_weights(
                &combiner.models, training_draws,
                &combiner.ball_weights, &combiner.star_weights,
                100, 0.10,
            );
            combiner.ball_weights = hedged_ball;
            combiner.star_weights = hedged_star;

            // Re-apply family cap after hedge
            let te_f: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
            let stresa_f: &[&str] = &["StresaSGD", "StresaChaos"];
            let fams: Vec<(&[&str], f64)> = vec![(te_f, 0.20), (stresa_f, 0.15)];
            let mn: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
            apply_family_cap_vecs(&mn, &mut combiner.ball_weights, &fams);
            apply_family_cap_vecs(&mn, &mut combiner.star_weights, &fams);
        }

        let mut ball_pred = combiner.predict_with_agreement_boost(training_draws, Pool::Balls, 0.20);
        let mut star_pred = combiner.predict_with_agreement_boost(training_draws, Pool::Stars, 0.20);

        // v15: online/offline blend
        {
            use lemillion_ensemble::ensemble::online::online_offline_blend;
            ball_pred.distribution = online_offline_blend(&ball_pred.distribution, training_draws, Pool::Balls, 8);
            // Stars: skip online blend (v15b)
        }

        // v16: Beta-transform post-pooling (aligned with production)
        if let Some(w) = weights {
            if let Some((alpha, beta)) = w.beta_balls {
                lemillion_ensemble::ensemble::beta_transform(&mut ball_pred.distribution, alpha, beta);
            }
            if let Some((alpha, beta)) = w.beta_stars {
                lemillion_ensemble::ensemble::beta_transform(&mut star_pred.distribution, alpha, beta);
            }
        }

        // Conviction → nombre de grilles
        let conviction = compute_conviction(
            &ball_pred.distribution,
            &star_pred.distribution,
            &ball_pred.spread,
            &star_pred.spread,
        );
        let n_grids = match conviction.verdict {
            ConvictionVerdict::HighConviction => 10,
            _ => 3,
        };
        let verdict_str = match conviction.verdict {
            ConvictionVerdict::HighConviction => "HAUTE",
            ConvictionVerdict::MediumConviction => "moyenne",
            ConvictionVerdict::LowConviction => "basse",
        };

        // Température adaptative
        let (eff_bt, eff_st) = if let Some(t) = temperature {
            (t, t)
        } else {
            conviction_temperature_split(&conviction)
        };
        let ball_dist = if (eff_bt - 1.0).abs() > 1e-9 {
            apply_temperature(&ball_pred.distribution, eff_bt)
        } else {
            ball_pred.distribution.clone()
        };
        let star_dist = if (eff_st - 1.0).abs() > 1e-9 {
            apply_temperature(&star_pred.distribution, eff_st)
        } else {
            star_pred.distribution.clone()
        };

        let popularity = PopularityModel::from_history(training_draws);
        let diverse = generate_diverse_grids_with_strategy(
            &ball_dist, &star_dist, training_draws, n_grids, 42 + i as u64, Some(&popularity), star_strategy,
        );

        // Évaluer toutes les grilles contre le tirage réel
        let cost = n_grids as f64 * TICKET_PRICE;
        let mut draw_gain = 0.0f64;
        let mut best_balls = 0u8;
        let mut best_stars = 0u8;
        let mut all_gains: Vec<(u8, u8, f64)> = Vec::new();

        for grid in &diverse.grids {
            let (bm, sm) = count_matches(&grid.balls, &grid.stars, test_draw);

            if let Some(tier_idx) = match_to_tier(bm, sm) {
                let prize = PRIZE_TIERS[tier_idx].fixed_prize;
                draw_gain += prize;
                tier_counts[tier_idx] += 1;
                all_gains.push((bm, sm, prize));
            }

            let prox = proximity_score(bm, sm);
            let best_prox = proximity_score(best_balls, best_stars);
            if prox > best_prox || (prox == best_prox && bm + sm > best_balls + best_stars) {
                best_balls = bm;
                best_stars = sm;
            }
        }

        results.push(DrawResult {
            date: test_draw.date.clone(),
            n_grids,
            conviction: conviction.overall,
            verdict: verdict_str.to_string(),
            best_balls,
            best_stars,
            cost,
            gain: draw_gain,
            all_gains,
        });

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Tableau des résultats
    use comfy_table::{Table, Cell, Color};
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_FULL);
    table.set_header(vec!["Date", "Conv.", "Grilles", "Best", "Gains", "Coût", "Solde", "Détail gains"]);

    let mut total_cost = 0.0f64;
    let mut total_gain = 0.0f64;

    for r in &results {
        total_cost += r.cost;
        total_gain += r.gain;

        let match_str = if r.best_balls > 0 || r.best_stars > 0 {
            format!("{}+{}", r.best_balls, r.best_stars)
        } else {
            "—".to_string()
        };

        let detail = if r.all_gains.is_empty() {
            "—".to_string()
        } else {
            r.all_gains.iter()
                .map(|(b, s, p)| format!("{}+{}={:.0}€", b, s, p))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let color = if r.gain > r.cost {
            Color::Green
        } else if r.gain > 0.0 {
            Color::Yellow
        } else {
            Color::DarkGrey
        };

        table.add_row(vec![
            Cell::new(&r.date),
            Cell::new(format!("{:.2} {}", r.conviction, r.verdict)),
            Cell::new(r.n_grids.to_string()),
            Cell::new(&match_str).fg(color),
            Cell::new(if r.gain > 0.0 { format!("{:.0}€", r.gain) } else { "—".into() }).fg(color),
            Cell::new(format!("{:.1}€", r.cost)),
            Cell::new(format!("{:+.1}€", r.gain - r.cost)).fg(color),
            Cell::new(&detail).fg(color),
        ]);
    }
    println!("{table}");

    // Statistiques par palier
    println!("\n── Gains par palier EuroMillions ──");
    for (idx, tier) in PRIZE_TIERS.iter().enumerate() {
        if tier_counts[idx] > 0 {
            println!("  {} ({}€) : {} fois = {:.0}€",
                tier.name, tier.fixed_prize, tier_counts[idx],
                tier_counts[idx] as f64 * tier.fixed_prize);
        }
    }

    // Résumé financier
    let n_high = results.iter().filter(|r| r.verdict == "HAUTE").count();
    let n_medium = results.iter().filter(|r| r.verdict == "moyenne").count();
    let n_low = results.iter().filter(|r| r.verdict == "basse").count();
    let total_grids: usize = results.iter().map(|r| r.n_grids).sum();
    let draws_with_gain = results.iter().filter(|r| r.gain > 0.0).count();
    let draws_profitable = results.iter().filter(|r| r.gain > r.cost).count();

    println!("\n── Bilan financier sur {} tirages ──", last);
    println!("  Conviction haute      : {} tirages (10 grilles)", n_high);
    println!("  Conviction moyenne    : {} tirages (3 grilles)", n_medium);
    println!("  Conviction basse      : {} tirages (3 grilles)", n_low);
    println!("  Total grilles jouées  : {}", total_grids);
    println!("  Coût total            : {:.1}€", total_cost);
    println!("  Gains totaux          : {:.1}€", total_gain);
    println!("  Solde net             : {:+.1}€", total_gain - total_cost);
    println!("  ROI                   : {:.1}%", (total_gain / total_cost - 1.0) * 100.0);
    println!("  Tirages avec gain     : {}/{} ({:.1}%)", draws_with_gain, last, 100.0 * draws_with_gain as f64 / last as f64);
    println!("  Tirages rentables     : {}/{} ({:.1}%)", draws_profitable, last, 100.0 * draws_profitable as f64 / last as f64);

    // Comparaison avec jeu aléatoire
    let random_ev_per_grid = PRIZE_TIERS.iter().map(|t| t.probability * t.fixed_prize).sum::<f64>();
    let random_gain = total_grids as f64 * random_ev_per_grid;
    println!("\n── Comparaison vs aléatoire ──");
    println!("  EV aléatoire/grille   : {:.2}€ (coût : {:.2}€)", random_ev_per_grid, TICKET_PRICE);
    println!("  Gain aléatoire estimé : {:.1}€ sur {} grilles", random_gain, total_grids);
    println!("  Gain réel             : {:.1}€", total_gain);
    println!("  Ratio vs aléatoire    : {:.2}x", if random_gain > 0.0 { total_gain / random_gain } else { 0.0 });

    Ok(())
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

        // MetaPredictor : ajustement contextuel des poids
        if let Some(w) = weights {
            if !w.detailed_ll.is_empty() {
                use lemillion_ensemble::ensemble::meta::{MetaPredictor, RegimeFeatures};
                let features = RegimeFeatures::from_draws(training_draws);
                if let Some(meta) = MetaPredictor::train(training_draws, &w.detailed_ll, 1.0) {
                    let adjustments = meta.weight_adjustments(&features);
                    for (name, adj) in &adjustments {
                        if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                            && idx < combiner.ball_weights.len()
                        {
                            combiner.ball_weights[idx] *= adj;
                        }
                    }
                    let total: f64 = combiner.ball_weights.iter().sum();
                    if total > 0.0 { for w in combiner.ball_weights.iter_mut() { *w /= total; } }
                }
                if !w.star_detailed_ll.is_empty() {
                    if let Some(meta) = MetaPredictor::train(training_draws, &w.star_detailed_ll, 1.0) {
                        let adjustments = meta.weight_adjustments(&features);
                        for (name, adj) in &adjustments {
                            if let Some(idx) = combiner.models.iter().position(|m| m.name() == name)
                                && idx < combiner.star_weights.len()
                            {
                                combiner.star_weights[idx] *= adj;
                            }
                        }
                        let total: f64 = combiner.star_weights.iter().sum();
                        if total > 0.0 { for w in combiner.star_weights.iter_mut() { *w /= total; } }
                    }
                }
            }
        }

        // Hedge weights
        {
            let (hedged_ball, hedged_star) = compute_hedge_weights(
                &combiner.models, training_draws,
                &combiner.ball_weights, &combiner.star_weights,
                100, 0.10,
            );
            combiner.ball_weights = hedged_ball;
            combiner.star_weights = hedged_star;

            // Re-apply family cap after hedge
            let te_f: &[&str] = &["TransferEntropy", "RényiTE", "CrossTE"];
            let stresa_f: &[&str] = &["StresaSGD", "StresaChaos"];
            let fams: Vec<(&[&str], f64)> = vec![(te_f, 0.20), (stresa_f, 0.15)];
            let mn: Vec<String> = combiner.models.iter().map(|m| m.name().to_string()).collect();
            apply_family_cap_vecs(&mn, &mut combiner.ball_weights, &fams);
            apply_family_cap_vecs(&mn, &mut combiner.star_weights, &fams);
        }

        let mut ball_pred = combiner.predict_with_agreement_boost(training_draws, Pool::Balls, 0.20);
        let mut star_pred = combiner.predict_with_agreement_boost(training_draws, Pool::Stars, 0.20);

        // v15: online/offline blend
        {
            use lemillion_ensemble::ensemble::online::online_offline_blend;
            ball_pred.distribution = online_offline_blend(&ball_pred.distribution, training_draws, Pool::Balls, 8);
            // Stars: skip online blend (v15b)
        }

        // v16: Beta-transform post-pooling (aligned with production)
        if let Some(w) = weights {
            if let Some((alpha, beta)) = w.beta_balls {
                lemillion_ensemble::ensemble::beta_transform(&mut ball_pred.distribution, alpha, beta);
            }
            if let Some((alpha, beta)) = w.beta_stars {
                lemillion_ensemble::ensemble::beta_transform(&mut star_pred.distribution, alpha, beta);
            }
        }

        let conviction = compute_conviction(
            &ball_pred.distribution,
            &star_pred.distribution,
            &ball_pred.spread,
            &star_pred.spread,
        );
        let (eff_bt, eff_st) = if let Some(t) = temperature {
            (t, t)
        } else {
            conviction_temperature_split(&conviction)
        };
        let ball_dist = if (eff_bt - 1.0).abs() > 1e-9 {
            apply_temperature(&ball_pred.distribution, eff_bt)
        } else {
            ball_pred.distribution.clone()
        };
        let star_dist = if (eff_st - 1.0).abs() > 1e-9 {
            apply_temperature(&star_pred.distribution, eff_st)
        } else {
            star_pred.distribution.clone()
        };

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
        "dfa" | "hurst" => ResearchCategory::Dfa,
        "rqa" | "recurrence" => ResearchCategory::Rqa,
        "all" | "tout" | "tous" => ResearchCategory::All,
        _ => bail!("Catégorie invalide : '{}'. Valeurs acceptées : physical, mathematical, informational, dfa, rqa, all", tests),
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

fn cmd_optimize(
    conn: &lemillion_db::rusqlite::Connection,
    n_grids: usize,
    last: usize,
    suggestions: usize,
    iterations: usize,
    calibration_path: &str,
    output_path: &str,
    seed: u64,
) -> Result<()> {
    use lemillion_ensemble::ensemble::bayesopt::{optimize, hyperparams_bounds, BacktestCache, fast_backtest_objective};
    use lemillion_ensemble::sampler::HyperParams;

    let n = count_draws(conn)?;
    if n == 0 {
        bail!("Base vide. Lancez d'abord : lemillion-cli import");
    }

    let draws = fetch_last_draws(conn, n)?;
    let weights = load_weights(&PathBuf::from(calibration_path))?;

    println!("== BayesOpt — Optimisation des hyperparamètres ==\n");
    println!("  Grilles: {}  |  Tirages: {}  |  Suggestions: {}  |  Itérations: {}\n",
        n_grids, last, suggestions, iterations);

    // Build cache ONCE (expensive: all model predictions + hedge losses)
    println!("  Construction du cache (prédictions modèles + hedge losses)...");
    let cache = BacktestCache::build(&draws, &weights, last, suggestions);
    println!("  Cache construit : {} points de backtest\n", cache.draw_caches.len());

    // Baseline with default params (fast — uses cache)
    let baseline_params = HyperParams::default();
    let baseline_score = fast_backtest_objective(&cache, &baseline_params, n_grids);
    println!("  Baseline (défaut) : P(5+2) moyen = {:.4e}", baseline_score);

    let bounds = hyperparams_bounds();
    let n_initial = (iterations / 3).max(5);

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    // Wrap objective for BayesOpt (fast — hedge replay + pool + temperature + jackpot only)
    let mut best_so_far = f64::NEG_INFINITY;
    let mut objective = |params_vec: &[f64]| -> f64 {
        let hp = HyperParams::from_vec(params_vec);
        let score = fast_backtest_objective(&cache, &hp, n_grids);
        pb.inc(1);
        if score > best_so_far {
            best_so_far = score;
            pb.set_message(format!("best={:.4e} T=({:.2},{:.2}) CW=({:.1},{:.1}) η={:.3}",
                score, hp.t_balls, hp.t_stars, hp.coherence_weight, hp.star_coherence_weight, hp.hedge_eta));
        }
        score
    };

    let (best_params_vec, best_value) = optimize(&bounds, &mut objective, iterations, n_initial, seed);
    pb.finish_with_message("done");

    let best_params = HyperParams::from_vec(&best_params_vec);

    println!("\n== Résultats ==\n");
    println!("  Meilleurs hyperparamètres :");
    println!("    T_balls:              {:.4}", best_params.t_balls);
    println!("    T_stars:              {:.4}", best_params.t_stars);
    println!("    coherence_weight:     {:.2}", best_params.coherence_weight);
    println!("    star_coherence_weight:{:.2}", best_params.star_coherence_weight);
    println!("    hedge_eta:            {:.4}", best_params.hedge_eta);
    println!();
    println!("  P(5+2) moyen optimisé : {:.4e}", best_value);
    println!("  P(5+2) moyen baseline : {:.4e}", baseline_score);
    if baseline_score > 0.0 {
        println!("  Gain : {:.1}%", (best_value / baseline_score - 1.0) * 100.0);
    }

    // Save
    best_params.save(&PathBuf::from(output_path).as_path())?;
    println!("\n  Sauvegardé dans : {}", output_path);

    Ok(())
}

