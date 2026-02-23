use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};

use crate::config::{EsnResult, GridSearchResults};

pub fn display_metrics(result: &EsnResult) {
    println!("\n== Metriques ESN ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metrique", "Validation", "Test"]);

    table.add_row(vec![
        Cell::new("Ball hit rate"),
        Cell::new(format!("{:.4}", result.val_ball_hit_rate)),
        Cell::new(format!("{:.4}", result.test_ball_hit_rate)),
    ]);
    table.add_row(vec![
        Cell::new("Star hit rate"),
        Cell::new(format!("{:.4}", result.val_star_hit_rate)),
        Cell::new(format!("{:.4}", result.test_star_hit_rate)),
    ]);
    table.add_row(vec![
        Cell::new("Ball top-K"),
        Cell::new(format!("{:.4}", result.val_ball_topk)),
        Cell::new(format!("{:.4}", result.test_ball_topk)),
    ]);
    table.add_row(vec![
        Cell::new("Star top-K"),
        Cell::new(format!("{:.4}", result.val_star_topk)),
        Cell::new(format!("{:.4}", result.test_star_topk)),
    ]);

    println!("{table}");

    display_lyapunov(result.lyapunov_exponent);

    println!(
        "\nTemps d'entrainement : {} ms",
        result.train_time_ms
    );
}

pub fn display_grid_search_top(results: &GridSearchResults, top_n: usize) {
    println!(
        "\n== Top {} configurations (sur {}) ==\n",
        top_n,
        results.results.len()
    );

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "#",
            "N_res",
            "rho",
            "spars",
            "alpha",
            "lambda",
            "in_sc",
            "enc",
            "wash",
            "val_B",
            "val_S",
            "test_B",
            "test_S",
            "lyap",
            "ms",
        ]);

    for (i, r) in results.results.iter().take(top_n).enumerate() {
        let enc_str = match r.config.encoding {
            crate::config::Encoding::OneHot => "OH",
            crate::config::Encoding::Normalized => "Norm",
        };
        let row = vec![
            format!("{}", i + 1),
            format!("{}", r.config.reservoir_size),
            format!("{:.2}", r.config.spectral_radius),
            format!("{:.2}", r.config.sparsity),
            format!("{:.1}", r.config.leaking_rate),
            format!("{:.0e}", r.config.ridge_lambda),
            format!("{:.2}", r.config.input_scaling),
            enc_str.to_string(),
            format!("{}", r.config.washout),
            format!("{:.4}", r.val_ball_hit_rate),
            format!("{:.4}", r.val_star_hit_rate),
            format!("{:.4}", r.test_ball_hit_rate),
            format!("{:.4}", r.test_star_hit_rate),
            format!("{:.2}", r.lyapunov_exponent),
            format!("{}", r.train_time_ms),
        ];

        if i == 0 {
            table.add_row(row.iter().map(|s| Cell::new(s).fg(Color::Green)).collect::<Vec<_>>());
        } else {
            table.add_row(row);
        }
    }

    println!("{table}");
}

pub fn display_prediction(ball_probs: &[f64], star_probs: &[f64]) {
    println!("\n== Prediction ESN ==\n");

    // Top-15 balls
    println!("-- Top 15 Boules --");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "Proba", "Ratio/unif"]);

    let mut ball_indices: Vec<usize> = (0..ball_probs.len()).collect();
    ball_indices.sort_by(|&a, &b| {
        ball_probs[b]
            .partial_cmp(&ball_probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let uniform_ball = 1.0 / 50.0;
    for (rank, &idx) in ball_indices.iter().take(15).enumerate() {
        let ratio = ball_probs[idx] / uniform_ball;
        let color = if rank < 5 { Color::Green } else { Color::White };
        table.add_row(vec![
            Cell::new(format!("{:2}", idx + 1)).fg(color),
            Cell::new(format!("{:.4}", ball_probs[idx])).fg(color),
            Cell::new(format!("{:.2}x", ratio)).fg(color),
        ]);
    }
    println!("{table}");

    // Top-6 stars
    println!("\n-- Top 6 Etoiles --");
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "Proba", "Ratio/unif"]);

    let mut star_indices: Vec<usize> = (0..star_probs.len()).collect();
    star_indices.sort_by(|&a, &b| {
        star_probs[b]
            .partial_cmp(&star_probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let uniform_star = 1.0 / 12.0;
    for (rank, &idx) in star_indices.iter().take(6).enumerate() {
        let ratio = star_probs[idx] / uniform_star;
        let color = if rank < 2 { Color::Yellow } else { Color::White };
        table.add_row(vec![
            Cell::new(format!("{:2}", idx + 1)).fg(color),
            Cell::new(format!("{:.4}", star_probs[idx])).fg(color),
            Cell::new(format!("{:.2}x", ratio)).fg(color),
        ]);
    }
    println!("{table}");
}

pub fn display_ensemble_metrics(results: &[EsnResult]) {
    let n = results.len() as f64;
    println!("\n== Metriques Ensemble ({} membres) ==\n", results.len());

    let mean = |f: fn(&EsnResult) -> f64| -> f64 { results.iter().map(f).sum::<f64>() / n };
    let std_dev = |f: fn(&EsnResult) -> f64| -> f64 {
        let m = mean(f);
        (results.iter().map(|r| (f(r) - m).powi(2)).sum::<f64>() / n).sqrt()
    };

    let metrics: Vec<(&str, fn(&EsnResult) -> f64)> = vec![
        ("Ball hit rate", |r| r.val_ball_hit_rate),
        ("Star hit rate", |r| r.val_star_hit_rate),
        ("Ball top-K", |r| r.val_ball_topk),
        ("Star top-K", |r| r.val_star_topk),
    ];
    let test_metrics: Vec<(&str, fn(&EsnResult) -> f64)> = vec![
        ("Ball hit rate", |r| r.test_ball_hit_rate),
        ("Star hit rate", |r| r.test_star_hit_rate),
        ("Ball top-K", |r| r.test_ball_topk),
        ("Star top-K", |r| r.test_star_topk),
    ];

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metrique", "Val (moy +/- std)", "Test (moy +/- std)"]);

    for (&(name, vf), &(_, tf)) in metrics.iter().zip(test_metrics.iter()) {
        table.add_row(vec![
            Cell::new(name),
            Cell::new(format!("{:.4} +/- {:.4}", mean(vf), std_dev(vf))),
            Cell::new(format!("{:.4} +/- {:.4}", mean(tf), std_dev(tf))),
        ]);
    }

    println!("{table}");

    // Lyapunov moyen
    let lyap_mean: f64 = results.iter().map(|r| r.lyapunov_exponent).sum::<f64>() / n;
    let lyap_std: f64 = (results
        .iter()
        .map(|r| (r.lyapunov_exponent - lyap_mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    println!(
        "\nLyapunov moyen : {:.4} +/- {:.4}",
        lyap_mean, lyap_std
    );

    let total_ms: u64 = results.iter().map(|r| r.train_time_ms).sum();
    println!("Temps total d'entrainement : {} ms", total_ms);
}

pub fn display_calibration(bins: &[(f64, f64, f64, usize)]) {
    println!("\n== Calibration ==\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Bin centre", "Pred moy", "Freq reelle", "N"]);

    for &(center, avg_pred, actual_freq, count) in bins {
        table.add_row(vec![
            format!("{:.3}", center),
            format!("{:.4}", avg_pred),
            format!("{:.4}", actual_freq),
            format!("{}", count),
        ]);
    }

    println!("{table}");
}

pub fn display_lyapunov(lambda: f64) {
    let interpretation = if lambda > 0.1 {
        "CHAOTIQUE - ESP non respectee"
    } else if lambda > 0.0 {
        "Marginalement chaotique"
    } else if lambda > -0.1 {
        "Stable (edge of chaos)"
    } else {
        "Tres stable (echo state property)"
    };

    println!(
        "\nExposant de Lyapunov : {:.4} ({})",
        lambda, interpretation
    );
}
