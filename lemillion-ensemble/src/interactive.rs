use std::io::{self, Write};

use anyhow::{Context, Result, bail};
use lemillion_db::db::insert_draw;
use lemillion_db::models::{Draw, validate_draw};

#[derive(Debug, PartialEq)]
enum InteractiveCommand {
    Add,
    Calibrate,
    Predict,
    History,
    Compare,
    Weights,
    Quit,
}

fn parse_command(input: &str) -> Option<InteractiveCommand> {
    match input.trim().to_lowercase().as_str() {
        "1" | "ajouter" | "add" => Some(InteractiveCommand::Add),
        "2" | "calibrer" | "calibrate" | "cal" => Some(InteractiveCommand::Calibrate),
        "3" | "predire" | "prédire" | "predict" | "pred" => Some(InteractiveCommand::Predict),
        "4" | "historique" | "history" | "hist" => Some(InteractiveCommand::History),
        "5" | "comparer" | "compare" | "comp" => Some(InteractiveCommand::Compare),
        "6" | "poids" | "weights" => Some(InteractiveCommand::Weights),
        "7" | "quitter" | "quit" | "q" | "exit" => Some(InteractiveCommand::Quit),
        _ => None,
    }
}

fn display_menu() {
    println!();
    println!("── Mode interactif ──");
    println!("  1. ajouter    Ajouter un tirage");
    println!("  2. calibrer   Calibrer les modèles");
    println!("  3. predire    Prédictions ensemble");
    println!("  4. historique Derniers tirages");
    println!("  5. comparer   Analyser une grille");
    println!("  6. poids      Afficher les poids");
    println!("  7. quitter    Quitter");
    println!();
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

fn prompt_with_default(msg: &str, default: &str) -> Result<String> {
    let input = prompt(&format!("{} [{}] : ", msg, default))?;
    if input.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(input)
    }
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

fn cmd_add_interactive(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
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
    println!(
        "  Boules: {}  Étoiles: {}",
        draw.balls.iter().map(|b| format!("{:2}", b)).collect::<Vec<_>>().join(" - "),
        draw.stars.iter().map(|s| format!("{:2}", s)).collect::<Vec<_>>().join(" - "),
    );

    let confirm = prompt("\nConfirmer l'insertion ? (o/n) : ")?;
    if confirm.to_lowercase() == "o" {
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

fn cmd_calibrate_interactive(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    let windows = prompt_with_default("Fenêtres (séparées par des virgules)", "20,30,40,50,60,80,100")?;
    let output = prompt_with_default("Fichier de sortie", "calibration.json")?;
    super::cmd_calibrate(conn, &windows, &output)
}

fn cmd_predict_interactive(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    let n_str = prompt_with_default("Nombre de suggestions", "5")?;
    let n: usize = n_str.parse().context("Nombre invalide")?;

    let seed_str = prompt_with_default("Seed (vide = date du jour)", "")?;
    let seed: Option<u64> = if seed_str.is_empty() {
        None
    } else {
        Some(seed_str.parse().context("Seed invalide")?)
    };

    super::cmd_predict(conn, "calibration.json", n, seed, 20, 2)
}

fn cmd_history_interactive(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    let n_str = prompt_with_default("Nombre de tirages", "10")?;
    let n: u32 = n_str.parse().context("Nombre invalide")?;
    super::cmd_history(conn, n)
}

fn cmd_compare_interactive(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    let balls = prompt_balls()?;
    let stars = prompt_stars()?;
    let numbers: Vec<u8> = balls.iter().chain(stars.iter()).copied().collect();
    super::cmd_compare(conn, &numbers)
}

pub fn run_interactive(conn: &lemillion_db::rusqlite::Connection) -> Result<()> {
    println!("Bienvenue dans le mode interactif de lemillion-ensemble !");

    loop {
        display_menu();
        let input = match prompt("> ") {
            Ok(s) => s,
            Err(_) => break, // EOF / Ctrl+D
        };

        if input.is_empty() {
            continue;
        }

        match parse_command(&input) {
            Some(InteractiveCommand::Quit) => {
                println!("Au revoir !");
                break;
            }
            Some(InteractiveCommand::Add) => {
                if let Err(e) = cmd_add_interactive(conn) {
                    println!("Erreur: {e:#}");
                }
            }
            Some(InteractiveCommand::Calibrate) => {
                if let Err(e) = cmd_calibrate_interactive(conn) {
                    println!("Erreur: {e:#}");
                }
            }
            Some(InteractiveCommand::Predict) => {
                if let Err(e) = cmd_predict_interactive(conn) {
                    println!("Erreur: {e:#}");
                }
            }
            Some(InteractiveCommand::History) => {
                if let Err(e) = cmd_history_interactive(conn) {
                    println!("Erreur: {e:#}");
                }
            }
            Some(InteractiveCommand::Compare) => {
                if let Err(e) = cmd_compare_interactive(conn) {
                    println!("Erreur: {e:#}");
                }
            }
            Some(InteractiveCommand::Weights) => {
                if let Err(e) = super::cmd_weights("calibration.json") {
                    println!("Erreur: {e:#}");
                }
            }
            None => {
                println!("Commande inconnue : '{}'. Tapez un numéro (1-7) ou un nom de commande.", input);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_command_by_number() {
        assert_eq!(parse_command("1"), Some(InteractiveCommand::Add));
        assert_eq!(parse_command("2"), Some(InteractiveCommand::Calibrate));
        assert_eq!(parse_command("3"), Some(InteractiveCommand::Predict));
        assert_eq!(parse_command("4"), Some(InteractiveCommand::History));
        assert_eq!(parse_command("5"), Some(InteractiveCommand::Compare));
        assert_eq!(parse_command("6"), Some(InteractiveCommand::Weights));
        assert_eq!(parse_command("7"), Some(InteractiveCommand::Quit));
    }

    #[test]
    fn test_parse_command_by_name() {
        assert_eq!(parse_command("ajouter"), Some(InteractiveCommand::Add));
        assert_eq!(parse_command("calibrer"), Some(InteractiveCommand::Calibrate));
        assert_eq!(parse_command("predire"), Some(InteractiveCommand::Predict));
        assert_eq!(parse_command("historique"), Some(InteractiveCommand::History));
        assert_eq!(parse_command("comparer"), Some(InteractiveCommand::Compare));
        assert_eq!(parse_command("poids"), Some(InteractiveCommand::Weights));
        assert_eq!(parse_command("quitter"), Some(InteractiveCommand::Quit));
    }

    #[test]
    fn test_parse_command_by_alias() {
        assert_eq!(parse_command("add"), Some(InteractiveCommand::Add));
        assert_eq!(parse_command("cal"), Some(InteractiveCommand::Calibrate));
        assert_eq!(parse_command("pred"), Some(InteractiveCommand::Predict));
        assert_eq!(parse_command("hist"), Some(InteractiveCommand::History));
        assert_eq!(parse_command("comp"), Some(InteractiveCommand::Compare));
        assert_eq!(parse_command("weights"), Some(InteractiveCommand::Weights));
        assert_eq!(parse_command("q"), Some(InteractiveCommand::Quit));
        assert_eq!(parse_command("exit"), Some(InteractiveCommand::Quit));
    }

    #[test]
    fn test_parse_command_case_insensitive() {
        assert_eq!(parse_command("QUIT"), Some(InteractiveCommand::Quit));
        assert_eq!(parse_command("Ajouter"), Some(InteractiveCommand::Add));
        assert_eq!(parse_command("CALIBRER"), Some(InteractiveCommand::Calibrate));
        assert_eq!(parse_command("Predire"), Some(InteractiveCommand::Predict));
    }

    #[test]
    fn test_parse_command_unknown() {
        assert_eq!(parse_command("foo"), None);
        assert_eq!(parse_command(""), None);
        assert_eq!(parse_command("8"), None);
        assert_eq!(parse_command("hello"), None);
    }
}
