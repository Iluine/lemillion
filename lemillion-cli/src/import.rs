use anyhow::{Context, Result, bail};
use lemillion_db::rusqlite::Connection;
use std::path::Path;

use lemillion_db::db::insert_draw;
use lemillion_db::models::Draw;

pub fn parse_french_decimal(s: &str) -> Result<f64> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(0.0);
    }
    let normalized = s.replace(',', ".");
    normalized
        .parse::<f64>()
        .with_context(|| format!("Impossible de parser le nombre: '{}'", s))
}

fn parse_record(record: &csv::StringRecord) -> Result<Draw> {
    let get = |idx: usize| -> Result<String> {
        record
            .get(idx)
            .map(|s| s.trim().to_string())
            .with_context(|| format!("Champ manquant à l'index {}", idx))
    };

    let get_u8 = |idx: usize| -> Result<u8> {
        let s = get(idx)?;
        s.parse::<u8>()
            .with_context(|| format!("Impossible de parser '{}' (index {})", s, idx))
    };

    let draw_id = get(0)?;
    let day = get(1)?;

    let raw_date = get(2)?;
    let date = parse_date(&raw_date)?;

    let balls: [u8; 5] = [
        get_u8(5)?,
        get_u8(6)?,
        get_u8(7)?,
        get_u8(8)?,
        get_u8(9)?,
    ];
    let stars: [u8; 2] = [get_u8(10)?, get_u8(11)?];

    let winner_count_str = get(15).unwrap_or_default();
    let winner_count: i32 = if winner_count_str.is_empty() {
        0
    } else {
        winner_count_str.parse().unwrap_or(0)
    };

    let winner_prize = parse_french_decimal(&get(16).unwrap_or_default()).unwrap_or(0.0);

    let my_million = get(73).unwrap_or_default();

    Ok(Draw {
        draw_id,
        day,
        date,
        balls,
        stars,
        winner_count,
        winner_prize,
        my_million,
    })
}

fn parse_date(raw: &str) -> Result<String> {
    let parts: Vec<&str> = raw.split('/').collect();
    if parts.len() != 3 {
        bail!("Format de date invalide: '{}'", raw);
    }
    Ok(format!("{}-{}-{}", parts[2], parts[1], parts[0]))
}

pub struct ImportResult {
    pub total_records: u32,
    pub inserted: u32,
    pub skipped: u32,
    pub errors: u32,
}

pub fn import_csv(conn: &Connection, path: &Path) -> Result<ImportResult> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Impossible d'ouvrir {:?}", path))?;

    let tx = conn.unchecked_transaction()
        .context("Impossible de démarrer la transaction")?;

    let mut result = ImportResult {
        total_records: 0,
        inserted: 0,
        skipped: 0,
        errors: 0,
    };

    for record_result in reader.records() {
        result.total_records += 1;
        match record_result {
            Ok(record) => {
                match parse_record(&record) {
                    Ok(draw) => {
                        match insert_draw(&tx, &draw) {
                            Ok(true) => result.inserted += 1,
                            Ok(false) => result.skipped += 1,
                            Err(e) => {
                                eprintln!("Erreur insertion tirage {}: {}", result.total_records, e);
                                result.errors += 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Erreur parsing ligne {}: {}", result.total_records, e);
                        result.errors += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("Erreur lecture ligne {}: {}", result.total_records, e);
                result.errors += 1;
            }
        }
    }

    tx.commit().context("Échec du commit")?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_french_decimal() {
        assert!((parse_french_decimal("109156,50").unwrap() - 109156.50).abs() < 0.001);
        assert!((parse_french_decimal("3,80").unwrap() - 3.80).abs() < 0.001);
        assert!((parse_french_decimal("0").unwrap() - 0.0).abs() < 0.001);
        assert!((parse_french_decimal("").unwrap() - 0.0).abs() < 0.001);
        assert!((parse_french_decimal("  42,5  ").unwrap() - 42.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_date() {
        assert_eq!(parse_date("17/02/2026").unwrap(), "2026-02-17");
        assert_eq!(parse_date("01/01/2020").unwrap(), "2020-01-01");
    }
}
