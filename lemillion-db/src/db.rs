use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::Path;

use crate::models::Draw;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS draws (
    draw_id       TEXT PRIMARY KEY,
    day           TEXT NOT NULL,
    date          TEXT NOT NULL,
    ball_1        INTEGER NOT NULL,
    ball_2        INTEGER NOT NULL,
    ball_3        INTEGER NOT NULL,
    ball_4        INTEGER NOT NULL,
    ball_5        INTEGER NOT NULL,
    star_1        INTEGER NOT NULL,
    star_2        INTEGER NOT NULL,
    winner_count  INTEGER NOT NULL DEFAULT 0,
    winner_prize  REAL NOT NULL DEFAULT 0.0,
    my_million    TEXT NOT NULL DEFAULT ''
);
";

pub fn db_path() -> std::path::PathBuf {
    let mut path = std::env::current_dir().unwrap_or_default();
    path.push("data");
    path.push("lemillion.db");
    path
}

pub fn open_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Impossible de créer le répertoire {:?}", parent))?;
    }
    let conn = Connection::open(path)
        .with_context(|| format!("Impossible d'ouvrir la base {:?}", path))?;
    Ok(conn)
}

pub fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(SCHEMA)
        .context("Échec de la migration")?;

    // Migration v9 : colonnes pour l'ordre d'extraction physique + cycle_number
    let new_columns = [
        "ball_order_1 INTEGER",
        "ball_order_2 INTEGER",
        "ball_order_3 INTEGER",
        "ball_order_4 INTEGER",
        "ball_order_5 INTEGER",
        "star_order_1 INTEGER",
        "star_order_2 INTEGER",
        "cycle_number INTEGER",
    ];
    for col in &new_columns {
        let sql = format!("ALTER TABLE draws ADD COLUMN {}", col);
        // Ignorer "duplicate column" si la migration a déjà été appliquée
        let _ = conn.execute_batch(&sql);
    }

    Ok(())
}

pub fn insert_draw(conn: &Connection, draw: &Draw) -> Result<bool> {
    let changed = conn.execute(
        "INSERT OR IGNORE INTO draws (draw_id, day, date, ball_1, ball_2, ball_3, ball_4, ball_5, star_1, star_2, winner_count, winner_prize, my_million,
         ball_order_1, ball_order_2, ball_order_3, ball_order_4, ball_order_5, star_order_1, star_order_2, cycle_number)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
        rusqlite::params![
            draw.draw_id,
            draw.day,
            draw.date,
            draw.balls[0],
            draw.balls[1],
            draw.balls[2],
            draw.balls[3],
            draw.balls[4],
            draw.stars[0],
            draw.stars[1],
            draw.winner_count,
            draw.winner_prize,
            draw.my_million,
            draw.ball_order.map(|o| o[0]),
            draw.ball_order.map(|o| o[1]),
            draw.ball_order.map(|o| o[2]),
            draw.ball_order.map(|o| o[3]),
            draw.ball_order.map(|o| o[4]),
            draw.star_order.map(|o| o[0]),
            draw.star_order.map(|o| o[1]),
            draw.cycle_number,
        ],
    ).context("Échec de l'insertion")?;

    // Si le draw existait déjà mais sans ordre d'extraction, mettre à jour
    if changed == 0 && draw.ball_order.is_some() {
        conn.execute(
            "UPDATE draws SET ball_order_1=?2, ball_order_2=?3, ball_order_3=?4, ball_order_4=?5, ball_order_5=?6,
             star_order_1=?7, star_order_2=?8, cycle_number=?9
             WHERE draw_id=?1 AND ball_order_1 IS NULL",
            rusqlite::params![
                draw.draw_id,
                draw.ball_order.map(|o| o[0]),
                draw.ball_order.map(|o| o[1]),
                draw.ball_order.map(|o| o[2]),
                draw.ball_order.map(|o| o[3]),
                draw.ball_order.map(|o| o[4]),
                draw.star_order.map(|o| o[0]),
                draw.star_order.map(|o| o[1]),
                draw.cycle_number,
            ],
        ).context("Échec de la mise à jour de l'ordre")?;
    }

    Ok(changed > 0)
}

pub fn delete_draw(conn: &Connection, draw_id: &str) -> Result<bool> {
    let changed = conn.execute(
        "DELETE FROM draws WHERE draw_id = ?1",
        rusqlite::params![draw_id],
    ).context("Échec de la suppression")?;
    Ok(changed > 0)
}

const SELECT_DRAW_COLS: &str = "draw_id, day, date, ball_1, ball_2, ball_3, ball_4, ball_5, star_1, star_2, winner_count, winner_prize, my_million, ball_order_1, ball_order_2, ball_order_3, ball_order_4, ball_order_5, star_order_1, star_order_2, cycle_number";

fn draw_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Draw> {
    let bo1: Option<u8> = row.get(13)?;
    let bo2: Option<u8> = row.get(14)?;
    let bo3: Option<u8> = row.get(15)?;
    let bo4: Option<u8> = row.get(16)?;
    let bo5: Option<u8> = row.get(17)?;
    let so1: Option<u8> = row.get(18)?;
    let so2: Option<u8> = row.get(19)?;

    let ball_order = match (bo1, bo2, bo3, bo4, bo5) {
        (Some(a), Some(b), Some(c), Some(d), Some(e)) => Some([a, b, c, d, e]),
        _ => None,
    };
    let star_order = match (so1, so2) {
        (Some(a), Some(b)) => Some([a, b]),
        _ => None,
    };

    Ok(Draw {
        draw_id: row.get(0)?,
        day: row.get(1)?,
        date: row.get(2)?,
        balls: [
            row.get::<_, u8>(3)?,
            row.get::<_, u8>(4)?,
            row.get::<_, u8>(5)?,
            row.get::<_, u8>(6)?,
            row.get::<_, u8>(7)?,
        ],
        stars: [
            row.get::<_, u8>(8)?,
            row.get::<_, u8>(9)?,
        ],
        winner_count: row.get(10)?,
        winner_prize: row.get(11)?,
        my_million: row.get(12)?,
        ball_order,
        star_order,
        cycle_number: row.get(20)?,
    })
}

pub fn fetch_last_draws(conn: &Connection, limit: u32) -> Result<Vec<Draw>> {
    let sql = format!("SELECT {} FROM draws ORDER BY date DESC, draw_id DESC LIMIT ?1", SELECT_DRAW_COLS);
    let mut stmt = conn.prepare(&sql)?;
    let mut draws: Vec<Draw> = stmt.query_map([limit], draw_from_row)?.collect::<Result<Vec<_>, _>>()?;
    for d in &mut draws { d.normalize(); }
    Ok(draws)
}

pub fn fetch_last_draws_numbers(conn: &Connection, limit: u32) -> Result<Vec<([u8; 5], [u8; 2])>> {
    let mut stmt = conn.prepare(
        "SELECT ball_1, ball_2, ball_3, ball_4, ball_5, star_1, star_2
         FROM draws ORDER BY date DESC, draw_id DESC LIMIT ?1"
    )?;
    let rows: Vec<([u8; 5], [u8; 2])> = stmt.query_map([limit], |row| {
        let mut balls = [
            row.get::<_, u8>(0)?,
            row.get::<_, u8>(1)?,
            row.get::<_, u8>(2)?,
            row.get::<_, u8>(3)?,
            row.get::<_, u8>(4)?,
        ];
        let mut stars = [
            row.get::<_, u8>(5)?,
            row.get::<_, u8>(6)?,
        ];
        balls.sort();
        stars.sort();
        Ok((balls, stars))
    })?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Récupère le tirage correspondant à une date (format YYYY-MM-DD).
pub fn fetch_draw_by_date(conn: &Connection, date: &str) -> Result<Option<Draw>> {
    let sql = format!("SELECT {} FROM draws WHERE date = ?1 ORDER BY draw_id DESC LIMIT 1", SELECT_DRAW_COLS);
    let mut stmt = conn.prepare(&sql)?;
    let mut draws: Vec<Draw> = stmt.query_map([date], draw_from_row)?.collect::<Result<Vec<_>, _>>()?;
    for d in &mut draws { d.normalize(); }
    Ok(draws.pop())
}

/// Récupère tous les tirages avec date strictement avant `before_date` (format YYYY-MM-DD).
pub fn fetch_draws_before_date(conn: &Connection, before_date: &str) -> Result<Vec<Draw>> {
    let sql = format!("SELECT {} FROM draws WHERE date < ?1 ORDER BY date DESC, draw_id DESC", SELECT_DRAW_COLS);
    let mut stmt = conn.prepare(&sql)?;
    let mut draws: Vec<Draw> = stmt.query_map([before_date], draw_from_row)?.collect::<Result<Vec<_>, _>>()?;
    for d in &mut draws { d.normalize(); }
    Ok(draws)
}

pub fn count_draws(conn: &Connection) -> Result<u32> {
    let count: u32 = conn.query_row("SELECT COUNT(*) FROM draws", [], |row| row.get(0))?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_draw(id: &str, date: &str) -> Draw {
        Draw {
            draw_id: id.to_string(),
            day: "MARDI".to_string(),
            date: date.to_string(),
            balls: [1, 2, 3, 4, 5],
            stars: [1, 2],
            winner_count: 0,
            winner_prize: 0.0,
            my_million: "AA 000 0000".to_string(),
            ball_order: None,
            star_order: None,
            cycle_number: None,
        }
    }

    #[test]
    fn test_insert_and_count() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        assert_eq!(count_draws(&conn).unwrap(), 0);

        insert_draw(&conn, &test_draw("001", "2024-01-01")).unwrap();
        assert_eq!(count_draws(&conn).unwrap(), 1);
    }

    #[test]
    fn test_duplicate_ignored() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        let inserted = insert_draw(&conn, &test_draw("001", "2024-01-01")).unwrap();
        assert!(inserted);
        let inserted = insert_draw(&conn, &test_draw("001", "2024-01-01")).unwrap();
        assert!(!inserted);
        assert_eq!(count_draws(&conn).unwrap(), 1);
    }

    #[test]
    fn test_delete_draw() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        insert_draw(&conn, &test_draw("001", "2024-01-01")).unwrap();
        insert_draw(&conn, &test_draw("002", "2024-01-05")).unwrap();
        assert_eq!(count_draws(&conn).unwrap(), 2);

        let deleted = delete_draw(&conn, "001").unwrap();
        assert!(deleted);
        assert_eq!(count_draws(&conn).unwrap(), 1);

        let deleted = delete_draw(&conn, "001").unwrap();
        assert!(!deleted);

        let draws = fetch_last_draws(&conn, 10).unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].draw_id, "002");
    }

    #[test]
    fn test_fetch_order() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        insert_draw(&conn, &test_draw("001", "2024-01-01")).unwrap();
        insert_draw(&conn, &test_draw("002", "2024-01-05")).unwrap();
        insert_draw(&conn, &test_draw("003", "2024-01-03")).unwrap();

        let draws = fetch_last_draws(&conn, 10).unwrap();
        assert_eq!(draws.len(), 3);
        assert_eq!(draws[0].date, "2024-01-05");
        assert_eq!(draws[1].date, "2024-01-03");
        assert_eq!(draws[2].date, "2024-01-01");
    }
}
