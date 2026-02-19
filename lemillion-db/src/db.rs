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
    Ok(())
}

pub fn insert_draw(conn: &Connection, draw: &Draw) -> Result<bool> {
    let changed = conn.execute(
        "INSERT OR IGNORE INTO draws (draw_id, day, date, ball_1, ball_2, ball_3, ball_4, ball_5, star_1, star_2, winner_count, winner_prize, my_million)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
        ],
    ).context("Échec de l'insertion")?;
    Ok(changed > 0)
}

pub fn fetch_last_draws(conn: &Connection, limit: u32) -> Result<Vec<Draw>> {
    let mut stmt = conn.prepare(
        "SELECT draw_id, day, date, ball_1, ball_2, ball_3, ball_4, ball_5, star_1, star_2, winner_count, winner_prize, my_million
         FROM draws ORDER BY date DESC, draw_id DESC LIMIT ?1"
    )?;
    let draws = stmt.query_map([limit], |row| {
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
        })
    })?.collect::<Result<Vec<_>, _>>()?;
    Ok(draws)
}

pub fn fetch_last_draws_numbers(conn: &Connection, limit: u32) -> Result<Vec<([u8; 5], [u8; 2])>> {
    let mut stmt = conn.prepare(
        "SELECT ball_1, ball_2, ball_3, ball_4, ball_5, star_1, star_2
         FROM draws ORDER BY date DESC, draw_id DESC LIMIT ?1"
    )?;
    let rows = stmt.query_map([limit], |row| {
        Ok((
            [
                row.get::<_, u8>(0)?,
                row.get::<_, u8>(1)?,
                row.get::<_, u8>(2)?,
                row.get::<_, u8>(3)?,
                row.get::<_, u8>(4)?,
            ],
            [
                row.get::<_, u8>(5)?,
                row.get::<_, u8>(6)?,
            ],
        ))
    })?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
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
