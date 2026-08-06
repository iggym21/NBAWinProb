"""SQLite schema and access layer for cached play-by-play data."""
import sqlite3
from typing import TypedDict

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_win INTEGER NOT NULL CHECK (home_win IN (0, 1))
);

CREATE TABLE IF NOT EXISTS events (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    event_index INTEGER NOT NULL,
    period INTEGER NOT NULL,
    clock TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    possession_team TEXT,
    PRIMARY KEY (game_id, event_index)
);
"""


class GameRow(TypedDict):
    game_id: str
    home_team: str
    away_team: str
    home_win: int


class EventRow(TypedDict):
    game_id: str
    event_index: int
    period: int
    clock: str
    home_score: int
    away_score: int
    event_type: str
    description: str
    possession_team: str | None


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_game(conn: sqlite3.Connection, game: GameRow) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO games (game_id, home_team, away_team, home_win) "
        "VALUES (:game_id, :home_team, :away_team, :home_win)",
        game,
    )
    conn.commit()


def insert_events(conn: sqlite3.Connection, events: list[EventRow]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO events "
        "(game_id, event_index, period, clock, home_score, away_score, "
        " event_type, description, possession_team) "
        "VALUES (:game_id, :event_index, :period, :clock, :home_score, :away_score, "
        " :event_type, :description, :possession_team)",
        events,
    )
    conn.commit()


def list_games(conn: sqlite3.Connection) -> list[GameRow]:
    rows = conn.execute("SELECT * FROM games ORDER BY game_id").fetchall()
    return [dict(row) for row in rows]


def get_game_events(conn: sqlite3.Connection, game_id: str) -> list[EventRow]:
    rows = conn.execute(
        "SELECT * FROM events WHERE game_id = ? ORDER BY event_index ASC", (game_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def game_exists(conn: sqlite3.Connection, game_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM games WHERE game_id = ?", (game_id,)).fetchone()
    return row is not None
