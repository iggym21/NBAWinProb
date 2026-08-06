import sqlite3
import pytest
from app.db import (
    get_connection, init_db, insert_game, insert_events,
    list_games, get_game_events, game_exists,
)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    init_db(c)
    yield c
    c.close()


def test_init_db_creates_tables(conn):
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"games", "events"} <= tables


def test_insert_and_list_games(conn):
    insert_game(conn, {"game_id": "001", "home_team": "BOS", "away_team": "LAL", "home_win": 1})
    games = list_games(conn)
    assert len(games) == 1
    assert games[0]["game_id"] == "001"
    assert games[0]["home_win"] == 1


def test_game_exists(conn):
    assert game_exists(conn, "001") is False
    insert_game(conn, {"game_id": "001", "home_team": "BOS", "away_team": "LAL", "home_win": 1})
    assert game_exists(conn, "001") is True


def test_insert_and_get_events_ordered(conn):
    insert_game(conn, {"game_id": "001", "home_team": "BOS", "away_team": "LAL", "home_win": 1})
    events = [
        {"game_id": "001", "event_index": 1, "period": 1, "clock": "PT12M00.00S",
         "home_score": 0, "away_score": 0, "event_type": "Jump Ball",
         "description": "Jump Ball", "possession_team": None},
        {"game_id": "001", "event_index": 0, "period": 1, "clock": "PT12M00.00S",
         "home_score": 0, "away_score": 0, "event_type": "OTHER",
         "description": "Start of 1st Period", "possession_team": None},
    ]
    insert_events(conn, events)
    fetched = get_game_events(conn, "001")
    assert [e["event_index"] for e in fetched] == [0, 1]


def test_get_game_events_empty_for_unknown_game(conn):
    assert get_game_events(conn, "does-not-exist") == []
