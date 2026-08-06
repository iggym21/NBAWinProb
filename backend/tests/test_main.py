import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the app at fixture DB/model via env vars before import.
    db_copy = tmp_path / "fixture.db"
    shutil.copy(FIXTURES / "fixture.db", db_copy)
    monkeypatch.setenv("NBA_DB_PATH", str(db_copy))
    monkeypatch.setenv("NBA_MODEL_PATH", str(FIXTURES / "fixture_model.pt"))
    monkeypatch.setenv("NBA_REPLAY_INTERVAL_MS", "1")  # fast for tests

    import importlib
    import app.main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as c:
        yield c


def test_get_games_returns_fixture_game(client):
    resp = client.get("/games")
    assert resp.status_code == 200
    data = resp.json()
    assert any(g["game_id"] == "fixture001" for g in data["replay_games"])
    assert isinstance(data["live_available"], bool)


def test_replay_websocket_streams_increasing_events(client):
    with client.websocket_connect("/replay/fixture001") as ws:
        received = []
        try:
            while True:
                msg = ws.receive_json()
                received.append(msg)
        except Exception:
            pass
    assert len(received) == 15
    assert [m["event_index"] for m in received] == list(range(15))
    assert all(0.0 <= m["win_prob"] <= 1.0 for m in received)


def test_replay_websocket_unknown_game_closes_immediately(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/replay/does-not-exist") as ws:
            ws.receive_json()


def test_replay_websocket_unknown_game_sends_close_code_4404_to_client(client):
    # The handshake must be accepted before closing with 4404, per ASGI
    # semantics -- closing before accept() rejects the handshake at the
    # HTTP level and the application close code never reaches the client
    # as a WS close event.
    with client.websocket_connect("/replay/does-not-exist") as ws:
        message = ws.receive()
    assert message["type"] == "websocket.close"
    assert message["code"] == 4404


def test_get_games_uses_cached_live_availability(client, monkeypatch):
    import app.main as main_module

    call_count = {"n": 0}

    def fake_is_live_game_available():
        call_count["n"] += 1
        return True

    monkeypatch.setattr(main_module, "is_live_game_available", fake_is_live_game_available)
    main_module._live_available_cache["timestamp"] = 0.0

    resp1 = client.get("/games")
    resp2 = client.get("/games")

    assert resp1.json()["live_available"] is True
    assert resp2.json()["live_available"] is True
    assert call_count["n"] == 1  # second call served from cache, not a fresh network hit


def test_get_games_returns_empty_lists_when_db_missing(tmp_path, monkeypatch):
    # A fresh clone has neither backend/data/nba.db nor backend/data/ itself
    # (both gitignored). GET /games must not 500 or fail to boot in that
    # case -- it should behave as if there are simply no cached replay
    # games yet. Deliberately do NOT pre-create the parent directory here --
    # that's the real fresh-clone state, and app startup must create it.
    missing_db = tmp_path / "does-not-exist" / "nba.db"
    monkeypatch.setenv("NBA_DB_PATH", str(missing_db))
    monkeypatch.setenv(
        "NBA_MODEL_PATH", str(FIXTURES / "fixture_model.pt")
    )
    monkeypatch.setenv("NBA_REPLAY_INTERVAL_MS", "1")

    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    monkeypatch.setattr(main_module, "is_live_game_available", lambda: False)

    with TestClient(main_module.app) as c:
        resp = c.get("/games")

    assert resp.status_code == 200
    assert resp.json() == {"replay_games": [], "live_available": False}
    assert missing_db.parent.is_dir()


def test_replay_websocket_empty_game_closes_without_error(client, tmp_path):
    import app.main as main_module
    from app.db import get_connection, insert_game

    conn = get_connection(main_module.DB_PATH)
    insert_game(conn, {"game_id": "empty-game", "home_team": "X", "away_team": "Y", "home_win": 1})
    conn.close()

    with client.websocket_connect("/replay/empty-game") as ws:
        with pytest.raises(Exception):
            ws.receive_json()
