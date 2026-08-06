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
