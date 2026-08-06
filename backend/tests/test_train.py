import sqlite3
import pytest
from app.db import get_connection, init_db, insert_game, insert_events
from scripts.train import split_games, build_game_tensors, train_model


def _make_fixture_db(path):
    conn = get_connection(path)
    init_db(conn)
    for i in range(8):
        game_id = f"g{i}"
        insert_game(conn, {"game_id": game_id, "home_team": "AAA", "away_team": "BBB",
                            "home_win": i % 2})
        events = []
        for t in range(10):
            events.append({
                "game_id": game_id, "event_index": t, "period": 1,
                "clock": f"PT{11 - t}M00.00S", "home_score": t, "away_score": t // 2,
                "event_type": "Made Shot" if t % 2 == 0 else "Missed Shot",
                "description": "x", "possession_team": "home" if t % 2 == 0 else "away",
            })
        insert_events(conn, events)
    conn.close()


def test_split_games_deterministic_and_covers_all():
    game_ids = [f"g{i}" for i in range(20)]
    train1, val1, test1 = split_games(game_ids, seed=42)
    train2, val2, test2 = split_games(game_ids, seed=42)
    assert (train1, val1, test1) == (train2, val2, test2)
    assert set(train1) | set(val1) | set(test1) == set(game_ids)
    assert not (set(train1) & set(val1)) and not (set(train1) & set(test1)) and not (set(val1) & set(test1))
    assert len(train1) == 16 and len(val1) == 2 and len(test1) == 2


def test_build_game_tensors_shapes():
    events = [
        {"game_id": "g0", "event_index": 0, "period": 1, "clock": "PT12M00.00S",
         "home_score": 0, "away_score": 0, "event_type": "Jump Ball",
         "description": "x", "possession_team": None},
        {"game_id": "g0", "event_index": 1, "period": 1, "clock": "PT11M00.00S",
         "home_score": 2, "away_score": 0, "event_type": "Made Shot",
         "description": "x", "possession_team": "home"},
    ]
    result = build_game_tensors(events, home_win=1)
    assert result is not None
    assert result["event_type_idx"].shape == (2,)
    assert result["numeric_features"].shape == (2, 3)
    assert result["target"] == 1.0


def test_train_model_runs_end_to_end(tmp_path):
    db_path = str(tmp_path / "fixture.db")
    _make_fixture_db(db_path)
    lstm_model, lstm_config, baseline_model, split_info = train_model(
        db_path=db_path, epochs=1, hidden_size=4, lr=1e-2, seed=42
    )
    assert lstm_config["hidden_size"] == 4
    assert set(split_info.keys()) == {"train_game_ids", "val_game_ids", "test_game_ids"}
    # baseline is a fitted sklearn model — predict_proba should work
    import numpy as np
    preds = baseline_model.predict_proba(np.zeros((1, 3 + 1)))  # +1 for event_type_idx as a raw feature
    assert preds.shape == (1, 2)
