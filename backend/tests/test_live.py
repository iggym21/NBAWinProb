# backend/tests/test_live.py
from unittest.mock import patch, MagicMock
import pytest
from app.live import fetch_live_game_id, fetch_live_events


def test_fetch_live_game_id_returns_none_on_no_games():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as MockBoard:
        MockBoard.return_value.get_dict.return_value = {"scoreboard": {"games": []}}
        assert fetch_live_game_id() is None


def test_fetch_live_game_id_returns_in_progress_game():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as MockBoard:
        MockBoard.return_value.get_dict.return_value = {
            "scoreboard": {"games": [
                {"gameId": "g1", "gameStatus": 3},
                {"gameId": "g2", "gameStatus": 2},
            ]}
        }
        assert fetch_live_game_id() == "g2"


def test_fetch_live_game_id_returns_none_on_exception():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard", side_effect=RuntimeError("network down")):
        assert fetch_live_game_id() is None


def test_fetch_live_events_maps_actions():
    with patch("nba_api.live.nba.endpoints.playbyplay.PlayByPlay") as MockPBP:
        MockPBP.return_value.get_dict.return_value = {
            "game": {
                "actions": [
                    {"actionNumber": 1, "clock": "PT12M00.00S", "period": 1, "teamTricode": "DEN",
                     "scoreHome": "0", "scoreAway": "0", "actionType": "Jump Ball", "description": "Tip"},
                    {"actionNumber": 2, "clock": "PT11M42.00S", "period": 1, "teamTricode": "LAL",
                     "scoreHome": "2", "scoreAway": "0", "actionType": "Made Shot", "description": "Dunk"},
                ],
                "homeTeam": {"teamTricode": "DEN"},
                "awayTeam": {"teamTricode": "LAL"},
            }
        }
        events = fetch_live_events("g2")
    assert len(events) == 2
    assert events[0]["event_index"] == 0
    assert events[1]["possession_team"] == "away"
    assert events[1]["home_score"] == 2
