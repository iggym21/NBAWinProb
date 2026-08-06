# backend/tests/test_live.py
from unittest.mock import patch
import pytest
from app.live import fetch_live_game_id, fetch_live_events, _map_live_action_type


def test_fetch_live_game_id_returns_none_on_no_games():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as MockBoard:
        MockBoard.return_value.get_dict.return_value = {"scoreboard": {"games": []}}
        assert fetch_live_game_id() is None


def test_fetch_live_game_id_returns_in_progress_game_and_tricodes():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as MockBoard:
        MockBoard.return_value.get_dict.return_value = {
            "scoreboard": {"games": [
                {"gameId": "g1", "gameStatus": 3,
                 "homeTeam": {"teamTricode": "XXX"}, "awayTeam": {"teamTricode": "YYY"}},
                {"gameId": "g2", "gameStatus": 2,
                 "homeTeam": {"teamTricode": "DEN"}, "awayTeam": {"teamTricode": "LAL"}},
            ]}
        }
        result = fetch_live_game_id()
    assert result == ("g2", "DEN", "LAL")


def test_fetch_live_game_id_returns_none_on_exception():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard", side_effect=RuntimeError("network down")):
        assert fetch_live_game_id() is None


def test_map_live_action_type_shots_use_shot_result():
    assert _map_live_action_type({"actionType": "2pt", "shotResult": "Made"}) == "Made Shot"
    assert _map_live_action_type({"actionType": "3pt", "shotResult": "Missed"}) == "Missed Shot"


def test_map_live_action_type_known_non_shot_types():
    assert _map_live_action_type({"actionType": "jumpball"}) == "Jump Ball"
    assert _map_live_action_type({"actionType": "rebound"}) == "Rebound"
    assert _map_live_action_type({"actionType": "freethrow"}) == "Free Throw"
    assert _map_live_action_type({"actionType": "foul"}) == "Foul"
    assert _map_live_action_type({"actionType": "turnover"}) == "Turnover"
    assert _map_live_action_type({"actionType": "substitution"}) == "Substitution"
    assert _map_live_action_type({"actionType": "timeout"}) == "Timeout"
    assert _map_live_action_type({"actionType": "violation"}) == "Violation"


def test_map_live_action_type_unknown_falls_through_for_other_bucket():
    # "steal"/"block"/"period"/"stoppage" have no canonical bucket; they
    # should pass through unmapped so encode_event_type's OTHER fallback
    # (exercised in app.features, not here) handles them.
    assert _map_live_action_type({"actionType": "steal"}) == "steal"
    assert _map_live_action_type({"actionType": "block"}) == "block"


def test_fetch_live_events_maps_actions_using_real_live_vocabulary():
    # Real nba_api LIVE endpoint actionType vocabulary is lowercase/compact
    # (e.g. "jumpball", "2pt"), NOT the Title-Case STATS vocabulary
    # ("Jump Ball", "Made Shot") used by PlayByPlayV3/scripts/fetch_data.py.
    # Confirmed via nba_api's own bundled source (live/nba/endpoints/
    # playbyplay.py's expected_data sample uses actionType "jumpball") and
    # published docs/example notebook.
    with patch("nba_api.live.nba.endpoints.playbyplay.PlayByPlay") as MockPBP:
        MockPBP.return_value.get_dict.return_value = {
            "game": {
                "actions": [
                    {"actionNumber": 4, "clock": "PT12M00.00S", "period": 1, "teamTricode": "DEN",
                     "scoreHome": "0", "scoreAway": "0", "actionType": "jumpball",
                     "description": "Jump Ball"},
                    {"actionNumber": 7, "clock": "PT11M42.00S", "period": 1, "teamTricode": "LAL",
                     "scoreHome": "2", "scoreAway": "0", "actionType": "2pt",
                     "shotResult": "Made", "description": "Dunk"},
                    {"actionNumber": 9, "clock": "PT11M20.00S", "period": 1, "teamTricode": "DEN",
                     "scoreHome": "", "scoreAway": "", "actionType": "rebound",
                     "description": "Rebound"},
                ],
            }
        }
        events = fetch_live_events("g2", home_tricode="DEN", away_tricode="LAL")

    assert len(events) == 3
    # event_index comes from the live feed's own stable actionNumber, not position
    assert [e["event_index"] for e in events] == [4, 7, 9]
    assert events[0]["event_type"] == "Jump Ball"
    assert events[1]["event_type"] == "Made Shot"
    assert events[1]["possession_team"] == "away"
    assert events[1]["home_score"] == 2
    # blank scoreHome/scoreAway on the rebound action carries forward the
    # last known real score instead of coercing to a fake 0-0 tie
    assert events[2]["home_score"] == 2
    assert events[2]["away_score"] == 0
    assert events[2]["event_type"] == "Rebound"


def test_fetch_live_events_returns_empty_list_on_exception():
    with patch(
        "nba_api.live.nba.endpoints.playbyplay.PlayByPlay",
        side_effect=RuntimeError("network down"),
    ):
        assert fetch_live_events("g2", "DEN", "LAL") == []
