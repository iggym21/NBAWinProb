import json
from pathlib import Path
import pytest
from scripts.fetch_data import parse_game_list, parse_playbyplay_rows

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_game_list_groups_pairs_and_derives_home_win():
    raw_rows = [
        {"GAME_ID": "0022300061", "TEAM_ID": 1, "TEAM_ABBREVIATION": "DEN",
         "MATCHUP": "DEN vs. LAL", "WL": "W"},
        {"GAME_ID": "0022300061", "TEAM_ID": 2, "TEAM_ABBREVIATION": "LAL",
         "MATCHUP": "LAL @ DEN", "WL": "L"},
    ]
    games = parse_game_list(raw_rows)
    assert len(games) == 1
    game = games[0]
    assert game["game_id"] == "0022300061"
    assert game["home_team"] == "DEN"
    assert game["away_team"] == "LAL"
    assert game["home_win"] == 1


def test_parse_game_list_away_win():
    raw_rows = [
        {"GAME_ID": "1", "TEAM_ID": 1, "TEAM_ABBREVIATION": "BOS",
         "MATCHUP": "BOS vs. MIA", "WL": "L"},
        {"GAME_ID": "1", "TEAM_ID": 2, "TEAM_ABBREVIATION": "MIA",
         "MATCHUP": "MIA @ BOS", "WL": "W"},
    ]
    games = parse_game_list(raw_rows)
    assert games[0]["home_win"] == 0


def test_parse_playbyplay_rows_from_fixture():
    raw_rows = json.loads((FIXTURES / "sample_playbyplay_v3.json").read_text())
    events = parse_playbyplay_rows(raw_rows, home_team_tricode="DEN", away_team_tricode="LAL")
    assert len(events) > 0
    assert all(e["game_id"] == "0022300061" for e in events)
    # event_index is a contiguous 0-based sequence after filtering
    assert [e["event_index"] for e in events] == list(range(len(events)))
    # possession_team correctly mapped from teamTricode
    home_possession_events = [e for e in events if e["possession_team"] == "home"]
    assert len(home_possession_events) > 0
    # unknown/blank action types fall back to OTHER at parse time is NOT required here —
    # raw event_type strings are stored as-is; encoding to OTHER happens in app.features
    # at train/serve time. Just assert descriptions are preserved.
    assert all("description" in e for e in events)


def test_parse_playbyplay_rows_drops_unparseable_clock():
    raw_rows = [
        {"actionNumber": 1, "clock": "PT12M00.00S", "period": 1, "teamTricode": "DEN",
         "scoreHome": 0, "scoreAway": 0, "actionType": "Jump Ball", "description": "Tip"},
        {"actionNumber": 2, "clock": "GARBAGE", "period": 1, "teamTricode": "DEN",
         "scoreHome": 0, "scoreAway": 0, "actionType": "Made Shot", "description": "bad clock"},
        {"actionNumber": 3, "clock": "PT11M42.00S", "period": 1, "teamTricode": "LAL",
         "scoreHome": 2, "scoreAway": 0, "actionType": "Missed Shot", "description": "ok"},
    ]
    events = parse_playbyplay_rows(raw_rows, home_team_tricode="DEN", away_team_tricode="LAL")
    assert len(events) == 2
    assert [e["description"] for e in events] == ["Tip", "ok"]
