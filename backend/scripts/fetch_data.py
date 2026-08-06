"""One-time historical data pull: nba_api -> data/raw/*.json -> data/nba.db.

Usage:
    python scripts/fetch_data.py --season 2023-24 --max-games 150 \
        --out-db data/nba.db --raw-dir data/raw
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import EventRow, GameRow, get_connection, init_db, insert_events, insert_game, game_exists
from app.features import parse_clock_to_seconds


def parse_game_list(rows: list[dict]) -> list[GameRow]:
    """Groups LeagueGameFinder's per-team rows into one row per game."""
    by_game: dict[str, list[dict]] = {}
    for row in rows:
        by_game.setdefault(row["GAME_ID"], []).append(row)

    games: list[GameRow] = []
    for game_id, pair in by_game.items():
        if len(pair) != 2:
            continue
        home_row = next((r for r in pair if "vs." in r["MATCHUP"]), None)
        away_row = next((r for r in pair if "@" in r["MATCHUP"]), None)
        if home_row is None or away_row is None:
            continue
        games.append(
            GameRow(
                game_id=game_id,
                home_team=home_row["TEAM_ABBREVIATION"],
                away_team=away_row["TEAM_ABBREVIATION"],
                home_win=1 if home_row["WL"] == "W" else 0,
            )
        )
    return games


def parse_playbyplay_rows(
    pbp_rows: list[dict], home_team_tricode: str, away_team_tricode: str
) -> list[EventRow]:
    """Converts raw PlayByPlayV3 rows to EventRows, dropping unparseable clocks."""
    events: list[EventRow] = []
    for row in pbp_rows:
        clock = row.get("clock")
        if parse_clock_to_seconds(clock) is None:
            continue
        team_tricode = row.get("teamTricode") or ""
        if team_tricode == home_team_tricode:
            possession_team = "home"
        elif team_tricode == away_team_tricode:
            possession_team = "away"
        else:
            possession_team = None
        events.append(
            EventRow(
                game_id=str(row.get("gameId", "")),
                event_index=len(events),
                period=int(row["period"]),
                clock=clock,
                home_score=int(row.get("scoreHome") or 0),
                away_score=int(row.get("scoreAway") or 0),
                event_type=row.get("actionType") or "",
                description=row.get("description") or "",
                possession_team=possession_team,
            )
        )
    return events


def fetch_and_store(season: str, max_games: int, out_db: str, raw_dir: str) -> None:
    from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    conn = get_connection(out_db)
    init_db(conn)

    print(f"Fetching game list for season {season}...")
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season, league_id_nullable="00", timeout=30)
    raw_game_rows = finder.get_data_frames()[0].to_dict(orient="records")
    games = parse_game_list(raw_game_rows)[:max_games]
    print(f"{len(games)} games to fetch.")

    for i, game in enumerate(games):
        game_id = game["game_id"]
        if game_exists(conn, game_id):
            print(f"[{i+1}/{len(games)}] {game_id} already cached, skipping.")
            continue

        cache_file = raw_path / f"{game_id}.json"
        if cache_file.exists():
            raw_pbp_rows = json.loads(cache_file.read_text())
        else:
            try:
                pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
                raw_pbp_rows = pbp.get_data_frames()[0].to_dict(orient="records")
            except Exception as exc:
                print(f"[{i+1}/{len(games)}] {game_id} FAILED: {exc}")
                continue
            cache_file.write_text(json.dumps(raw_pbp_rows, indent=2, default=str))
            time.sleep(0.6)  # be polite to stats.nba.com

        events = parse_playbyplay_rows(raw_pbp_rows, game["home_team"], game["away_team"])
        if not events:
            print(f"[{i+1}/{len(games)}] {game_id} produced 0 usable events, skipping.")
            continue

        insert_game(conn, game)
        insert_events(conn, events)
        print(f"[{i+1}/{len(games)}] {game_id} stored ({len(events)} events).")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--max-games", type=int, default=150)
    parser.add_argument("--out-db", default="data/nba.db")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    fetch_and_store(args.season, args.max_games, args.out_db, args.raw_dir)


if __name__ == "__main__":
    main()
