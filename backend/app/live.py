"""WS /live producer — polls nba_api's live scoreboard/play-by-play for an
in-progress game and streams new events through the same inference session
and message schema as /replay."""
import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from app.features import parse_clock_to_seconds
from app.inference import InferenceSession
from app.schemas import WinProbMessage

# nba_api's LIVE endpoint (nba_api.live.nba.endpoints.playbyplay) uses a
# different, lowercase/compact actionType vocabulary than the STATS
# PlayByPlayV3 endpoint used by scripts/fetch_data.py (which is Title-Case:
# "Made Shot", "Missed Shot", "Rebound", etc -- see app.features.EVENT_TYPES,
# the vocabulary the model was trained on). This maps the live vocabulary to
# those same canonical Title-Case categories before encode_event_type() sees
# it, so live mode doesn't silently map every event to OTHER.
#
# Confirmed from nba_api's own bundled source (live/nba/endpoints/
# playbyplay.py's "expected_data" sample response has actionType "jumpball")
# and its published docs/example notebook, which show actionType values:
# period, jumpball, foul, freethrow, rebound, 2pt, 3pt, turnover, steal,
# timeout, substitution, block, stoppage, violation -- plus a "shotResult":
# "Made"/"Missed" field on shot actions. Direct live calls to
# cdn.nba.com/stats.nba.com were not reachable from this environment
# (403/blocked egress) to verify against a real in-progress or recent game,
# so this mapping is built from nba_api's bundled source/docs rather than a
# live network call.
LIVE_ACTION_TYPE_MAP: dict[str, str] = {
    "rebound": "Rebound",
    "freethrow": "Free Throw",
    "foul": "Foul",
    "turnover": "Turnover",
    "substitution": "Substitution",
    "timeout": "Timeout",
    "jumpball": "Jump Ball",
    "violation": "Violation",
}
# "steal", "block", "period", "stoppage" (and anything else) have no
# equivalent bucket in the STATS-derived EVENT_TYPES vocabulary and are
# intentionally left unmapped -- they fall through to OTHER via
# encode_event_type's existing fallback, which is correct/expected.


def _map_live_action_type(action: dict) -> str:
    """Translates a live-endpoint action's actionType (+ shotResult, for
    shot actions) into the canonical Title-Case event_type string used by
    app.features.EVENT_TYPES. Unrecognized types are passed through as-is
    so encode_event_type's OTHER fallback handles them."""
    raw_type = (action.get("actionType") or "").strip().lower()
    if raw_type in ("2pt", "3pt"):
        shot_result = (action.get("shotResult") or "").strip().lower()
        if shot_result == "made":
            return "Made Shot"
        if shot_result == "missed":
            return "Missed Shot"
        return action.get("actionType") or ""
    return LIVE_ACTION_TYPE_MAP.get(raw_type, action.get("actionType") or "")


def fetch_live_game_id() -> tuple[str, str, str] | None:
    """Returns (game_id, home_tricode, away_tricode) for the first
    in-progress game (gameStatus == 2), or None. Never raises.

    The scoreboard endpoint's per-game entry already carries homeTeam/
    awayTeam tricodes directly, so this fetches them once here rather than
    needing a second call -- the live PlayByPlay endpoint's "game" object
    does NOT contain homeTeam/awayTeam (confirmed via nba_api's own bundled
    ScoreBoard endpoint source, whose "expected_data" sample shows
    homeTeam/awayTeam only on scoreboard game entries, not on playbyplay's
    "game" object)."""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        board = scoreboard.ScoreBoard()
        games = board.get_dict().get("scoreboard", {}).get("games", [])
        for g in games:
            if g.get("gameStatus") == 2:
                game_id = g.get("gameId")
                if game_id is None:
                    continue
                home_tricode = (g.get("homeTeam") or {}).get("teamTricode")
                away_tricode = (g.get("awayTeam") or {}).get("teamTricode")
                return game_id, home_tricode, away_tricode
        return None
    except Exception:
        return None


def fetch_live_events(
    game_id: str, home_tricode: str | None, away_tricode: str | None
) -> list[dict]:
    """Fetches and maps live play-by-play actions to the same event dict
    shape used by InferenceSession.step. home_tricode/away_tricode must be
    supplied by the caller (from fetch_live_game_id / scoreboard) since the
    live playbyplay endpoint's response doesn't carry them itself.

    Like scripts.fetch_data.parse_playbyplay_rows, scoreHome/scoreAway are
    only populated on actual scoring actions -- every other action has a
    blank/missing score, so the last known real score is carried forward
    rather than coerced to a fake 0-0 tie."""
    try:
        from nba_api.live.nba.endpoints import playbyplay
        data = playbyplay.PlayByPlay(game_id=game_id).get_dict()
    except Exception:
        return []

    game = data.get("game", {})

    events: list[dict] = []
    last_home_score = 0
    last_away_score = 0
    for action in game.get("actions", []):
        clock = action.get("clock")
        if parse_clock_to_seconds(clock) is None:
            continue
        team_tricode = action.get("teamTricode") or ""
        if team_tricode == home_tricode:
            possession_team = "home"
        elif team_tricode == away_tricode:
            possession_team = "away"
        else:
            possession_team = None
        if action.get("scoreHome"):
            last_home_score = int(action["scoreHome"])
        if action.get("scoreAway"):
            last_away_score = int(action["scoreAway"])
        # actionNumber is the live feed's own stable per-action identifier --
        # used as event_index instead of a positional len(events) so that if
        # filtering (e.g. clock parseability) ever shifts between polls,
        # run_live's high-water-mark tracking doesn't skip or reprocess events.
        action_number = action.get("actionNumber")
        events.append({
            "event_index": int(action_number) if action_number is not None else len(events),
            "period": int(action.get("period", 1)),
            "clock": clock,
            "home_score": last_home_score,
            "away_score": last_away_score,
            "event_type": _map_live_action_type(action),
            "description": action.get("description") or "",
            "possession_team": possession_team,
        })
    return events


async def run_live(websocket: WebSocket, session: InferenceSession, poll_interval_s: float = 3.0) -> None:
    # fetch_live_game_id/fetch_live_events make synchronous nba_api HTTP
    # calls; running them directly here would block the entire FastAPI event
    # loop (freezing all concurrent /replay connections too) for the
    # duration of each network call, every poll cycle. asyncio.to_thread
    # runs them off the event loop while keeping the functions themselves
    # plain sync code.
    result = await asyncio.to_thread(fetch_live_game_id)
    if result is None:
        await websocket.close(code=4204, reason="no live game in progress")
        return
    game_id, home_tricode, away_tricode = result

    last_sent_index = -1
    try:
        while True:
            events = await asyncio.to_thread(
                fetch_live_events, game_id, home_tricode, away_tricode
            )
            new_events = [e for e in events if e["event_index"] > last_sent_index]
            for event in new_events:
                win_prob = session.step(event)
                if win_prob is None:
                    continue
                message = WinProbMessage(
                    event_index=event["event_index"],
                    period=event["period"],
                    clock=event["clock"],
                    home_score=event["home_score"],
                    away_score=event["away_score"],
                    event_type=event["event_type"],
                    description=event["description"],
                    win_prob=win_prob,
                )
                await websocket.send_json(message.model_dump())
                last_sent_index = event["event_index"]
            await asyncio.sleep(poll_interval_s)
    except WebSocketDisconnect:
        return
