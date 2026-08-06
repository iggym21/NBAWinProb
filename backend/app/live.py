"""WS /live producer — polls nba_api's live scoreboard/play-by-play for an
in-progress game and streams new events through the same inference session
and message schema as /replay."""
import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from app.features import parse_clock_to_seconds
from app.inference import InferenceSession
from app.schemas import WinProbMessage


def fetch_live_game_id() -> str | None:
    try:
        from nba_api.live.nba.endpoints import scoreboard
        board = scoreboard.ScoreBoard()
        games = board.get_dict().get("scoreboard", {}).get("games", [])
        for g in games:
            if g.get("gameStatus") == 2:
                return g.get("gameId")
        return None
    except Exception:
        return None


def fetch_live_events(game_id: str) -> list[dict]:
    try:
        from nba_api.live.nba.endpoints import playbyplay
        data = playbyplay.PlayByPlay(game_id=game_id).get_dict()
    except Exception:
        return []

    game = data.get("game", {})
    home_tricode = game.get("homeTeam", {}).get("teamTricode")
    away_tricode = game.get("awayTeam", {}).get("teamTricode")

    events: list[dict] = []
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
        events.append({
            "event_index": len(events),
            "period": int(action.get("period", 1)),
            "clock": clock,
            "home_score": int(action.get("scoreHome") or 0),
            "away_score": int(action.get("scoreAway") or 0),
            "event_type": action.get("actionType") or "",
            "description": action.get("description") or "",
            "possession_team": possession_team,
        })
    return events


async def run_live(websocket: WebSocket, session: InferenceSession, poll_interval_s: float = 3.0) -> None:
    game_id = fetch_live_game_id()
    if game_id is None:
        await websocket.close(code=4204, reason="no live game in progress")
        return

    last_sent_index = -1
    try:
        while True:
            events = fetch_live_events(game_id)
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
