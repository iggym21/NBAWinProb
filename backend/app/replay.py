"""WS /replay producer — streams a cached game's events at an accelerated interval."""
import asyncio

from fastapi import WebSocket

from app.db import get_game_events
from app.inference import InferenceSession
from app.schemas import WinProbMessage


async def run_replay(websocket: WebSocket, conn, game_id: str, session: InferenceSession, interval_ms: int) -> None:
    events = get_game_events(conn, game_id)
    for event in events:
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
        await asyncio.sleep(interval_ms / 1000)
    await websocket.close()
