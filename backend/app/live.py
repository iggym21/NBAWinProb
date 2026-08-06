"""WS /live producer — polls nba_api's live scoreboard/play-by-play. Implemented in Task 10."""
from fastapi import WebSocket

from app.inference import InferenceSession


async def run_live(websocket: WebSocket, session: InferenceSession) -> None:
    await websocket.close(code=4204, reason="live mode not yet available")
