"""FastAPI app: model loaded once at startup, WS routes stream WinProbMessages."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.db import get_connection, get_game_events, list_games, game_exists
from app.inference import InferenceSession
from app.model import load_model
from app.schemas import GamesResponse, GameSummary, WinProbMessage

DB_PATH = os.environ.get("NBA_DB_PATH", "data/nba.db")
MODEL_PATH = os.environ.get("NBA_MODEL_PATH", "model/win_prob_lstm.pt")
REPLAY_INTERVAL_MS = int(os.environ.get("NBA_REPLAY_INTERVAL_MS", "200"))

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = load_model(MODEL_PATH)
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


def is_live_game_available() -> bool:
    try:
        from nba_api.live.nba.endpoints import scoreboard
        board = scoreboard.ScoreBoard()
        games = board.get_dict().get("scoreboard", {}).get("games", [])
        return any(g.get("gameStatus") == 2 for g in games)  # 2 = in progress
    except Exception:
        return False


@app.get("/games", response_model=GamesResponse)
def get_games() -> GamesResponse:
    conn = get_connection(DB_PATH)
    games = list_games(conn)
    conn.close()
    return GamesResponse(
        replay_games=[GameSummary(**g) for g in games],
        live_available=is_live_game_available(),
    )


@app.websocket("/replay/{game_id}")
async def replay_websocket(websocket: WebSocket, game_id: str):
    from app.replay import run_replay

    conn = get_connection(DB_PATH)
    exists = game_exists(conn, game_id)
    if not exists:
        conn.close()
        await websocket.close(code=4404)
        return

    await websocket.accept()
    session = InferenceSession(_state["model"])
    try:
        await run_replay(websocket, conn, game_id, session, REPLAY_INTERVAL_MS)
    except WebSocketDisconnect:
        pass
    finally:
        conn.close()


@app.websocket("/live")
async def live_websocket(websocket: WebSocket):
    from app.live import run_live

    await websocket.accept()
    session = InferenceSession(_state["model"])
    try:
        await run_live(websocket, session)
    except WebSocketDisconnect:
        pass
