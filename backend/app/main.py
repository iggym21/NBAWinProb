"""FastAPI app: model loaded once at startup, WS routes stream WinProbMessages."""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.db import get_connection, get_game_events, init_db, list_games, game_exists
from app.inference import InferenceSession
from app.model import load_model
from app.schemas import GamesResponse, GameSummary, WinProbMessage

DB_PATH = os.environ.get("NBA_DB_PATH", "data/nba.db")
MODEL_PATH = os.environ.get("NBA_MODEL_PATH", "model/win_prob_lstm.pt")
REPLAY_INTERVAL_MS = int(os.environ.get("NBA_REPLAY_INTERVAL_MS", "200"))

# is_live_game_available() hits nba.com on every call; cache its result for
# a short TTL so GET /games doesn't make a fresh network round-trip (with
# nba_api's ~30s default timeout) on every single request.
_LIVE_AVAILABLE_CACHE_TTL_S = 30.0
_live_available_cache: dict = {"timestamp": 0.0, "result": False}

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A fresh clone has no backend/data/ directory at all (it's gitignored),
    # so create it before connecting -- otherwise sqlite3.connect() itself
    # fails and the app never boots. init_db is idempotent
    # (CREATE TABLE IF NOT EXISTS), so calling it once here (rather than
    # per-request) makes GET /games resilient to a missing/fresh nba.db --
    # it returns {"replay_games": [], "live_available": false} instead of a
    # 500 from sqlite3.OperationalError: no such table.
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(DB_PATH)
    try:
        init_db(conn)
    finally:
        conn.close()
    _state["model"] = load_model(MODEL_PATH)
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


def is_live_game_available() -> bool:
    try:
        from nba_api.live.nba.endpoints import scoreboard
        # Explicit short timeout so a slow/unreachable network fails fast
        # rather than hanging the request for nba_api's ~30s default.
        board = scoreboard.ScoreBoard(timeout=5)
        games = board.get_dict().get("scoreboard", {}).get("games", [])
        return any(g.get("gameStatus") == 2 for g in games)  # 2 = in progress
    except Exception:
        return False


def _cached_is_live_game_available() -> bool:
    now = time.monotonic()
    if now - _live_available_cache["timestamp"] > _LIVE_AVAILABLE_CACHE_TTL_S:
        _live_available_cache["result"] = is_live_game_available()
        _live_available_cache["timestamp"] = now
    return _live_available_cache["result"]


@app.get("/games", response_model=GamesResponse)
def get_games() -> GamesResponse:
    conn = get_connection(DB_PATH)
    try:
        games = list_games(conn)
    finally:
        conn.close()
    return GamesResponse(
        replay_games=[GameSummary(**g) for g in games],
        live_available=_cached_is_live_game_available(),
    )


@app.websocket("/replay/{game_id}")
async def replay_websocket(websocket: WebSocket, game_id: str):
    from app.replay import run_replay

    # Per ASGI semantics, closing before accept() rejects the WS handshake
    # with an HTTP-level 403 and the application close code never reaches
    # the client as a WS close event. Accept first, then close with 4404 if
    # the game doesn't exist -- matching the pattern /live already uses for
    # its "no live game" case.
    await websocket.accept()

    conn = get_connection(DB_PATH)
    exists = game_exists(conn, game_id)
    if not exists:
        conn.close()
        await websocket.close(code=4404)
        return

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
