"""Pydantic request/response/message models shared by REST and WebSocket routes."""
from pydantic import BaseModel


class GameSummary(BaseModel):
    game_id: str
    home_team: str
    away_team: str


class GamesResponse(BaseModel):
    replay_games: list[GameSummary]
    live_available: bool


class WinProbMessage(BaseModel):
    event_index: int
    period: int
    clock: str
    home_score: int
    away_score: int
    event_type: str
    description: str
    win_prob: float
