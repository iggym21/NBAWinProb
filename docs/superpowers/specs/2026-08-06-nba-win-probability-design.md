# NBA Win Probability — Design

## Summary

A live NBA win-probability model: a PyTorch LSTM ingests play-by-play events one at a time and outputs P(home team wins) at every point in the game. Two ingestion modes feed the same pipeline — a live poller (real games, when one is in progress) and a replay mode (any cached historical game, accelerated) — so the system is always demoable even outside game hours. Full-stack: FastAPI backend, React/TypeScript frontend with a live-updating probability chart. All data sourced from the free, unofficial `nba_api` package; no paid APIs; hosted on free tiers.

## Goals

- A sequence model (not just hand-features + gradient boosting) that meaningfully outperforms a logistic-regression baseline, with the comparison documented.
- A live, watchable demo: either a real in-progress game or an accelerated replay of a historical one, via the same frontend.
- Zero ongoing API or hosting cost.

## Non-goals

- Betting odds / spread prediction, multi-sport support, player-level props, native mobile app.

## Architecture

```
React frontend (win-prob line chart, score/clock header, scrolling play log)
        │  WebSocket
        ▼
FastAPI backend
  ├─ GET  /games                → list of cached replay games + whether a live game is active
  ├─ WS   /replay/{game_id}     → replays a cached historical game, one event at a time, accelerated
  ├─ WS   /live                 → polls nba_api's live scoreboard/play-by-play during an actual game
  └─ both emit the same event schema → LSTM inference → win_prob → broadcast to client
        │
        ▼
PyTorch LSTM (trained offline, checkpoint loaded once at server startup)
```

Live and replay are two producers of the same event stream; the frontend and the model don't know or care which one is feeding them. This mirrors the live/fallback split already used in Sentinel.

## Data pipeline

**Source:** `nba_api` (free, unofficial wrapper around stats.nba.com; no key required).

**One-time historical pull** (`scripts/fetch_data.py`): pulls one full recent NBA season — `leaguegamefinder` for the season's game list, `playbyplayv2` per game for event-level data. Raw JSON responses are cached to `data/raw/` so the pull never needs to repeat. Parsed events are written to a local SQLite database (`data/nba.db`) with two tables: `games` (game_id, home_team, away_team, final home_win) and `events` (game_id, event_index, period, clock, home_score, away_score, event_type, description).

**Feature engineering** (applied at both training and serving time, shared code path):
- `seconds_remaining_total`: computed from period + game clock, OT-aware (regulation periods are 12 min × 4, OT periods are 5 min each).
- `score_diff`: home_score − away_score at that event.
- `event_type`: categorical (made shot, missed shot, rebound, turnover, foul, timeout, substitution, etc.), embedded in the model rather than one-hot encoded. Any event type not seen at training time maps to a fixed `OTHER` bucket at inference — the pipeline must never crash on an unrecognized event.
- `possession_team`: derived from the event's team relative to home/away.

**Label:** final `home_win` (0/1) for the game, applied as the training target to every event row in that game — standard win-probability training setup (same approach real win-prob models like ESPN's BPI use).

**Split:** train/val/test split by *game* (80/10/10), never by individual event — splitting by event would leak information across a game's own timeline.

**Known data quirk to handle:** nba_api's clock strings are occasionally malformed at period boundaries (e.g., end-of-quarter buzzer events). The pipeline normalizes or drops these rather than letting them corrupt `seconds_remaining_total`.

## Model & evaluation

**Model:** PyTorch LSTM. Per-event feature vector (embedded event_type concatenated with score_diff, seconds_remaining, possession) feeds the LSTM; a linear+sigmoid head on the hidden state at every timestep produces `win_prob(t)`. Games are padded to a common length within a batch (masked so padding doesn't contribute to loss); loss is BCE averaged over all real (non-padded) timesteps.

**Baseline:** scikit-learn logistic regression on the same per-event features, treated as independent rows (no sequence, no LSTM) — same train/val/test split.

**Evaluation report** (`scripts/evaluate.py`, produces a markdown/JSON report analogous to BacktestIQ's tearsheet):
- Brier score and log-loss, LSTM vs. baseline.
- Calibration curve (predicted probability vs. observed frequency).
- Accuracy broken out by time-remaining bucket (e.g., >36 min, 12–36 min, 3–12 min, <3 min remaining) — this is where a sequence model should visibly beat the baseline, since it can use momentum/run information the baseline can't.

## Serving

FastAPI backend, model checkpoint (`model/win_prob_lstm.pt`) loaded once at startup — no runtime training or fine-tuning.

- **Replay mode** (`WS /replay/{game_id}`): streams a cached game's events from SQLite at an accelerated, configurable interval (e.g., one event per 200ms). Always available, since it depends only on already-fetched data — this is the primary demo path.
- **Live mode** (`WS /live`): polls nba_api's live scoreboard to detect an in-progress game, then polls its play-by-play endpoint for new events. Only meaningful when an NBA game is actually being played; `GET /games` tells the frontend whether live mode currently has anything to show, so the UI can steer toward replay otherwise.
- Per-connection LSTM hidden state is carried across the WebSocket session so each new event is a single incremental forward pass, not a replay of the full prefix.
- On WebSocket disconnect/reconnect, replay restarts from the beginning of the selected game (acceptable for a demo tool; no resume-from-offset complexity needed).

## Frontend

React + TypeScript, matching the BacktestIQ stack. A game picker (choose any cached historical game for replay, or "watch live" if `/games` reports one in progress), a live-updating win-probability line chart (Recharts), a score/clock header, and a scrolling play-by-play log driven by the same WebSocket stream.

## Deployment

Trained checkpoint is small (a few MB) and is bundled directly into the backend Docker image — no Git LFS needed. Backend deploys to Render or Fly.io's free tier (may sleep when idle — acceptable for a portfolio demo); frontend deploys to Vercel's free tier. GitHub Actions CI runs pytest (backend) and Vitest (frontend) on every push, matching the CI pattern already used on BacktestIQ. Total ongoing cost: $0.

## Testing

- **Unit (pytest):** clock-to-seconds conversion including OT edge cases and malformed-clock-string handling; event feature encoding; unknown `event_type` fallback to `OTHER`.
- **Model smoke test:** inference runs end-to-end on a fixture event sequence and returns valid probabilities in [0, 1] at every timestep.
- **Integration (pytest + FastAPI TestClient):** connect to `/replay/{fixture_game_id}`, assert a full, monotonically increasing sequence of `event_index` / `win_prob` messages is received for the fixture game.
- **Frontend (Vitest):** chart component renders correctly given a mocked win-probability stream.
