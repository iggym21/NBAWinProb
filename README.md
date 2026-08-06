# NBA Win Probability

A live NBA win-probability model — a PyTorch LSTM watches play-by-play events one at a time and outputs P(home team wins) at every point in the game, served through a FastAPI backend and a React live-updating chart.

[![CI](https://github.com/iggym21/NBAWinProb/actions/workflows/ci.yml/badge.svg)](https://github.com/iggym21/NBAWinProb/actions/workflows/ci.yml)

![Win probability across a full game replay](docs/screenshots/replay-chart.png)
*Win probability across a completed replay of GSW @ DAL — the chart, score/clock header, and play log are all driven by the same WebSocket event stream.*

## What this is

A live NBA win-probability model: a PyTorch LSTM ingests play-by-play events one at a time and outputs P(home team wins) at every point in the game. Two ingestion modes feed the same pipeline — a live poller (real games, when one is in progress) and a replay mode (any cached historical game, accelerated) — so the system is always demoable even outside game hours. Full-stack: FastAPI backend, React/TypeScript frontend with a live-updating probability chart. All data sourced from the free, unofficial `nba_api` package; no paid APIs; hosted on free tiers.

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

Live and replay are two producers of the same event stream; the frontend and the model don't know or care which one is feeding them.

## Results

Evaluated on 200 games from the 2023-24 NBA season (160/20/20 train/val/test split by game — never by event, since splitting by event would leak a game's own timeline across splits). Full report: [`backend/reports/evaluation_report.md`](backend/reports/evaluation_report.md).

| Metric | LSTM | Logistic Regression baseline |
|---|---|---|
| Brier score (lower better) | 0.2476 | 0.2282 |
| Log-loss (lower better) | 0.6882 | 0.6401 |

| Time remaining | LSTM accuracy | Baseline accuracy | N |
|---|---|---|---|
| >36 min | 0.547 | 0.575 | 2357 |
| 12-36 min | 0.539 | 0.588 | 4735 |
| 3-12 min | 0.541 | 0.614 | 1698 |
| <3 min | 0.561 | 0.658 | 660 |

**Honest read: on this 200-game sample, the logistic-regression baseline beats the LSTM on every metric.** The design goal was a sequence model that meaningfully outperforms the baseline — that didn't happen here, and it's reported as-is rather than reframed. A few plausible reasons: 200 games / 30 full-batch epochs is a small training run for a sequence model to learn from, relative to a baseline that's already hard to beat on well-engineered per-event features (score differential and time remaining alone carry most of the signal in basketball win-prob); and the LSTM's predictions were narrowly calibrated, clustering roughly in the 50-70% range rather than spanning the full probability range the baseline covers. More games, more epochs, mini-batching instead of full-batch, or a smaller-capacity/regularized LSTM are the natural next steps to close this gap.

## Quickstart

**1. Fetch data** (one-time historical pull, cached to `data/raw/` and `data/nba.db`):

```bash
cd backend && .venv/bin/python scripts/fetch_data.py --season 2023-24 --max-games 200 --out-db data/nba.db --raw-dir data/raw
```

**2. Train** (LSTM + logistic-regression baseline):

```bash
.venv/bin/python scripts/train.py --db data/nba.db --epochs 30 --hidden-size 64 --out-model model/win_prob_lstm.pt --out-baseline model/baseline_logreg.joblib --out-split model/split.json
```

**3. Evaluate**:

```bash
.venv/bin/python scripts/evaluate.py --db data/nba.db --model model/win_prob_lstm.pt --baseline model/baseline_logreg.joblib --split model/split.json --out-md reports/evaluation_report.md --out-json reports/evaluation_report.json
```

**4. Run the backend**:

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```

**5. Run the frontend**:

```bash
cd frontend && npm run dev
```

## Testing

```bash
cd backend && .venv/bin/pytest -v
cd frontend && npm run test
```

CI (`.github/workflows/ci.yml`) runs both on every push to `main` and every pull request.

## Screenshots

| Game picker | Full game replay | Play log |
|---|---|---|
| ![Game picker](docs/screenshots/game-picker.png) | ![Win probability chart after a completed full game replay](docs/screenshots/replay-chart.png) | ![Scrolling play-by-play log](docs/screenshots/play-log.png) |

The replay and play-log screenshots were captured at the end of a full replay of GSW @ DAL (final: Q4 00:00, home win probability 59%) — the chart shows the win-probability line across the entire game and the play log is fully populated.

## Project Structure

```
NBAWinProb/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, /games, /replay/{id}, /live routes
│   │   ├── db.py             # SQLite access (games, events tables)
│   │   ├── features.py       # shared feature engineering (train + serve)
│   │   ├── model.py          # PyTorch LSTM definition
│   │   ├── inference.py      # incremental forward-pass inference
│   │   ├── replay.py         # accelerated historical replay producer
│   │   ├── live.py           # nba_api live-game poller
│   │   └── schemas.py        # pydantic request/response/event models
│   ├── scripts/
│   │   ├── fetch_data.py     # one-time historical data pull → data/nba.db
│   │   ├── train.py          # trains LSTM + logistic-regression baseline
│   │   └── evaluate.py       # produces reports/evaluation_report.{md,json}
│   ├── tests/                 # pytest: unit, model-smoke, integration
│   ├── model/                 # trained checkpoint, baseline, split, event vocab
│   ├── reports/                # evaluation_report.md / .json
│   └── data/                   # nba.db + raw/ (cached API responses)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── GamePicker.tsx
│   │   │   ├── WinProbChart.tsx
│   │   │   ├── ScoreHeader.tsx
│   │   │   └── PlayLog.tsx
│   │   ├── hooks/useGameSocket.ts   # WebSocket connection + message state
│   │   └── test/                     # Vitest component/hook tests
│   └── package.json
├── docs/
│   ├── screenshots/            # game-picker.png, replay-chart.png, play-log.png
│   └── superpowers/specs/      # design spec
└── .github/workflows/ci.yml
```

## Tech Stack

- **Model:** PyTorch (LSTM), scikit-learn (logistic-regression baseline)
- **Backend:** FastAPI, WebSockets, SQLite, pydantic, `nba_api`
- **Frontend:** React 19, TypeScript, Vite, Recharts
- **Testing:** pytest / pytest-asyncio / httpx (backend), Vitest + Testing Library (frontend)
- **CI:** GitHub Actions (pytest + Vitest/build on every push and PR to `main`)

## License

MIT — see [LICENSE](LICENSE).
