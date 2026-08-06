# NBA Win Probability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and ship a working, tested, documented NBA live win-probability app (PyTorch LSTM + FastAPI + React) as a public GitHub repo with a README that includes real screenshots and evaluation results.

**Architecture:** FastAPI backend serves a PyTorch LSTM over WebSocket (`/replay/{game_id}` for accelerated historical replay, `/live` for real games), backed by a SQLite cache of `nba_api` play-by-play data. React/TypeScript frontend renders a live win-probability chart, score header, and play log off the same WebSocket message schema. See `docs/superpowers/specs/2026-08-06-nba-win-probability-design.md` for the full design rationale.

**Tech Stack:** Python 3.12, PyTorch, FastAPI, uvicorn, websockets, scikit-learn, pandas, nba_api, pytest; React 18 + TypeScript + Vite, Recharts, Vitest + @testing-library/react; GitHub Actions CI.

## Global Constraints

- No paid APIs, no API keys. Only `nba_api` (free, unofficial, no auth) for data.
- Zero ongoing hosting cost — local-first; any deploy step is optional/manual, documented in README, not required for "done."
- Train/val/test split is by **game**, never by event (spec §Data pipeline).
- Malformed/unparseable clock strings must be dropped, never crash the pipeline (spec §Known data quirk).
- Unknown `event_type` at inference time must map to a fixed `OTHER` bucket, never crash (spec §Feature engineering).
- Feature engineering code is shared between training and serving — one implementation, imported by both (spec §Feature engineering).
- Model checkpoint loaded once at FastAPI startup; no runtime training (spec §Serving).
- Per-connection LSTM hidden state carried across a WebSocket session — incremental forward pass per event, not full-prefix replay (spec §Serving).
- Real API shapes verified live during planning (2026-08-06): `nba_api.stats.endpoints.playbyplayv3.PlayByPlayV3(game_id=...)` returns columns including `period`, `clock` (ISO-8601 duration, e.g. `"PT11M42.00S"`), `teamId`, `teamTricode`, `scoreHome`, `scoreAway`, `actionType` (values seen: `''`, `'Foul'`, `'Free Throw'`, `'Jump Ball'`, `'Made Shot'`, `'Missed Shot'`, `'Rebound'`, `'Substitution'`, `'Timeout'`, `'Turnover'`, `'Violation'`, `'period'`), `description`. `PlayByPlayV2` is deprecated and returns empty data — do not use it.
- `nba_api.stats.endpoints.leaguegamefinder.LeagueGameFinder(season_nullable=..., league_id_nullable="00")` returns **two rows per game** (one per team) with `GAME_ID`, `TEAM_ID`, `MATCHUP` (contains `"vs."` for the home team, `"@"` for the away team), `WL` (`"W"`/`"L"`). Home team = the row whose `MATCHUP` contains `"vs."`; `home_win` = 1 if that row's `WL == "W"`.

---

## File Structure

```
NBAWinProb/
  backend/
    pyproject.toml
    requirements.txt
    app/
      __init__.py
      features.py       # shared clock parsing + feature encoding (training & serving)
      db.py              # SQLite schema + access layer
      schemas.py         # pydantic request/response/message models
      model.py            # WinProbLSTM nn.Module + checkpoint load
      inference.py         # per-connection incremental inference session
      replay.py             # WS /replay producer
      live.py                # WS /live producer (nba_api polling)
      main.py                 # FastAPI app, routes, lifespan model load
    scripts/
      fetch_data.py            # nba_api -> data/raw/*.json -> data/nba.db
      train.py                  # train LSTM + logistic-regression baseline
      evaluate.py                 # Brier/log-loss/calibration/time-bucket report
    tests/
      fixtures/
        sample_playbyplay_v3.json   # small real-shaped fixture (from live pull)
        fixture.db                    # tiny SQLite db with 1-2 fixture games
        fixture_model.pt                # tiny randomly-initialized checkpoint
      test_features.py
      test_db.py
      test_fetch_data.py
      test_model.py
      test_main.py
    data/                          # gitignored (raw/ and nba.db)
    model/                         # win_prob_lstm.pt, baseline_logreg.joblib, event_types.json (committed, small)
    reports/                       # evaluation_report.md / .json (committed)
  frontend/
    package.json
    vite.config.ts
    tsconfig.json
    index.html
    src/
      main.tsx
      App.tsx
      types.ts
      hooks/useGameSocket.ts
      components/GamePicker.tsx
      components/WinProbChart.tsx
      components/ScoreHeader.tsx
      components/PlayLog.tsx
      test/
        setup.ts
        useGameSocket.test.ts
        WinProbChart.test.tsx
        GamePicker.test.tsx
  .github/workflows/ci.yml
  docs/screenshots/
  README.md
  .gitignore
```

---

### Task 1: Repo scaffolding and GitHub repo creation

**Files:**
- Create: `.gitignore`
- Create: `backend/requirements.txt`
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `README.md` (placeholder, rewritten in Task 16)

**Interfaces:**
- Produces: a `backend/` Python package importable as `app.*` from within `backend/`; a git repo with a `main` branch pushed to a new GitHub repo `NBAWinProb` under the authenticated `gh` account.

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Data (regenerated by scripts/fetch_data.py, not committed)
backend/data/raw/
backend/data/*.db

# Node
node_modules/
frontend/dist/

# OS
.DS_Store
```

- [ ] **Step 2: Write `backend/requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
websockets==15.0.1
pydantic==2.7.4
torch==2.3.1
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
nba_api==1.5.2
joblib==1.4.2
pytest==8.2.2
pytest-asyncio==0.23.7
httpx==0.27.0
```

- [ ] **Step 3: Write `backend/pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 4: Create `backend/app/__init__.py`** (empty file, makes `app` a package)

- [ ] **Step 5: Create venv and install deps**

Run:
```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Expected: installs without error (torch CPU wheel is large — allow a few minutes).

- [ ] **Step 6: Write placeholder `README.md`**

```markdown
# NBA Win Probability

Work in progress — see `docs/superpowers/plans/2026-08-06-nba-win-probability-implementation.md`.
```

- [ ] **Step 7: Initial commit**

```bash
git add .gitignore backend/requirements.txt backend/pyproject.toml backend/app/__init__.py README.md docs
git commit -m "chore: scaffold backend package, deps, gitignore"
```

- [ ] **Step 8: Create the GitHub repo and push**

Run:
```bash
gh repo create NBAWinProb --public --source=. --remote=origin --description "Live NBA win-probability model: PyTorch LSTM over play-by-play, FastAPI + WebSocket serving, React frontend" --push
```
Expected: repo created under the authenticated account, `origin` remote added, `main` pushed. Confirm with `git remote -v` and `gh repo view --web=false`.

---

### Task 2: Feature engineering module (clock parsing + event/possession encoding)

**Files:**
- Create: `backend/app/features.py`
- Test: `backend/tests/test_features.py`

**Interfaces:**
- Produces:
  - `EVENT_TYPES: list[str]` — `["Made Shot", "Missed Shot", "Rebound", "Turnover", "Foul", "Free Throw", "Jump Ball", "Substitution", "Timeout", "Violation", "OTHER"]`
  - `EVENT_TYPE_TO_IDX: dict[str, int]`
  - `NUM_EVENT_TYPES: int = len(EVENT_TYPES)`
  - `REGULATION_PERIOD_SECONDS: int = 720`
  - `OT_PERIOD_SECONDS: int = 300`
  - `REGULATION_TOTAL_SECONDS: int = 2880`
  - `SCORE_DIFF_SCALE: float = 30.0`
  - `parse_clock_to_seconds(clock: str) -> float | None`
  - `total_seconds_remaining(period: int, clock: str) -> float | None`
  - `encode_event_type(raw_action_type: str) -> int`
  - `encode_possession(possession_team: str | None) -> float`  (`"home"`→`1.0`, `"away"`→`-1.0`, `None`→`0.0`)
  - `build_numeric_features(period: int, clock: str, home_score: int, away_score: int, possession_team: str | None) -> tuple[float, float, float] | None` — returns `(score_diff_scaled, seconds_remaining_scaled, possession_encoded)` or `None` if the clock is unparseable (caller must drop the event).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_features.py
import math
import pytest
from app.features import (
    EVENT_TYPES, EVENT_TYPE_TO_IDX, NUM_EVENT_TYPES,
    REGULATION_PERIOD_SECONDS, OT_PERIOD_SECONDS, REGULATION_TOTAL_SECONDS,
    parse_clock_to_seconds, total_seconds_remaining,
    encode_event_type, encode_possession, build_numeric_features,
)


def test_parse_clock_to_seconds_well_formed():
    assert parse_clock_to_seconds("PT11M42.00S") == pytest.approx(11 * 60 + 42.0)
    assert parse_clock_to_seconds("PT00M00.00S") == pytest.approx(0.0)
    assert parse_clock_to_seconds("PT12M00.00S") == pytest.approx(720.0)


@pytest.mark.parametrize("bad_clock", ["", "garbage", "0:00", "PTM S", None])
def test_parse_clock_to_seconds_malformed_returns_none(bad_clock):
    assert parse_clock_to_seconds(bad_clock) is None


def test_total_seconds_remaining_regulation():
    # Start of period 1: all 4 quarters ahead
    assert total_seconds_remaining(1, "PT12M00.00S") == pytest.approx(4 * 720)
    # Start of period 4, 5 minutes left: only this period remains
    assert total_seconds_remaining(4, "PT05M00.00S") == pytest.approx(300)
    # End of period 2: 2 quarters remain (3, 4) plus 0 left in this one
    assert total_seconds_remaining(2, "PT00M00.00S") == pytest.approx(2 * 720)


def test_total_seconds_remaining_overtime_only_current_ot_known():
    # First OT (period 5), 3 minutes left: future OTs are unknowable, so only
    # the current period's remaining time counts.
    assert total_seconds_remaining(5, "PT03M00.00S") == pytest.approx(180)
    assert total_seconds_remaining(6, "PT01M00.00S") == pytest.approx(60)


def test_total_seconds_remaining_malformed_clock_returns_none():
    assert total_seconds_remaining(1, "not a clock") is None


def test_encode_event_type_known():
    assert encode_event_type("Made Shot") == EVENT_TYPE_TO_IDX["Made Shot"]
    assert encode_event_type("Rebound") == EVENT_TYPE_TO_IDX["Rebound"]


def test_encode_event_type_unknown_maps_to_other():
    assert encode_event_type("Some New Event Type Nobody Trained On") == EVENT_TYPE_TO_IDX["OTHER"]
    assert encode_event_type("") == EVENT_TYPE_TO_IDX["OTHER"]
    assert encode_event_type("period") == EVENT_TYPE_TO_IDX["OTHER"]


def test_encode_possession():
    assert encode_possession("home") == 1.0
    assert encode_possession("away") == -1.0
    assert encode_possession(None) == 0.0


def test_build_numeric_features_valid():
    feats = build_numeric_features(
        period=2, clock="PT06M00.00S", home_score=50, away_score=45, possession_team="home"
    )
    assert feats is not None
    score_diff_scaled, seconds_remaining_scaled, possession = feats
    assert score_diff_scaled == pytest.approx(5 / 30.0)
    expected_seconds = total_seconds_remaining(2, "PT06M00.00S")
    assert seconds_remaining_scaled == pytest.approx(expected_seconds / REGULATION_TOTAL_SECONDS)
    assert possession == 1.0


def test_build_numeric_features_malformed_clock_returns_none():
    assert build_numeric_features(1, "bad", 0, 0, None) is None


def test_event_types_constants_consistent():
    assert NUM_EVENT_TYPES == len(EVENT_TYPES)
    assert EVENT_TYPES[-1] == "OTHER"
    assert all(EVENT_TYPE_TO_IDX[t] == i for i, t in enumerate(EVENT_TYPES))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.features'`.

- [ ] **Step 3: Implement `backend/app/features.py`**

```python
"""Shared feature engineering — used identically at training and serving time."""
import re

EVENT_TYPES: list[str] = [
    "Made Shot", "Missed Shot", "Rebound", "Turnover", "Foul",
    "Free Throw", "Jump Ball", "Substitution", "Timeout", "Violation",
    "OTHER",
]
EVENT_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(EVENT_TYPES)}
NUM_EVENT_TYPES: int = len(EVENT_TYPES)

REGULATION_PERIOD_SECONDS: int = 12 * 60
OT_PERIOD_SECONDS: int = 5 * 60
NUM_REGULATION_PERIODS: int = 4
REGULATION_TOTAL_SECONDS: int = REGULATION_PERIOD_SECONDS * NUM_REGULATION_PERIODS
SCORE_DIFF_SCALE: float = 30.0

_CLOCK_RE = re.compile(r"^PT(\d{1,2})M(\d{1,2}(?:\.\d+)?)S$")


def parse_clock_to_seconds(clock: str | None) -> float | None:
    """Parse an ISO-8601-style NBA clock string ('PT11M42.00S') to seconds
    remaining in the current period. Returns None if malformed."""
    if not clock:
        return None
    match = _CLOCK_RE.match(clock.strip())
    if not match:
        return None
    minutes, seconds = match.groups()
    return float(minutes) * 60 + float(seconds)


def total_seconds_remaining(period: int, clock: str | None) -> float | None:
    """Seconds remaining in the game from this event onward, OT-aware.
    Future overtime periods are unknowable in advance, so once play reaches
    OT (period > 4) only the current OT period's remaining time is counted."""
    clock_seconds = parse_clock_to_seconds(clock)
    if clock_seconds is None:
        return None
    if period <= NUM_REGULATION_PERIODS:
        remaining_full_periods = NUM_REGULATION_PERIODS - period
        return remaining_full_periods * REGULATION_PERIOD_SECONDS + clock_seconds
    return clock_seconds


def encode_event_type(raw_action_type: str | None) -> int:
    if raw_action_type in EVENT_TYPE_TO_IDX:
        return EVENT_TYPE_TO_IDX[raw_action_type]
    return EVENT_TYPE_TO_IDX["OTHER"]


def encode_possession(possession_team: str | None) -> float:
    if possession_team == "home":
        return 1.0
    if possession_team == "away":
        return -1.0
    return 0.0


def build_numeric_features(
    period: int,
    clock: str | None,
    home_score: int,
    away_score: int,
    possession_team: str | None,
) -> tuple[float, float, float] | None:
    """Returns (score_diff_scaled, seconds_remaining_scaled, possession_encoded)
    or None if the clock could not be parsed (caller must drop this event)."""
    seconds_remaining = total_seconds_remaining(period, clock)
    if seconds_remaining is None:
        return None
    score_diff_scaled = (home_score - away_score) / SCORE_DIFF_SCALE
    seconds_remaining_scaled = seconds_remaining / REGULATION_TOTAL_SECONDS
    possession_encoded = encode_possession(possession_team)
    return (score_diff_scaled, seconds_remaining_scaled, possession_encoded)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/features.py backend/tests/test_features.py backend/tests/__init__.py 2>/dev/null; git add backend/app/features.py backend/tests/test_features.py
git commit -m "feat: shared clock parsing and feature encoding"
```

---

### Task 3: SQLite schema and data access layer

**Files:**
- Create: `backend/app/db.py`
- Test: `backend/tests/test_db.py`

**Interfaces:**
- Consumes: nothing from prior tasks (standalone).
- Produces:
  - `SCHEMA_SQL: str`
  - `get_connection(db_path: str) -> sqlite3.Connection` (row_factory = sqlite3.Row, foreign_keys on)
  - `init_db(conn: sqlite3.Connection) -> None`
  - `class GameRow(TypedDict): game_id: str; home_team: str; away_team: str; home_win: int`
  - `class EventRow(TypedDict): game_id: str; event_index: int; period: int; clock: str; home_score: int; away_score: int; event_type: str; description: str; possession_team: str | None`
  - `insert_game(conn, game: GameRow) -> None`
  - `insert_events(conn, events: list[EventRow]) -> None`
  - `list_games(conn) -> list[GameRow]`
  - `get_game_events(conn, game_id: str) -> list[EventRow]` (ordered by `event_index` ascending)
  - `game_exists(conn, game_id: str) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_db.py
import sqlite3
import pytest
from app.db import (
    get_connection, init_db, insert_game, insert_events,
    list_games, get_game_events, game_exists,
)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    init_db(c)
    yield c
    c.close()


def test_init_db_creates_tables(conn):
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"games", "events"} <= tables


def test_insert_and_list_games(conn):
    insert_game(conn, {"game_id": "001", "home_team": "BOS", "away_team": "LAL", "home_win": 1})
    games = list_games(conn)
    assert len(games) == 1
    assert games[0]["game_id"] == "001"
    assert games[0]["home_win"] == 1


def test_game_exists(conn):
    assert game_exists(conn, "001") is False
    insert_game(conn, {"game_id": "001", "home_team": "BOS", "away_team": "LAL", "home_win": 1})
    assert game_exists(conn, "001") is True


def test_insert_and_get_events_ordered(conn):
    insert_game(conn, {"game_id": "001", "home_team": "BOS", "away_team": "LAL", "home_win": 1})
    events = [
        {"game_id": "001", "event_index": 1, "period": 1, "clock": "PT12M00.00S",
         "home_score": 0, "away_score": 0, "event_type": "Jump Ball",
         "description": "Jump Ball", "possession_team": None},
        {"game_id": "001", "event_index": 0, "period": 1, "clock": "PT12M00.00S",
         "home_score": 0, "away_score": 0, "event_type": "OTHER",
         "description": "Start of 1st Period", "possession_team": None},
    ]
    insert_events(conn, events)
    fetched = get_game_events(conn, "001")
    assert [e["event_index"] for e in fetched] == [0, 1]


def test_get_game_events_empty_for_unknown_game(conn):
    assert get_game_events(conn, "does-not-exist") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db'`.

- [ ] **Step 3: Implement `backend/app/db.py`**

```python
"""SQLite schema and access layer for cached play-by-play data."""
import sqlite3
from typing import TypedDict

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_win INTEGER NOT NULL CHECK (home_win IN (0, 1))
);

CREATE TABLE IF NOT EXISTS events (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    event_index INTEGER NOT NULL,
    period INTEGER NOT NULL,
    clock TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    possession_team TEXT,
    PRIMARY KEY (game_id, event_index)
);
"""


class GameRow(TypedDict):
    game_id: str
    home_team: str
    away_team: str
    home_win: int


class EventRow(TypedDict):
    game_id: str
    event_index: int
    period: int
    clock: str
    home_score: int
    away_score: int
    event_type: str
    description: str
    possession_team: str | None


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_game(conn: sqlite3.Connection, game: GameRow) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO games (game_id, home_team, away_team, home_win) "
        "VALUES (:game_id, :home_team, :away_team, :home_win)",
        game,
    )
    conn.commit()


def insert_events(conn: sqlite3.Connection, events: list[EventRow]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO events "
        "(game_id, event_index, period, clock, home_score, away_score, "
        " event_type, description, possession_team) "
        "VALUES (:game_id, :event_index, :period, :clock, :home_score, :away_score, "
        " :event_type, :description, :possession_team)",
        events,
    )
    conn.commit()


def list_games(conn: sqlite3.Connection) -> list[GameRow]:
    rows = conn.execute("SELECT * FROM games ORDER BY game_id").fetchall()
    return [dict(row) for row in rows]


def get_game_events(conn: sqlite3.Connection, game_id: str) -> list[EventRow]:
    rows = conn.execute(
        "SELECT * FROM events WHERE game_id = ? ORDER BY event_index ASC", (game_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def game_exists(conn: sqlite3.Connection, game_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM games WHERE game_id = ?", (game_id,)).fetchone()
    return row is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_db.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/db.py backend/tests/test_db.py
git commit -m "feat: SQLite schema and data access layer"
```

---

### Task 4: Data fetch script (nba_api → raw cache → SQLite)

**Files:**
- Create: `backend/scripts/fetch_data.py`
- Create: `backend/scripts/__init__.py` (empty)
- Create: `backend/tests/fixtures/sample_playbyplay_v3.json` (small, hand-trimmed real-shaped fixture — see Step 1)
- Test: `backend/tests/test_fetch_data.py`

**Interfaces:**
- Consumes: `app.db.EventRow`, `app.db.GameRow`, `app.db.insert_game`, `app.db.insert_events`, `app.features.parse_clock_to_seconds`, `app.features.EVENT_TYPES`.
- Produces:
  - `parse_game_list(rows: list[dict]) -> list[GameRow]` — takes raw `LeagueGameFinder` rows (list of dicts with `GAME_ID`, `TEAM_ID`, `TEAM_ABBREVIATION`, `MATCHUP`, `WL`), groups pairs by `GAME_ID`, returns one `GameRow` per game.
  - `parse_playbyplay_rows(pbp_rows: list[dict], home_team_tricode: str, away_team_tricode: str) -> list[EventRow]` — takes raw `PlayByPlayV3` rows (dicts with `actionNumber`, `clock`, `period`, `teamTricode`, `scoreHome`, `scoreAway`, `actionType`, `description`), drops rows with unparseable clocks, returns ordered `EventRow`s with `event_index` = 0-based position after filtering.
  - CLI `main()` invoked as `python scripts/fetch_data.py --season 2023-24 --max-games 150 --out-db data/nba.db --raw-dir data/raw`.

- [ ] **Step 1: Create the fixture file**

Run this one-off script to produce a small, real-shaped fixture (first 20 rows of a real game's play-by-play) — do this once, save the output:

```bash
cd backend
.venv/bin/python - <<'EOF'
import json
from nba_api.stats.endpoints import playbyplayv3
r = playbyplayv3.PlayByPlayV3(game_id="0022300061", timeout=15)
df = r.get_data_frames()[0]
rows = df.head(20).to_dict(orient="records")
with open("tests/fixtures/sample_playbyplay_v3.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)
print("wrote", len(rows), "rows")
EOF
```
Expected: `tests/fixtures/sample_playbyplay_v3.json` created with ~20 rows, home team DEN, away team LAL (from the earlier design exploration — game `0022300061`).

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_fetch_data.py
import json
from pathlib import Path
import pytest
from scripts.fetch_data import parse_game_list, parse_playbyplay_rows

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_game_list_groups_pairs_and_derives_home_win():
    raw_rows = [
        {"GAME_ID": "0022300061", "TEAM_ID": 1, "TEAM_ABBREVIATION": "DEN",
         "MATCHUP": "DEN vs. LAL", "WL": "W"},
        {"GAME_ID": "0022300061", "TEAM_ID": 2, "TEAM_ABBREVIATION": "LAL",
         "MATCHUP": "LAL @ DEN", "WL": "L"},
    ]
    games = parse_game_list(raw_rows)
    assert len(games) == 1
    game = games[0]
    assert game["game_id"] == "0022300061"
    assert game["home_team"] == "DEN"
    assert game["away_team"] == "LAL"
    assert game["home_win"] == 1


def test_parse_game_list_away_win():
    raw_rows = [
        {"GAME_ID": "1", "TEAM_ID": 1, "TEAM_ABBREVIATION": "BOS",
         "MATCHUP": "BOS vs. MIA", "WL": "L"},
        {"GAME_ID": "1", "TEAM_ID": 2, "TEAM_ABBREVIATION": "MIA",
         "MATCHUP": "MIA @ BOS", "WL": "W"},
    ]
    games = parse_game_list(raw_rows)
    assert games[0]["home_win"] == 0


def test_parse_playbyplay_rows_from_fixture():
    raw_rows = json.loads((FIXTURES / "sample_playbyplay_v3.json").read_text())
    events = parse_playbyplay_rows(raw_rows, home_team_tricode="DEN", away_team_tricode="LAL")
    assert len(events) > 0
    assert all(e["game_id"] == "0022300061" for e in events)
    # event_index is a contiguous 0-based sequence after filtering
    assert [e["event_index"] for e in events] == list(range(len(events)))
    # possession_team correctly mapped from teamTricode
    home_possession_events = [e for e in events if e["possession_team"] == "home"]
    assert len(home_possession_events) > 0
    # unknown/blank action types fall back to OTHER at parse time is NOT required here —
    # raw event_type strings are stored as-is; encoding to OTHER happens in app.features
    # at train/serve time. Just assert descriptions are preserved.
    assert all("description" in e for e in events)


def test_parse_playbyplay_rows_drops_unparseable_clock():
    raw_rows = [
        {"actionNumber": 1, "clock": "PT12M00.00S", "period": 1, "teamTricode": "DEN",
         "scoreHome": 0, "scoreAway": 0, "actionType": "Jump Ball", "description": "Tip"},
        {"actionNumber": 2, "clock": "GARBAGE", "period": 1, "teamTricode": "DEN",
         "scoreHome": 0, "scoreAway": 0, "actionType": "Made Shot", "description": "bad clock"},
        {"actionNumber": 3, "clock": "PT11M42.00S", "period": 1, "teamTricode": "LAL",
         "scoreHome": 2, "scoreAway": 0, "actionType": "Missed Shot", "description": "ok"},
    ]
    events = parse_playbyplay_rows(raw_rows, home_team_tricode="DEN", away_team_tricode="LAL")
    assert len(events) == 2
    assert [e["description"] for e in events] == ["Tip", "ok"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_fetch_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_data'`.

- [ ] **Step 4: Implement `backend/scripts/fetch_data.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_fetch_data.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/fetch_data.py backend/tests/test_fetch_data.py backend/tests/fixtures/sample_playbyplay_v3.json
git commit -m "feat: nba_api fetch script with cached raw JSON and SQLite parsing"
```

---

### Task 5: PyTorch LSTM model

**Files:**
- Create: `backend/app/model.py`
- Test: `backend/tests/test_model.py`

**Interfaces:**
- Consumes: `app.features.NUM_EVENT_TYPES`.
- Produces:
  - `class WinProbLSTM(nn.Module)`:
    - `__init__(self, num_event_types: int, event_embed_dim: int = 8, num_numeric_features: int = 3, hidden_size: int = 64, num_layers: int = 1)`
    - `forward(self, event_type_idx: LongTensor[B,T], numeric_features: FloatTensor[B,T,3], hidden: tuple[Tensor,Tensor] | None = None) -> tuple[FloatTensor[B,T], tuple[Tensor,Tensor]]` — returns `(win_prob, new_hidden)`.
    - `init_hidden(self, batch_size: int, device) -> tuple[Tensor, Tensor]`
  - `masked_bce_loss(win_prob: FloatTensor[B,T], targets: FloatTensor[B,T], mask: FloatTensor[B,T]) -> Tensor` (scalar)
  - `save_checkpoint(model: WinProbLSTM, config: dict, path: str) -> None` — saves `{"state_dict": ..., "config": config}`.
  - `load_model(checkpoint_path: str, device: str = "cpu") -> WinProbLSTM` — reconstructs from saved `config`, loads `state_dict`, calls `.eval()`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_model.py
import torch
import pytest
from app.model import WinProbLSTM, masked_bce_loss, save_checkpoint, load_model
from app.features import NUM_EVENT_TYPES


def test_forward_shapes_and_range():
    model = WinProbLSTM(num_event_types=NUM_EVENT_TYPES, hidden_size=16, num_layers=1)
    batch, seq_len = 3, 5
    event_type_idx = torch.randint(0, NUM_EVENT_TYPES, (batch, seq_len))
    numeric_features = torch.randn(batch, seq_len, 3)
    win_prob, hidden = model(event_type_idx, numeric_features)
    assert win_prob.shape == (batch, seq_len)
    assert torch.all(win_prob >= 0) and torch.all(win_prob <= 1)
    assert hidden[0].shape == (1, batch, 16)


def test_forward_incremental_matches_full_sequence():
    """Feeding one timestep at a time with carried hidden state must match
    feeding the whole sequence at once — this is what /replay and /live rely on."""
    model = WinProbLSTM(num_event_types=NUM_EVENT_TYPES, hidden_size=16, num_layers=1)
    model.eval()
    seq_len = 4
    event_type_idx = torch.randint(0, NUM_EVENT_TYPES, (1, seq_len))
    numeric_features = torch.randn(1, seq_len, 3)

    with torch.no_grad():
        full_probs, _ = model(event_type_idx, numeric_features)

        hidden = None
        incremental_probs = []
        for t in range(seq_len):
            step_prob, hidden = model(
                event_type_idx[:, t : t + 1], numeric_features[:, t : t + 1, :], hidden
            )
            incremental_probs.append(step_prob)
        incremental_probs = torch.cat(incremental_probs, dim=1)

    assert torch.allclose(full_probs, incremental_probs, atol=1e-5)


def test_masked_bce_loss_ignores_padding():
    win_prob = torch.tensor([[0.9, 0.1, 0.5]])
    targets = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # last timestep is padding
    loss = masked_bce_loss(win_prob, targets, mask)
    # Only first two timesteps count: -log(0.9) and -log(0.1)
    expected = -(torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.1))) / 2
    assert torch.allclose(loss, expected, atol=1e-4)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    config = {"num_event_types": NUM_EVENT_TYPES, "event_embed_dim": 4,
              "num_numeric_features": 3, "hidden_size": 8, "num_layers": 1}
    model = WinProbLSTM(**config)
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(model, config, str(ckpt_path))

    loaded = load_model(str(ckpt_path), device="cpu")
    assert not loaded.training  # .eval() was called

    event_type_idx = torch.randint(0, NUM_EVENT_TYPES, (1, 3))
    numeric_features = torch.randn(1, 3, 3)
    with torch.no_grad():
        original_out, _ = model(event_type_idx, numeric_features)
        loaded_out, _ = loaded(event_type_idx, numeric_features)
    assert torch.allclose(original_out, loaded_out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.model'`.

- [ ] **Step 3: Implement `backend/app/model.py`**

```python
"""PyTorch LSTM win-probability model."""
import torch
import torch.nn as nn


class WinProbLSTM(nn.Module):
    def __init__(
        self,
        num_event_types: int,
        event_embed_dim: int = 8,
        num_numeric_features: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_event_types = num_event_types
        self.event_embed_dim = event_embed_dim
        self.num_numeric_features = num_numeric_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.event_embedding = nn.Embedding(num_event_types, event_embed_dim)
        self.lstm = nn.LSTM(
            input_size=event_embed_dim + num_numeric_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, event_type_idx, numeric_features, hidden=None):
        embedded = self.event_embedding(event_type_idx)  # (B, T, embed_dim)
        lstm_input = torch.cat([embedded, numeric_features], dim=-1)  # (B, T, embed_dim+3)
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)  # (B, T, hidden_size)
        logits = self.head(lstm_out).squeeze(-1)  # (B, T)
        win_prob = torch.sigmoid(logits)
        return win_prob, new_hidden

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)


def masked_bce_loss(win_prob: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    win_prob = win_prob.clamp(eps, 1 - eps)
    per_step = -(targets * torch.log(win_prob) + (1 - targets) * torch.log(1 - win_prob))
    return (per_step * mask).sum() / mask.sum().clamp(min=1)


def save_checkpoint(model: WinProbLSTM, config: dict, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_model(checkpoint_path: str, device: str = "cpu") -> WinProbLSTM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WinProbLSTM(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_model.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/model.py backend/tests/test_model.py
git commit -m "feat: WinProbLSTM model, masked loss, checkpoint save/load"
```

---

### Task 6: Training script (LSTM + logistic-regression baseline)

**Files:**
- Create: `backend/scripts/train.py`
- Test: `backend/tests/test_train.py`

**Interfaces:**
- Consumes: `app.db.get_connection`, `app.db.list_games`, `app.db.get_game_events`, `app.features.build_numeric_features`, `app.features.encode_event_type`, `app.features.NUM_EVENT_TYPES`, `app.model.WinProbLSTM`, `app.model.masked_bce_loss`, `app.model.save_checkpoint`.
- Produces:
  - `split_games(game_ids: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]` — 80/10/10 train/val/test, deterministic given seed.
  - `build_game_tensors(events: list[EventRow]) -> dict | None` — returns `{"event_type_idx": LongTensor[T], "numeric_features": FloatTensor[T,3], "target": float}` for one game (target = final `home_win`, same value used at training for every timestep per spec), or `None` if the game has zero usable events after feature building.
  - `train_model(db_path: str, epochs: int, hidden_size: int, lr: float, seed: int) -> tuple[WinProbLSTM, dict, LogisticRegression, dict]` — returns `(lstm_model, lstm_config, baseline_model, split_info)`. `split_info = {"train_game_ids": [...], "val_game_ids": [...], "test_game_ids": [...]}`.
  - CLI `main()`: `python scripts/train.py --db data/nba.db --epochs 15 --hidden-size 64 --out-model model/win_prob_lstm.pt --out-baseline model/baseline_logreg.joblib --out-split model/split.json`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_train.py
import sqlite3
import pytest
from app.db import get_connection, init_db, insert_game, insert_events
from scripts.train import split_games, build_game_tensors, train_model


def _make_fixture_db(path):
    conn = get_connection(path)
    init_db(conn)
    for i in range(8):
        game_id = f"g{i}"
        insert_game(conn, {"game_id": game_id, "home_team": "AAA", "away_team": "BBB",
                            "home_win": i % 2})
        events = []
        for t in range(10):
            events.append({
                "game_id": game_id, "event_index": t, "period": 1,
                "clock": f"PT{11 - t}M00.00S", "home_score": t, "away_score": t // 2,
                "event_type": "Made Shot" if t % 2 == 0 else "Missed Shot",
                "description": "x", "possession_team": "home" if t % 2 == 0 else "away",
            })
        insert_events(conn, events)
    conn.close()


def test_split_games_deterministic_and_covers_all():
    game_ids = [f"g{i}" for i in range(20)]
    train1, val1, test1 = split_games(game_ids, seed=42)
    train2, val2, test2 = split_games(game_ids, seed=42)
    assert (train1, val1, test1) == (train2, val2, test2)
    assert set(train1) | set(val1) | set(test1) == set(game_ids)
    assert not (set(train1) & set(val1)) and not (set(train1) & set(test1)) and not (set(val1) & set(test1))
    assert len(train1) == 16 and len(val1) == 2 and len(test1) == 2


def test_build_game_tensors_shapes():
    events = [
        {"game_id": "g0", "event_index": 0, "period": 1, "clock": "PT12M00.00S",
         "home_score": 0, "away_score": 0, "event_type": "Jump Ball",
         "description": "x", "possession_team": None},
        {"game_id": "g0", "event_index": 1, "period": 1, "clock": "PT11M00.00S",
         "home_score": 2, "away_score": 0, "event_type": "Made Shot",
         "description": "x", "possession_team": "home"},
    ]
    result = build_game_tensors(events, home_win=1)
    assert result is not None
    assert result["event_type_idx"].shape == (2,)
    assert result["numeric_features"].shape == (2, 3)
    assert result["target"] == 1.0


def test_train_model_runs_end_to_end(tmp_path):
    db_path = str(tmp_path / "fixture.db")
    _make_fixture_db(db_path)
    lstm_model, lstm_config, baseline_model, split_info = train_model(
        db_path=db_path, epochs=1, hidden_size=4, lr=1e-2, seed=42
    )
    assert lstm_config["hidden_size"] == 4
    assert set(split_info.keys()) == {"train_game_ids", "val_game_ids", "test_game_ids"}
    # baseline is a fitted sklearn model — predict_proba should work
    import numpy as np
    preds = baseline_model.predict_proba(np.zeros((1, 3 + 1)))  # +1 for event_type_idx as a raw feature
    assert preds.shape == (1, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.train'`.

- [ ] **Step 3: Implement `backend/scripts/train.py`**

```python
"""Trains the WinProbLSTM and a logistic-regression baseline on cached play-by-play data.

Usage:
    python scripts/train.py --db data/nba.db --epochs 15 --hidden-size 64 \
        --out-model model/win_prob_lstm.pt --out-baseline model/baseline_logreg.joblib \
        --out-split model/split.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_connection, get_game_events, list_games
from app.features import build_numeric_features, encode_event_type
from app.model import WinProbLSTM, masked_bce_loss, save_checkpoint


def split_games(game_ids: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    ids = sorted(game_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def build_game_tensors(events: list[dict], home_win: int) -> dict | None:
    event_type_indices = []
    numeric_rows = []
    for e in events:
        feats = build_numeric_features(
            e["period"], e["clock"], e["home_score"], e["away_score"], e["possession_team"]
        )
        if feats is None:
            continue
        event_type_indices.append(encode_event_type(e["event_type"]))
        numeric_rows.append(feats)

    if not event_type_indices:
        return None

    return {
        "event_type_idx": torch.tensor(event_type_indices, dtype=torch.long),
        "numeric_features": torch.tensor(numeric_rows, dtype=torch.float32),
        "target": float(home_win),
    }


def _collate(games: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a list of per-game tensors to the batch's max length. Returns
    (event_type_idx, numeric_features, targets, mask)."""
    max_len = max(g["event_type_idx"].shape[0] for g in games)
    batch = len(games)
    event_type_idx = torch.zeros(batch, max_len, dtype=torch.long)
    numeric_features = torch.zeros(batch, max_len, 3, dtype=torch.float32)
    targets = torch.zeros(batch, max_len, dtype=torch.float32)
    mask = torch.zeros(batch, max_len, dtype=torch.float32)
    for i, g in enumerate(games):
        length = g["event_type_idx"].shape[0]
        event_type_idx[i, :length] = g["event_type_idx"]
        numeric_features[i, :length, :] = g["numeric_features"]
        targets[i, :length] = g["target"]
        mask[i, :length] = 1.0
    return event_type_idx, numeric_features, targets, mask


def train_model(db_path: str, epochs: int, hidden_size: int, lr: float, seed: int):
    torch.manual_seed(seed)
    conn = get_connection(db_path)
    games = list_games(conn)
    game_ids = [g["game_id"] for g in games]
    home_win_by_id = {g["game_id"]: g["home_win"] for g in games}
    train_ids, val_ids, test_ids = split_games(game_ids, seed=seed)

    def load_split(ids: list[str]) -> list[dict]:
        out = []
        for gid in ids:
            events = get_game_events(conn, gid)
            tensors = build_game_tensors(events, home_win_by_id[gid])
            if tensors is not None:
                out.append(tensors)
        return out

    train_games = load_split(train_ids)
    conn.close()

    from app.features import NUM_EVENT_TYPES

    lstm_config = {
        "num_event_types": NUM_EVENT_TYPES,
        "event_embed_dim": 8,
        "num_numeric_features": 3,
        "hidden_size": hidden_size,
        "num_layers": 1,
    }
    model = WinProbLSTM(**lstm_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(train_games)
        event_type_idx, numeric_features, targets, mask = _collate(train_games)
        optimizer.zero_grad()
        win_prob, _ = model(event_type_idx, numeric_features)
        loss = masked_bce_loss(win_prob, targets, mask)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    model.eval()

    # Baseline: flatten every event of every training game into an independent row.
    baseline_X, baseline_y = [], []
    for g in train_games:
        n = g["event_type_idx"].shape[0]
        event_type_col = g["event_type_idx"].numpy().reshape(-1, 1).astype(float)
        numeric_cols = g["numeric_features"].numpy()
        baseline_X.append(np.hstack([numeric_cols, event_type_col]))
        baseline_y.append(np.full(n, g["target"]))
    baseline_X = np.vstack(baseline_X)
    baseline_y = np.concatenate(baseline_y)
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(baseline_X, baseline_y)

    split_info = {"train_game_ids": train_ids, "val_game_ids": val_ids, "test_game_ids": test_ids}
    return model, lstm_config, baseline_model, split_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/nba.db")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-model", default="model/win_prob_lstm.pt")
    parser.add_argument("--out-baseline", default="model/baseline_logreg.joblib")
    parser.add_argument("--out-split", default="model/split.json")
    args = parser.parse_args()

    model, config, baseline_model, split_info = train_model(
        args.db, args.epochs, args.hidden_size, args.lr, args.seed
    )

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, config, args.out_model)
    joblib.dump(baseline_model, args.out_baseline)
    Path(args.out_split).write_text(json.dumps(split_info, indent=2))
    print(f"Saved LSTM to {args.out_model}, baseline to {args.out_baseline}, split to {args.out_split}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_train.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/train.py backend/tests/test_train.py
git commit -m "feat: LSTM + logistic-regression baseline training script"
```

---

### Task 7: Evaluation script

**Files:**
- Create: `backend/scripts/evaluate.py`
- Test: `backend/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `app.model.load_model`, `app.db.get_connection/get_game_events`, `scripts.train.build_game_tensors`, joblib-loaded `LogisticRegression`.
- Produces:
  - `brier_score(probs: np.ndarray, targets: np.ndarray) -> float`
  - `log_loss_score(probs: np.ndarray, targets: np.ndarray) -> float`
  - `calibration_curve(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> list[dict]` — each dict `{"bin_start": float, "bin_end": float, "predicted_mean": float, "observed_frequency": float, "count": int}`, bins with zero samples omitted.
  - `TIME_BUCKETS: list[tuple[str, float, float]]` — `[(">36 min", 36*60, float("inf")), ("12-36 min", 12*60, 36*60), ("3-12 min", 3*60, 12*60), ("<3 min", 0, 3*60)]` (seconds remaining, using `total_seconds_remaining` — bucket edges chosen so `>36 min` only applies pre-game/very-early since regulation total is 48 min).
  - `accuracy_by_time_bucket(probs, targets, seconds_remaining: np.ndarray) -> dict[str, dict]` — `{"bucket_name": {"accuracy": float, "count": int}}` (accuracy = fraction where `(prob > 0.5) == target`).
  - `generate_report(db_path, model_path, baseline_path, split_path) -> dict` — evaluates both models on the test split, returns a report dict; also called by `main()` to write `reports/evaluation_report.json` and a human-readable `reports/evaluation_report.md`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_evaluate.py
import numpy as np
import pytest
from scripts.evaluate import brier_score, log_loss_score, calibration_curve, accuracy_by_time_bucket, TIME_BUCKETS


def test_brier_score_perfect_predictions_is_zero():
    probs = np.array([1.0, 0.0, 1.0])
    targets = np.array([1.0, 0.0, 1.0])
    assert brier_score(probs, targets) == pytest.approx(0.0)


def test_brier_score_worst_predictions_is_one():
    probs = np.array([0.0, 1.0])
    targets = np.array([1.0, 0.0])
    assert brier_score(probs, targets) == pytest.approx(1.0)


def test_log_loss_score_reasonable():
    probs = np.array([0.9, 0.1])
    targets = np.array([1.0, 0.0])
    loss = log_loss_score(probs, targets)
    assert loss > 0
    assert loss == pytest.approx(-np.log(0.9), abs=1e-4)


def test_calibration_curve_bins_and_shape():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 200)
    targets = (rng.uniform(0, 1, 200) < probs).astype(float)
    curve = calibration_curve(probs, targets, n_bins=5)
    assert len(curve) > 0
    for bucket in curve:
        assert 0.0 <= bucket["predicted_mean"] <= 1.0
        assert 0.0 <= bucket["observed_frequency"] <= 1.0
        assert bucket["count"] > 0


def test_accuracy_by_time_bucket_perfect_predictions():
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    seconds_remaining = np.array([2000, 2000, 100, 100])  # first two in one bucket, last two in another
    result = accuracy_by_time_bucket(probs, targets, seconds_remaining)
    assert set(result.keys()) == {name for name, _, _ in TIME_BUCKETS}
    populated = {k: v for k, v in result.items() if v["count"] > 0}
    assert all(v["accuracy"] == pytest.approx(1.0) for v in populated.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.evaluate'`.

- [ ] **Step 3: Implement `backend/scripts/evaluate.py`**

```python
"""Evaluation report: Brier score, log-loss, calibration, accuracy by time bucket.

Usage:
    python scripts/evaluate.py --db data/nba.db --model model/win_prob_lstm.pt \
        --baseline model/baseline_logreg.joblib --split model/split.json \
        --out-md reports/evaluation_report.md --out-json reports/evaluation_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_connection, get_game_events, list_games
from app.model import load_model
from scripts.train import build_game_tensors

TIME_BUCKETS: list[tuple[str, float, float]] = [
    (">36 min", 36 * 60, float("inf")),
    ("12-36 min", 12 * 60, 36 * 60),
    ("3-12 min", 3 * 60, 12 * 60),
    ("<3 min", 0, 3 * 60),
]


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def log_loss_score(probs: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))


def calibration_curve(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        out.append({
            "bin_start": float(lo),
            "bin_end": float(hi),
            "predicted_mean": float(probs[in_bin].mean()),
            "observed_frequency": float(targets[in_bin].mean()),
            "count": count,
        })
    return out


def accuracy_by_time_bucket(probs: np.ndarray, targets: np.ndarray, seconds_remaining: np.ndarray) -> dict:
    result = {}
    for name, lo, hi in TIME_BUCKETS:
        in_bucket = (seconds_remaining >= lo) & (seconds_remaining < hi)
        count = int(in_bucket.sum())
        if count == 0:
            result[name] = {"accuracy": None, "count": 0}
            continue
        correct = ((probs[in_bucket] > 0.5).astype(float) == targets[in_bucket]).mean()
        result[name] = {"accuracy": float(correct), "count": count}
    return result


def _evaluate_lstm(model, test_games: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs, all_targets, all_seconds = [], [], []
    with torch.no_grad():
        for g in test_games:
            event_type_idx = g["event_type_idx"].unsqueeze(0)
            numeric_features = g["numeric_features"].unsqueeze(0)
            probs, _ = model(event_type_idx, numeric_features)
            probs = probs.squeeze(0).numpy()
            n = len(probs)
            all_probs.append(probs)
            all_targets.append(np.full(n, g["target"]))
            # numeric_features[..., 1] is seconds_remaining_scaled; undo the scale for bucketing
            from app.features import REGULATION_TOTAL_SECONDS
            all_seconds.append(g["numeric_features"][:, 1].numpy() * REGULATION_TOTAL_SECONDS)
    return np.concatenate(all_probs), np.concatenate(all_targets), np.concatenate(all_seconds)


def _evaluate_baseline(baseline_model, test_games: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_targets = [], []
    for g in test_games:
        n = g["event_type_idx"].shape[0]
        event_type_col = g["event_type_idx"].numpy().reshape(-1, 1).astype(float)
        numeric_cols = g["numeric_features"].numpy()
        X = np.hstack([numeric_cols, event_type_col])
        probs = baseline_model.predict_proba(X)[:, 1]
        all_probs.append(probs)
        all_targets.append(np.full(n, g["target"]))
    return np.concatenate(all_probs), np.concatenate(all_targets)


def generate_report(db_path: str, model_path: str, baseline_path: str, split_path: str) -> dict:
    conn = get_connection(db_path)
    games = {g["game_id"]: g for g in list_games(conn)}
    split_info = json.loads(Path(split_path).read_text())
    test_ids = split_info["test_game_ids"]

    test_games = []
    for gid in test_ids:
        events = get_game_events(conn, gid)
        tensors = build_game_tensors(events, games[gid]["home_win"])
        if tensors is not None:
            test_games.append(tensors)
    conn.close()

    lstm_model = load_model(model_path)
    baseline_model = joblib.load(baseline_path)

    lstm_probs, lstm_targets, seconds_remaining = _evaluate_lstm(lstm_model, test_games)
    baseline_probs, baseline_targets = _evaluate_baseline(baseline_model, test_games)

    report = {
        "num_test_games": len(test_games),
        "lstm": {
            "brier_score": brier_score(lstm_probs, lstm_targets),
            "log_loss": log_loss_score(lstm_probs, lstm_targets),
            "calibration": calibration_curve(lstm_probs, lstm_targets),
            "accuracy_by_time_bucket": accuracy_by_time_bucket(lstm_probs, lstm_targets, seconds_remaining),
        },
        "baseline_logreg": {
            "brier_score": brier_score(baseline_probs, baseline_targets),
            "log_loss": log_loss_score(baseline_probs, baseline_targets),
            "calibration": calibration_curve(baseline_probs, baseline_targets),
            "accuracy_by_time_bucket": accuracy_by_time_bucket(baseline_probs, baseline_targets, seconds_remaining),
        },
    }
    return report


def _report_to_markdown(report: dict) -> str:
    lines = ["# Evaluation Report", "", f"Test games: {report['num_test_games']}", ""]
    lines.append("| Metric | LSTM | Logistic Regression baseline |")
    lines.append("|---|---|---|")
    lines.append(f"| Brier score (lower better) | {report['lstm']['brier_score']:.4f} | {report['baseline_logreg']['brier_score']:.4f} |")
    lines.append(f"| Log-loss (lower better) | {report['lstm']['log_loss']:.4f} | {report['baseline_logreg']['log_loss']:.4f} |")
    lines.append("")
    lines.append("## Accuracy by time remaining")
    lines.append("| Time remaining | LSTM accuracy | Baseline accuracy | N |")
    lines.append("|---|---|---|---|")
    for name, _, _ in TIME_BUCKETS:
        l = report["lstm"]["accuracy_by_time_bucket"][name]
        b = report["baseline_logreg"]["accuracy_by_time_bucket"][name]
        l_acc = f"{l['accuracy']:.3f}" if l["accuracy"] is not None else "n/a"
        b_acc = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "n/a"
        lines.append(f"| {name} | {l_acc} | {b_acc} | {l['count']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/nba.db")
    parser.add_argument("--model", default="model/win_prob_lstm.pt")
    parser.add_argument("--baseline", default="model/baseline_logreg.joblib")
    parser.add_argument("--split", default="model/split.json")
    parser.add_argument("--out-md", default="reports/evaluation_report.md")
    parser.add_argument("--out-json", default="reports/evaluation_report.json")
    args = parser.parse_args()

    report = generate_report(args.db, args.model, args.baseline, args.split)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    Path(args.out_md).write_text(_report_to_markdown(report))
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_evaluate.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/evaluate.py backend/tests/test_evaluate.py
git commit -m "feat: evaluation report (Brier, log-loss, calibration, time-bucket accuracy)"
```

---

### Task 8: FastAPI schemas, `GET /games`, and incremental inference session

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/app/inference.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/fixtures/fixture.db` (built by a helper script, see Step 1)
- Create: `backend/tests/fixtures/fixture_model.pt` (built by a helper script, see Step 1)
- Test: `backend/tests/test_main.py`

**Interfaces:**
- Consumes: `app.db.*`, `app.model.WinProbLSTM`, `app.model.load_model`, `app.features.build_numeric_features`, `app.features.encode_event_type`.
- Produces:
  - `schemas.GameSummary(BaseModel)`: `game_id: str; home_team: str; away_team: str`
  - `schemas.GamesResponse(BaseModel)`: `replay_games: list[GameSummary]; live_available: bool`
  - `schemas.WinProbMessage(BaseModel)`: `event_index: int; period: int; clock: str; home_score: int; away_score: int; event_type: str; description: str; win_prob: float`
  - `inference.InferenceSession` class: `__init__(self, model: WinProbLSTM)`; `step(self, event: EventRow) -> float | None` — builds features for one event, runs one incremental forward pass carrying `self._hidden`, returns `win_prob` or `None` if the event's clock was unparseable (caller should skip emitting a message for that event).
  - `main.app: FastAPI` with `GET /games -> GamesResponse` and route stubs for `/replay/{game_id}` and `/live` (implemented in Tasks 9-10; this task wires the app, model loading via lifespan, and `/games`).
  - `main.is_live_game_available() -> bool` — wraps an `nba_api` live-scoreboard check in try/except, returns `False` on any error (network, no live games, etc.) so `/games` never fails because live detection failed.

- [ ] **Step 1: Build fixture DB and fixture model checkpoint**

```bash
cd backend
.venv/bin/python - <<'EOF'
import sys
sys.path.insert(0, ".")
from app.db import get_connection, init_db, insert_game, insert_events
from app.model import WinProbLSTM, save_checkpoint
from app.features import NUM_EVENT_TYPES

conn = get_connection("tests/fixtures/fixture.db")
init_db(conn)
insert_game(conn, {"game_id": "fixture001", "home_team": "DEN", "away_team": "LAL", "home_win": 1})
events = []
for t in range(15):
    events.append({
        "game_id": "fixture001", "event_index": t, "period": 1,
        "clock": f"PT{max(0, 11 - t)}M00.00S", "home_score": t * 2, "away_score": t,
        "event_type": "Made Shot" if t % 3 == 0 else "Missed Shot",
        "description": f"event {t}", "possession_team": "home" if t % 2 == 0 else "away",
    })
insert_events(conn, events)
conn.close()

config = {"num_event_types": NUM_EVENT_TYPES, "event_embed_dim": 4,
          "num_numeric_features": 3, "hidden_size": 8, "num_layers": 1}
model = WinProbLSTM(**config)
save_checkpoint(model, config, "tests/fixtures/fixture_model.pt")
print("fixtures written")
EOF
```
Expected: `tests/fixtures/fixture.db` and `tests/fixtures/fixture_model.pt` created.

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_main.py
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the app at fixture DB/model via env vars before import.
    db_copy = tmp_path / "fixture.db"
    shutil.copy(FIXTURES / "fixture.db", db_copy)
    monkeypatch.setenv("NBA_DB_PATH", str(db_copy))
    monkeypatch.setenv("NBA_MODEL_PATH", str(FIXTURES / "fixture_model.pt"))
    monkeypatch.setenv("NBA_REPLAY_INTERVAL_MS", "1")  # fast for tests

    import importlib
    import app.main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as c:
        yield c


def test_get_games_returns_fixture_game(client):
    resp = client.get("/games")
    assert resp.status_code == 200
    data = resp.json()
    assert any(g["game_id"] == "fixture001" for g in data["replay_games"])
    assert isinstance(data["live_available"], bool)


def test_replay_websocket_streams_increasing_events(client):
    with client.websocket_connect("/replay/fixture001") as ws:
        received = []
        try:
            while True:
                msg = ws.receive_json()
                received.append(msg)
        except Exception:
            pass
    assert len(received) == 15
    assert [m["event_index"] for m in received] == list(range(15))
    assert all(0.0 <= m["win_prob"] <= 1.0 for m in received)


def test_replay_websocket_unknown_game_closes_immediately(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/replay/does-not-exist") as ws:
            ws.receive_json()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 4: Implement `backend/app/schemas.py`**

```python
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
```

- [ ] **Step 5: Implement `backend/app/inference.py`**

```python
"""Per-WebSocket-connection incremental inference — one forward pass per event,
hidden state carried across the session (spec: no full-prefix replay per event)."""
import torch

from app.features import build_numeric_features, encode_event_type
from app.model import WinProbLSTM


class InferenceSession:
    def __init__(self, model: WinProbLSTM):
        self._model = model
        self._hidden = None

    def step(self, event: dict) -> float | None:
        feats = build_numeric_features(
            event["period"], event["clock"], event["home_score"],
            event["away_score"], event["possession_team"],
        )
        if feats is None:
            return None
        event_type_idx = torch.tensor([[encode_event_type(event["event_type"])]], dtype=torch.long)
        numeric_features = torch.tensor([[list(feats)]], dtype=torch.float32)
        with torch.no_grad():
            win_prob, self._hidden = self._model(event_type_idx, numeric_features, self._hidden)
        return float(win_prob.item())
```

- [ ] **Step 6: Implement `backend/app/main.py`** (base app + `/games`; `/replay` and `/live` route bodies added in Tasks 9-10, but the route registration and a minimal working `/replay` are included now since Task 8's tests exercise it — Task 9 will move the replay body into `app/replay.py` and import it here)

```python
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
```

- [ ] **Step 7: Implement a minimal `backend/app/replay.py` stub so Task 8's tests pass** (fleshed out fully in Task 9)

```python
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
```

- [ ] **Step 8: Implement a minimal `backend/app/live.py` stub so imports resolve** (fleshed out fully in Task 10)

```python
"""WS /live producer — polls nba_api's live scoreboard/play-by-play. Implemented in Task 10."""
from fastapi import WebSocket

from app.inference import InferenceSession


async def run_live(websocket: WebSocket, session: InferenceSession) -> None:
    await websocket.close(code=4204, reason="live mode not yet available")
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_main.py -v`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add backend/app/schemas.py backend/app/inference.py backend/app/main.py backend/app/replay.py backend/app/live.py backend/tests/test_main.py backend/tests/fixtures/fixture.db backend/tests/fixtures/fixture_model.pt
git commit -m "feat: FastAPI app, GET /games, replay websocket, inference session"
```

---

### Task 9: Full replay endpoint hardening

**Files:**
- Modify: `backend/app/replay.py`
- Modify: `backend/tests/test_main.py`

**Interfaces:**
- Consumes: same as Task 8.
- Produces: same `run_replay` signature; adds handling for client disconnect mid-stream (must not raise unhandled exceptions) and games with zero events (closes immediately, no crash).

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_main.py

def test_replay_websocket_empty_game_closes_without_error(client, tmp_path):
    import app.main as main_module
    from app.db import get_connection, insert_game

    conn = get_connection(main_module.DB_PATH)
    insert_game(conn, {"game_id": "empty-game", "home_team": "X", "away_team": "Y", "home_win": 1})
    conn.close()

    with client.websocket_connect("/replay/empty-game") as ws:
        with pytest.raises(Exception):
            ws.receive_json()
```

- [ ] **Step 2: Run test to verify it fails or passes trivially, then confirm behavior explicitly**

Run: `cd backend && .venv/bin/pytest tests/test_main.py::test_replay_websocket_empty_game_closes_without_error -v`
Expected: likely already PASS given Task 8's implementation (empty `events` list means the for-loop body never runs and `websocket.close()` is called) — this step is a regression guard, not new functionality. If it fails, the `run_replay` loop is not reached correctly; fix by ensuring `get_game_events` on a gameless id returns `[]` (already guaranteed by `app.db.get_game_events`).

- [ ] **Step 3: Harden `backend/app/replay.py` against client disconnect mid-stream**

```python
"""WS /replay producer — streams a cached game's events at an accelerated interval."""
import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from app.inference import InferenceSession
from app.schemas import WinProbMessage
from app.db import get_game_events


async def run_replay(websocket: WebSocket, conn, game_id: str, session: InferenceSession, interval_ms: int) -> None:
    events = get_game_events(conn, game_id)
    try:
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
    except WebSocketDisconnect:
        return
```

- [ ] **Step 4: Run full test suite**

Run: `cd backend && .venv/bin/pytest tests/test_main.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/replay.py backend/tests/test_main.py
git commit -m "fix: replay websocket handles client disconnect and empty games cleanly"
```

---

### Task 10: Live endpoint (nba_api polling)

**Files:**
- Modify: `backend/app/live.py`
- Test: `backend/tests/test_live.py`

**Interfaces:**
- Consumes: `app.inference.InferenceSession`, `app.schemas.WinProbMessage`.
- Produces:
  - `fetch_live_game_id() -> str | None` — wraps `nba_api.live.nba.endpoints.scoreboard.ScoreBoard`, returns the `gameId` of the first in-progress game (`gameStatus == 2`), or `None`. Never raises.
  - `fetch_live_events(game_id: str) -> list[dict]` — wraps `nba_api.live.nba.endpoints.playbyplay.PlayByPlay(game_id=...).get_dict()`, maps the live endpoint's action list into the same event dict shape used by `InferenceSession.step` (`period`, `clock`, `home_score`, `away_score`, `event_type`, `description`, `possession_team`, `event_index`). The live endpoint's clock field is also `PT..M..S` format, and its actions carry `actionType` — reuse the same mapping logic as `scripts/fetch_data.parse_playbyplay_rows` conceptually, but operating on the live dict shape (`teamTricode` present per action, same as v3 stats endpoint since both are NBA's modern schema).
  - `run_live(websocket: WebSocket, session: InferenceSession, poll_interval_s: float = 3.0) -> None` — if no live game, closes with code `4204`; otherwise polls, sends only new events (tracked by `event_index` high-water mark) as `WinProbMessage`s, loops until the client disconnects.

- [ ] **Step 1: Write the failing tests (mocking nba_api, no real network dependency)**

```python
# backend/tests/test_live.py
from unittest.mock import patch, MagicMock
import pytest
from app.live import fetch_live_game_id, fetch_live_events


def test_fetch_live_game_id_returns_none_on_no_games():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as MockBoard:
        MockBoard.return_value.get_dict.return_value = {"scoreboard": {"games": []}}
        assert fetch_live_game_id() is None


def test_fetch_live_game_id_returns_in_progress_game():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as MockBoard:
        MockBoard.return_value.get_dict.return_value = {
            "scoreboard": {"games": [
                {"gameId": "g1", "gameStatus": 3},
                {"gameId": "g2", "gameStatus": 2},
            ]}
        }
        assert fetch_live_game_id() == "g2"


def test_fetch_live_game_id_returns_none_on_exception():
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard", side_effect=RuntimeError("network down")):
        assert fetch_live_game_id() is None


def test_fetch_live_events_maps_actions():
    with patch("nba_api.live.nba.endpoints.playbyplay.PlayByPlay") as MockPBP:
        MockPBP.return_value.get_dict.return_value = {
            "game": {
                "actions": [
                    {"actionNumber": 1, "clock": "PT12M00.00S", "period": 1, "teamTricode": "DEN",
                     "scoreHome": "0", "scoreAway": "0", "actionType": "Jump Ball", "description": "Tip"},
                    {"actionNumber": 2, "clock": "PT11M42.00S", "period": 1, "teamTricode": "LAL",
                     "scoreHome": "2", "scoreAway": "0", "actionType": "Made Shot", "description": "Dunk"},
                ],
                "homeTeam": {"teamTricode": "DEN"},
                "awayTeam": {"teamTricode": "LAL"},
            }
        }
        events = fetch_live_events("g2")
    assert len(events) == 2
    assert events[0]["event_index"] == 0
    assert events[1]["possession_team"] == "away"
    assert events[1]["home_score"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_live.py -v`
Expected: FAIL — `fetch_live_game_id`/`fetch_live_events` not defined (current `app/live.py` only has `run_live` stub).

- [ ] **Step 3: Implement `backend/app/live.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_live.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && .venv/bin/pytest -v`
Expected: all PASS (Tasks 2-10 combined).

- [ ] **Step 6: Commit**

```bash
git add backend/app/live.py backend/tests/test_live.py
git commit -m "feat: live websocket polling nba_api scoreboard/play-by-play"
```

---

### Task 11: Frontend scaffold, types, and WebSocket hook

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/hooks/useGameSocket.ts`
- Create: `frontend/src/test/setup.ts`
- Test: `frontend/src/test/useGameSocket.test.ts`

**Interfaces:**
- Produces:
  - `types.ts`: `GameSummary { game_id: string; home_team: string; away_team: string }`, `GamesResponse { replay_games: GameSummary[]; live_available: boolean }`, `WinProbMessage { event_index: number; period: number; clock: string; home_score: number; away_score: number; event_type: string; description: string; win_prob: number }`.
  - `useGameSocket(wsUrl: string | null): { messages: WinProbMessage[]; connected: boolean; error: string | null }` — opens a WebSocket when `wsUrl` is non-null, appends parsed `WinProbMessage`s to `messages`, closes and resets `messages` when `wsUrl` changes or the component unmounts.

- [ ] **Step 1: Scaffold the Vite project**

Run:
```bash
cd frontend 2>/dev/null || mkdir -p frontend
cd /Users/ignatiusmartin/Documents/Personal/Projects/NBAWinProb
npm create vite@latest frontend -- --template react-ts
```
Expected: `frontend/` populated with a standard Vite React-TS template (this will create `package.json`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, etc. — subsequent steps overwrite/extend specific files).

- [ ] **Step 2: Install dependencies including Recharts and Vitest**

Run:
```bash
cd frontend
npm install
npm install recharts
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom @vitest/ui
```

- [ ] **Step 3: Configure Vitest in `frontend/vite.config.ts`**

```typescript
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/games': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
})
```

- [ ] **Step 4: Write `frontend/src/test/setup.ts`**

```typescript
import '@testing-library/jest-dom'
```

- [ ] **Step 5: Add test script to `frontend/package.json`**

Add to the `"scripts"` section: `"test": "vitest run"`.

- [ ] **Step 6: Write `frontend/src/types.ts`**

```typescript
export interface GameSummary {
  game_id: string
  home_team: string
  away_team: string
}

export interface GamesResponse {
  replay_games: GameSummary[]
  live_available: boolean
}

export interface WinProbMessage {
  event_index: number
  period: number
  clock: string
  home_score: number
  away_score: number
  event_type: string
  description: string
  win_prob: number
}
```

- [ ] **Step 7: Write the failing test for `useGameSocket`**

```typescript
// frontend/src/test/useGameSocket.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useGameSocket } from '../hooks/useGameSocket'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((ev: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  readyState = 0
  url: string
  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }
  close() {
    this.readyState = 3
    this.onclose?.()
  }
}

beforeEach(() => {
  MockWebSocket.instances = []
  // @ts-expect-error test override
  global.WebSocket = MockWebSocket
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useGameSocket', () => {
  it('starts disconnected with no messages when wsUrl is null', () => {
    const { result } = renderHook(() => useGameSocket(null))
    expect(result.current.connected).toBe(false)
    expect(result.current.messages).toEqual([])
  })

  it('appends parsed messages as they arrive', async () => {
    const { result } = renderHook(() => useGameSocket('ws://localhost:8000/replay/g1'))
    const socket = MockWebSocket.instances[0]
    socket.onopen?.()
    await waitFor(() => expect(result.current.connected).toBe(true))

    const message = {
      event_index: 0, period: 1, clock: 'PT12M00.00S', home_score: 0, away_score: 0,
      event_type: 'Jump Ball', description: 'Tip', win_prob: 0.5,
    }
    socket.onmessage?.({ data: JSON.stringify(message) })

    await waitFor(() => expect(result.current.messages).toHaveLength(1))
    expect(result.current.messages[0]).toEqual(message)
  })

  it('resets messages and connected state when the socket closes', async () => {
    const { result } = renderHook(() => useGameSocket('ws://localhost:8000/replay/g1'))
    const socket = MockWebSocket.instances[0]
    socket.onopen?.()
    await waitFor(() => expect(result.current.connected).toBe(true))
    socket.onclose?.()
    await waitFor(() => expect(result.current.connected).toBe(false))
  })
})
```

- [ ] **Step 8: Run test to verify it fails**

Run: `cd frontend && npm run test -- useGameSocket`
Expected: FAIL — `Cannot find module '../hooks/useGameSocket'`.

- [ ] **Step 9: Implement `frontend/src/hooks/useGameSocket.ts`**

```typescript
import { useEffect, useRef, useState } from 'react'
import type { WinProbMessage } from '../types'

export interface UseGameSocketResult {
  messages: WinProbMessage[]
  connected: boolean
  error: string | null
}

export function useGameSocket(wsUrl: string | null): UseGameSocketResult {
  const [messages, setMessages] = useState<WinProbMessage[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    setMessages([])
    setConnected(false)
    setError(null)

    if (!wsUrl) {
      return
    }

    const socket = new WebSocket(wsUrl)
    socketRef.current = socket

    socket.onopen = () => setConnected(true)
    socket.onmessage = (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data) as WinProbMessage
        setMessages((prev) => [...prev, parsed])
      } catch {
        setError('Failed to parse server message')
      }
    }
    socket.onerror = () => setError('WebSocket error')
    socket.onclose = () => setConnected(false)

    return () => {
      socket.close()
    }
  }, [wsUrl])

  return { messages, connected, error }
}
```

- [ ] **Step 10: Run test to verify it passes**

Run: `cd frontend && npm run test -- useGameSocket`
Expected: all PASS.

- [ ] **Step 11: Commit**

```bash
cd /Users/ignatiusmartin/Documents/Personal/Projects/NBAWinProb
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/tsconfig*.json frontend/index.html frontend/src/main.tsx frontend/src/types.ts frontend/src/hooks/useGameSocket.ts frontend/src/test/setup.ts frontend/src/test/useGameSocket.test.ts
git commit -m "feat: Vite React TS scaffold, shared types, useGameSocket hook"
```

---

### Task 12: Frontend components (chart, header, play log, game picker) and App composition

**Files:**
- Create: `frontend/src/components/WinProbChart.tsx`
- Create: `frontend/src/components/ScoreHeader.tsx`
- Create: `frontend/src/components/PlayLog.tsx`
- Create: `frontend/src/components/GamePicker.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css` (basic layout styling)
- Test: `frontend/src/test/WinProbChart.test.tsx`
- Test: `frontend/src/test/GamePicker.test.tsx`

**Interfaces:**
- Consumes: `types.WinProbMessage`, `types.GameSummary`, `types.GamesResponse`, `hooks/useGameSocket`.
- Produces:
  - `WinProbChart({ messages: WinProbMessage[] }): JSX.Element` — Recharts `LineChart` plotting `win_prob` (0-1, y-axis) vs `event_index` (x-axis).
  - `ScoreHeader({ latest: WinProbMessage | null; homeTeam: string; awayTeam: string }): JSX.Element`.
  - `PlayLog({ messages: WinProbMessage[] }): JSX.Element` — scrolling list, newest entry first.
  - `GamePicker({ onSelect: (wsPath: string) => void }): JSX.Element` — fetches `GET /games` on mount, renders a button per replay game (`onSelect('/replay/' + game_id)`) and a "Watch Live" button (`onSelect('/live')`) disabled when `live_available` is false.
  - `App`: composes `GamePicker` (selection sets `wsPath` state) → builds full ws URL (`ws://<host>:8000` + path in dev, same-origin in prod) → `useGameSocket` → renders `ScoreHeader`, `WinProbChart`, `PlayLog` once messages exist.

- [ ] **Step 1: Write the failing tests**

```typescript
// frontend/src/test/WinProbChart.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { WinProbChart } from '../components/WinProbChart'
import type { WinProbMessage } from '../types'

const sampleMessages: WinProbMessage[] = [
  { event_index: 0, period: 1, clock: 'PT12M00.00S', home_score: 0, away_score: 0, event_type: 'Jump Ball', description: 'Tip', win_prob: 0.5 },
  { event_index: 1, period: 1, clock: 'PT11M42.00S', home_score: 2, away_score: 0, event_type: 'Made Shot', description: 'Dunk', win_prob: 0.55 },
]

describe('WinProbChart', () => {
  it('renders an empty state with no messages', () => {
    render(<WinProbChart messages={[]} />)
    expect(screen.getByTestId('win-prob-chart-empty')).toBeInTheDocument()
  })

  it('renders a chart container when messages are present', () => {
    render(<WinProbChart messages={sampleMessages} />)
    expect(screen.getByTestId('win-prob-chart')).toBeInTheDocument()
  })
})
```

```typescript
// frontend/src/test/GamePicker.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { GamePicker } from '../components/GamePicker'

beforeEach(() => {
  global.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({
      replay_games: [{ game_id: 'g1', home_team: 'DEN', away_team: 'LAL' }],
      live_available: false,
    }),
  })) as unknown as typeof fetch
})

describe('GamePicker', () => {
  it('fetches and renders replay games, disables live when unavailable', async () => {
    const onSelect = vi.fn()
    render(<GamePicker onSelect={onSelect} />)

    await waitFor(() => expect(screen.getByText(/DEN vs LAL/i)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /watch live/i })).toBeDisabled()

    fireEvent.click(screen.getByText(/DEN vs LAL/i))
    expect(onSelect).toHaveBeenCalledWith('/replay/g1')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- WinProbChart GamePicker`
Expected: FAIL — components not found.

- [ ] **Step 3: Implement `frontend/src/components/WinProbChart.tsx`**

```tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import type { WinProbMessage } from '../types'

interface WinProbChartProps {
  messages: WinProbMessage[]
}

export function WinProbChart({ messages }: WinProbChartProps) {
  if (messages.length === 0) {
    return <div data-testid="win-prob-chart-empty">Waiting for game data…</div>
  }

  const data = messages.map((m) => ({ event_index: m.event_index, win_prob: m.win_prob }))

  return (
    <div data-testid="win-prob-chart" style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="event_index" label={{ value: 'Event', position: 'insideBottom', offset: -4 }} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <ReferenceLine y={0.5} stroke="#999" strokeDasharray="4 4" />
          <Tooltip formatter={(value: number) => `${Math.round(value * 100)}%`} />
          <Line type="monotone" dataKey="win_prob" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 4: Implement `frontend/src/components/ScoreHeader.tsx`**

```tsx
import type { WinProbMessage } from '../types'

interface ScoreHeaderProps {
  latest: WinProbMessage | null
  homeTeam: string
  awayTeam: string
}

export function ScoreHeader({ latest, homeTeam, awayTeam }: ScoreHeaderProps) {
  if (!latest) {
    return <div className="score-header">Waiting for tip-off…</div>
  }
  const homeWinPct = Math.round(latest.win_prob * 100)
  return (
    <div className="score-header">
      <div className="score-header-teams">
        <span>{awayTeam} {latest.away_score}</span>
        <span className="score-header-divider">—</span>
        <span>{latest.home_score} {homeTeam}</span>
      </div>
      <div className="score-header-clock">Q{latest.period} · {latest.clock.replace('PT', '').replace('S', '').replace('M', ':')}</div>
      <div className="score-header-winprob">{homeTeam} win probability: {homeWinPct}%</div>
    </div>
  )
}
```

- [ ] **Step 5: Implement `frontend/src/components/PlayLog.tsx`**

```tsx
import type { WinProbMessage } from '../types'

interface PlayLogProps {
  messages: WinProbMessage[]
}

export function PlayLog({ messages }: PlayLogProps) {
  const newestFirst = [...messages].reverse()
  return (
    <ul className="play-log" data-testid="play-log">
      {newestFirst.map((m) => (
        <li key={m.event_index}>
          <span className="play-log-clock">Q{m.period} {m.clock.replace('PT', '').replace('S', '').replace('M', ':')}</span>
          <span className="play-log-desc">{m.description}</span>
        </li>
      ))}
    </ul>
  )
}
```

- [ ] **Step 6: Implement `frontend/src/components/GamePicker.tsx`**

```tsx
import { useEffect, useState } from 'react'
import type { GamesResponse, GameSummary } from '../types'

interface GamePickerProps {
  onSelect: (wsPath: string) => void
}

export function GamePicker({ onSelect }: GamePickerProps) {
  const [games, setGames] = useState<GameSummary[]>([])
  const [liveAvailable, setLiveAvailable] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/games')
      .then((res) => res.json())
      .then((data: GamesResponse) => {
        setGames(data.replay_games)
        setLiveAvailable(data.live_available)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return <div>Loading games…</div>
  }

  return (
    <div className="game-picker">
      <button disabled={!liveAvailable} onClick={() => onSelect('/live')}>
        Watch Live
      </button>
      <ul>
        {games.map((g) => (
          <li key={g.game_id}>
            <button onClick={() => onSelect(`/replay/${g.game_id}`)}>
              {g.away_team} vs {g.home_team}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd frontend && npm run test -- WinProbChart GamePicker`
Expected: all PASS.

- [ ] **Step 8: Wire up `frontend/src/App.tsx`**

```tsx
import { useMemo, useState } from 'react'
import { GamePicker } from './components/GamePicker'
import { ScoreHeader } from './components/ScoreHeader'
import { WinProbChart } from './components/WinProbChart'
import { PlayLog } from './components/PlayLog'
import { useGameSocket } from './hooks/useGameSocket'
import './App.css'

function buildWsUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = import.meta.env.DEV ? 'localhost:8000' : window.location.host
  return `${protocol}://${host}${path}`
}

function App() {
  const [wsPath, setWsPath] = useState<string | null>(null)
  const wsUrl = useMemo(() => (wsPath ? buildWsUrl(wsPath) : null), [wsPath])
  const { messages, connected, error } = useGameSocket(wsUrl)
  const latest = messages.length > 0 ? messages[messages.length - 1] : null

  return (
    <div className="app">
      <h1>NBA Live Win Probability</h1>
      <GamePicker onSelect={setWsPath} />
      {wsPath && (
        <div className="game-view">
          <div className="connection-status">{connected ? 'Connected' : 'Connecting…'}{error && ` — ${error}`}</div>
          <ScoreHeader latest={latest} homeTeam="Home" awayTeam="Away" />
          <WinProbChart messages={messages} />
          <PlayLog messages={messages} />
        </div>
      )}
    </div>
  )
}

export default App
```

- [ ] **Step 9: Add minimal layout styling to `frontend/src/App.css`**

```css
.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
  font-family: system-ui, sans-serif;
}
.game-picker button { margin: 4px; padding: 8px 12px; cursor: pointer; }
.game-picker button:disabled { cursor: not-allowed; opacity: 0.5; }
.score-header { display: flex; gap: 16px; align-items: baseline; padding: 12px 0; font-size: 1.1rem; }
.score-header-teams { font-weight: 700; font-size: 1.4rem; }
.play-log { list-style: none; padding: 0; max-height: 240px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; }
.play-log li { display: flex; gap: 12px; padding: 6px 12px; border-bottom: 1px solid #eee; }
.play-log-clock { color: #666; min-width: 90px; }
.connection-status { color: #666; font-size: 0.9rem; margin-bottom: 8px; }
```

- [ ] **Step 10: Manual smoke check the build compiles**

Run: `cd frontend && npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 11: Commit**

```bash
cd /Users/ignatiusmartin/Documents/Personal/Projects/NBAWinProb
git add frontend/src/components frontend/src/App.tsx frontend/src/App.css frontend/src/test/WinProbChart.test.tsx frontend/src/test/GamePicker.test.tsx
git commit -m "feat: chart, score header, play log, game picker, App composition"
```

---

### Task 13: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: a GitHub Actions workflow with two jobs, `backend` and `frontend`, triggered on push and pull_request to `main`.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest -v

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - run: npm ci
      - run: npm run test
      - run: npm run build
```

- [ ] **Step 2: Verify workflow YAML is well-formed**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>&1 || cat .github/workflows/ci.yml`
Expected: no error (install `pyyaml` ad hoc if missing: `pip3 install --quiet pyyaml`).

- [ ] **Step 3: Commit and push, then confirm CI runs**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add backend pytest and frontend vitest/build workflow"
git push origin main
```
Then check: `gh run list --limit 3` and `gh run watch` (or `gh run view --log-failed` if it fails) to confirm both jobs go green. Fix any CI-only failures (e.g., missing lockfile — ensure `frontend/package-lock.json` is committed; missing `model/` checkpoint — the backend test suite must not depend on a trained production checkpoint, only on the committed test fixtures, so this should already pass at this point in the plan since Task 14's checkpoint doesn't exist yet).

---

### Task 14: Run the real data pipeline — fetch, train, evaluate

**Files:**
- No new files; produces `backend/data/nba.db` (gitignored), `backend/data/raw/*.json` (gitignored), `backend/model/win_prob_lstm.pt`, `backend/model/baseline_logreg.joblib`, `backend/model/split.json`, `backend/model/event_types.json`, `backend/reports/evaluation_report.md`, `backend/reports/evaluation_report.json` (all committed — small).

- [ ] **Step 1: Fetch a real data sample**

Run (from `backend/`, with venv active):
```bash
cd backend
.venv/bin/python scripts/fetch_data.py --season 2023-24 --max-games 200 --out-db data/nba.db --raw-dir data/raw
```
Expected: prints progress per game, ends with `data/nba.db` containing ~150-200 games (some may fail/skip due to transient API errors — that's fine, the script continues). This will take a while (network I/O + polite rate-limit sleep) — if it's taking excessively long, it's safe to interrupt (Ctrl-C) once at least ~60 games are cached, since `game_exists` makes the script resumable: rerun the same command to continue from where it left off.

- [ ] **Step 2: Sanity check the cached data**

Run:
```bash
.venv/bin/python -c "
from app.db import get_connection, list_games, get_game_events
conn = get_connection('data/nba.db')
games = list_games(conn)
print('games:', len(games))
print('sample events for first game:', len(get_game_events(conn, games[0]['game_id'])))
"
```
Expected: `games:` count in the dozens-to-hundreds, with events present. If `games: 0`, stop and debug `fetch_data.py` (network reachability, or the `parse_game_list`/`parse_playbyplay_rows` logic against the real response shape) before continuing — do not proceed to training on empty data.

- [ ] **Step 3: Write `event_types.json` vocabulary file (used by README/frontend docs, not by serving code, which hardcodes the same constant)**

Run:
```bash
.venv/bin/python -c "
import json
from app.features import EVENT_TYPES
with open('model/event_types.json', 'w') as f:
    json.dump(EVENT_TYPES, f, indent=2)
"
```

- [ ] **Step 4: Train**

Run:
```bash
.venv/bin/python scripts/train.py --db data/nba.db --epochs 30 --hidden-size 64 \
  --out-model model/win_prob_lstm.pt --out-baseline model/baseline_logreg.joblib --out-split model/split.json
```
Expected: per-epoch loss printed, generally decreasing; ends with "Saved LSTM to model/win_prob_lstm.pt, baseline to model/baseline_logreg.joblib, split to model/split.json". If loss is NaN or not decreasing at all after a few epochs, reduce `--lr` (try `5e-4`) and rerun — do not proceed with a broken checkpoint.

- [ ] **Step 5: Evaluate**

Run:
```bash
.venv/bin/python scripts/evaluate.py --db data/nba.db --model model/win_prob_lstm.pt \
  --baseline model/baseline_logreg.joblib --split model/split.json \
  --out-md reports/evaluation_report.md --out-json reports/evaluation_report.json
cat reports/evaluation_report.md
```
Expected: a markdown table comparing LSTM vs. baseline Brier score, log-loss, and accuracy-by-time-bucket. Read it — this is the real result that goes in the README (Task 16). It's fine if the LSTM doesn't dramatically beat the baseline on a ~150-game sample; report the real numbers either way.

- [ ] **Step 6: Re-run the full backend test suite against the real checkpoint to make sure nothing broke**

Run: `.venv/bin/pytest -v`
Expected: all PASS (tests use fixtures, not the production checkpoint, so this should be unaffected — this step is a final regression guard before committing model artifacts).

- [ ] **Step 7: Commit the trained artifacts and report**

```bash
cd /Users/ignatiusmartin/Documents/Personal/Projects/NBAWinProb
git add backend/model/win_prob_lstm.pt backend/model/baseline_logreg.joblib backend/model/split.json backend/model/event_types.json backend/reports/evaluation_report.md backend/reports/evaluation_report.json
git commit -m "feat: train LSTM + baseline on 2023-24 season sample, add evaluation report"
git push origin main
```

---

### Task 15: Run the app end-to-end and capture screenshots

**Files:**
- Create: `docs/screenshots/replay-chart.png`
- Create: `docs/screenshots/play-log.png`
- Create: `docs/screenshots/game-picker.png`

**Interfaces:** none (manual verification + asset capture task).

- [ ] **Step 1: Start the backend**

Run (background):
```bash
cd backend
.venv/bin/uvicorn app.main:app --reload --port 8000
```
Expected: "Application startup complete" with no errors (model loads from `model/win_prob_lstm.pt`).

- [ ] **Step 2: Confirm `GET /games` works against the real data**

Run: `curl -s http://localhost:8000/games | python3 -m json.tool | head -30`
Expected: JSON with a non-empty `replay_games` array.

- [ ] **Step 3: Start the frontend**

Run (background):
```bash
cd frontend
npm run dev
```
Expected: Vite dev server on `http://localhost:5173`.

- [ ] **Step 4: Drive the app in a browser and capture screenshots**

Use the browser automation tools (`mcp__claude-in-chrome__*`, loading them via `ToolSearch` first): navigate to `http://localhost:5173`, screenshot the game picker, click a replay game, wait for a few seconds of streaming (the chart line should be visibly building and the play log populating), screenshot the full game view (chart + score header + play log). Save into `docs/screenshots/` with the filenames above. If browser tools are unavailable or fail after 2-3 attempts, fall back to macOS `screencapture` on the visible browser window and tell the user screenshots need a manual pass.

- [ ] **Step 5: Stop the dev servers**

Kill both background processes once screenshots are captured.

- [ ] **Step 6: Commit screenshots**

```bash
cd /Users/ignatiusmartin/Documents/Personal/Projects/NBAWinProb
git add docs/screenshots
git commit -m "docs: add app screenshots"
```

---

### Task 16: Final README and repo polish

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Rewrite `README.md`**

Include, in order: project title + one-line pitch; CI badge (`![CI](https://github.com/<owner>/NBAWinProb/actions/workflows/ci.yml/badge.svg)`); an embedded screenshot from `docs/screenshots/replay-chart.png`; a short "what this is" paragraph pulled from the design spec's Summary; an architecture section (the ASCII diagram from `docs/superpowers/specs/2026-08-06-nba-win-probability-design.md`); a "Results" section with the actual table copied from `backend/reports/evaluation_report.md` (Task 14's real numbers, not placeholders); a Quickstart section with the exact commands to fetch data, train, run backend, run frontend (copy verbatim from Tasks 1, 11, 14, 15's Run blocks); a Testing section (`pytest` / `npm run test` commands); a Project Structure tree; a Tech Stack list; and a short License section (MIT, add `LICENSE` file with MIT text and the current year/owner).

- [ ] **Step 2: Add `LICENSE`**

Write standard MIT license text with copyright holder = the GitHub account used in Task 1 and year 2026.

- [ ] **Step 3: Verify the README renders sensibly**

Run: `gh repo view --web=false` and re-read the raw `README.md` file top to bottom, confirming no broken relative image links (`docs/screenshots/...` paths must match exactly what Task 15 committed) and no leftover placeholder text like "TBD" or "coming soon."

- [ ] **Step 4: Final commit and push**

```bash
git add README.md LICENSE
git commit -m "docs: final README with screenshots, architecture, and real evaluation results"
git push origin main
```

- [ ] **Step 5: Confirm CI is green on the final `main`**

Run: `gh run list --limit 3`
Expected: latest run for `main` shows both `backend` and `frontend` jobs succeeded. If not, fix and repeat until green — this is the definition of done for the plan.

---

## Self-Review Notes

- **Spec coverage:** data pipeline (Tasks 3-4), feature engineering (Task 2), model + baseline (Tasks 5-6), evaluation report (Task 7), serving/replay/live (Tasks 8-10), frontend (Tasks 11-12), deployment/CI (Task 13; actual paid-tier deploy to Render/Vercel intentionally omitted — no CLI auth available in this environment and it's not required for "zero ongoing cost" local-first completion, documented as a manual optional step in README), testing (unit/model-smoke/integration/frontend all covered across Tasks 2, 5, 8-10, 11-12).
- **Real API verified**, not guessed: `playbyplayv3` and `leaguegamefinder` shapes confirmed against a live call before writing this plan; `nba_api.live.nba.endpoints` scoreboard/playbyplay module paths used in Task 10 match the installed `nba_api==1.5.2` package layout.
- **Interface consistency checked:** `EventRow` fields used identically in `db.py`, `fetch_data.py`, `train.py`, `inference.py`, `live.py`; `WinProbLSTM.forward` signature identical across `model.py`, `train.py`, `inference.py`, and the incremental-vs-batch equivalence test in Task 5.
