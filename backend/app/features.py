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
