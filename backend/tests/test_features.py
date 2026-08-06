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
