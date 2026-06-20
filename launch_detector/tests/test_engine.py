"""Tests for the correlation engine — grouping, scoring, tiering."""

import pytest
from datetime import datetime, timedelta

from launch_detector.models import SignalEvent, SignalType, AlertTier
from launch_detector.engine import (
    correlate,
    score_group,
    classify_tier,
    narrow_window,
    evaluate,
)
from launch_detector.site_config import WFF, ScoringPolicy


def _make_signal(
    signal_type: SignalType = SignalType.RF_LICENSE,
    confidence: float = 0.8,
    start_offset_hours: int = 0,
    duration_hours: int = 24,
    source: str = "test",
) -> SignalEvent:
    base = datetime(2025, 6, 15, 12, 0)
    return SignalEvent(
        site_id="WFF",
        source=source,
        signal_type=signal_type,
        window_start=base + timedelta(hours=start_offset_hours),
        window_end=base + timedelta(hours=start_offset_hours + duration_hours),
        confidence=confidence,
        raw_ref=f"test-{start_offset_hours}-{signal_type.value}",
    )


class TestCorrelation:
    def test_single_signal_one_group(self):
        groups = correlate([_make_signal()], WFF)
        assert len(groups) == 1
        assert len(groups[0].signals) == 1

    def test_overlapping_signals_merge(self):
        s1 = _make_signal(start_offset_hours=0)
        s2 = _make_signal(
            signal_type=SignalType.AIRSPACE_ACTIVATION,
            start_offset_hours=12,
        )
        groups = correlate([s1, s2], WFF)
        assert len(groups) == 1
        assert len(groups[0].signals) == 2

    def test_distant_signals_separate(self):
        s1 = _make_signal(start_offset_hours=0, duration_hours=6)
        s2 = _make_signal(start_offset_hours=500, duration_hours=6)
        groups = correlate([s1, s2], WFF)
        assert len(groups) == 2

    def test_empty_signals(self):
        assert correlate([], WFF) == []

    def test_group_window_covers_all(self):
        s1 = _make_signal(start_offset_hours=0, duration_hours=10)
        s2 = _make_signal(start_offset_hours=5, duration_hours=20)
        groups = correlate([s1, s2], WFF)
        assert groups[0].window_start <= s1.window_start
        assert groups[0].window_end >= s2.window_end


class TestScoring:
    def test_rf_license_alone(self):
        s = _make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.8)
        groups = correlate([s], WFF)
        score = score_group(groups[0], WFF.scoring)
        expected = 0.45 * 0.8
        assert abs(score - expected) < 0.01

    def test_multiple_types_boost(self):
        signals = [
            _make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.8),
            _make_signal(signal_type=SignalType.AIRSPACE_ACTIVATION, confidence=0.5, start_offset_hours=6),
        ]
        groups = correlate(signals, WFF)
        score = score_group(groups[0], WFF.scoring)
        base = 0.45 * 0.8 + 0.15 * 0.5
        boosted = base * 1.05
        assert abs(score - boosted) < 0.01

    def test_three_types_higher_boost(self):
        signals = [
            _make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.8),
            _make_signal(signal_type=SignalType.AIRSPACE_ACTIVATION, confidence=0.5, start_offset_hours=6),
            _make_signal(signal_type=SignalType.WARNING_AREA_HOT, confidence=0.6, start_offset_hours=12),
        ]
        groups = correlate(signals, WFF)
        score = score_group(groups[0], WFF.scoring)
        base = 0.45 * 0.8 + 0.15 * 0.5 + 0.20 * 0.6
        boosted = base * 1.15
        assert abs(score - min(boosted, 1.0)) < 0.01

    def test_same_type_takes_highest_confidence(self):
        signals = [
            _make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.5),
            _make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.9, start_offset_hours=1),
        ]
        groups = correlate(signals, WFF)
        score = score_group(groups[0], WFF.scoring)
        assert abs(score - 0.45 * 0.9) < 0.01


class TestTierClassification:
    def test_below_watch(self):
        assert classify_tier(0.05, WFF.scoring) is None

    def test_watch(self):
        assert classify_tier(0.15, WFF.scoring) == AlertTier.WATCH

    def test_likely(self):
        assert classify_tier(0.50, WFF.scoring) == AlertTier.LIKELY

    def test_imminent(self):
        assert classify_tier(0.80, WFF.scoring) == AlertTier.IMMINENT


class TestWindowNarrowing:
    def test_overlapping_narrows(self):
        s1 = _make_signal(start_offset_hours=0, duration_hours=48)
        s2 = _make_signal(start_offset_hours=24, duration_hours=48)
        groups = correlate([s1, s2], WFF)
        narrow_start, narrow_end = narrow_window(groups[0])
        assert narrow_start == s2.window_start
        assert narrow_end == s1.window_end

    def test_non_overlapping_returns_union(self):
        s1 = _make_signal(start_offset_hours=0, duration_hours=6)
        s2 = _make_signal(start_offset_hours=50, duration_hours=6)
        groups = correlate([s1, s2], WFF)
        if len(groups) == 1:
            narrow_start, narrow_end = narrow_window(groups[0])
            assert narrow_start <= s1.window_start
            assert narrow_end >= s2.window_end


class TestEvaluate:
    def test_high_confidence_rf_triggers_watch(self):
        signals = [_make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.8)]
        alerts = evaluate(signals, WFF)
        assert len(alerts) >= 1
        assert alerts[0].tier == AlertTier.WATCH

    def test_multi_source_triggers_likely(self):
        signals = [
            _make_signal(signal_type=SignalType.RF_LICENSE, confidence=0.9, source="fcc_els"),
            _make_signal(signal_type=SignalType.AIRSPACE_ACTIVATION, confidence=0.7, start_offset_hours=6, source="faa_notam"),
            _make_signal(signal_type=SignalType.WARNING_AREA_HOT, confidence=0.8, start_offset_hours=12, source="faa_notam"),
        ]
        alerts = evaluate(signals, WFF)
        assert len(alerts) >= 1
        assert alerts[0].tier in (AlertTier.LIKELY, AlertTier.IMMINENT)

    def test_no_signals_no_alerts(self):
        assert evaluate([], WFF) == []

    def test_alert_has_explanation(self):
        signals = [_make_signal()]
        alerts = evaluate(signals, WFF)
        if alerts:
            assert "WFF" in alerts[0].explanation
            assert alerts[0].signal_refs
