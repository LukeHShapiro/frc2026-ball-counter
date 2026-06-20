"""Tests for the backtest harness."""

import pytest
from datetime import datetime, timedelta

from launch_detector.models import SignalEvent, SignalType
from launch_detector.site_config import WFF
from launch_detector.backtest.harness import run_backtest, BacktestResult


def _make_signal(
    start: datetime,
    duration_hours: int = 24,
    signal_type: SignalType = SignalType.RF_LICENSE,
    confidence: float = 0.8,
) -> SignalEvent:
    return SignalEvent(
        site_id="WFF",
        source="test",
        signal_type=signal_type,
        window_start=start,
        window_end=start + timedelta(hours=duration_hours),
        confidence=confidence,
        raw_ref=f"test-{start.isoformat()}-{signal_type.value}",
    )


class TestBacktestResult:
    def test_precision(self):
        r = BacktestResult("test", 0.5, true_positives=3, false_positives=1)
        assert abs(r.precision - 0.75) < 0.01

    def test_recall(self):
        r = BacktestResult("test", 0.5, true_positives=3, false_negatives=2)
        assert abs(r.recall - 0.60) < 0.01

    def test_f1(self):
        r = BacktestResult("test", 0.5, true_positives=4, false_positives=1, false_negatives=1)
        p = 4 / 5
        rec = 4 / 5
        expected_f1 = 2 * p * rec / (p + rec)
        assert abs(r.f1 - expected_f1) < 0.01

    def test_zero_division(self):
        r = BacktestResult("test", 0.5)
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0


class TestBacktest:
    def test_true_positive(self):
        launch_date = datetime(2025, 6, 15)
        signals = [
            _make_signal(launch_date - timedelta(hours=48), confidence=0.9),
            _make_signal(
                launch_date - timedelta(hours=24),
                signal_type=SignalType.AIRSPACE_ACTIVATION,
                confidence=0.6,
            ),
        ]
        ground_truth = [{"site_id": "WFF", "date": "2025-06-15"}]
        results = run_backtest(signals, ground_truth, WFF, thresholds=[0.3])
        assert results[0].true_positives >= 1

    def test_false_negative(self):
        launch_date = datetime(2025, 6, 15)
        signals = [_make_signal(launch_date + timedelta(days=100), confidence=0.3)]
        ground_truth = [{"site_id": "WFF", "date": "2025-06-15"}]
        results = run_backtest(signals, ground_truth, WFF, thresholds=[0.5])
        assert results[0].false_negatives >= 1

    def test_no_ground_truth(self):
        signals = [_make_signal(datetime(2025, 6, 15), confidence=0.9)]
        results = run_backtest(signals, [], WFF, thresholds=[0.3])
        assert results[0].true_positives == 0
        assert results[0].false_positives >= 1

    def test_multiple_thresholds(self):
        results = run_backtest(
            [_make_signal(datetime(2025, 6, 15))],
            [{"site_id": "WFF", "date": "2025-06-15"}],
            WFF,
            thresholds=[0.1, 0.5, 0.9],
        )
        assert len(results) == 3
