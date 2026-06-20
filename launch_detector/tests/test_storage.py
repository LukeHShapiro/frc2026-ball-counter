"""Tests for SQLite storage."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from launch_detector.models import SignalEvent, SignalType
from launch_detector.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(db_path=tmp_path / "test.db")


def _make_signal(**kwargs) -> SignalEvent:
    defaults = dict(
        site_id="WFF",
        source="test",
        signal_type=SignalType.RF_LICENSE,
        window_start=datetime(2025, 6, 15),
        window_end=datetime(2025, 6, 20),
        confidence=0.8,
        raw_ref="test-ref-1",
        metadata={"key": "value"},
    )
    defaults.update(kwargs)
    return SignalEvent(**defaults)


class TestSignalStorage:
    def test_store_and_retrieve(self, storage):
        sig = _make_signal()
        sid = storage.store_signal(sig)
        assert sid > 0

        signals = storage.get_signals_for_site("WFF")
        assert len(signals) == 1
        assert signals[0].source == "test"
        assert signals[0].confidence == 0.8

    def test_deduplication(self, storage):
        sig = _make_signal()
        id1 = storage.store_signal(sig)
        id2 = storage.store_signal(sig)
        assert id1 == id2

        signals = storage.get_signals_for_site("WFF")
        assert len(signals) == 1

    def test_different_refs_stored_separately(self, storage):
        storage.store_signal(_make_signal(raw_ref="ref-a"))
        storage.store_signal(_make_signal(raw_ref="ref-b"))
        signals = storage.get_signals_for_site("WFF")
        assert len(signals) == 2

    def test_filter_by_date(self, storage):
        storage.store_signal(_make_signal(
            window_start=datetime(2025, 1, 1),
            window_end=datetime(2025, 1, 5),
            raw_ref="old",
        ))
        storage.store_signal(_make_signal(
            window_start=datetime(2025, 6, 15),
            window_end=datetime(2025, 6, 20),
            raw_ref="new",
        ))

        signals = storage.get_signals_for_site(
            "WFF", after=datetime(2025, 6, 1)
        )
        assert len(signals) == 1
        assert signals[0].raw_ref == "new"

    def test_metadata_roundtrip(self, storage):
        sig = _make_signal(metadata={"azimuth": 126.0, "applicant": "Rocket Lab"})
        storage.store_signal(sig)
        signals = storage.get_signals_for_site("WFF")
        assert signals[0].metadata["azimuth"] == 126.0


class TestGroundTruth:
    def test_store_and_retrieve(self, storage):
        storage.store_ground_truth(
            site_id="WFF",
            launch_date="2024-06-15",
            vehicle="Electron",
            mission="PREFIRE-2",
        )
        gt = storage.get_ground_truth("WFF")
        assert len(gt) == 1
        assert gt[0]["vehicle"] == "Electron"

    def test_deduplication(self, storage):
        storage.store_ground_truth("WFF", "2024-06-15", "Electron")
        storage.store_ground_truth("WFF", "2024-06-15", "Electron")
        gt = storage.get_ground_truth("WFF")
        assert len(gt) == 1
