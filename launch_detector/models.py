"""Core data models for the launch detection system."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


class SignalType(str, enum.Enum):
    AIRSPACE_ACTIVATION = "airspace_activation"
    WARNING_AREA_HOT = "warning_area_hot"
    MARITIME_HAZARD = "maritime_hazard"
    RF_LICENSE = "rf_license"
    VESSEL_CLEARING = "vessel_clearing"
    LOCAL_CLOSURE = "local_closure"


class AlertTier(str, enum.Enum):
    WATCH = "WATCH"
    LIKELY = "LIKELY"
    IMMINENT = "IMMINENT"


@dataclass
class SignalEvent:
    site_id: str
    source: str
    signal_type: SignalType
    window_start: datetime
    window_end: datetime
    confidence: float
    raw_ref: str
    geo: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal_type"] = self.signal_type.value
        d["window_start"] = self.window_start.isoformat()
        d["window_end"] = self.window_end.isoformat()
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        return d

    def overlaps(self, other: SignalEvent) -> bool:
        return self.window_start <= other.window_end and other.window_start <= self.window_end


@dataclass
class CorrelationGroup:
    site_id: str
    window_start: datetime
    window_end: datetime
    signals: list[SignalEvent] = field(default_factory=list)
    score: float = 0.0
    tier: Optional[AlertTier] = None

    @property
    def signal_types_present(self) -> set[SignalType]:
        return {s.signal_type for s in self.signals}

    @property
    def sources_present(self) -> set[str]:
        return {s.source for s in self.signals}


@dataclass
class Alert:
    tier: AlertTier
    site_id: str
    group: CorrelationGroup
    window_start: datetime
    window_end: datetime
    explanation: str
    signal_refs: list[str] = field(default_factory=list)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
