"""SQLite storage for SignalEvents and Alerts. Stores all raw data for audit and replay."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SignalEvent, SignalType, Alert, AlertTier


DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "launch_detector.db"


class Storage:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    raw_ref TEXT NOT NULL,
                    geo TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(source, raw_ref, window_start)
                );

                CREATE INDEX IF NOT EXISTS idx_signal_site
                    ON signal_events(site_id, window_start, window_end);
                CREATE INDEX IF NOT EXISTS idx_signal_source
                    ON signal_events(source);

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tier TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    signal_refs TEXT NOT NULL,
                    signal_event_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_alert_site
                    ON alerts(site_id, created_at);

                CREATE TABLE IF NOT EXISTS ground_truth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    launch_date TEXT NOT NULL,
                    vehicle TEXT,
                    mission TEXT,
                    outcome TEXT,
                    notes TEXT,
                    UNIQUE(site_id, launch_date, vehicle)
                );
            """)

    def store_signal(self, event: SignalEvent) -> int:
        with self._conn() as conn:
            try:
                cur = conn.execute(
                    """INSERT INTO signal_events
                       (site_id, source, signal_type, window_start, window_end,
                        confidence, raw_ref, geo, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.site_id,
                        event.source,
                        event.signal_type.value,
                        event.window_start.isoformat(),
                        event.window_end.isoformat(),
                        event.confidence,
                        event.raw_ref,
                        event.geo,
                        json.dumps(event.metadata) if event.metadata else None,
                    ),
                )
                return cur.lastrowid
            except sqlite3.IntegrityError:
                row = conn.execute(
                    """SELECT id FROM signal_events
                       WHERE source = ? AND raw_ref = ? AND window_start = ?""",
                    (event.source, event.raw_ref, event.window_start.isoformat()),
                ).fetchone()
                return row["id"] if row else -1

    def get_signals_for_site(
        self,
        site_id: str,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> list[SignalEvent]:
        with self._conn() as conn:
            query = "SELECT * FROM signal_events WHERE site_id = ?"
            params: list = [site_id]
            if after:
                query += " AND window_end >= ?"
                params.append(after.isoformat())
            if before:
                query += " AND window_start <= ?"
                params.append(before.isoformat())
            query += " ORDER BY window_start"
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_signal(r) for r in rows]

    def get_all_signals(self) -> list[SignalEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signal_events ORDER BY window_start"
            ).fetchall()
            return [self._row_to_signal(r) for r in rows]

    def store_alert(self, alert: Alert) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO alerts
                   (tier, site_id, window_start, window_end, explanation,
                    signal_refs, signal_event_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert.tier.value,
                    alert.site_id,
                    alert.window_start.isoformat(),
                    alert.window_end.isoformat(),
                    alert.explanation,
                    json.dumps(alert.signal_refs),
                    json.dumps([s.id for s in alert.group.signals if s.id]),
                ),
            )
            return cur.lastrowid

    def get_recent_alerts(
        self, site_id: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        with self._conn() as conn:
            if site_id:
                rows = conn.execute(
                    """SELECT * FROM alerts WHERE site_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (site_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def store_ground_truth(
        self, site_id: str, launch_date: str, vehicle: str = "",
        mission: str = "", outcome: str = "success", notes: str = "",
    ) -> int:
        with self._conn() as conn:
            try:
                cur = conn.execute(
                    """INSERT INTO ground_truth
                       (site_id, launch_date, vehicle, mission, outcome, notes)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (site_id, launch_date, vehicle, mission, outcome, notes),
                )
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return -1

    def get_ground_truth(self, site_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ground_truth WHERE site_id = ? ORDER BY launch_date",
                (site_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def _row_to_signal(self, row: sqlite3.Row) -> SignalEvent:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return SignalEvent(
            id=row["id"],
            site_id=row["site_id"],
            source=row["source"],
            signal_type=SignalType(row["signal_type"]),
            window_start=datetime.fromisoformat(row["window_start"]),
            window_end=datetime.fromisoformat(row["window_end"]),
            confidence=row["confidence"],
            raw_ref=row["raw_ref"],
            geo=row["geo"],
            metadata=meta,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
