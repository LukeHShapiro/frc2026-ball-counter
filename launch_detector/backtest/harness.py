"""Backtest harness — replay stored SignalEvents against known past launches.

Measures false-positive and false-negative rates per scoring threshold.
Outputs a confusion matrix and lets you tune weights/thresholds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..engine import correlate, score_group, classify_tier
from ..models import SignalEvent, AlertTier
from ..site_config import SiteConfig, ScoringPolicy
from ..storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    threshold_name: str
    threshold_value: float
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def confusion_matrix(self) -> str:
        lines = [
            f"--- {self.threshold_name} (threshold={self.threshold_value:.2f}) ---",
            f"              Predicted+   Predicted-",
            f"  Actual+     {self.true_positives:>8d}     {self.false_negatives:>8d}",
            f"  Actual-     {self.false_positives:>8d}     {self.true_negatives:>8d}",
            f"",
            f"  Precision: {self.precision:.3f}",
            f"  Recall:    {self.recall:.3f}",
            f"  F1:        {self.f1:.3f}",
        ]
        return "\n".join(lines)


def load_ground_truth_file(path: Path) -> list[dict]:
    """Load ground truth from a JSON file.

    Expected format:
    [
        {
            "site_id": "WFF",
            "date": "2024-06-15",
            "vehicle": "Electron",
            "mission": "PREFIRE-2",
            "outcome": "success"
        },
        ...
    ]
    """
    with open(path) as f:
        return json.load(f)


def load_ground_truth_db(storage: Storage, site_id: str) -> list[dict]:
    return storage.get_ground_truth(site_id)


def run_backtest(
    signals: list[SignalEvent],
    ground_truth: list[dict],
    site: SiteConfig,
    match_window_hours: int = 48,
    thresholds: Optional[list[float]] = None,
) -> list[BacktestResult]:
    """Replay signals against ground truth at various thresholds.

    For each threshold, we:
    1. Correlate signals into groups
    2. Score each group
    3. Check if groups above threshold correspond to real launches (TP)
       or false alarms (FP)
    4. Check if real launches had a group above threshold (FN if not)
    """
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    launch_dates = []
    for gt in ground_truth:
        if gt.get("site_id", site.site_id) == site.site_id:
            launch_dates.append(datetime.strptime(gt["date"], "%Y-%m-%d"))

    groups = correlate(signals, site)
    scored_groups = []
    for group in groups:
        score = score_group(group, site.scoring)
        group.score = score
        scored_groups.append(group)

    results = []
    match_window = timedelta(hours=match_window_hours)

    for threshold in thresholds:
        result = BacktestResult(
            threshold_name=f"score>={threshold:.2f}",
            threshold_value=threshold,
        )

        above = [g for g in scored_groups if g.score >= threshold]
        below = [g for g in scored_groups if g.score < threshold]

        matched_launches: set[int] = set()
        matched_groups: set[int] = set()

        for gi, group in enumerate(above):
            is_tp = False
            for li, launch_dt in enumerate(launch_dates):
                if (
                    group.window_start - match_window <= launch_dt
                    and launch_dt <= group.window_end + match_window
                ):
                    is_tp = True
                    matched_launches.add(li)
                    matched_groups.add(gi)
                    break

            if is_tp:
                result.true_positives += 1
                result.details.append({
                    "type": "TP",
                    "group_window": f"{group.window_start.isoformat()}..{group.window_end.isoformat()}",
                    "score": group.score,
                    "n_signals": len(group.signals),
                })
            else:
                result.false_positives += 1
                result.details.append({
                    "type": "FP",
                    "group_window": f"{group.window_start.isoformat()}..{group.window_end.isoformat()}",
                    "score": group.score,
                    "n_signals": len(group.signals),
                })

        for li, launch_dt in enumerate(launch_dates):
            if li not in matched_launches:
                result.false_negatives += 1
                result.details.append({
                    "type": "FN",
                    "launch_date": launch_dt.isoformat(),
                })

        n_non_launch_windows = max(len(below), 1)
        tn_count = 0
        for group in below:
            is_real = False
            for launch_dt in launch_dates:
                if (
                    group.window_start - match_window <= launch_dt
                    and launch_dt <= group.window_end + match_window
                ):
                    is_real = True
                    break
            if not is_real:
                tn_count += 1
        result.true_negatives = tn_count

        results.append(result)

    return results


def print_backtest_report(results: list[BacktestResult]):
    print("=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    for r in results:
        print()
        print(r.confusion_matrix())
    print()
    print("=" * 60)

    best = max(results, key=lambda r: r.f1)
    print(f"\nBest F1: {best.f1:.3f} at threshold {best.threshold_value:.2f}")
    print(f"  Precision={best.precision:.3f}  Recall={best.recall:.3f}")
