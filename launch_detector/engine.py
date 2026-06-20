"""Correlation engine — groups SignalEvents by overlapping time window,
scores against site-specific policy, emits tiered alerts.

The engine is site-agnostic: it uses SiteConfig.scoring to determine
weights and thresholds. Adding a new site never requires engine changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from .models import SignalEvent, SignalType, AlertTier, CorrelationGroup, Alert
from .site_config import SiteConfig, ScoringPolicy

logger = logging.getLogger(__name__)


def correlate(
    signals: list[SignalEvent],
    site: SiteConfig,
) -> list[CorrelationGroup]:
    """Group overlapping signals into correlation groups.

    Algorithm: sort signals by window_start, greedily merge signals
    whose windows overlap within the site's correlation_window_hours.
    """
    if not signals:
        return []

    sorted_signals = sorted(signals, key=lambda s: s.window_start)
    max_gap = timedelta(hours=site.scoring.correlation_window_hours)

    groups: list[CorrelationGroup] = []
    current = CorrelationGroup(
        site_id=site.site_id,
        window_start=sorted_signals[0].window_start,
        window_end=sorted_signals[0].window_end,
        signals=[sorted_signals[0]],
    )

    for sig in sorted_signals[1:]:
        extended_end = current.window_end + max_gap
        if sig.window_start <= extended_end:
            current.signals.append(sig)
            if sig.window_end > current.window_end:
                current.window_end = sig.window_end
            if sig.window_start < current.window_start:
                current.window_start = sig.window_start
        else:
            groups.append(current)
            current = CorrelationGroup(
                site_id=site.site_id,
                window_start=sig.window_start,
                window_end=sig.window_end,
                signals=[sig],
            )

    groups.append(current)
    return groups


def score_group(group: CorrelationGroup, policy: ScoringPolicy) -> float:
    """Score a correlation group against site-specific weights.

    For each signal type present, take the highest-confidence signal
    of that type and multiply by the type's weight. Sum all.
    """
    best_by_type: dict[str, float] = {}
    for sig in group.signals:
        st = sig.signal_type.value
        if st not in best_by_type or sig.confidence > best_by_type[st]:
            best_by_type[st] = sig.confidence

    total = 0.0
    for signal_type, confidence in best_by_type.items():
        weight = policy.weights.get(signal_type, 0.0)
        total += weight * confidence

    n_types = len(best_by_type)
    if n_types >= 3:
        total *= 1.15
    elif n_types >= 2:
        total *= 1.05

    return min(total, 1.0)


def classify_tier(score: float, policy: ScoringPolicy) -> Optional[AlertTier]:
    if score >= policy.imminent_threshold:
        return AlertTier.IMMINENT
    elif score >= policy.likely_threshold:
        return AlertTier.LIKELY
    elif score >= policy.watch_threshold:
        return AlertTier.WATCH
    return None


def narrow_window(group: CorrelationGroup) -> tuple[datetime, datetime]:
    """Narrow the predicted launch window based on signal overlap.

    Start wide (earliest start, latest end), then shrink to the
    intersection of all signal windows. If no intersection exists,
    fall back to the union.
    """
    if not group.signals:
        return group.window_start, group.window_end

    latest_start = max(s.window_start for s in group.signals)
    earliest_end = min(s.window_end for s in group.signals)

    if latest_start < earliest_end:
        return latest_start, earliest_end

    return group.window_start, group.window_end


def build_explanation(group: CorrelationGroup, score: float, tier: AlertTier) -> str:
    lines = [f"[{tier.value}] Launch activity detected at {group.site_id}"]
    lines.append(f"Score: {score:.2f}")
    lines.append(f"Window: {group.window_start.isoformat()} to {group.window_end.isoformat()}")
    lines.append(f"Signal types: {', '.join(sorted(st.value for st in group.signal_types_present))}")
    lines.append(f"Sources: {', '.join(sorted(group.sources_present))}")
    lines.append("")
    lines.append("Signals:")
    for sig in sorted(group.signals, key=lambda s: s.confidence, reverse=True):
        lines.append(
            f"  - [{sig.signal_type.value}] conf={sig.confidence:.2f} "
            f"src={sig.source} window={sig.window_start.isoformat()}..{sig.window_end.isoformat()} "
            f"ref={sig.raw_ref}"
        )
    return "\n".join(lines)


def evaluate(
    signals: list[SignalEvent],
    site: SiteConfig,
) -> list[Alert]:
    """Full pipeline: correlate, score, classify, build alerts."""
    groups = correlate(signals, site)
    alerts: list[Alert] = []

    for group in groups:
        score = score_group(group, site.scoring)
        group.score = score
        tier = classify_tier(score, site.scoring)
        if tier is None:
            continue

        group.tier = tier
        narrow_start, narrow_end = narrow_window(group)
        explanation = build_explanation(group, score, tier)

        alert = Alert(
            tier=tier,
            site_id=site.site_id,
            group=group,
            window_start=narrow_start,
            window_end=narrow_end,
            explanation=explanation,
            signal_refs=[s.raw_ref for s in group.signals],
        )
        alerts.append(alert)

    return alerts
