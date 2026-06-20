"""Alert delivery — pluggable backends.

V1: ntfy (push notifications) and webhook.
Email left as config option for v2.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from .models import Alert

logger = logging.getLogger(__name__)


class AlertBackend(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        ...


class NtfyBackend(AlertBackend):
    """Push notifications via ntfy.sh (or self-hosted ntfy)."""

    def __init__(
        self,
        topic: Optional[str] = None,
        server: str = "https://ntfy.sh",
    ):
        self.topic = topic or os.environ.get("NTFY_TOPIC", "")
        self.server = server

    def send(self, alert: Alert) -> bool:
        if not self.topic:
            logger.warning("ntfy: no topic configured, skipping")
            return False

        url = f"{self.server}/{self.topic}"
        priority_map = {"WATCH": "default", "LIKELY": "high", "IMMINENT": "urgent"}
        tag_map = {"WATCH": "eyes", "LIKELY": "rocket", "IMMINENT": "warning"}

        title = f"[{alert.tier.value}] Launch activity at {alert.site_id}"
        body_lines = [
            f"Window: {alert.window_start.isoformat()} to {alert.window_end.isoformat()}",
            f"Score: {alert.group.score:.2f}",
            f"Sources: {', '.join(sorted(alert.group.sources_present))}",
        ]
        if alert.signal_refs:
            body_lines.append(f"Refs: {', '.join(alert.signal_refs[:3])}")

        body = "\n".join(body_lines)

        headers = {
            "Title": title,
            "Priority": priority_map.get(alert.tier.value, "default"),
            "Tags": tag_map.get(alert.tier.value, ""),
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, content=body, headers=headers)
                resp.raise_for_status()
                logger.info("ntfy: sent %s alert for %s", alert.tier.value, alert.site_id)
                return True
        except httpx.HTTPError:
            logger.exception("ntfy: failed to send alert")
            return False


class WebhookBackend(AlertBackend):
    """POST alert JSON to a webhook URL."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("WEBHOOK_URL", "")

    def send(self, alert: Alert) -> bool:
        if not self.url:
            logger.warning("webhook: no URL configured, skipping")
            return False

        payload = {
            "tier": alert.tier.value,
            "site_id": alert.site_id,
            "window_start": alert.window_start.isoformat(),
            "window_end": alert.window_end.isoformat(),
            "score": alert.group.score,
            "explanation": alert.explanation,
            "signal_refs": alert.signal_refs,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    self.url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                logger.info("webhook: sent %s alert for %s", alert.tier.value, alert.site_id)
                return True
        except httpx.HTTPError:
            logger.exception("webhook: failed to send alert")
            return False


class LogBackend(AlertBackend):
    """Print alerts to log — always-on default backend."""

    def send(self, alert: Alert) -> bool:
        logger.info("\n%s", alert.explanation)
        return True


def get_backends() -> list[AlertBackend]:
    backends: list[AlertBackend] = [LogBackend()]

    if os.environ.get("NTFY_TOPIC"):
        backends.append(NtfyBackend())
    if os.environ.get("WEBHOOK_URL"):
        backends.append(WebhookBackend())

    return backends


def send_alerts(alerts: list[Alert], backends: Optional[list[AlertBackend]] = None):
    if backends is None:
        backends = get_backends()
    for alert in alerts:
        for backend in backends:
            backend.send(alert)
