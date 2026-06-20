"""FAA NOTAM adapter.

Queries the FAA NOTAM API for airspace restrictions near a launch site.
Filters for restricted area activations (R-6604, W-386 for WFF).

Access method: REST API at external-api.faa.gov with OAuth2 client credentials.
See docs/datasources.md for details.

Known limitation: R-6604 activation is NOISY — fires for non-launch range ops.
This signal needs corroboration from other sources to indicate a launch.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from ..models import SignalEvent, SignalType
from ..site_config import SiteConfig
from .base import BaseAdapter

logger = logging.getLogger(__name__)

NOTAM_API_BASE = "https://external-api.faa.gov/notamapi/v1/notams"
TOKEN_URL = "https://external-api.faa.gov/oauth/client_sso/accessToken"

REQUEST_DELAY_S = 1.0


class FAANOTAMAdapter(BaseAdapter):
    """Queries FAA NOTAM API for airspace activations near a launch site."""

    name = "faa_notam"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.client_id = client_id or os.environ.get("FAA_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("FAA_CLIENT_SECRET", "")
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    def is_available(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_token(self, client: httpx.Client) -> str:
        if self._access_token and self._token_expires and datetime.utcnow() < self._token_expires:
            return self._access_token

        resp = client.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        expires_in = int(data.get("expires_in", 3600))
        self._token_expires = datetime.utcnow() + timedelta(seconds=expires_in - 60)
        return self._access_token

    def poll(self, site: SiteConfig) -> list[SignalEvent]:
        if not self.is_available():
            logger.warning("FAA NOTAM: no credentials configured, skipping")
            return []

        logger.info("FAA NOTAM: polling for NOTAMs near %s", site.site_id)
        events: list[SignalEvent] = []

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                token = self._get_token(client)
                notams = self._fetch_notams(client, token, site)
                for notam in notams:
                    signal = self._parse_notam(notam, site)
                    if signal:
                        events.append(signal)
        except Exception:
            logger.exception("FAA NOTAM: failed to poll")

        logger.info("FAA NOTAM: found %d signals for %s", len(events), site.site_id)
        return events

    def _fetch_notams(
        self, client: httpx.Client, token: str, site: SiteConfig
    ) -> list[dict]:
        all_notams: list[dict] = []

        params = {
            "locationLatitude": str(site.center_lat),
            "locationLongitude": str(site.center_lon),
            "locationRadius": str(site.notam_radius_nm),
            "pageSize": "1000",
            "pageNum": "1",
            "sortBy": "effectiveStartDate",
            "sortOrder": "DESC",
        }

        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = client.get(NOTAM_API_BASE, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", data.get("notams", []))
            if isinstance(items, list):
                all_notams.extend(items)
        except httpx.HTTPError:
            logger.exception("FAA NOTAM: location-based query failed")

        for loc_id in site.notam_location_ids:
            params_loc = {
                "domesticLocation": loc_id,
                "pageSize": "1000",
                "pageNum": "1",
            }
            try:
                resp = client.get(NOTAM_API_BASE, params=params_loc, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", data.get("notams", []))
                if isinstance(items, list):
                    all_notams.extend(items)
            except httpx.HTTPError:
                logger.warning("FAA NOTAM: location ID query for %s failed", loc_id)

        seen_ids: set[str] = set()
        deduped = []
        for n in all_notams:
            nid = n.get("notamNumber", n.get("id", ""))
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                deduped.append(n)
        return deduped

    def _parse_notam(self, notam: dict, site: SiteConfig) -> Optional[SignalEvent]:
        """Parse a NOTAM record into a SignalEvent if it matches restricted areas."""

        text = notam.get("traditionalMessage", notam.get("text", notam.get("message", "")))
        notam_id = notam.get("notamNumber", notam.get("id", "unknown"))

        all_areas = site.restricted_areas + site.warning_areas
        matched_area = None
        for area in all_areas:
            area_pattern = area.replace("-", r"[\s-]?")
            if re.search(area_pattern, text, re.IGNORECASE):
                matched_area = area
                break

        if not matched_area:
            return None

        effective = notam.get("effectiveStart", notam.get("startDate", ""))
        expiry = notam.get("effectiveEnd", notam.get("endDate", ""))

        try:
            window_start = self._parse_datetime(effective)
        except (ValueError, TypeError):
            window_start = datetime.utcnow()

        try:
            window_end = self._parse_datetime(expiry)
        except (ValueError, TypeError):
            window_end = window_start + timedelta(hours=24)

        if matched_area in site.warning_areas:
            signal_type = SignalType.WARNING_AREA_HOT
            confidence = 0.5
        else:
            signal_type = SignalType.AIRSPACE_ACTIVATION
            confidence = 0.3

        text_lower = text.lower()
        if any(kw in text_lower for kw in ["rocket", "launch", "missile", "space"]):
            confidence = min(confidence + 0.3, 0.85)
        if "unlimited" in text_lower or "sfc" in text_lower:
            confidence = min(confidence + 0.1, 0.85)

        raw_ref = f"NOTAM:{notam_id}"

        return SignalEvent(
            site_id=site.site_id,
            source=self.name,
            signal_type=signal_type,
            window_start=window_start,
            window_end=window_end,
            confidence=confidence,
            raw_ref=raw_ref,
            geo=matched_area,
            metadata={
                "notam_id": notam_id,
                "matched_area": matched_area,
                "text_snippet": text[:300],
            },
        )

    @staticmethod
    def _parse_datetime(s: str) -> datetime:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
