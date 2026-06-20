"""FCC Experimental Licensing System adapter.

Scrapes the FCC ELS web interface for Special Temporary Authority (STA) grants
that indicate upcoming launch communications.

Access method: HTTP scraping of apps.fcc.gov/oetcf/els/reports/ pages.
No public API exists. See docs/datasources.md for details.

Known limitation: US-government-operated RF is exempt from FCC licensing,
so NASA/DoD launches on government frequencies will not appear here.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ..models import SignalEvent, SignalType
from ..site_config import SiteConfig
from .base import BaseAdapter

logger = logging.getLogger(__name__)

GENERIC_SEARCH_URL = "https://apps.fcc.gov/oetcf/els/reports/GenericSearch.cfm"
STA_PRINT_URL = "https://apps.fcc.gov/oetcf/els/reports/STA_Print.cfm"
ATTACHMENT_URL = "https://apps.fcc.gov/els/GetAtt.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

REQUEST_DELAY_S = 2.0

COORD_PATTERN = re.compile(
    r"(\d{1,3})\s*[-°\s]\s*(\d{1,2})\s*[-'′\s]\s*(\d{1,2}(?:\.\d+)?)\s*[\"″]?\s*([NSns])"
    r".*?"
    r"(\d{1,3})\s*[-°\s]\s*(\d{1,2})\s*[-'′\s]\s*(\d{1,2}(?:\.\d+)?)\s*[\"″]?\s*([EWew])",
    re.DOTALL,
)

COORD_NL_PATTERN = re.compile(
    r"NL\s*(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2}(?:\.\d+)?)"
    r".*?"
    r"WL\s*(\d{1,3})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2}(?:\.\d+)?)",
    re.DOTALL,
)

AZIMUTH_PATTERN = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)\s*azimuth",
    re.IGNORECASE,
)

DATE_RANGE_PATTERN = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:to|-|through)\s*(\d{1,2}/\d{1,2}/\d{4})"
)


def _dms_to_decimal(deg: float, minutes: float, seconds: float, direction: str) -> float:
    decimal = deg + minutes / 60.0 + seconds / 3600.0
    if direction.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def parse_coordinates_standard(text: str) -> Optional[tuple[float, float]]:
    m = COORD_PATTERN.search(text)
    if m:
        lat = _dms_to_decimal(float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))
        lon = _dms_to_decimal(float(m.group(5)), float(m.group(6)), float(m.group(7)), m.group(8))
        return (lat, lon)
    return None


def parse_coordinates_nl(text: str) -> Optional[tuple[float, float]]:
    """Parse NL/WL format like 'NL 37-49-55 ... WL 75-27-30'."""
    m = COORD_NL_PATTERN.search(text)
    if m:
        lat = _dms_to_decimal(float(m.group(1)), float(m.group(2)), float(m.group(3)), "N")
        lon = _dms_to_decimal(float(m.group(4)), float(m.group(5)), float(m.group(6)), "W")
        return (lat, lon)
    return None


def parse_coordinates(text: str) -> Optional[tuple[float, float]]:
    return parse_coordinates_standard(text) or parse_coordinates_nl(text)


def parse_azimuth(text: str) -> Optional[float]:
    m = AZIMUTH_PATTERN.search(text)
    if m:
        return float(m.group(1))
    simple = re.search(r"~?\s*(\d{2,3})\s*(?:deg|°)", text, re.IGNORECASE)
    if simple:
        val = float(simple.group(1))
        if 0 <= val <= 360:
            return val
    return None


def parse_sta_dates(text: str) -> Optional[tuple[datetime, datetime]]:
    m = DATE_RANGE_PATTERN.search(text)
    if m:
        try:
            start = datetime.strptime(m.group(1), "%m/%d/%Y")
            end = datetime.strptime(m.group(2), "%m/%d/%Y")
            return (start, end)
        except ValueError:
            pass
    return None


class FCCELSAdapter(BaseAdapter):
    """Scrapes FCC ELS for STAs near a launch site."""

    name = "fcc_els"

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)
        return self._client

    def poll(self, site: SiteConfig) -> list[SignalEvent]:
        logger.info("FCC ELS: polling for STAs near %s", site.site_id)
        events: list[SignalEvent] = []

        try:
            sta_records = self._search_recent_stas()
        except Exception:
            logger.exception("FCC ELS: failed to search for STAs")
            return events

        for record in sta_records:
            try:
                signal = self._process_sta_record(record, site)
                if signal:
                    events.append(signal)
            except Exception:
                logger.exception("FCC ELS: failed to process STA record %s", record.get("file_number", "?"))

        logger.info("FCC ELS: found %d signals for %s", len(events), site.site_id)
        return events

    def _search_recent_stas(self) -> list[dict]:
        """Search FCC ELS GenericSearch for recent STA grants.

        The GenericSearch.cfm page is a ColdFusion form. We POST to it
        with minimal filters to get recent STA records, then parse the
        results table.
        """
        client = self._get_client()
        records = []

        try:
            resp = client.get(GENERIC_SEARCH_URL)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.warning("FCC ELS: could not load search page, trying direct STA search")
            return self._search_via_application_search()

        soup = BeautifulSoup(resp.text, "html.parser")

        form = soup.find("form")
        form_data = {}
        if form:
            for inp in form.find_all("input"):
                name = inp.get("name")
                if name:
                    form_data[name] = inp.get("value", "")
            for sel in form.find_all("select"):
                name = sel.get("name")
                if name:
                    opt = sel.find("option", selected=True)
                    form_data[name] = opt.get("value", "") if opt else ""

        form_data.update({
            "app_type": "STA",
            "status": "G",
        })

        time.sleep(REQUEST_DELAY_S)

        try:
            resp = client.post(GENERIC_SEARCH_URL, data=form_data)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.warning("FCC ELS: search POST failed")
            return self._search_via_application_search()

        records = self._parse_search_results(resp.text)
        logger.info("FCC ELS: found %d STA records from GenericSearch", len(records))
        return records

    def _search_via_application_search(self) -> list[dict]:
        """Fallback: try the ApplicationSearch.cfm endpoint."""
        client = self._get_client()
        url = "https://apps.fcc.gov/oetcf/els/reports/ApplicationSearch.cfm"

        try:
            resp = client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.warning("FCC ELS: ApplicationSearch also failed")
            return []

        time.sleep(REQUEST_DELAY_S)
        return self._parse_search_results(resp.text)

    def _parse_search_results(self, html: str) -> list[dict]:
        """Parse an ELS search results page into a list of STA record dicts."""
        soup = BeautifulSoup(html, "html.parser")
        records = []

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            header_cells = rows[0].find_all(["th", "td"])
            headers = [c.get_text(strip=True).lower() for c in header_cells]

            if not any(kw in " ".join(headers) for kw in ["file", "call", "applicant", "status"]):
                continue

            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                record: dict = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        record[headers[i]] = cell.get_text(strip=True)

                    link = cell.find("a", href=True)
                    if link:
                        href = link["href"]
                        if "STA_Print" in href or "application_seq" in href:
                            record["detail_url"] = href
                            seq_match = re.search(r"application_seq=(\d+)", href)
                            if seq_match:
                                record["application_seq"] = seq_match.group(1)
                        if "GetAtt" in href:
                            record["attachment_url"] = href

                if record.get("application_seq") or record.get("detail_url"):
                    records.append(record)

        return records

    def _fetch_sta_detail(self, application_seq: str) -> Optional[str]:
        """Fetch the STA detail page text."""
        client = self._get_client()
        url = f"{STA_PRINT_URL}?mode=current&application_seq={application_seq}"

        time.sleep(REQUEST_DELAY_S)
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError:
            logger.warning("FCC ELS: failed to fetch STA detail %s", application_seq)
            return None

    def _fetch_attachment(self, attachment_id: str) -> Optional[str]:
        """Fetch STA attachment document text."""
        client = self._get_client()
        url = f"{ATTACHMENT_URL}?id={attachment_id}&x="

        time.sleep(REQUEST_DELAY_S)
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError:
            logger.warning("FCC ELS: failed to fetch attachment %s", attachment_id)
            return None

    def _process_sta_record(self, record: dict, site: SiteConfig) -> Optional[SignalEvent]:
        """Fetch STA detail, parse for launch-related data, check proximity to site."""

        all_text = ""
        application_seq = record.get("application_seq")
        raw_ref = ""

        if application_seq:
            detail_html = self._fetch_sta_detail(application_seq)
            if detail_html:
                detail_soup = BeautifulSoup(detail_html, "html.parser")
                all_text += detail_soup.get_text(" ", strip=True)
                raw_ref = f"{STA_PRINT_URL}?mode=current&application_seq={application_seq}"

        attachment_url = record.get("attachment_url")
        if attachment_url:
            att_match = re.search(r"id=(\d+)", attachment_url)
            if att_match:
                att_text = self._fetch_attachment(att_match.group(1))
                if att_text:
                    att_soup = BeautifulSoup(att_text, "html.parser")
                    all_text += " " + att_soup.get_text(" ", strip=True)
                    if not raw_ref:
                        raw_ref = f"{ATTACHMENT_URL}?id={att_match.group(1)}&x="

        if not all_text:
            return None

        coords = parse_coordinates(all_text)
        if not coords:
            return None

        lat, lon = coords
        dist_km = _haversine_km(lat, lon, site.center_lat, site.center_lon)
        if dist_km > site.fcc_search_radius_km:
            return None

        dates = parse_sta_dates(all_text)
        if dates:
            window_start, window_end = dates
        else:
            window_start = datetime.utcnow()
            window_end = window_start + timedelta(days=30)

        cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)
        if window_end < cutoff:
            return None

        azimuth = parse_azimuth(all_text)
        file_number = record.get("file number", record.get("file_number", ""))
        applicant = record.get("applicant", record.get("applicant name", ""))

        metadata = {
            "applicant": applicant,
            "file_number": file_number,
            "coordinates": {"lat": lat, "lon": lon},
            "distance_km": round(dist_km, 2),
        }
        if azimuth is not None:
            metadata["azimuth_deg"] = azimuth

        confidence = 0.75
        text_lower = all_text.lower()
        if any(kw in text_lower for kw in ["launch", "rocket", "orbital", "suborbital"]):
            confidence = 0.85
        if azimuth is not None:
            confidence = min(confidence + 0.10, 0.95)

        return SignalEvent(
            site_id=site.site_id,
            source=self.name,
            signal_type=SignalType.RF_LICENSE,
            window_start=window_start,
            window_end=window_end,
            confidence=confidence,
            raw_ref=raw_ref,
            geo=f"{lat:.4f},{lon:.4f}",
            metadata=metadata,
        )

    def close(self):
        if self._client and not self._client.is_closed:
            self._client.close()
