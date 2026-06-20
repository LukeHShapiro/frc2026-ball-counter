"""Stub adapters for v2 data sources.

These define the interface and signal types but do not fetch real data.
See docs/datasources.md for access method notes per source.
"""

from __future__ import annotations

import logging

from ..models import SignalEvent
from ..site_config import SiteConfig
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class NGABroadcastAdapter(BaseAdapter):
    """NGA NAVAREA IV / HYDROLANT downrange hazard box warnings.

    TODO: Verify access method at https://msi.nga.mil/
    Would emit maritime_hazard SignalEvents for downrange warning areas.
    """

    name = "nga_broadcast"

    def poll(self, site: SiteConfig) -> list[SignalEvent]:
        logger.info("NGA Broadcast: stub — not implemented (v2)")
        return []

    def is_available(self) -> bool:
        return False


class USCGLNMAdapter(BaseAdapter):
    """USCG District 5 Local Notice to Mariners (weekly PDF).

    TODO: Verify access at https://www.navcen.uscg.gov/
    Would download weekly PDF, parse with pdfplumber for hazard areas
    near WFF, emit maritime_hazard SignalEvents.
    """

    name = "uscg_lnm"

    def poll(self, site: SiteConfig) -> list[SignalEvent]:
        logger.info("USCG LNM: stub — not implemented (v2)")
        return []

    def is_available(self) -> bool:
        return False


class AISAdapter(BaseAdapter):
    """AIS vessel traffic — detect range-clearing behavior.

    TODO: Requires commercial AIS data feed (MarineTraffic, AISHub, etc.)
    Would monitor vessel positions in downrange warning area,
    detect clearing pattern (vessels leaving area), emit vessel_clearing.
    """

    name = "ais"

    def poll(self, site: SiteConfig) -> list[SignalEvent]:
        logger.info("AIS: stub — not implemented (v2)")
        return []

    def is_available(self) -> bool:
        return False
