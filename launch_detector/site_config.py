"""Site configuration registry. Adding a new launch site = a new SiteConfig entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoringPolicy:
    """Per-site scoring weights and thresholds for correlation."""
    weights: dict[str, float] = field(default_factory=dict)
    watch_threshold: float = 0.3
    likely_threshold: float = 0.55
    imminent_threshold: float = 0.8
    correlation_window_hours: int = 72


@dataclass
class SiteConfig:
    site_id: str
    name: str
    center_lat: float
    center_lon: float
    restricted_areas: list[str] = field(default_factory=list)
    warning_areas: list[str] = field(default_factory=list)
    maritime_authority: Optional[str] = None
    downrange_warning_source: Optional[str] = None
    tier: str = "T1_federal"
    has_maritime_signal: bool = False
    scoring: ScoringPolicy = field(default_factory=ScoringPolicy)
    notam_radius_nm: int = 30
    notam_location_ids: list[str] = field(default_factory=list)
    fcc_search_radius_km: float = 15.0


SITE_REGISTRY: dict[str, SiteConfig] = {}


def register_site(config: SiteConfig) -> None:
    SITE_REGISTRY[config.site_id] = config


def get_site(site_id: str) -> SiteConfig:
    return SITE_REGISTRY[site_id]


def list_sites() -> list[str]:
    return list(SITE_REGISTRY.keys())


WFF_SCORING = ScoringPolicy(
    weights={
        "rf_license": 0.45,
        "airspace_activation": 0.15,
        "warning_area_hot": 0.20,
        "maritime_hazard": 0.15,
        "vessel_clearing": 0.05,
    },
    watch_threshold=0.15,
    likely_threshold=0.45,
    imminent_threshold=0.75,
    correlation_window_hours=72,
)

WFF = SiteConfig(
    site_id="WFF",
    name="NASA Wallops Flight Facility",
    center_lat=37.8319,
    center_lon=-75.4877,
    restricted_areas=["R-6604A", "R-6604B", "R-6604C", "R-6604D", "R-6604E"],
    warning_areas=["W-386"],
    maritime_authority="USCG District 5 (Portsmouth, VA)",
    downrange_warning_source="NGA NAVAREA IV / HYDROLANT",
    tier="T1_federal",
    has_maritime_signal=True,
    scoring=WFF_SCORING,
    notam_radius_nm=30,
    notam_location_ids=["KWAL"],
    fcc_search_radius_km=15.0,
)

register_site(WFF)
