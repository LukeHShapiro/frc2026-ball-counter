"""
src/statbotics_client.py - Statbotics API v3 client

Fetches EPA (Statbotics' OPR equivalent) for teams.
No API key required — public endpoint.

Functions:
  get_team_event_epa(team_number, event_key) - EPA for one team at one event
  get_team_epa(team_number, year)            - Season-wide EPA fallback
  build_opr_map(team_numbers, event_key)     - {team: epa} for a list of teams

EPA (Expected Points Added) is Statbotics' modern replacement for OPR.
It measures how many points a team is expected to contribute per match.
Used here as the OPR weight in dense-shot scoring attribution.

Caches responses to data/statbotics_cache/ (TTL: 1 hour).
Gracefully degrades to weight=1.0 if API is unreachable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

BASE_URL  = "https://api.statbotics.io/v3"
CACHE_DIR = Path("data/statbotics_cache")
CACHE_TTL = 3600  # seconds


# ---- Cache helpers ----------------------------------------------------------

def _cache_path(endpoint: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(endpoint.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _get(endpoint: str, timeout: int = 6) -> dict | list | None:
    """GET from Statbotics with file-based caching. Returns None on 404 or error."""
    cache = _cache_path(endpoint)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL:
        with open(cache) as f:
            return json.load(f)

    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        with open(cache, "w") as f:
            json.dump(data, f)
        return data
    except requests.HTTPError:
        return None
    except Exception as exc:
        print(f"  [Statbotics] Warning - could not reach API: {exc}")
        return None


def _extract_epa(data: dict) -> float | None:
    """Pull total_points mean out of a team_event or team_year response."""
    if not isinstance(data, dict):
        return None
    epa = data.get("epa", {})
    if not isinstance(epa, dict):
        return None
    tp = epa.get("total_points", {})
    if isinstance(tp, dict):
        val = tp.get("mean")
        return float(val) if val is not None else None
    # Some older responses store a flat float
    if isinstance(tp, (int, float)):
        return float(tp)
    return None


# ---- Public API -------------------------------------------------------------

def get_team_event_epa(team_number: str | int, event_key: str) -> float | None:
    """
    Fetch EPA for a single team at a specific event.

    Args:
        team_number: Team number string or int (e.g. "1234" or 1234).
        event_key:   TBA-format event key (e.g. "2026txhou").

    Returns:
        EPA total_points mean as a float, or None if not available.
    """
    team = str(team_number)
    data = _get(f"/team_event/{team}/{event_key}")
    return _extract_epa(data) if data else None


def get_team_epa(team_number: str | int, year: int = 2026) -> float | None:
    """
    Fetch season-wide EPA for a team (fallback when event data is absent).

    Args:
        team_number: Team number string or int.
        year:        Season year.

    Returns:
        EPA total_points mean, or None if unavailable.
    """
    team = str(team_number)
    data = _get(f"/team_year/{team}/{year}")
    return _extract_epa(data) if data else None


def get_event_epas(event_key: str) -> dict[str, float]:
    """
    Bulk-fetch EPA for all teams at an event via /event/{event_key}/team_events.

    Returns:
        {team_number_str: epa_float}  (may be empty if endpoint unavailable)
    """
    data = _get(f"/event/{event_key}/team_events")
    if not isinstance(data, list):
        return {}

    result: dict[str, float] = {}
    for entry in data:
        # Statbotics team_event objects have a "team" field (just the number)
        team = str(entry.get("team", "")).replace("frc", "").strip()
        if not team:
            continue
        val = _extract_epa(entry)
        if val is not None:
            result[team] = val
    return result


def build_opr_map(
    team_numbers:   list[str],
    event_key:      str | None = None,
    year:           int = 2026,
    tba_oprs:       dict[str, float] | None = None,
    tba_blend:      float = 0.4,
) -> dict[str, float]:
    """
    Build a {team_number: weight} map for a list of teams.

    Data sources (in priority order):
      1. Statbotics bulk event EPA   — most accurate for this event
      2. Statbotics individual EPA   — per-team fallback
      3. Statbotics season-wide EPA  — last Statbotics fallback
      4. TBA OPR                     — blended in when available (see tba_blend)
      5. Default weight = 1.0        — neutral when all APIs unreachable

    When both Statbotics EPA and TBA OPR are available for a team the final
    weight is a weighted average:
        weight = (1 - tba_blend) * statbotics_epa + tba_blend * tba_opr

    Args:
        team_numbers: List of team number strings.
        event_key:    TBA/Statbotics event key (e.g. "2026txhou").
        year:         Season year for fallback fetches.
        tba_oprs:     Pre-fetched {team_number: opr} from tba_client.get_event_oprs().
                      Pass None to skip TBA blending.
        tba_blend:    Weight given to TBA OPR when blending (0.0 = Statbotics only,
                      1.0 = TBA only).  Default 0.4.

    Returns:
        {team_number: weight_float}  — every team in team_numbers has an entry.
    """
    statbotics_map: dict[str, float] = {}

    # Step 1: Statbotics bulk event fetch
    if event_key:
        bulk = get_event_epas(event_key)
        if bulk:
            print(f"  [Statbotics] Bulk EPA loaded: {len(bulk)} teams "
                  f"at event {event_key}")
            for t in team_numbers:
                if t in bulk:
                    statbotics_map[t] = bulk[t]

    # Step 2 & 3: individual Statbotics fetches for still-missing teams
    for team in team_numbers:
        if team in statbotics_map:
            continue
        val = None
        if event_key:
            val = get_team_event_epa(team, event_key)
        if val is None:
            val = get_team_epa(team, year)
        if val is not None:
            statbotics_map[team] = val

    # Step 4: blend with TBA OPR where both are available
    opr_map: dict[str, float] = {}
    for team in team_numbers:
        sb_val  = statbotics_map.get(team)
        tba_val = (tba_oprs or {}).get(team)

        if sb_val is not None and tba_val is not None:
            # Both sources available: weighted average
            blended = (1.0 - tba_blend) * sb_val + tba_blend * tba_val
            opr_map[team] = blended
        elif sb_val is not None:
            opr_map[team] = sb_val
        elif tba_val is not None:
            opr_map[team] = tba_val
        # else: handled below

    # Step 5: neutral fallback for any team with no data from either source
    missing = [t for t in team_numbers if t not in opr_map]
    if missing:
        for t in missing:
            opr_map[t] = 1.0
        print(f"  [Statbotics] {len(missing)} teams with no EPA/OPR data "
              f"-> default weight 1.0")

    if tba_oprs:
        blended_count = sum(
            1 for t in team_numbers
            if t in statbotics_map and t in (tba_oprs or {})
        )
        if blended_count:
            print(f"  [Statbotics] Blended Statbotics EPA + TBA OPR "
                  f"({tba_blend:.0%} TBA) for {blended_count} teams")

    return opr_map
