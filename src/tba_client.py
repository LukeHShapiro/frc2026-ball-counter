"""
src/tba_client.py — Phase 14: The Blue Alliance API v3 client

All requests:
  - Include X-TBA-Auth-Key header from configs/tba_config.json
  - Use ETag caching (If-None-Match) to avoid redundant fetches
  - Handle 304 Not Modified gracefully
  - Raise TBAAuthError on 401, TBANotFoundError on 404
  - Cache responses to data/tba_cache/{endpoint_hash}.json

Depends on: configs/tba_config.json (INPUT CHECKPOINT #8)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://www.thebluealliance.com/api/v3"
_CACHE_DIR = Path("data/tba_cache")


# ── Custom exceptions ─────────────────────────────────────────────────────────

class TBAAuthError(Exception):
    """Raised when the TBA API key is invalid (HTTP 401)."""


class TBANotFoundError(Exception):
    """Raised when the requested resource does not exist (HTTP 404)."""


class TBAError(Exception):
    """Generic TBA API error."""


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "tba_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            "configs/tba_config.json not found. "
            "Complete INPUT CHECKPOINT #8 before using the alliance builder."
        )
    with open(cfg_path) as f:
        return json.load(f)


def _api_key() -> str:
    cfg = _load_config()
    key = cfg.get("api_key", "")
    if not key or key == "YOUR_TBA_KEY_HERE":
        raise TBAAuthError(
            "No TBA API key set. Add your key to configs/tba_config.json. "
            "(Generate one at https://www.thebluealliance.com/account)"
        )
    return key


def _cache_ttl() -> int:
    return _load_config().get("cache_ttl_seconds", 300)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(endpoint: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(endpoint.encode()).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def _load_cache(endpoint: str) -> dict | None:
    """Return cached response dict or None if expired / missing."""
    path = _cache_path(endpoint)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
        age = time.time() - cached.get("_cached_at", 0)
        if age > _cache_ttl():
            return None
        return cached
    except (json.JSONDecodeError, KeyError):
        return None


def _save_cache(endpoint: str, data: Any, etag: str | None) -> None:
    path = _cache_path(endpoint)
    payload = {
        "_cached_at": time.time(),
        "_etag": etag,
        "data": data,
    }
    path.write_text(json.dumps(payload, indent=2))


def _get_cached_etag(endpoint: str) -> str | None:
    path = _cache_path(endpoint)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("_etag")
    except (json.JSONDecodeError, KeyError):
        return None


# ── Core request function ─────────────────────────────────────────────────────

def _get(endpoint: str) -> Any:
    """
    Perform a GET request against the TBA API with ETag caching.

    Args:
        endpoint: Path after BASE_URL (e.g. "/team/frc1234").

    Returns:
        Parsed JSON response.

    Raises:
        TBAAuthError:     HTTP 401 — bad API key.
        TBANotFoundError: HTTP 404 — resource not found.
        TBAError:         Other HTTP errors.
    """
    # Try cache first
    cached = _load_cache(endpoint)
    cached_etag = _get_cached_etag(endpoint)

    headers: dict[str, str] = {
        "X-TBA-Auth-Key": _api_key(),
        "Accept": "application/json",
    }
    if cached_etag:
        headers["If-None-Match"] = cached_etag

    url = BASE_URL + endpoint
    try:
        response = requests.get(url, headers=headers, timeout=8)
    except requests.RequestException as e:
        # Fall back to cache if request fails
        if cached:
            return cached["data"]
        raise TBAError(f"Network error fetching {url}: {e}") from e

    if response.status_code == 304:
        # Not modified — return cached data if available, else re-fetch without ETag
        if cached:
            return cached["data"]
        # Cache file missing despite sending ETag — re-fetch without conditional header
        headers.pop("If-None-Match", None)
        try:
            response = requests.get(url, headers=headers, timeout=8)
        except requests.RequestException as e:
            raise TBAError(f"Network error re-fetching {url}: {e}") from e

    if response.status_code == 401:
        raise TBAAuthError(
            "TBA API returned 401 Unauthorized. "
            "Check your api_key in configs/tba_config.json."
        )

    if response.status_code == 404:
        raise TBANotFoundError(
            f"TBA returned 404 for {endpoint}. "
            "Check that your event_key or team number is correct."
        )

    if response.status_code != 200:
        raise TBAError(
            f"TBA API error {response.status_code} for {endpoint}: {response.text[:200]}"
        )

    data = response.json()
    etag = response.headers.get("ETag")
    _save_cache(endpoint, data, etag)
    return data


# ── Public API functions ──────────────────────────────────────────────────────

def get_team_info(team_number: str | int) -> dict:
    """
    Fetch basic info for a single team.

    GET /team/frc{team_number}

    Args:
        team_number: FRC team number (int or string, without "frc" prefix).

    Returns:
        {nickname, city, state_prov, rookie_year, website}

    Depends on: configs/tba_config.json api_key.
    """
    raw = _get(f"/team/frc{team_number}")
    return {
        "team_number": str(team_number),
        "nickname":    raw.get("nickname", ""),
        "city":        raw.get("city", ""),
        "state_prov":  raw.get("state_prov", ""),
        "rookie_year": raw.get("rookie_year"),
        "website":     raw.get("website", ""),
    }


def get_event_teams(event_key: str) -> list[dict]:
    """
    Fetch all teams attending an event.

    GET /event/{event_key}/teams

    Args:
        event_key: TBA event key (e.g. "2026txhou").

    Returns:
        List of {team_number, nickname, city, state_prov}

    Depends on: configs/tba_config.json api_key.
    """
    raw_list = _get(f"/event/{event_key}/teams")
    return [
        {
            "team_number": str(t.get("team_number", "")),
            "nickname":    t.get("nickname", ""),
            "city":        t.get("city", ""),
            "state_prov":  t.get("state_prov", ""),
        }
        for t in raw_list
    ]


def get_event_rankings(event_key: str) -> list[dict]:
    """
    Fetch rankings for an event.

    GET /event/{event_key}/rankings

    Args:
        event_key: TBA event key.

    Returns:
        [{rank, team_number, wins, losses, ties, ranking_points,
          avg_match_points, avg_bonus_points}]

    Depends on: configs/tba_config.json api_key.
    """
    raw = _get(f"/event/{event_key}/rankings")
    rankings_list = raw.get("rankings", []) if isinstance(raw, dict) else raw
    results = []
    for entry in rankings_list:
        record = entry.get("record", {})
        extra  = entry.get("extra_stats", [0, 0])
        results.append({
            "rank":              entry.get("rank"),
            "team_number":       str(entry.get("team_key", "")).replace("frc", ""),
            "wins":              record.get("wins", 0),
            "losses":            record.get("losses", 0),
            "ties":              record.get("ties", 0),
            "ranking_points":    entry.get("sort_orders", [0])[0],
            "avg_match_points":  extra[0] if len(extra) > 0 else 0,
            "avg_bonus_points":  extra[1] if len(extra) > 1 else 0,
        })
    return results


def get_event_oprs(event_key: str) -> dict:
    """
    Fetch OPR / DPR / CCWM for all teams at an event.

    GET /event/{event_key}/oprs

    Args:
        event_key: TBA event key.

    Returns:
        {
          oprs:  {team_key: float},
          dprs:  {team_key: float},
          ccwms: {team_key: float}
        }

    Depends on: configs/tba_config.json api_key.
    """
    raw = _get(f"/event/{event_key}/oprs")
    return {
        "oprs":  raw.get("oprs",  {}),
        "dprs":  raw.get("dprs",  {}),
        "ccwms": raw.get("ccwms", {}),
    }


def get_event_matches(event_key: str) -> list[dict]:
    """
    Fetch all matches for an event.

    GET /event/{event_key}/matches

    Args:
        event_key: TBA event key.

    Returns:
        Full match list with alliance compositions + scores.

    Depends on: configs/tba_config.json api_key.
    """
    return _get(f"/event/{event_key}/matches")


def get_team_event_status(team_number: str | int, event_key: str) -> dict:
    """
    Fetch a team's current status at an event.

    GET /team/frc{team_number}/event/{event_key}/status

    Args:
        team_number: FRC team number.
        event_key:   TBA event key.

    Returns:
        {qual_rank, qual_average, playoff_status, alliance}

    Depends on: configs/tba_config.json api_key.
    """
    raw = _get(f"/team/frc{team_number}/event/{event_key}/status")
    qual  = raw.get("qual", {}) or {}
    playoff = raw.get("playoff", {}) or {}
    return {
        "qual_rank":       qual.get("ranking", {}).get("rank"),
        "qual_average":    qual.get("ranking", {}).get("sort_orders", [None])[0],
        "playoff_status":  playoff.get("status"),
        "alliance":        raw.get("alliance"),
    }


def get_team_history(team_number: str | int, year: int = 2026) -> list[dict]:
    """
    Fetch a team's event performance history for a given season.

    GET /team/frc{team_number}/events/{year}/statuses

    Args:
        team_number: FRC team number.
        year:        Season year (default 2026).

    Returns:
        Performance data across all events this season.

    Depends on: configs/tba_config.json api_key.
    """
    raw = _get(f"/team/frc{team_number}/events/{year}/statuses")
    results = []
    if isinstance(raw, dict):
        for event_key, status in raw.items():
            if status is None:
                continue
            qual    = (status.get("qual") or {})
            playoff = (status.get("playoff") or {})
            results.append({
                "event_key":      event_key,
                "qual_rank":      (qual.get("ranking") or {}).get("rank"),
                "playoff_status": playoff.get("status"),
                "alliance":       status.get("alliance"),
            })
    return results


def get_event_predictions(event_key: str) -> dict:
    """
    Fetch TBA's match predictions for an event.

    GET /event/{event_key}/predictions

    Args:
        event_key: TBA event key.

    Returns:
        TBA's predicted win probabilities (used as one input signal).

    Depends on: configs/tba_config.json api_key.
    """
    return _get(f"/event/{event_key}/predictions") or {}


# ── Video functions ───────────────────────────────────────────────────────────

def get_match(match_key: str) -> dict:
    """
    Fetch full data for a single match including videos and score breakdown.

    GET /match/{match_key}

    Args:
        match_key: TBA match key (e.g. "2026txhou_qm14" or "2026txhou_sf1m1").

    Returns:
        Full match dict including:
          key, comp_level, match_number, set_number,
          alliances (red/blue scores + team keys),
          score_breakdown (per-game-piece counts),
          videos ([{type, key}] — type is usually "youtube")

    Depends on: configs/tba_config.json api_key.
    """
    return _get(f"/match/{match_key}") or {}


def get_match_videos(event_key: str) -> list[dict]:
    """
    Return all match video links for an event.

    Iterates the event match list and extracts video entries.

    Args:
        event_key: TBA event key (e.g. "2026txhou").

    Returns:
        [{
            match_key:   str,   # e.g. "2026txhou_qm1"
            comp_level:  str,   # "qm" | "sf" | "f"
            match_number: int,
            set_number:  int,
            videos: [{type: str, key: str}],
            youtube_urls: [str],   # resolved full YouTube URLs (may be empty)
        }]
        Sorted by comp_level then match_number.

    Depends on: configs/tba_config.json api_key.
    """
    matches = get_event_matches(event_key)
    results = []
    _level_order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}

    for m in matches:
        videos = m.get("videos") or []
        yt_urls = [
            f"https://www.youtube.com/watch?v={v['key']}"
            for v in videos
            if v.get("type") == "youtube" and v.get("key")
        ]
        results.append({
            "match_key":    m.get("key", ""),
            "comp_level":   m.get("comp_level", "qm"),
            "match_number": m.get("match_number", 0),
            "set_number":   m.get("set_number", 1),
            "videos":       videos,
            "youtube_urls": yt_urls,
        })

    results.sort(key=lambda r: (
        _level_order.get(r["comp_level"], 9),
        r["set_number"],
        r["match_number"],
    ))
    return results


def get_team_media(team_number: str | int, year: int = 2026) -> list[dict]:
    """
    Fetch media entries for a team in a given year (highlight videos, photos, etc.)

    GET /team/frc{team_number}/media/{year}

    Args:
        team_number: FRC team number.
        year:        Season year.

    Returns:
        [{
            type:        str,    # "youtube", "imgur", "cdphotothread", "instagram-image", etc.
            foreign_key: str,    # YouTube video ID or URL fragment
            preferred:   bool,
            youtube_url: str | None,   # resolved URL if type=="youtube"
        }]

    Depends on: configs/tba_config.json api_key.
    """
    raw = _get(f"/team/frc{team_number}/media/{year}") or []
    results = []
    for item in raw:
        media_type   = item.get("type", "")
        foreign_key  = item.get("foreign_key", "") or item.get("details", {}).get("youtube_key", "")
        youtube_url  = (
            f"https://www.youtube.com/watch?v={foreign_key}"
            if media_type == "youtube" and foreign_key
            else None
        )
        results.append({
            "type":        media_type,
            "foreign_key": foreign_key,
            "preferred":   bool(item.get("preferred", False)),
            "youtube_url": youtube_url,
        })
    return results


# ── Team stats functions ──────────────────────────────────────────────────────

def get_team_awards(team_number: str | int, year: int | None = None) -> list[dict]:
    """
    Fetch awards won by a team, optionally filtered to one year.

    GET /team/frc{team_number}/awards
    GET /team/frc{team_number}/awards/{year}

    Args:
        team_number: FRC team number.
        year:        Filter to this year (None = all time).

    Returns:
        [{name, award_type, event_key, year}]

    Depends on: configs/tba_config.json api_key.
    """
    endpoint = (
        f"/team/frc{team_number}/awards/{year}"
        if year else
        f"/team/frc{team_number}/awards"
    )
    raw = _get(endpoint) or []
    return [
        {
            "name":       a.get("name", ""),
            "award_type": a.get("award_type"),
            "event_key":  a.get("event_key", ""),
            "year":       a.get("year"),
        }
        for a in raw
    ]


def get_event_team_stats(event_key: str) -> dict[str, dict]:
    """
    Aggregate OPR, DPR, CCWM, and ranking data for every team at an event
    into a single dict keyed by team number.

    Args:
        event_key: TBA event key.

    Returns:
        {
          "1234": {
              opr: float, dpr: float, ccwm: float,
              rank: int, wins: int, losses: int, ties: int,
              ranking_points: float,
          },
          ...
        }

    Depends on: configs/tba_config.json api_key.
    """
    oprs_data = get_event_oprs(event_key)
    rankings  = get_event_rankings(event_key)

    # Index rankings by team number
    rank_by_team: dict[str, dict] = {}
    for r in rankings:
        rank_by_team[r["team_number"]] = r

    # Collect all team numbers mentioned in OPRs or rankings
    all_teams: set[str] = set(rank_by_team.keys())
    for frc_key in oprs_data.get("oprs", {}):
        all_teams.add(frc_key.replace("frc", ""))

    result: dict[str, dict] = {}
    for team in all_teams:
        frc_key = f"frc{team}"
        r = rank_by_team.get(team, {})
        result[team] = {
            "opr":            oprs_data["oprs"].get(frc_key),
            "dpr":            oprs_data["dprs"].get(frc_key),
            "ccwm":           oprs_data["ccwms"].get(frc_key),
            "rank":           r.get("rank"),
            "wins":           r.get("wins"),
            "losses":         r.get("losses"),
            "ties":           r.get("ties"),
            "ranking_points": r.get("ranking_points"),
        }
    return result


def get_team_full_stats(
    team_number: str | int,
    event_key:   str,
    year:        int = 2026,
) -> dict:
    """
    Compile a comprehensive stats dict for one team at one event.

    Combines: team info, OPR/DPR/CCWM, event ranking, event status,
              season history, awards (current year), and media.

    Args:
        team_number: FRC team number.
        event_key:   TBA event key.
        year:        Season year.

    Returns:
        {
            team_number, nickname, city, state_prov, rookie_year,
            opr, dpr, ccwm,
            rank, wins, losses, ties, ranking_points,
            qual_average, playoff_status, alliance,
            awards_this_year: [{name, award_type, event_key, year}],
            season_events: [{event_key, qual_rank, playoff_status}],
            media: [{type, foreign_key, preferred, youtube_url}],
        }

    Depends on: configs/tba_config.json api_key.
    """
    team   = str(team_number)
    frc_key = f"frc{team}"

    info    = get_team_info(team)
    oprs    = get_event_oprs(event_key)
    rank_list = get_event_rankings(event_key)
    rank_entry = next(
        (r for r in rank_list if r["team_number"] == team), {}
    )

    try:
        status = get_team_event_status(team, event_key)
    except (TBANotFoundError, TBAError):
        status = {}

    try:
        history = get_team_history(team, year)
    except (TBANotFoundError, TBAError):
        history = []

    try:
        awards = get_team_awards(team, year)
    except (TBANotFoundError, TBAError):
        awards = []

    try:
        media = get_team_media(team, year)
    except (TBANotFoundError, TBAError):
        media = []

    return {
        "team_number":    team,
        "nickname":       info.get("nickname", ""),
        "city":           info.get("city", ""),
        "state_prov":     info.get("state_prov", ""),
        "rookie_year":    info.get("rookie_year"),
        "opr":            oprs["oprs"].get(frc_key),
        "dpr":            oprs["dprs"].get(frc_key),
        "ccwm":           oprs["ccwms"].get(frc_key),
        "rank":           rank_entry.get("rank"),
        "wins":           rank_entry.get("wins"),
        "losses":         rank_entry.get("losses"),
        "ties":           rank_entry.get("ties"),
        "ranking_points": rank_entry.get("ranking_points"),
        "qual_average":   status.get("qual_average"),
        "playoff_status": status.get("playoff_status"),
        "alliance":       status.get("alliance"),
        "awards_this_year": awards,
        "season_events":    history,
        "media":            media,
    }
