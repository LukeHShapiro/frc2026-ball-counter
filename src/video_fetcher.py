"""
src/video_fetcher.py - Pull match videos from The Blue Alliance / YouTube

Uses TBA API to discover video links, then yt-dlp to download them.
No video is re-downloaded if a file with the same name already exists.

Functions:
  list_event_videos(event_key)              - print/return all video links, no download
  fetch_event_videos(event_key, output_dir) - download all match videos for an event
  fetch_match_video(match_info, output_dir) - download one match's video
  fetch_team_highlight(team_number, year)   - download a team's highlight reel

Requires: yt-dlp  (pip install yt-dlp)
Depends on: tba_client.py  (for API calls)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable


# ── yt-dlp wrapper ────────────────────────────────────────────────────────────

def _yt_dlp_available() -> bool:
    """Return True if yt-dlp is importable / on PATH."""
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def _download_youtube(
    url:        str,
    output_dir: Path,
    filename:   str,
    quality:    str = "best[height<=1080]",
) -> Path | None:
    """
    Download a single YouTube URL to output_dir using yt-dlp.

    Args:
        url:        Full YouTube URL.
        output_dir: Directory to save the file.
        filename:   Output filename without extension (yt-dlp adds it).
        quality:    yt-dlp format selector string.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / f"{filename}.%(ext)s")

    # Check if already downloaded (any extension)
    existing = list(output_dir.glob(f"{filename}.*"))
    if existing:
        print(f"  [VideoFetch] Already exists: {existing[0].name}")
        return existing[0]

    try:
        import yt_dlp

        ydl_opts = {
            "format":          quality,
            "outtmpl":         outtmpl,
            "quiet":           True,
            "no_warnings":     True,
            "ignoreerrors":    True,
            "writeinfojson":   False,
            "noplaylist":      True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return None
            ext  = info.get("ext", "mp4")
            dest = output_dir / f"{filename}.{ext}"
            return dest if dest.exists() else None

    except Exception as exc:
        print(f"  [VideoFetch] (!) Download failed for {url}: {exc}")
        return None


# ── Sanitise filenames ────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    """Strip characters that are illegal in Windows/Mac/Linux filenames."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
               "0123456789-_.")
    return "".join(c if c in keep else "_" for c in s)


# ── Public functions ──────────────────────────────────────────────────────────

def list_event_videos(event_key: str) -> list[dict]:
    """
    List all match videos for an event without downloading anything.

    Prints a formatted table to stdout and returns the raw list.

    Args:
        event_key: TBA event key (e.g. "2026txhou").

    Returns:
        List of match video dicts from tba_client.get_match_videos().
    """
    from tba_client import get_match_videos, TBAAuthError, TBANotFoundError

    try:
        videos = get_match_videos(event_key)
    except TBAAuthError as e:
        print(f"  [VideoFetch] TBA auth error: {e}")
        return []
    except TBANotFoundError:
        print(f"  [VideoFetch] Event not found: {event_key}")
        return []

    total_links = sum(len(v["youtube_urls"]) for v in videos)
    print(f"\n  [VideoFetch] {event_key} - {len(videos)} matches, "
          f"{total_links} YouTube links\n")

    for v in videos:
        level_label = {"qm": "Qual", "sf": "Semifinal",
                       "f": "Final", "qf": "Quarterfinal",
                       "ef": "Octo-final"}.get(v["comp_level"], v["comp_level"])
        match_label = (
            f"{level_label} {v['match_number']}"
            if v["comp_level"] == "qm"
            else f"{level_label} {v['set_number']}m{v['match_number']}"
        )
        if v["youtube_urls"]:
            for url in v["youtube_urls"]:
                print(f"    {v['match_key']:<22}  {match_label:<22}  {url}")
        else:
            print(f"    {v['match_key']:<22}  {match_label:<22}  (no video)")

    return videos


def fetch_match_video(
    match_info: dict,
    output_dir: Path | str,
    quality:    str = "best[height<=1080]",
) -> list[Path]:
    """
    Download all YouTube videos for a single match.

    Args:
        match_info: Dict from tba_client.get_match_videos() (one element).
        output_dir: Directory to save downloaded files.
        quality:    yt-dlp format selector.

    Returns:
        List of Paths to downloaded files (one per video link).
    """
    output_dir = Path(output_dir)
    downloaded: list[Path] = []

    for i, url in enumerate(match_info.get("youtube_urls", [])):
        suffix    = f"_v{i+1}" if len(match_info["youtube_urls"]) > 1 else ""
        filename  = _safe_name(f"{match_info['match_key']}{suffix}")
        dest      = _download_youtube(url, output_dir, filename, quality)
        if dest:
            downloaded.append(dest)
            print(f"  [VideoFetch] [OK] {dest.name}")

    return downloaded


def fetch_event_videos(
    event_key:    str,
    output_dir:   Path | str = "data/match_videos",
    comp_levels:  list[str] | None = None,
    quality:      str = "best[height<=1080]",
    on_progress:  Callable[[int, int], None] | None = None,
) -> dict[str, list[Path]]:
    """
    Download all match videos for an event from TBA / YouTube.

    Args:
        event_key:   TBA event key (e.g. "2026txhou").
        output_dir:  Root directory — files go into output_dir/{event_key}/.
        comp_levels: Which match types to download.
                     None = all. Common values: ["qm"], ["qm","sf","f"].
        quality:     yt-dlp format selector.
        on_progress: Optional callback(done, total) for UI progress bars.

    Returns:
        {match_key: [Path, ...]}  for every match that was downloaded.

    Raises:
        RuntimeError if yt-dlp is not installed.
    """
    if not _yt_dlp_available():
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )

    from tba_client import get_match_videos, TBAAuthError, TBANotFoundError

    try:
        all_matches = get_match_videos(event_key)
    except TBAAuthError as e:
        raise RuntimeError(f"TBA auth error: {e}") from e
    except TBANotFoundError:
        raise RuntimeError(f"Event not found: {event_key}")

    if comp_levels:
        all_matches = [m for m in all_matches if m["comp_level"] in comp_levels]

    has_video   = [m for m in all_matches if m["youtube_urls"]]
    save_dir    = Path(output_dir) / event_key
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [VideoFetch] {event_key}: {len(has_video)} matches with video "
          f"(of {len(all_matches)} total)")

    results: dict[str, list[Path]] = {}
    for i, match in enumerate(has_video, 1):
        if on_progress:
            on_progress(i, len(has_video))
        paths = fetch_match_video(match, save_dir, quality)
        if paths:
            results[match["match_key"]] = paths

    # Save index file
    index = {
        "event_key": event_key,
        "total_matches": len(all_matches),
        "downloaded": {k: [str(p) for p in v] for k, v in results.items()},
    }
    (save_dir / "video_index.json").write_text(json.dumps(index, indent=2))
    print(f"  [VideoFetch] Done: {len(results)}/{len(has_video)} videos saved "
          f"-> {save_dir}")
    return results


def fetch_team_highlight(
    team_number: str | int,
    year:        int = 2026,
    output_dir:  Path | str = "data/team_media",
    quality:     str = "best[height<=1080]",
) -> Path | None:
    """
    Download the preferred highlight video for a team from TBA team media.

    Tries the "preferred" media entry first, then falls back to the first
    YouTube video found in the team's media list.

    Args:
        team_number: FRC team number.
        year:        Season year.
        output_dir:  Directory to save to.
        quality:     yt-dlp format selector.

    Returns:
        Path to downloaded file, or None if no video is available.
    """
    if not _yt_dlp_available():
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    from tba_client import get_team_media, TBAAuthError, TBANotFoundError

    try:
        media = get_team_media(team_number, year)
    except (TBAAuthError, TBANotFoundError) as e:
        print(f"  [VideoFetch] Could not fetch media for team {team_number}: {e}")
        return None

    yt_media = [m for m in media if m.get("youtube_url")]
    if not yt_media:
        print(f"  [VideoFetch] No YouTube media for team {team_number} in {year}")
        return None

    # Prefer the "preferred" entry
    preferred = next((m for m in yt_media if m.get("preferred")), yt_media[0])
    url       = preferred["youtube_url"]
    filename  = _safe_name(f"team{team_number}_{year}_highlight")
    dest      = _download_youtube(url, Path(output_dir), filename, quality)

    if dest:
        print(f"  [VideoFetch] Team {team_number} highlight -> {dest}")
    return dest


def fetch_all_team_highlights(
    team_numbers: list[str | int],
    year:         int = 2026,
    output_dir:   Path | str = "data/team_media",
    quality:      str = "best[height<=1080]",
) -> dict[str, Path | None]:
    """
    Download highlight videos for a list of teams.

    Args:
        team_numbers: List of FRC team numbers.
        year:         Season year.
        output_dir:   Root output directory.
        quality:      yt-dlp format selector.

    Returns:
        {team_number_str: Path | None}
    """
    results: dict[str, Path | None] = {}
    for team in team_numbers:
        results[str(team)] = fetch_team_highlight(team, year, output_dir, quality)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pull match videos and team media from The Blue Alliance"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List all video links for an event")
    p_list.add_argument("event_key")

    # fetch-event
    p_event = sub.add_parser("fetch-event", help="Download all match videos for an event")
    p_event.add_argument("event_key")
    p_event.add_argument("--out",     default="data/match_videos")
    p_event.add_argument("--levels",  nargs="*",
                         help="comp levels to download (e.g. qm sf f)")
    p_event.add_argument("--quality", default="best[height<=1080]")

    # fetch-team
    p_team = sub.add_parser("fetch-team", help="Download a team's highlight video")
    p_team.add_argument("team_number")
    p_team.add_argument("--year",    type=int, default=2026)
    p_team.add_argument("--out",     default="data/team_media")
    p_team.add_argument("--quality", default="best[height<=1080]")

    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    if args.cmd == "list":
        list_event_videos(args.event_key)

    elif args.cmd == "fetch-event":
        fetch_event_videos(
            args.event_key,
            output_dir  = args.out,
            comp_levels = args.levels,
            quality     = args.quality,
        )

    elif args.cmd == "fetch-team":
        fetch_team_highlight(
            args.team_number,
            year       = args.year,
            output_dir = args.out,
            quality    = args.quality,
        )
