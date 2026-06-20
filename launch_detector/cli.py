"""CLI entry points: poll, run, backtest, alerts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .adapters.base import BaseAdapter
from .adapters.fcc_els import FCCELSAdapter
from .adapters.faa_notam import FAANOTAMAdapter
from .adapters.stubs import NGABroadcastAdapter, USCGLNMAdapter, AISAdapter
from .alerting import send_alerts, get_backends
from .backtest.harness import (
    load_ground_truth_file,
    run_backtest,
    print_backtest_report,
)
from .engine import evaluate
from .models import SignalEvent
from .site_config import SITE_REGISTRY, get_site, list_sites
from .storage import Storage

logger = logging.getLogger("launch_detector")


def get_adapters() -> list[BaseAdapter]:
    return [
        FCCELSAdapter(),
        FAANOTAMAdapter(),
        NGABroadcastAdapter(),
        USCGLNMAdapter(),
        AISAdapter(),
    ]


def cmd_poll(args: argparse.Namespace):
    """Run all adapters once for a site, store results, evaluate, alert."""
    storage = Storage(db_path=Path(args.db) if args.db else None)
    site = get_site(args.site)
    adapters = get_adapters()

    all_signals: list[SignalEvent] = []
    for adapter in adapters:
        if not adapter.is_available():
            logger.info("Skipping %s (not available)", adapter.name)
            continue
        logger.info("Polling %s for %s...", adapter.name, site.site_id)
        try:
            signals = adapter.poll(site)
            for sig in signals:
                storage.store_signal(sig)
            all_signals.extend(signals)
            logger.info("  %s: %d signals", adapter.name, len(signals))
        except Exception:
            logger.exception("Adapter %s failed", adapter.name)

    stored = storage.get_signals_for_site(site.site_id)
    alerts = evaluate(stored, site)

    if alerts:
        print(f"\n{'='*60}")
        print(f"ALERTS for {site.name} ({site.site_id})")
        print(f"{'='*60}")
        for alert in alerts:
            print(f"\n{alert.explanation}")
        send_alerts(alerts)
    else:
        print(f"\nNo alerts triggered for {site.name}.")

    print(f"\nTotal signals in DB: {len(stored)}")


def cmd_run(args: argparse.Namespace):
    """Daemon mode: poll on schedule."""
    interval = args.interval * 60
    print(f"Running in daemon mode. Polling every {args.interval} minutes.")
    print(f"Site: {args.site}")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            cmd_poll(args)
        except KeyboardInterrupt:
            print("\nStopping.")
            break
        except Exception:
            logger.exception("Poll cycle failed")

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping.")
            break


def cmd_backtest(args: argparse.Namespace):
    """Replay stored signals against ground truth."""
    storage = Storage(db_path=Path(args.db) if args.db else None)
    site = get_site(args.site)

    if args.ground_truth:
        gt = load_ground_truth_file(Path(args.ground_truth))
    else:
        gt = storage.get_ground_truth(args.site)
        if not gt:
            print("No ground truth data. Provide --ground-truth <file.json>")
            print('Format: [{"site_id":"WFF","date":"2024-06-15","vehicle":"Electron",...}]')
            sys.exit(1)

    signals = storage.get_signals_for_site(args.site)
    if not signals:
        print(f"No signals in DB for {args.site}. Run `poll` first.")
        sys.exit(1)

    print(f"Backtesting with {len(signals)} signals and {len(gt)} known launches")

    thresholds = None
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]

    results = run_backtest(
        signals, gt, site,
        match_window_hours=args.match_window,
        thresholds=thresholds,
    )
    print_backtest_report(results)


def cmd_alerts(args: argparse.Namespace):
    """Show recent alerts."""
    storage = Storage(db_path=Path(args.db) if args.db else None)
    alerts = storage.get_recent_alerts(
        site_id=args.site if args.site != "all" else None,
        limit=args.limit,
    )

    if not alerts:
        print("No recent alerts.")
        return

    for a in alerts:
        print(f"\n[{a['tier']}] {a['site_id']} — {a['created_at']}")
        print(f"  Window: {a['window_start']} to {a['window_end']}")
        print(f"  {a['explanation'][:200]}")
        refs = json.loads(a["signal_refs"]) if isinstance(a["signal_refs"], str) else a["signal_refs"]
        if refs:
            print(f"  Refs: {', '.join(refs[:3])}")


def cmd_add_ground_truth(args: argparse.Namespace):
    """Add a known launch to the ground truth table."""
    storage = Storage(db_path=Path(args.db) if args.db else None)

    if args.file:
        gt = load_ground_truth_file(Path(args.file))
        count = 0
        for entry in gt:
            rid = storage.store_ground_truth(
                site_id=entry.get("site_id", args.site),
                launch_date=entry["date"],
                vehicle=entry.get("vehicle", ""),
                mission=entry.get("mission", ""),
                outcome=entry.get("outcome", "success"),
                notes=entry.get("notes", ""),
            )
            if rid > 0:
                count += 1
        print(f"Added {count} ground truth entries.")
    else:
        if not args.date:
            print("Provide --date YYYY-MM-DD or --file <path>")
            sys.exit(1)
        storage.store_ground_truth(
            site_id=args.site,
            launch_date=args.date,
            vehicle=args.vehicle or "",
            mission=args.mission or "",
            outcome=args.outcome or "success",
        )
        print(f"Added ground truth: {args.site} {args.date}")


def cmd_sites(args: argparse.Namespace):
    """List configured sites."""
    for sid in list_sites():
        site = get_site(sid)
        print(f"  {sid}: {site.name} ({site.center_lat:.4f}, {site.center_lon:.4f})")
        print(f"    Restricted: {', '.join(site.restricted_areas)}")
        print(f"    Warning: {', '.join(site.warning_areas)}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="launch-detector",
        description="Detect upcoming launches from NASA Wallops and other spaceports",
    )
    parser.add_argument("--db", help="SQLite database path", default=None)
    sub = parser.add_subparsers(dest="command")

    p_poll = sub.add_parser("poll", help="Run all adapters once")
    p_poll.add_argument("--site", default="WFF", help="Site ID (default: WFF)")

    p_run = sub.add_parser("run", help="Daemon mode — poll on interval")
    p_run.add_argument("--site", default="WFF")
    p_run.add_argument("--interval", type=int, default=60, help="Minutes between polls")

    p_bt = sub.add_parser("backtest", help="Replay signals against ground truth")
    p_bt.add_argument("--site", default="WFF")
    p_bt.add_argument("--ground-truth", help="Path to ground truth JSON file")
    p_bt.add_argument("--thresholds", help="Comma-separated thresholds (e.g. 0.1,0.3,0.5)")
    p_bt.add_argument("--match-window", type=int, default=48, help="Hours to match signal to launch")

    p_alerts = sub.add_parser("alerts", help="Show recent alerts")
    p_alerts.add_argument("--site", default="all")
    p_alerts.add_argument("--limit", type=int, default=20)

    p_gt = sub.add_parser("add-ground-truth", help="Add known launch to ground truth")
    p_gt.add_argument("--site", default="WFF")
    p_gt.add_argument("--date", help="Launch date YYYY-MM-DD")
    p_gt.add_argument("--vehicle", default="")
    p_gt.add_argument("--mission", default="")
    p_gt.add_argument("--outcome", default="success")
    p_gt.add_argument("--file", help="Path to ground truth JSON file")

    p_sites = sub.add_parser("sites", help="List configured launch sites")

    args = parser.parse_args()

    commands = {
        "poll": cmd_poll,
        "run": cmd_run,
        "backtest": cmd_backtest,
        "alerts": cmd_alerts,
        "add-ground-truth": cmd_add_ground_truth,
        "sites": cmd_sites,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
