# Wallops Launch Detector

Detects upcoming rocket launches from NASA Wallops Flight Facility (WFF) by
correlating signals from multiple public data sources — including launches
that are not publicly announced.

Built site-agnostic: adding another US spaceport is a config entry, not a code change.

## How It Works

Launches require regulatory filings days/weeks before the event:

1. **FCC STA** — Commercial launch operators file Special Temporary Authority
   requests with the FCC ≥10 days before launch for telemetry/tracking comms.
   These include the launch site coordinates and azimuth.

2. **FAA NOTAMs** — Airspace restrictions (R-6604, W-386) are activated before
   range operations. Noisy on their own, but corroborating with FCC data is
   high signal.

3. **Maritime warnings** (v2) — NGA/USCG hazard areas for downrange safety.

4. **AIS vessel clearing** (v2) — Commercial shipping cleared from downrange.

The engine correlates overlapping signals, scores them against per-site
weights, and emits tiered alerts: **WATCH → LIKELY → IMMINENT**.

## Quick Start

```bash
# Install
pip install -r launch_detector/requirements.txt

# List configured sites
python -m launch_detector sites

# Poll all sources once (FCC ELS works without credentials)
python -m launch_detector poll --site WFF

# Daemon mode (polls every 60 minutes)
python -m launch_detector run --site WFF --interval 60

# Show recent alerts
python -m launch_detector alerts
```

## API Credentials

### FAA NOTAM API (recommended but optional)

1. Register at https://api.faa.gov/s/
2. Set environment variables:
   ```bash
   export FAA_CLIENT_ID=your_id
   export FAA_CLIENT_SECRET=your_secret
   ```

### Push Notifications (ntfy)

```bash
export NTFY_TOPIC=wallops-launches
```

Alerts are sent to `https://ntfy.sh/wallops-launches`. Subscribe in the
ntfy app or at that URL.

## Backtest

Replay stored signals against known past launches to measure accuracy:

```bash
# Add ground truth
python -m launch_detector add-ground-truth --file ground_truth_wff.json

# Run backtest with various thresholds
python -m launch_detector backtest --site WFF --ground-truth ground_truth_wff.json
```

Ground truth format:
```json
[
    {"site_id": "WFF", "date": "2024-06-15", "vehicle": "Electron", "mission": "PREFIRE-2"},
    {"site_id": "WFF", "date": "2025-03-20", "vehicle": "Electron", "mission": "Live and Let Fly"}
]
```

Output: confusion matrix per threshold, precision/recall/F1 for tuning.

## Adding a New Site

Edit `launch_detector/site_config.py` — add a new `SiteConfig` and call
`register_site()`. No engine changes needed. Example for Cape Canaveral:

```python
CCAFS = SiteConfig(
    site_id="CCAFS",
    name="Cape Canaveral Space Force Station",
    center_lat=28.4889,
    center_lon=-80.5778,
    restricted_areas=["R-2932", "R-2931"],
    warning_areas=["W-497A", "W-497B"],
    scoring=ScoringPolicy(
        weights={...},  # Tune for this site
    ),
)
register_site(CCAFS)
```

## Architecture

```
adapters/           One per data source, normalizes to SignalEvent
  fcc_els.py        FCC Experimental Licensing System (scraping)
  faa_notam.py      FAA NOTAM API (REST + OAuth2)
  stubs.py          NGA, USCG, AIS — interface only, v2

models.py           SignalEvent, CorrelationGroup, Alert
site_config.py      SiteConfig registry (WFF is first entry)
engine.py           Correlate → Score → Classify → Alert
storage.py          SQLite — all signals stored for audit + replay
alerting.py         ntfy, webhook, log backends
backtest/harness.py Replay + confusion matrix
cli.py              poll | run | backtest | alerts | sites
```

## Known Limitations

- **Government launches invisible to FCC**: NASA/DoD launches using
  government-operated RF are exempt from FCC licensing. Only commercial
  operators (Rocket Lab, SpaceX, Firefly, etc.) file STAs.
- **R-6604 is noisy**: The restricted area activates for non-launch range
  operations. NOTAM signals need corroboration.
- **FCC ELS has no API**: We scrape server-rendered pages with rate limiting.
  The FCC may change page structure without notice.
- **FAA NOTAM API requires registration**: Free but manual approval.

See `docs/datasources.md` for detailed source verification notes.

## Tests

```bash
python -m pytest launch_detector/tests/ -v
```

62 tests covering: FCC STA parsing (including Rocket Lab WFF validation),
correlation engine, scoring, tier classification, window narrowing,
SQLite storage, and backtest harness.
