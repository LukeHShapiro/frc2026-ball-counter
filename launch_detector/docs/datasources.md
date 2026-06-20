# Data Source Access Methods — Verified 2026-06-20

## A. FCC Experimental Licensing System (ELS) — IMPLEMENTED

### Access Method: Web Scraping (no public API)

The FCC ELS has **no documented REST API** for programmatic search.
Access is via the ColdFusion web application at:
- Search form: `https://apps.fcc.gov/oetcf/els/reports/GenericSearch.cfm`
- STA detail pages: `https://apps.fcc.gov/oetcf/els/reports/STA_Print.cfm?mode=current&application_seq={id}`
- STA attachments: `https://apps.fcc.gov/els/GetAtt.html?id={attachment_id}&x=`

**Verified search approach:**
The GenericSearch.cfm page accepts POST form submissions with fields including
applicant name, file number, call sign, and state. The exact POST field names
must be reverse-engineered from the form HTML (the FCC does not document them).
Results are returned as HTML tables.

**Third-party mirror:** fcc.report (https://fcc.report/ELS/) mirrors ELS data
in a cleaner format with predictable URL patterns
(e.g., `/ELS/Rocket-Lab-USA-Inc`). However, fcc.report returns 403 to
automated fetchers — it requires browser-like headers or may block scrapers.

**Our approach:**
1. Primary: Scrape `GenericSearch.cfm` and `STA_Print.cfm` pages directly.
   These are server-rendered ColdFusion pages (no JavaScript required).
   FCC sites block many automated user agents — we use a browser-like
   User-Agent header and respect rate limits.
2. Parse STA grant text for: dates, frequencies, coordinates, azimuth,
   applicant name.
3. Match coordinates against site config to associate STAs with launch sites.

**Known limitation:** US government launches using government-operated RF
(e.g., NASA, DoD) are exempt from FCC licensing and will NOT appear in ELS.
Only commercial operators (Rocket Lab, SpaceX, Firefly, etc.) file STAs.

**Rate limits:** No documented rate limit, but we self-impose 2-second delays
between requests. FCC may block aggressive crawling.

**Reference STA (validated):**
- File: 0659-EX-ST-2025, Call Sign: WZ9XOM (Rocket Lab, Wallops Island)
- Location: ~126° azimuth from true north, within 10 km, centered NL 37-49-55
- URL: `STA_Print.cfm?mode=current&application_seq=146495`

---

## B. FAA NOTAM API — IMPLEMENTED

### Access Method: REST API with OAuth2 client credentials

The FAA provides a NOTAM API at:
- Endpoint: `https://external-api.faa.gov/notamapi/v1/notams`
- Auth: OAuth2 client credentials (client_id + client_secret)
- Registration: https://api.faa.gov/s/ (free, manual approval)

**Query parameters (verified from multiple sources):**
- `locationLatitude`, `locationLongitude`, `locationRadius` (NM, max 100)
- `domesticLocation` (e.g., location identifier)
- `effectiveStartDate`, `effectiveEndDate`
- `notamType` (N=New, R=Replaced, C=Canceled)
- `classification`
- `pageNum`, `pageSize` (max 1000)
- `sortBy`, `sortOrder`
- `responseFormat`

**Our approach:**
1. Query by lat/lon radius around WFF center (37.8319, -75.4877), radius 30 NM
2. Filter results for restricted area references: R-6604A/B/C/D/E, W-386
3. Parse NOTAM text for activation times, altitude restrictions
4. Emit `airspace_activation` or `warning_area_hot` SignalEvents

**Known limitation:** R-6604 activations are NOISY — the range activates for
sounding rockets, drone tests, and non-launch range operations. Airspace
activation alone is a weak signal; it needs corroboration from other sources
(FCC STA, maritime warnings) to indicate a launch.

**Auth requirement:** User must register at api.faa.gov and set
`FAA_CLIENT_ID` and `FAA_CLIENT_SECRET` environment variables (or in config).

---

## C. NGA NAVAREA IV / HYDROLANT — STUB (v2)

### Access Method: UNVERIFIED — needs manual research

NGA publishes maritime safety broadcasts including HYDROLANT warnings for the
western Atlantic. These include downrange hazard areas for launches.

**Possible access points (not yet verified):**
- NGA Maritime Safety Information: https://msi.nga.mil/
- NAVAREA IV coordinator broadcasts
- Possibly available via NavTex or SafetyNET

**Status:** Stubbed. Adapter interface defined, no fetch logic implemented.

---

## D. USCG Local Notice to Mariners (LNM) — STUB (v2)

### Access Method: UNVERIFIED — weekly PDF publication

USCG District 5 (Portsmouth, VA) publishes weekly LNM PDFs that may include
temporary hazard areas for launches from Wallops.

**Possible access:**
- https://www.navcen.uscg.gov/ — Local Notice to Mariners section
- District 5 LNM archives

**Status:** Stubbed. Would require PDF download + pdfplumber parsing.

---

## E. AIS (Automatic Identification System) — STUB (v2)

### Access Method: UNVERIFIED

Vessel traffic near WFF downrange areas — clearing of commercial shipping
before launch. Would require AIS data feed (e.g., MarineTraffic API,
AISHub, or VesselFinder API — all commercial, paid).

**Status:** Stubbed. Would detect vessel clearing pattern in downrange box.

---

## Assumptions

1. FCC ELS pages are server-rendered HTML (ColdFusion) — no JS rendering needed.
2. FAA NOTAM API requires OAuth2 client credentials, not API key.
3. Government/military launches will NOT have FCC STA signals.
4. Maritime and NGA signals are v2 scope — stubbed only.
5. All adapters self-impose rate limiting to be good citizens.
