# UTC research data contract and operator runbook

Implemented and integrity-hardened October 28, 2026. This path is separate from the synthetic `make all`
demo. A fixture run proves integration; a historical run measures the supplied sample.
Neither creates broker orders.

## Commands

Run from the repository root with the existing environment:

```bash
make test
make research-demo
make counterfactuals
```

`research-demo` creates four fictional issuers, a fabricated weekday calendar, 260
sessions, and 128 filing-shaped events. It does not use any live prices or filings.
Outputs live under `results/research/<run_id>/`; a repeated run ID fails rather than
overwriting a previous run. Input files and evaluation source are hashed in the manifest.
Set a Make run ID with:

```bash
make RUN_ID=study-2026-10-27 research-demo
```

For a direct Python invocation, use:

```bash
SIGNAL_GRAPH_RUN_ID=study-2026-10-27 .venv/bin/python -m research.pipeline --demo
```

Direct Python commands do not read `RUN_ID`.

For a historical study, copy/edit `config/research.yaml`, set explicit ticker/CIK mappings and
inclusive filing-date bounds, and supply your real name/contact as `SEC_USER_AGENT`.
An empty contact is rejected. Explicit CIKs are mandatory: the collector deliberately
does not fall back to today's SEC ticker index, because renamed/reused symbols can bind
the wrong historical issuer. The configured mapping remains a reviewed security-master
assertion, not historical proof of the price instrument's identity; effective-dated
permanent instrument IDs remain a required production extension. Choose a new
`research.events_path` for each collection snapshot.

```bash
make research-fetch RCFG=config/research.yaml
.venv/bin/python -m data.market_data validate results/research_inputs/market.json
make research RCFG=config/research.yaml
```

Initial historical graph construction is also supported:

```bash
.venv/bin/python -m graph.build_graph --config config/research.yaml
```

That graph contains observed issuer filings plus the minimum shared-ontology structure
needed to represent them: an `issuer` player, a `certified_filing` signal type, and a
configured filing access/transmission relation. The issuer cost is an explicit modeling
assumption, not an observed EDGAR field. It does not create the synthetic multi-role
population or synthetic market outcomes. The legacy signaling/agents/brief CLI sequence
still expects the synthetic configuration; do not use `make all` with the historical
profile. The new evaluator consumes `research.features` directly.

## EDGAR event envelope

`research.events_path` is JSON with `metadata.data_kind` equal to `historical` or
`fixture`, an explicit availability mode, and an `events` list. Required event fields:

| Field | Meaning |
| --- | --- |
| `event_id` | Stable issuer/accession identity; duplicates rejected |
| `ticker`, `cik`, `accession`, `form`, `primary_document`, `source` | Issuer/accession identity; amendments remain distinct |
| `event_time` | Acceptance instant from the SEC SGML header, normalized to UTC |
| `available_at` | Earliest permitted use under the selected availability policy |
| `retrieved_at` | Latest collection time needed to reconstruct this event |
| `availability_mode` | `historical` or `live` |
| `document_text` | Actual extracted primary HTML/text content |
| `document_url`, `document_sha256` | Original SEC URL and SHA256 of original bytes |
| `provenance` | Immutable source receipts, submission/header identity, parser version, API/header clock comparison and latency |

UTC serialization uses full ISO timestamps such as `2024-11-01T10:01:36Z`.
The parser rejects date-only and naive timestamps. SEC header acceptance wall clocks
are interpreted in `America/New_York` with DST validation; API clocks are retained for
comparison. Missing acceptance headers fail rather than becoming midnight events.

Historical availability is acceptance plus a declared ingestion delay. This is a
reconstruction assumption, not proof of when a live system received the document.
Live availability is the maximum of that assumed publication time and the metadata,
header, and document retrieval clocks. Thus a document retrieved today cannot be
entered into yesterday's prospective paper record.

The collector discovers recent and archived submission pages, filters the date/form
range, and downloads each primary document plus `.hdr.sgml`. Original bytes and
retrieval receipts are immutable content-addressed cache objects. HTTP operations
have timeouts, bounded retries/response sizes, disabled redirects and per-process
pacing. Multiple concurrent collectors still require a shared aggregate rate limiter.

Replay treats the normalized event envelope as untrusted. Each receipt must match the
exact content-addressed receipt sidecar, object hash, canonical SEC URL, current
submissions CIK, selected submission row, primary-document filename, and SGML header
accession/form/filer-CIK. An archived submission file must still be named by the saved
captured current submissions index. An explicit ticker/CIK mapping is a configured
security-master label—not evidence that today's SEC ticker index proves a historical
ticker mapping. CIK is the durable issuer identity in the filing record; the current
ticker join is a reviewed configuration assertion, not an instrument-level join. A
versioned security master must map issuers to permanent price-instrument IDs with
effective intervals. The writable local cache remains a trust boundary: a production
audit trail should additionally seal/sign input manifests or use append-only storage
outside the collector's writer.

The only admitted historical event adapter today is `sec_edgar`. The supported
historical admission path (`load_inputs` / `make research`) and direct historical
evaluation with its required source configuration replay every SEC receipt before
features/evaluation proceed. Historical envelope metadata and every event must declare
`sec_edgar`. Fixture sources must remain explicitly fixture/fabricated.
Adding news, transcripts, XBRL, or another filing provider requires a named adapter and
an equally strict source-byte replay verifier; relabeling JSON is intentionally blocked.

Primary HTML/text is supported. PDF primaries, exhibits, press-release attachments,
and accession-specific XBRL fact vintages are not yet supported. Many earnings releases
put substantive figures in an exhibit; a primary 8-K may therefore contain very little
directional content. There is no fallback to generated summaries or synthetic filings.

## Market bundle

`research.market_path` is one JSON object. All returns are decimals: 1% = `0.01`.
`metadata`, `bars`, `benchmark`, and `factors` are required. Historical evaluation and
prospective paper recording additionally require a versioned, source-identified,
schema-validated `session_schedule`; date-only benchmark rows cannot establish whether
an expected close has elapsed.
`sector_returns` and `memberships` are optional controls; the full example below
includes both.

```json
{
  "metadata": {
    "data_kind": "historical",
    "source": "Your provider and export version",
    "adjustment": "split_and_dividend",
    "universe_policy": "Explicit historical selection and delisting coverage",
    "retrieved_at": "2026-10-26T00:00:00Z",
    "price_lineage": [{
      "id": "alpha_vantage:TIME_SERIES_DAILY_ADJUSTED:AAPL",
      "source": "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
      "identity": {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": "AAPL"}
    }]
  },
  "session_schedule": {
    "version": 1,
    "calendar_id": "XNYS-example-vintage",
    "source": "Exchange calendar provider and version",
    "retrieved_at": "2024-01-01T00:00:00Z",
    "sessions": [{"session": "2024-01-02", "open_at": "2024-01-02T14:30:00Z",
                  "close_at": "2024-01-02T21:00:00Z"}]
  },
  "bars": [
    {"ticker": "AAPL", "session": "2024-01-02",
     "open_at": "2024-01-02T14:30:00Z", "close_at": "2024-01-02T21:00:00Z",
     "available_at": "2024-01-02T21:01:00Z", "adjusted_close": 100.0}
  ],
  "benchmark": [{"session": "2024-01-02", "return": 0.001}],
  "factors": [{"session": "2024-01-02", "mkt_rf": 0.0009,
               "smb": 0.001, "hml": -0.001, "rf": 0.0001}],
  "sector_returns": [{"session": "2024-01-02", "sector": "Technology", "return": 0.001}],
  "memberships": [{"ticker": "AAPL", "sector": "Technology",
                   "valid_from": "2020-01-01T00:00:00Z", "valid_to": null,
                   "available_at": "2020-01-01T00:00:00Z"}]
}
```

These numbers illustrate the schema; they are not historical AAPL observations and are
**not** a replay-admissible historical bundle. A real historical bundle assembled by
`data.market_data assemble` additionally contains `component_metadata`, `factor_lineage`,
and the raw receipt/sidecar references required to replay supported price and factor
inputs. A usable study needs sufficient estimation sessions, event windows, and distinct test
dates. Entries must be unique and chronologically ordered per series. Prices must be
positive/finite, all UTC session clocks explicit, and company sector assignments
available at the decision time. Missing classification becomes `unknown`, never the
current sector attached retroactively. Overlapping membership intervals are rejected;
full bitemporal membership revisions are not modeled.

Use a provider or exchange calendar for open/close timestamps, including holidays,
early closes and DST. The importer never assumes every weekday is a trading day. A
schedule may include announced future sessions before their bars, factors, or benchmark
returns exist; observed rows must match a schedule session and bar clocks must match it
exactly. This lets a paper decision freeze a future entry/exit plan without inventing
future prices. The calendar ID and source should identify a versioned schedule snapshot.
Supply delisted instruments and delisting outcomes where your historical universe
requires them. The validator can check a declaration, not certify vendor adjustment,
licensing, ticker lineage, security-class identity, or survivorship completeness. A
production security master must join price instruments to CIKs with permanent IDs and
effective intervals; a ticker alone is insufficient for rename/reuse/class-change history.

The optional factor fetcher produces a fragment:

```bash
.venv/bin/python -m data.market_data fetch-factors \
  --cache-dir results/raw/factors \
  --output results/research_inputs/factors-new-snapshot.json
```

Fama–French daily percentages are converted to decimals. The default market benchmark
is `mkt_rf + rf`, not SPY. The adapter stores the raw ZIP/CSV in a content-addressed
object plus a parser-versioned receipt; historical admission reparses it and requires
the complete factor and derived benchmark rows to match exactly. Factor files downloaded
today can contain revisions; this proves local raw-byte-to-row reproducibility, not the
original publication time or vintage. These factors are used for ex-post risk adjustment,
not as point-in-time predictive features.

The optional Alpha Vantage adapter uses `ALPHAVANTAGE_API_KEY` and its daily-adjusted
endpoint, which the provider marks premium. Supply a JSON session list containing
`session`, `open_at`, and `close_at`, or the `session_schedule` object shown above;
no session timestamps are guessed. The same versioned schedule object can be used for
both price selection and assembly:

```bash
.venv/bin/python -m data.market_data fetch-prices --ticker AAPL \
  --sessions results/research_inputs/exchange-session-schedule.json \
  --cache-dir results/raw/prices \
  --output results/research_inputs/aapl-prices.json

.venv/bin/python -m data.market_data assemble \
  --prices results/research_inputs/aapl-prices.json \
  --factors results/research_inputs/factors-new-snapshot.json \
  --session-schedule results/research_inputs/exchange-session-schedule.json \
  --universe-policy 'Selected historical issuers; document delisting coverage' \
  --output results/research_inputs/market.json
```

Repeat `--prices` for multiple issuers. The Alpha Vantage adapter stores a
credential-free request identity, canonical selected sessions, raw response bytes, and
an exact receipt sidecar under the content-addressed `--cache-dir`; it intentionally
omits the key-bearing request URL and API key from receipt fields. Provider responses
are retained verbatim, so treat the raw cache as potentially sensitive. Its fragment
records the adapter/parser version and a per-ticker
`price_lineage_id`. During assembly and later historical admission, the raw response is
reparsed with the saved sessions and the exact union of replayed bars—including clocks,
adjusted closes, volume, and `available_at`—must equal the assembled bars. A changed,
omitted, added, or unowned bar fails closed.

Historical price admission currently supports this replayable Alpha Vantage adapter.
Another vendor must receive its own versioned parser and receipt-replay implementation;
plain provider-labelled JSON is rejected for a historical run. Optional `--memberships`
and `--sector-returns` accept JSON lists matching the schema above. Assembly retains
component source metadata and exact file hashes, validates the combined bundle, and
marks a bundle as a fixture when a price or factor fragment is fixture-labelled.
Historical admission replays the
supported Alpha Vantage price components and the Fama–French factor component. It
separately schema-validates and records schedule, membership, and sector-control inputs;
their upstream raw artifacts are not yet independently replayed or authenticated. Output
paths are exclusive; choose a new filename for each refreshed snapshot.

Historical price or Fama–French fragments created before the receipt schema are not
grandfathered in: re-fetch the raw provider artifact and reassemble the bundle. This
prevents a legacy hash label from being mistaken for a replay-verifiable source record.

Preserve or export the referenced raw cache with a market bundle. Current price/factor
receipts use absolute local paths to objects, selected sessions, and receipt sidecars, so
replay is same-cache/host reproducibility until a portable artifact store or URI mapping
is added.

Use only ratios of adjusted prices for returns. Current back-adjusted price levels
can reflect later splits/dividends, so they must not be treated as prices known at a
historical decision time. No adjusted close is claimed to be an executable quote.
The adapter sets each bar's `available_at` to its recorded local response-receipt time.
A response collected today is valid outcome evidence but is not historical availability
evidence for momentum. Price-based controls therefore use only bars whose recorded local
receipt clock is no later than the event; this is not vendor publication timing.
Benchmark-regime slices likewise require explicit benchmark `available_at` values.
Fama–French files lack that predictive availability clock here and remain ex-post
risk-adjustment inputs.

## Feature and evaluation contract

`research.features` discards imported scores and regenerates all features under
`filing-rules-v1`. Features and code hashes are stored per run. Credibility and filing
cost use declared signaling assumptions. Direction uses word counts. Novelty compares
with strictly earlier documents from that issuer; simultaneous documents are peers.
Earnings identification requires 8-K Item 2.02 text, not just the form name.
Propagation is explicitly unavailable/zero for the single-source EDGAR collector.

The fixed score is direction times a weighted sum of credibility, bounded cost,
novelty, and propagation. It is a research exposure/ranking hypothesis, not a fitted
or calibrated return probability. Constant costs/credibility and absent propagation
cannot identify the economic contribution of those features on this dataset.
When SEC event snapshots are loaded, the primary text and acceptance/availability clocks
are regenerated from raw cache objects and checked against the saved event, so editing
normalized text cannot silently retain an authentic but unrelated source hash.
Direct historical calls to `evaluation.walk_forward.evaluate_research` additionally
recompute the derived feature fields from replay-verified SEC records and the supplied
config. A caller cannot substitute a favorable direction, novelty, sector, sender,
feature version or availability clock while retaining the raw source receipts.

Entry uses the first **scheduled** close strictly after evidence availability plus the
configured execution delay. A horizon is a count of scheduled sessions after that
entry. The evaluator then requires observed benchmark/factor rows and price bars for
the frozen window; missing observations are excluded with reasons, not filled. OLS fits
stock excess returns on an intercept plus market, size and value factors using only
pre-entry observations. The output subtracts compounded expected returns from actual
compounded returns; it is a factor residual, not an estimated strategy alpha intercept.

Scores stay frozen. Expanding chronological train/test groups define evaluation and
future model-training boundaries; no return predictor is currently trained on the
train set. Training labels overlapping the test boundary are purged, with an additional
session embargo. Identical-time releases remain in one group.

Metrics include net event return, factor residual, rank IC, directional hit rate,
costs, turnover, date-cluster intervals and fixed-notional event-sleeve drawdown.
Sleeves can overlap and do not enforce a capital/borrow constraint. Their configured
capital denominator is a diagnostic, not an investable fund NAV. Intervals do not
correct cross-date serial correlation, overlapping horizons or multiple comparisons.

Baselines include market/sector, prior momentum, an explicitly identified earnings
filter and deterministic random scores. Ablations remove one feature on the same
test sample. Results are sliced by sector, source, trailing-market regime and horizon.
Missing controls and sparse groups must remain visible. A regime is explicitly
`unavailable_without_point_in_time_benchmark` when its benchmark history lacks explicit
availability clocks; the provenance of any supplied clock still needs independent
validation. It is not silently treated as an observed market state. Historical complete-case
exclusions can themselves bias results; review exclusion counts before interpretation.

## Prospective paper record

Use a fresh config/input path with EDGAR `availability_mode: live`, real market inputs,
and a current collection window. Record decisions before their outcome closes:

```bash
.venv/bin/python -m research.paper record --config config/research.yaml \
  --ledger results/paper/decisions.json

.venv/bin/python -m research.paper mark \
  --ledger results/paper/decisions.json \
  --market results/research_inputs/market-later.json \
  --horizon 5 --output results/paper/marks-new-snapshot.json
```

The recording CLI takes the actual UTC clock and rejects fixture inputs, events with
reconstructed `availability_mode: historical`, future evidence, and a market/calendar
snapshot retrieved after the decision. It requires replay-verified historical price and
factor artifacts for the market bundle. It also rejects a late first fetch of an old
filing: `max_filing_age_seconds` is measured from the SEC acceptance `event_time` to
the actual recording clock, not from the later live-mode `available_at`. The default
and sample configuration are 259,200 seconds (72 hours). This is an explicit research
freshness policy, not evidence of dissemination latency. Ledger schema v4 freezes
the acceptance timestamp and freshness threshold alongside each event's score,
evidence hash, planned horizons, execution delay, costs, calendar identity/source,
replayed price lineage, entry-to-max-horizon session window, and window hash in a hash
chain. Updates use a POSIX advisory writer lock and atomic replacement.
The ledger detects accidental edits; it is not an externally trusted timestamp or
tamper-proof audit service. Existing v3 local ledgers do not satisfy the v4 schema;
preserve them as historical artifacts and start a new v4 paper ledger.

Marking uses only the ledger's frozen scheduled entry/exit sessions and a later market
snapshot retrieved no later than the requested mark cutoff. It requires the same price
source, adjustment convention, universe policy, and per-ticker replayed price lineage.
An extended later schedule is accepted only when every frozen session clock is unchanged.
The mark remains pending before its scheduled entry close and fails closed if a scheduled
entry/interim/exit bar is missing after its close. Price lineage—not factor lineage—is
frozen across marks because factor rows are not paper-fill evidence; each supplied bundle's
factor artifact is nevertheless replay-verified. A new open mark does not require future
factor rows. Fees/slippage are charged at entry and exit.
Marks are hypothetical, not broker fills; borrow, financing, impact, capacity and order
rejection remain unmodeled. A later provider response can revise an earlier adjusted
price under the same price-lineage identity. Before treating paper performance as
audit-grade evidence, freeze each first observed entry/mark price with its raw receipt
in an independent append-only store and reconcile subsequent provider revisions.

## Promotion rule and evidence boundary

No automatic route turns a successful backtest into a deployment. The research result
records evidence checks and remains `deployable: false` until independent prospective
paper evidence, a credible mechanism, model review and drift monitoring exist.
The current code has no broker connection. Actual elapsed paper performance cannot be
manufactured by replaying a historical bundle.

## Primary references

- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [SEC developer and fair-access guidance](https://www.sec.gov/about/developer-resources)
- [Fama–French factor definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
- [Alpha Vantage daily-adjusted API](https://www.alphavantage.co/documentation/#dailyadj)
