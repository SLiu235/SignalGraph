# SignalGraph: Multi-Agent Financial Intelligence with Knowledge Graphs and Bayesian Signaling Games

## Research question

> **How does the credibility and association with subsequent returns of a signal depend on who transmits it,
> who else has access to it, and what it costs them to send it?**

Classic signaling games model how an informed sender transmits a signal that a receiver
decodes under uncertainty. In financial markets, information is rarely a single firm's
public filing — it is produced and relayed by a population of players with heterogeneous
signal access and credibility: corporate insiders, sell-side analysts, competitors,
suppliers, regulators, journalists, activist funds, and the market itself.

This repository implements that setting as a knowledge graph whose nodes include **both
companies** (filings, events, fundamentals, price history) **and players** (insiders,
analysts, competitors, suppliers, regulators, journalists, funds), each with an edge
describing their **signal-access**: which signals they can observe, which they can
transmit, and at what cost. A receiver (the quant researcher) interprets transmitted
signals, updates beliefs via Bayes, and forms a cited decision brief; subsequent returns
are an observational outcome proxy. This system does not identify causal price impact
without a price-formation or causal-identification design.

This is a **research / decision-support artifact**, not a trading signal.

## Theoretical foundations

- **Spence (1973), job-market signaling.** Costly signals separate types: the sender
  bears a real cost that differs by type, so low-cost types send and high-cost types
  decline to mimic. We instantiate this as the **cost function** — informed senders bear
  lower marginal cost (they know the truth), so a costly signal (an insider's certified
  filing revision, a regulator's enforcement action, a fund's disclosed position)
  *separates* types and is credible.
- **Crawford & Sobel (1982), cheap talk.** This simplified implementation models
  near-zero-cost channels as **pooling**. Cheap-talk theory can also admit informative
  partition equilibria; here credibility rests on the sender's type and preference
  alignment (a low-cost analyst post from an aligned player differs from a costly insider
  revision).
- **Bayesian belief updating.** The receiver holds a prior over sender type and a prior
  confidence in the hypothesis ("this company will outperform"), and updates both by
  Bayes' rule as signals arrive.

## Architecture — four layers

1. **Knowledge graph (players + companies + signal-access).** A directed dynamic graph
   (networkx `MultiDiGraph`, serialized to JSON). Company nodes carry fundamentals,
   filings, events, price history; player nodes carry identity, role, credibility prior,
   and **signal-access edges** (`signal_access` player→signal_type) encoding
   `can_observe` / `can_transmit` / `transmit_cost`. Two storage backends behind the same
   ontology (`graph/graph_io.py` + `graph/storage.py`): the reproducible **JSON** snapshot
   (default, offline) and a live **Neo4j** backend (`graph/neo4j_backend.py`,
   `graph.storage=neo4j`). Incremental updates arrive via `graph/update.py`, which ingests
   new events from `data/ingest.py` (live SEC EDGAR or deterministic feeds) and applies
   them incrementally.
2. **Signal-extraction & synthesis.** `extraction/` retrieves the relevant subgraph
   (the company, the transmitting players, their signal-access edges) plus evidence
   passages (filings/news), and synthesizes a **cited decision brief** — the evidence,
   which players transmitted what, and how the signal should be interpreted.
3. **Signaling-game layer.** `signaling/` implements the sender–receiver game: sender
   types, the signal-cost function, receiver strategy, and Bayesian updating. Each
   signal's **credibility score derives from the game** (the receiver's posterior
   `P(H | signal)` via a noisy-signal Bayes update, with reliability fixed by the
   equilibrium regime), and the equilibrium regime (pooling vs separating) is
   characterized per signal type **using the sender's own signal-access transmit cost** —
   so the same costly type can separate for a high-cost channel and pool for a low-cost
   channel (Spence's lever, applied per sender).
4. **Multi-agent simulation layer.** `agents/` promotes the graph's players into **active
   agents** that analyze each other's strategies and make their own reactions, with
   heterogeneous **information access** (full / partial / limited), **response time**
   (fast / slow), **roles**, and **learning strategies** (Bayesian updating, Inverse
   Reinforcement Learning, fictitious play, heuristic). Each agent maintains an
   opponent model (inferred type + inferred payoff params + predicted action), reacts
   each round, and its reactions feed back as observable signals for others.

## Usage

```bash
# run from the repo root (the project is a flat package: graph/, signaling/, ...)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

make graph        # stage 1: build the knowledge graph (players + companies + signal-access)
make update       # stage 1b: incremental event ingestion (new filings/news/events -> graph)
make ingest       # ingest only: fetch + advance the cursor, without touching the graph
make signaling    # stage 2: sender-receiver game, equilibria, Bayesian belief update
make agents       # stage 2b: multi-agent simulation (agents analyze each other + react)
make extract      # stage 3: retrieve + cited decision brief per entity
make evaluate     # stage 4: no-lookahead benchmark + structural citation-coverage checks
make all          # run the full pipeline end-to-end
make test         # regression checks + agent-strategy unit tests
make research-demo # UTC research pipeline with a clearly labeled fabricated fixture
make research-fetch # real EDGAR primary documents; uses SEC_USER_AGENT and config/research.yaml
make research      # evaluate prepared real events + adjusted market bundle
make counterfactuals # paired synthetic interventions across seeds; no profitability claim
```

All parameters are in `config/config.yaml` (seeds, data mode, game costs, priors,
retrieval knobs, evaluation window, graph storage backend, ingest sources). Every stage
logs a shared `run_id` when invoked through `make`, so artifacts from one pipeline run
can be traced together; synthetic data is seeded so offline runs reproduce exactly.

## Historical research path

The UTC research pipeline is configured separately in `config/research.yaml` (`RCFG`
for Make targets). It uses real SEC primary documents and acceptance headers, immutable
raw-data hashes, explicit SEC event and local price-receipt clocks, adjusted close-price
ratios, benchmark returns, and Fama–French factors. Fama–French rows have no per-row
historical availability clock in the current adapter and remain ex-post risk-adjustment
inputs. The historical admission path never falls back to synthetic data.
`make research-demo` fabricates **all** of its inputs and is only an integration test.

For real research, set a genuine `SEC_USER_AGENT`, explicit reviewed ticker-to-CIK
mappings and the study dates, then run `make research-fetch`. Supply a validated market
bundle at `research.market_path` and run `make research`. The current historical price
admission path supports replayable Alpha Vantage adjusted prices and Fama–French daily
factors: both require content-addressed raw-response caches and parser-versioned
receipts, so the assembled price bars and factor/benchmark rows are reproduced exactly.
Alpha Vantage requires an entitled API key and a versioned, source-identified,
schema-validated exchange session schedule. Additional providers need their own
versioned receipt/replay adapter; a provider label alone is not admitted.
See [the data contract and runbook](docs/RESEARCH_DATA_CONTRACT.md) for schemas, commands,
prospective paper recording, provenance, and limitations.

Each run writes a new `results/research/<run_id>/` with observed-issuer `graph.json`,
versioned `events.json`, `evaluation.json`, and a replay manifest. The evaluator supplies
purged expanding chronological splits, an embargo, factor residuals, fees and slippage,
IC, hit rate, turnover, drawdown diagnostics, baselines, ablations, and sector/source/regime
slices. Paper decisions use a separately frozen, hash-chained local ledger with a
versioned, source-identified, schema-validated entry/exit schedule and price lineage.
There is no
broker integration or capital deployment. The legacy `make all` remains the synthetic
teaching pipeline; use `make research`, not `make all CFG=config/research.yaml`, for real data.

## Neo4j backend & incremental ingestion

The knowledge graph supports two storage backends behind the same ontology, selected by
`graph.storage`:

- **`json`** (default): networkx `MultiDiGraph` serialized deterministically to
  `graph.path` (`graph/graph_io.py`). Zero external infra, bit-for-bit reproducible.
- **`neo4j`**: live Neo4j graph via the `neo4j` driver (`graph/neo4j_backend.py`). Node
  label = entity kind, relationship type = relation, `id`/`eid` unique properties make
  writes idempotent MERGE upserts; complex attrs (fundamentals, price history) are
  JSON-encoded with a sentinel and decoded on read, so round-trips are faithful. A JSON
  snapshot of the same graph is always written alongside. Configure `graph.neo4j.{uri,
  user, password, database, encrypted}` (password may come from `NEO4J_PASSWORD`). If the
  DB is unreachable the backend raises a clear error — it never silently degrades.

**Real incremental event ingestion** (`graph/update.py` + `data/ingest.py`): each run
polls the configured ingest sources (`data.ingest.sources`) and applies **only new**
events. A per-source cursor (`data.ingest.cursor_path`, default
`results/ingest/cursor.json`) records how far ingestion has advanced — for the live
`edgar` source it tracks seen accession numbers + last filed date; the deterministic
`feed`/`news` sources emit a new batch per poll. Feed timestamps start at
`data.ingest.feed_start_day` (default 70), above `evaluation.brief_day` (40), so ingested
events stay strictly post-brief and never leak into existing decision briefs
(no-lookahead preserved). `--dry-run` fetches but does not persist or advance the cursor.

## Multi-agent simulation layer (`agents/`)

The static sender→receiver game is promoted into an **active multi-agent simulation**:
the graph's players become agents that **analyze each other's strategies** and **make
their own reactions** each round. All heterogeneity is configurable under `agents:`
(`agents/role_defaults`, `agents/payoff`, `agents/irl`):

- **Roles** — the seven graph roles (insider, analyst, competitor, supplier, regulator,
  journalist, fund) each get a strategy/access/response assignment and role priors.
- **Information access** — `full` (all simulated signals + actions for the same company), `partial` (own
  signal-access `can_observe` channels + same-company reactions), or `limited` (own
  channels only, no reactions).
- **Response time** — `fast` agents react in the current round; `slow` agents react on a
  one-round lag (they see only prior-round signals).
- **Learning strategies** (`agents.role_defaults.<role>.strategy`):
  - `bayes` — Bayesian posterior over opponent type (H/L) and over the market hypothesis;
    reacts on the posterior sign.
  - `irl` — **Inverse Reinforcement Learning**: each agent infers its opponents' payoff
    (`theta_truth` / `theta_contrarian` / `theta_stay`) by maximum-likelihood over their
    observed (belief, action) pairs under a logit choice model, then predicts their next
    action and best-responds.
  - `fictitious_play` — empirical-frequency best response to opponents' action
    distributions (classic game-theory learning).
  - `heuristic` — role-prior baseline (no learning) for contrast.

Each agent keeps a per-opponent **strategy model** (inferred type, inferred payoff
params, predicted action, confidence, and a plain-language "read"), and its actions
feed back as observable signals for the next round. The opponent generative model is a
logit choice: `utility(a|belief) = θt·[a==sign(b)] + θc·[a==-sign(b)] + θs·[a==0]`,
`P(a) = softmax(U/temperature)`. A strategy produces the public belief context and the
latent payoff generates the realized action, so IRL has a coherent data-generating model.
`theta_error` is finite-sample/grid parameter-recovery error; it is expected to be nonzero
but should be analyzed against simulation baselines, not treated as a trading metric.

## What the default run produces

With the synthetic default (12 companies × 7 players, deterministic seed), `make all`
produces, under `results/`:

- `graph/` — the serialized graph + a build manifest (node/edge counts by kind/relation).
- `signaling/` — per-signal **credibility** scores and the **equilibrium characterization**
  (costly `certified_filing`/`disclosure` → *separating* for high-cost channels;
  cheap-talk `commentary`/`media` → *pooling*; a few low-cost costly senders pool too),
  plus per-company **posterior confidence** trajectories that update only on signals
  observed at or before the brief time (no lookahead).
- `agents/` — the multi-agent simulation: `agents.json` (roster: roles, strategies,
  info-access/response tiers, latent payoff), `simulation.json` (per-company round-by-round
  observations + reactions), `learning.json` (per-agent strategy state: market posterior,
  opponent models), `strategy_analysis.json` (the **"analyze each other's strategies"**
  deliverable — per agent, per opponent: inferred type, inferred payoff, predicted action,
  confidence, read), and `comparison.json` (IRL-fidelity and reaction-rate aggregates by
  strategy / info-access / response-time tier).
- `briefs/` — one **cited decision brief** per company (evidence + citations, transmitting
  players + signal-access, credibility scores, posterior confidence, research direction).
- `evaluation/` — the **no-lookahead benchmark** (brief at day t scored against outcomes
  strictly after t) and per-metric structural citation-coverage / uncited-claim summaries
  (semantic entailment is not evaluated). The
  synthetic outcome is coupled to a latent per-company direction that its costly signals
  observe, so the benchmark is meaningful: default direction accuracy ≈ 0.750 and
  directional-confidence-weighted accuracy ≈ 0.760, both well above the 0.5 random baseline
  (and not trivially 1.0 — a credible signal that disagrees with the true outcome is a real
  miss).

## Limitations

- Default data is **synthetic and deterministic**. The historical research collector
  (`data/edgar_documents.py`) requires an identified SEC user agent and fails on
  missing timestamps/documents. The old incremental EDGAR metadata adapter remains
  a separate legacy route; it is not input to the new empirical evaluation.
- The new research path is implemented and tested offline, but an actual SEC-plus-price
  study still requires live provider verification and sourced historical stock data.
  Historical acceptance plus latency is an explicit availability assumption. Current
  adjusted-price and factor snapshots can be revised. Their receipt replay makes the
  local input reproducible, not historically available; they support ex-post return
  analysis but do not establish live point-in-time market availability. A price bar's
  `available_at` is its recorded local response-receipt clock, not vendor dissemination
  time. Momentum requires every price bar in its lookback to have that recorded
  availability clock; regime requires an explicit benchmark availability clock. Neither
  clock is independently authenticated vendor dissemination timing in the current
  adapters. Preserve the raw cache with the bundle: current receipts
  contain absolute local paths and are not portable by themselves.
- Filing credibility/cost are configured assumptions, text direction is a simple word
  count, and EDGAR alone does not observe propagation. The propagation feature remains
  unavailable/zero; its ablation cannot demonstrate value on a single-source dataset.
- The synthetic price outcome is coupled to a latent per-company direction (`data.signal_outcome_agree`,
  `data.outcome_drift`) so the decision benchmark measures signal-reading, not noise; accuracy is
  bounded by that synthetic signal-to-outcome reliability.
- LLM synthesis is currently a **mock (rule-based) backend**. Selecting a non-mock
  backend fails fast rather than silently substituting mock output; a production LLM
  adapter, evaluation harness, and secret management remain future work.
- The graph backend defaults to **networkx-in-JSON** (reproducible, no external DB); a
  **Neo4j** backend (`graph.storage=neo4j`) is implemented behind the same ontology but is
  opt-in and requires a running Neo4j instance.
- Sentiment direction for the Bayes update is a crude deterministic word heuristic —
  a research placeholder, not a trained NLP model.
- The agent layer is a **synthetic, deterministic simulation** of strategy analysis and
  actions. The latent payoff generates actions through a logit policy, while the learning
  strategy supplies its public belief context; `theta_error` therefore measures finite
  sample/grid recovery error. Learning is one-shot per run (no cross-run accumulation or
  real-time adaptation).

## What I'd do next

- Wire a real LLM synthesis backend and evaluate citation precision against a gold set.
- Calibrate the cost function / priors against observed price reactions.
- Add a semi-separating equilibrium regime and comparative-static sensitivity analysis.
- Verify the new document collector against live SEC responses, add exhibits and
  accession-specific XBRL fact reconstruction, and add a licensed news/transcript source.
- Run a preregistered historical study with delisted securities and point-in-time
  membership, then collect a prospective paper record before any deployment decision.
- Extend the agent layer: continuous (non-grid) IRL via gradient ascent, cross-company
  opponent modeling (agents learn about rivals across tickers), and a real-time adaptive
  response tier where fast agents' reactions move the simulated market.
