> ORIGINAL PROJECT NARRATIVE — retained for context. These claims predate the September 7, 2026 review. Use the reviewed README, HIRING_REVIEW.md and validation/validation.json for current status.

# PROJECT_SUMMARY — Handoff document for quant_signal_game

This document is the canonical handoff for a fresh researcher or LLM to continue the
project without asking questions. It covers the research framing, the exact methodology
and code layout, the engineering decisions (and why), the default results and how to read
them, known limitations, and the precise state of every module.

---

## 1. Research question and goal

**RQ:** How does the credibility and price impact of a signal depend on who transmits it,
who else has access to it, and what it costs them to send it?

We build a reproducible, evaluable research artifact that models financial information as
a **signaling game** over a **knowledge graph** whose nodes include both companies and the
players who can observe or transmit signals. The output per entity is a **decision brief**:
the retrieved evidence, the transmitting players and their signal-access, the
signal-credibility scores, the Bayesian posterior confidence, and a recommended research
direction. This is a research/decision-support artifact, **not a trading signal**.

Theoretical anchors: Spence (1973) costly-signal separation; Crawford & Sobel (1982)
cheap talk with preference divergence; Sobel (2007) equilibria of signaling games; Bayes
posterior updating by the receiver.

## 2. Repository structure and how each module fits the three-layer design

```
quant_signal_game/
├── config/config.yaml          # all tunable parameters (seeds, costs, priors, modes)
├── common.py                   # config loading, deterministic RNG, logging, JSON IO
├── graph/                      # LAYER 1: knowledge graph
│   ├── ontology.py             #   entities + relations + signal-access model
│   ├── graph_io.py             #   deterministic JSON serialization of the MultiDiGraph
│   ├── storage.py              #   backend dispatch (json | neo4j) — open_graph/save_graph
│   ├── neo4j_backend.py        #   Neo4j storage backend (sync, incremental upsert, load-back)
│   ├── build_graph.py          #   assemble seed + sources -> graph + manifest
│   └── update.py               #   incremental updates (real event ingestion -> graph)
├── signaling/                  # LAYER 3: the sender-receiver game
│   ├── types.py                #   sender types, signal types, cost function
│   ├── signal_game.py          #   game: strategies, equilibria (pooling/separating),
│   │                           #   and game-derived credibility scores
│   └── belief_update.py        #   Bayesian posterior updating -> confidence score
├── extraction/                 # LAYER 2: signal extraction & synthesis
│   ├── retrieve.py             #   subgraph + passage retrieval (no lookahead)
│   ├── embed.py                #   TF-IDF embeddings for filings/news/transcripts
│   └── synthesis.py            #   cited decision brief (mock LLM backend)
├── evaluation/                 # cross-layer scoring
│   ├── decision_bench.py       #   score briefs vs realized outcomes (no lookahead)
│   └── citation_check.py       #   per-metric citation/hallucination failure summaries
├── data/                       # data sources + seed + ingestion
│   ├── seed_graph.py           #   deterministic companies + players + signal-access
│   ├── ingest.py               #   cursor-based incremental event ingestion (feed/news/edgar)
│   ├── fetch_filings.py        #   SEC EDGAR (live) with synthetic fallback
│   ├── fetch_news.py           #   news/transcripts (synthetic, deterministic)
│   └── _edgar.py               #   live SEC EDGAR (full + incremental, cursor-filtered)
├── results/                    # decision briefs, credibility, posteriors, evaluation
├── docs/PROJECT_SUMMARY.md     # this document
├── README.md                   # research framing, architecture, usage, limitations
├── requirements.txt            # pinned dependencies
└── Makefile                    # stages: graph, signaling, extract, evaluate, all
```

Layer mapping: `graph/*` = Layer 1 (players + companies + signal-access); `extraction/*`
= Layer 2 (retrieval + synthesis); `signaling/*` = Layer 3 (game + belief update);
`evaluation/*` = cross-layer scoring; `data/*` feeds Layer 1.

## 3. Data sources and run steps

### Data sources
- **Default mode (`data.mode: synthetic`)** — fully deterministic, offline, seeded from
  `config.seed`. `data/seed_graph.py` generates 12 companies (fundamentals + price
  history + filings/news) and 7 players per company (insider, analyst, competitor,
  supplier, regulator, journalist, fund), each with a role-specific signal-access
  profile.
- **Live mode (`data.mode: edgar`)** — `data/_edgar.py` attempts SEC EDGAR retrieval
  (requires `requests` and a descriptive `data.edgar.user_agent`; SEC mandates ≥100ms
  between requests). Any failure falls back to synthetic so the pipeline never blocks.
- **Incremental ingestion (`data.ingest`)** — `graph/update.py` polls the configured
  sources (`data/ingest.py`) and applies only NEW events. The cursor
  (`data.ingest.cursor_path`) records per-source progress: for live `edgar` it tracks
  seen accession numbers + last filed date; the deterministic `feed`/`news` sources emit
  a new batch per poll. Feed timestamps start at `feed_start_day` (70), above
  `evaluation.brief_day` (40), so ingested events stay strictly post-brief.

### Run steps
```bash
cd quant_signal_game
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
make graph        # stage 1: build graph (json or neo4j per graph.storage)
make update       # stage 1b: incremental event ingestion -> graph
make signaling    # stage 2: game + belief update
make extract      # stage 3: decision briefs
make evaluate     # stage 4: benchmark + citation checks
make all          # full pipeline
```
Each stage reads `config/config.yaml`, logs its `run_id`, writes JSON under `results/`,
and is independently re-runnable (stages only depend on prior stage outputs).

## 4. Methodology (exact)

### 4.1 Ontology (`graph/ontology.py`)
Entity kinds: `company`, `player`, `signal_type`, `signal`, `event`. Relations:
`signal_access` (player→signal_type), `transmits` (player→signal), `observes`
(player→signal), `linked_to` (player→company), `about` (signal→company), `has_source`
(company→signal for filings/news), `instance_of` (signal→signal_type). Graph backend is
`networkx.MultiDiGraph` (directed) serialized to JSON — chosen for reproducibility and
zero external dependencies; the ontology is backend-agnostic so a Neo4j backend could be
added later.

**Signal-access model (the focus).** A player's information set and transmission channel
are encoded as `signal_access` edges player→signal_type carrying `can_observe`,
`can_transmit`, `transmit_cost`, and `channel`. Role defaults live in `ROLE_ACCESS`:
insider (observe/transmit costly `certified_filing`, cheap `commentary`), analyst
(observe filings, transmit cheap `commentary`), competitor/supplier (transmit cheap talk
only), regulator (transmit costly `disclosure`), journalist (transmit cheap `media`),
fund (transmit costly `disclosure` + cheap `commentary`). `ROLE_CREDIBILITY_PRIOR` gives
a base prior per role (regulator highest, journalist lowest). Per-player variation is
jittered deterministically in `seed_graph.py` so credibility scores are non-degenerate.

### 4.2 Graph build/update/storage
`build_graph.py` generates seed records, instantiates every entity and relation (each
player's `access_profile` becomes `signal_access` edges), validates
(`validate()` — every `transmits` edge must be backed by a `signal_access.can_transmit`),
and persists via `graph/storage.py` (a JSON snapshot is always written; Neo4j is synced
when `graph.storage=neo4j`) with a reproducibility manifest that records the backend.
`update.py` opens the configured backend, ingests only new events (`data/ingest.py`,
cursor-based), applies them through the ontology `Updater` with validation, and
re-persists. `graph/neo4j_backend.py` maps the MultiDiGraph to Neo4j: node label = kind,
relationship type = relation, `id`/`eid` unique properties make writes idempotent MERGE
upserts, and complex attrs are JSON-sentinel encoded/decoded so round-trips are faithful.

### 4.3 Retrieval (`extraction/retrieve.py`)
`retrieve_subgraph(G, ticker, t)` returns the company node, every linked player and their
`signal_access` edges, and all signals with `timestamp <= t`. `retrieve_passages`
builds a deterministic TF-IDF index (`extraction/embed.py`) over filings/news whose
timestamp ≤ t and returns the top-k passages for a query. **No lookahead** is enforced at
the timestamp level: nothing strictly after t enters the brief.

### 4.4 Synthesis (`extraction/synthesis.py`)
For each company at day `t = evaluation.brief_day`, assembles a cited decision brief:
evidence passages (each claim carries its `source_id` citations), transmitting players
+ their signal-access, the game-derived credibility scores, the Bayesian posterior
confidence, and a research direction. Uncited claims are flagged `low_confidence`.
Default backend is `mock` (deterministic rule-based); `openai` can be selected in config.

### 4.5 Signaling game (`signaling/`) — the theoretical core
Sender types are a 2D space: `informed` × `aligned`; a sender is **high type (H)** iff
informed AND aligned, else **low type (L)**. `P(H) = informed_prior * aligned_prior`.
Signal types split by Spence vs Crawford-Sobel: `certified_filing`/`disclosure` are
**costly**; `commentary`/`media` are **cheap talk**.

**Cost function** (`types.signal_cost`): `cost = base_cost * type_mult`.
- costly: H bears `informed_cost_mult` (0.4) — lower marginal cost (knows the truth);
  L bears `misaligned_cost_mult` (1.0).
- cheap talk: `cheap_talk_base` (0.02) for both → no separation.

**Equilibrium characterization** (`signal_game.characterize_equilibrium`), from first
principles with normalized benefit of being perceived H, `b = 1.0`:
- *separating* iff `cost_H < b <= cost_L` (H sends, L declines to mimic);
- otherwise *pooling* (cheap talk always pooling).
The sender's costs are scaled from **their own signal-access transmit cost**
(`signal_access` edge → `score_signals` falls back to `spec.base_cost` if the edge is
missing), so the regime depends on WHO transmits and WHAT it costs them: the same costly
type can *separate* for a sender on a high-cost channel and *pool* for one on a low-cost
channel. Cheap talk always reports `cost_H == cost_L == access_cost` (equal near-zero
costs → Crawford-Sobel pooling).

**Credibility score** (game-derived, not a rubric): the receiver's Bayes posterior
`P(H | m) = (p_H * r) / (p_H*r + (1-p_H)(1-r))` — a noisy-signal update where
reliability `r` is fixed by the regime: `r = receiver.precision_prior` (0.8) for
separating; `r = 0.5` (uninformative) for pooling → posterior collapses to prior `p_H`.

### 4.6 Belief updating (`signaling/belief_update.py`)
Sequential Bayes over signals (chronological): prior confidence `prior_confidence`
(0.5), each signal contributes a likelihood conditional on the hypothesis vs its
negation based on credibility and content **direction** (−1/0/+1 from a crude
deterministic sentiment heuristic), producing a posterior trajectory and final
confidence per company. **No lookahead:** `build_confidence_report` drops any signal
with `timestamp > evaluation.brief_day` before the sequential update, so a brief at
time t updates only on signals observed at or before t.

### 4.7 Evaluation (`evaluation/`)
- `decision_bench.py`: brief at day `t` scored against the realized price return over
  `(t, t+window_days]`, using price observations **strictly after** t. Predicted impact is
  the credibility-weighted direction of the highest-credibility signal; we compare signs
  (direction accuracy), also report a **posterior-confidence-weighted accuracy** (each
  direction hit re-weighted by the brief's stated Bayesian posterior confidence), and
  check **no-lookahead violations** (any brief source with timestamp > t).
- `citation_check.py`: per-metric failure summaries — total/cited/uncited claims,
  hallucination count, citation precision, failed briefs.

**Meaningful-outcome design.** The synthetic price history is a random walk, but after
`evaluation.brief_day` each company's path carries a drift in a latent **outcome
direction** `y` (a deterministic sign derived from the ticker, `seed_graph._outcome_direction`).
Each company's **costly** transmitted signals (`certified_filing`/`disclosure`) are noisy
observations of `y` — their direction word agrees with `y` with probability
`data.signal_outcome_agree` — while **cheap talk** stays independent noise (uninformative,
Crawford-Sobel). The realized return over `(t, t+window]` therefore has a predictable
sign that the brief's credible-signal reading should recover. This makes the direction
benchmark meaningful (a model that reads credible signals lands well above the 0.5 random
baseline, and not trivially at 1.0 — a credible signal that disagrees with the true outcome
is a real miss). No-lookahead is preserved: the brief at `t` sees only signals ≤ t, and the
drift is the outcome realized strictly after `t`.

## 5. Engineering decisions and why

- **networkx-in-JSON graph as the default backend.** Reproducible, deterministic, zero
  external infra, ideal for offline research runs. The ontology is backend-agnostic and a
  **Neo4j backend** (`graph/neo4j_backend.py`, `graph.storage=neo4j`) is implemented behind
  the same `graph/storage.py` dispatch; a JSON snapshot is always written alongside so
  offline stages and reproducibility checks keep working. The graph is a **directed**
  `MultiDiGraph` — direction encodes who transmits to whom, essential for the signaling
  semantics.
- **Incremental ingestion is cursor-based.** `data/ingest.py` persists a per-source
  watermark (`results/ingest/cursor.json`) so each `update` run applies only NEW events;
  live `edgar` tracks seen accession numbers + last filed date, deterministic feeds emit
  one batch per poll. Feed timestamps start above `evaluation.brief_day`, so ingested
  events never leak into existing briefs (no-lookahead preserved).
- **No-lookahead evaluation is structural.** Retrieval filters `timestamp <= t` at the
  graph level; synthesis only uses those; evaluation scores against outcomes strictly
  after `t`. This makes the artifact honest and the benchmark meaningful.
- **Credibility derives from the game, not a rubric.** The score is a Bayes posterior
  whose reliability comes from the equilibrium regime — the theory, not hand-tuning.
- **Deterministic synthetic data.** Seeds are pinned (`config.seed` + salted RNGs), so
  offline runs reproduce bit-for-bit (verified). Live SEC is an optional fallback.
- **Citations required.** Every synthesis claim cites a retrieved source; uncited claims
  are marked low-confidence/dropped, and evaluation reports per-metric failure counts.
- **Mock LLM backend as the default** keeps the offline artifact fully runnable; real
  LLM synthesis is a config swap, not a hard dependency.

## 6. Default results and how to interpret them

With the synthetic default (12 companies × 7 players), `make all` produces:

- `graph/`: 352 nodes, 1020 edges, 0 validation issues. Counts by kind: 4 signal types,
  12 companies, 84 players, 252 signals. By relation: 180 `signal_access`, 84 `transmits`,
  84 `linked_to`, 252 `about`, 168 `has_source`, 252 `instance_of`.
- `signaling/credibility.json`: 84 transmitted signals, credibility range ≈ [0.095, 0.817],
  ~79 unique values (non-degenerate). Equilibrium regimes: 34 **separating**, 50 **pooling**.
  Because the regime uses each sender's *own* signal-access transmit cost, a handful of
  costly senders on low-cost channels fall into pooling; credibility depends on WHO
  transmits (e.g. 12 distinct scores among the 12 `certified_filing` senders). `equilibrium.json`
  shows the aggregate per-signal-type characterization at a representative access cost
  (costly ≈ 1.5 → separating; cheap talk ≈ 0.02 → pooling).
- `signaling/posterior.json`: per-company posterior confidence after sequential Bayes.
- `briefs/decision_briefs.json`: 12 cited decision briefs, each with evidence+citations,
  transmitting players + signal-access, credibility scores, posterior confidence, and a
  research direction.
- `evaluation/decision_bench.json`: benchmark rows, **direction accuracy ≈ 0.750** and
  **posterior-confidence-weighted accuracy ≈ 0.855** in the default run (well above the
  0.5 random baseline — the synthetic outcome is coupled to the credible signals' latent
  direction), and 0 no-lookahead violations; `citation_report.json`: citation precision
  1.000 (96 claims, all cited, 0 hallucinated).

**Reading the numbers.** Separating (costly) signals carry credibility well above prior
(≈0.5–0.85), pooling (cheap talk) signals sit ≈ prior (≈0.1–0.57). This is the Spence
prediction: cost separates types and raises receiver confidence; cheap talk moves the
receiver only through the sender's type/alignment priors. The posterior confidence is a
Bayesian accumulation of evidence direction × credibility, not a trading recommendation.

## 7. Known limitations

- Synthetic data is a stand-in; live EDGAR ingestion requires a real user-agent
  (`data.edgar.user_agent`) and currently parses submission metadata only (documents,
  not filing text) — see §8.
- The Neo4j backend (`graph.storage=neo4j`) is implemented and code-tested, but an
  end-to-end integration test against a running database is left for an environment with
  a live Neo4j instance (the backend raises a clear error if the DB is unreachable).
- The synthetic outcome is coupled to a latent direction (see §4.7); the realized-return
  sign is predictable from credible signals, so the benchmark measures signal-reading, not
  market noise. Direction accuracy ≈ `signal_outcome_agree` (0.75 default), which bounds
  the benchmark by the synthetic signal-to-outcome reliability — a placeholder for a real
  calibration target.
- LLM synthesis is mock/deterministic; real-LLM citation precision is unmeasured.
- Sentiment direction is a crude keyword heuristic, not trained NLP.
- Cheap-talk credibility currently uses the prior only (full Crawford–Sobel preference
  divergence games with misreporting incentives are a future extension).
- No semi-separating regime yet; comparative statics on cost/priors are unrun.

## 8. State of the work

| Module | Status |
|---|---|
| `graph/ontology.py` | **complete** — entities, relations, signal-access model |
| `graph/graph_io.py` | **complete** — deterministic JSON (de)serialization |
| `graph/storage.py` | **complete** — backend dispatch (json | neo4j) |
| `graph/neo4j_backend.py` | **complete** — Neo4j sync / incremental upsert / load-back (code-tested; live-DB integration pending) |
| `graph/build_graph.py` | **complete** — assembles + validates + manifest (backend-aware) |
| `graph/update.py` | **complete** — real incremental event ingestion -> configured backend |
| `data/seed_graph.py` | **complete** — deterministic seed (companies+players+access); synthetic outcome coupled to latent direction |
| `data/ingest.py` | **complete** — cursor-based incremental ingestion (feed/news/edgar) |
| `data/fetch_filings.py` / `fetch_news.py` | **complete** — synthetic default; live path via `data/_edgar.py` |
| `data/_edgar.py` | **complete** — live SEC full + cursor-based incremental fetch (needs user_agent; metadata-only parsing) |
| `signaling/types.py` | **complete** — types + cost function |
| `signaling/signal_game.py` | **complete** — equilibria + game-derived credibility |
| `signaling/belief_update.py` | **complete** — sequential Bayes posterior |
| `extraction/embed.py` | **complete** — TF-IDF index (deterministic) |
| `extraction/retrieve.py` | **complete** — subgraph + passage retrieval (no lookahead) |
| `extraction/synthesis.py` | **complete** — cited brief (mock backend); openai = config swap |
| `evaluation/decision_bench.py` | **complete** — no-lookahead benchmark |
| `evaluation/citation_check.py` | **complete** — per-metric failure summaries |
| `docs/PROJECT_SUMMARY.md` / `README.md` | **complete** |

Left to do: live-Neo4j integration test, real LLM synthesis + citation evaluation, EDGAR
document-text parsing, semi-separating equilibrium, comparative statics, real
sentiment/embedding backends.

## 9. Config knobs and their locations (`config/config.yaml`)

- `seed` — master RNG seed (reproducibility). `data.num_companies`,
  `num_players_per_company`, `price_history_len`, `filings_per_company`,
  `news_per_company` — dataset size. `data.signal_outcome_agree`,
  `data.outcome_drift` — the synthetic outcome coupling (§4.7). `data.mode` — `synthetic`
  | `edgar`; `data.edgar.user_agent`, `rate_limit_s`, `submissions_url`.
- `data.ingest.{enabled, sources, cursor_path, feed_events_per_poll, feed_start_day,
  feed_step_days}` — incremental event ingestion (feed/news deterministic; edgar live).
- `graph.storage` (`json` | `neo4j`), `graph.path` — graph output;
  `graph.neo4j.{uri, user, password, database, encrypted}` — live backend connection
  (password may come from `NEO4J_PASSWORD`).
- `signaling.type_priors.{informed_prior, aligned_prior}` — sender-type priors.
- `signaling.cost.{cheap_talk_base, costly_base, informed_cost_mult, misaligned_cost_mult}`
  — the cost function; tuning these changes which signals separate.
- `signaling.receiver.{belief_threshold, precision_prior}` — receiver strategy + the
  reliability of separating signals.
- `signaling.equilibrium.tolerance` — reserved for equilibrium convergence.
- `extraction.retrieval.{top_k_subgraph, top_k_passages}`; `extraction.embed.backend`
  (`tfidf` | ...); `extraction.synthesis.{llm_backend, model, temperature, cite_required}`.
- `evaluation.{brief_day, outcome_window_days, price_impact_threshold, metrics}` —
  no-lookahead benchmark timing.
