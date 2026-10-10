> ORIGINAL PROJECT NARRATIVE — retained for context. These claims predate the September 7, 2026 review. Use the reviewed README, HIRING_REVIEW.md and validation/validation.json for current status.

# SignalGraph: Multi-Agent Financial Intelligence with Knowledge Graphs and Bayesian Signaling Games

## Research question

> **How does the credibility and price impact of a signal depend on who transmits it,
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
signals, updates beliefs via Bayes, and forms a cited decision brief; the market's
response (price impact) is the observable outcome.

This is a **research / decision-support artifact**, not a trading signal.

## Theoretical foundations

- **Spence (1973), job-market signaling.** Costly signals separate types: the sender
  bears a real cost that differs by type, so low-cost types send and high-cost types
  decline to mimic. We instantiate this as the **cost function** — informed senders bear
  lower marginal cost (they know the truth), so a costly signal (an insider's certified
  filing revision, a regulator's enforcement action, a fund's disclosed position)
  *separates* types and is credible.
- **Crawford & Sobel (1982), cheap talk.** When the sender's cost is ~0, separation is
  impossible: the equilibrium is **pooling**, and credibility rests on the sender's type
  and preference alignment (a low-cost analyst post from an aligned player differs from
  a costly insider revision).
- **Bayesian belief updating.** The receiver holds a prior over sender type and a prior
  confidence in the hypothesis ("this company will outperform"), and updates both by
  Bayes' rule as signals arrive.

## Architecture — three layers

1. **Knowledge graph (players + companies + signal-access).** A directed dynamic graph
   (networkx `MultiDiGraph`, serialized to JSON). Company nodes carry fundamentals,
   filings, events, price history; player nodes carry identity, role, credibility prior,
   and **signal-access edges** (`signal_access` player→signal_type) encoding
   `can_observe` / `can_transmit` / `transmit_cost`. Incremental updates arrive via
   `graph/update.py`.
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

## Usage

```bash
cd quant_signal_game
python3 -m venv ../.venv
../.venv/bin/python -m pip install -r requirements.txt

make graph        # stage 1: build the knowledge graph (players + companies + signal-access)
make signaling    # stage 2: sender-receiver game, equilibria, Bayesian belief update
make extract      # stage 3: retrieve + cited decision brief per entity
make evaluate     # stage 4: no-lookahead benchmark + citation/hallucination checks
make all          # run the full pipeline end-to-end
```

All parameters are in `config/config.yaml` (seeds, data mode, game costs, priors,
retrieval knobs, evaluation window). Every stage logs its `run_id` and is re-runnable;
synthetic data is seeded so offline runs reproduce exactly.

## What the default run produces

With the synthetic default (12 companies × 7 players, deterministic seed), `make all`
produces, under `results/`:

- `graph/` — the serialized graph + a build manifest (node/edge counts by kind/relation).
- `signaling/` — per-signal **credibility** scores and the **equilibrium characterization**
  (costly `certified_filing`/`disclosure` → *separating* for high-cost channels;
  cheap-talk `commentary`/`media` → *pooling*; a few low-cost costly senders pool too),
  plus per-company **posterior confidence** trajectories that update only on signals
  observed at or before the brief time (no lookahead).
- `briefs/` — one **cited decision brief** per company (evidence + citations, transmitting
  players + signal-access, credibility scores, posterior confidence, research direction).
- `evaluation/` — the **no-lookahead benchmark** (brief at day t scored against outcomes
  strictly after t) and per-metric **citation/hallucination failure summaries**. The
  synthetic outcome is coupled to a latent per-company direction that its costly signals
  observe, so the benchmark is meaningful: default direction accuracy ≈ 0.750 and
  posterior-confidence-weighted accuracy ≈ 0.855, both well above the 0.5 random baseline
  (and not trivially 1.0 — a credible signal that disagrees with the true outcome is a real
  miss).

## Limitations

- Default data is **synthetic and deterministic**; live SEC EDGAR retrieval is a
  best-effort stub (`data/_edgar.py`) that falls back to synthetic.
- The synthetic price outcome is coupled to a latent per-company direction (`data.signal_outcome_agree`,
  `data.outcome_drift`) so the decision benchmark measures signal-reading, not noise; accuracy is
  bounded by that synthetic signal-to-outcome reliability.
- LLM synthesis defaults to a **mock (rule-based) backend**; an `openai` backend can be
  wired in via `config` but is not required for the offline default.
- The graph backend is networkx-in-JSON (reproducible, no external DB); a Neo4j backend
  could be added behind the same ontology.
- Sentiment direction for the Bayes update is a crude deterministic word heuristic —
  a research placeholder, not a trained NLP model.

## What I'd do next

- Add a Neo4j backend and real incremental event ingestion.
- Wire a real LLM synthesis backend and evaluate citation precision against a gold set.
- Calibrate the cost function / priors against observed price reactions.
- Add a semi-separating equilibrium regime and comparative-static sensitivity analysis.
- See `docs/PROJECT_SUMMARY.md` for a full handoff and the state of each module.
