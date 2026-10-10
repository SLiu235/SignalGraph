# SignalGraph — Makefile
# Run from the repo root:  make all  (or make graph, etc.).
# Modules now live flat at the repo root (graph/, signaling/, extraction/,
# evaluation/, data/), so each recipe runs them with `-m` from here.
# PY uses the project-local venv (has all deps incl. the neo4j driver).
PY := $(abspath .venv/bin/python)
CFG := config/config.yaml

.PHONY: venv graph update ingest signaling extract evaluate all clean

venv:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# ---- stage 1: knowledge graph (players + companies + signal-access) ----
graph:
	$(PY) -m graph.build_graph --config $(CFG) --stage graph

# ---- incremental event ingestion (real, cursor-based) ----
update:
	$(PY) -m graph.update --config $(CFG) --stage update

ingest:
	$(PY) -m data.ingest --config $(CFG)

# ---- stage 2: signaling game (sender-receiver, equilibria, belief update) ----
signaling:
	$(PY) -m signaling.signal_game --config $(CFG) --stage signaling
	$(PY) -m signaling.belief_update --config $(CFG) --stage signaling

# ---- stage 3: extraction & synthesis (retrieve + cited decision brief) ----
extract:
	$(PY) -m extraction.synthesis --config $(CFG) --stage extract

# ---- stage 4: evaluation (no-lookahead bench + citation checks) ----
evaluate:
	$(PY) -m evaluation.decision_bench --config $(CFG) --stage evaluate

all: graph signaling extract evaluate

clean:
	rm -rf results
