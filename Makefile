# SignalGraph — Makefile
# Run from the repo root:  make all  (or make graph, etc.).
# Modules now live flat at the repo root (graph/, signaling/, extraction/,
# evaluation/, data/), so each recipe runs them with `-m` from here.
# PY uses the project-local venv (has all deps incl. the neo4j driver).
PY ?= $(abspath .venv/bin/python)
CFG ?= config/config.yaml
RCFG ?= config/research.yaml
# One trace ID per `make` invocation. It is passed to every stage so artifacts from
# a pipeline run can be joined reliably, while callers may still override RUN_ID.
RUN_ID ?= $(or $(SIGNAL_GRAPH_RUN_ID),$(shell date -u +%Y%m%dT%H%M%S))
# `?=` creates a recursive variable; freeze it now so a multi-target `make all`
# does not evaluate `date` separately for every recipe.
RUN_ID := $(RUN_ID)
RUN_ENV = SIGNAL_GRAPH_RUN_ID="$(RUN_ID)"

.PHONY: venv graph update ingest signaling agents extract evaluate test all clean research research-fetch research-demo counterfactuals

venv:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# ---- stage 1: knowledge graph (players + companies + signal-access) ----
graph:
	$(RUN_ENV) $(PY) -m graph.build_graph --config $(CFG) --stage graph

# ---- incremental event ingestion (real, cursor-based) ----
update:
	$(RUN_ENV) $(PY) -m graph.update --config $(CFG) --stage update

ingest:
	$(RUN_ENV) $(PY) -m data.ingest --config $(CFG)

# ---- stage 2: signaling game (sender-receiver, equilibria, belief update) ----
signaling: graph
	$(RUN_ENV) $(PY) -m signaling.signal_game --config $(CFG) --stage signaling
	$(RUN_ENV) $(PY) -m signaling.belief_update --config $(CFG) --stage signaling

# ---- stage 2b: multi-agent simulation (agents analyze each other + react) ----
agents: signaling
	$(RUN_ENV) $(PY) -m agents.run --config $(CFG) --stage agents

# ---- stage 3: extraction & synthesis (retrieve + cited decision brief) ----
extract: signaling
	$(RUN_ENV) $(PY) -m extraction.synthesis --config $(CFG) --stage extract

# ---- stage 4: evaluation (no-lookahead bench + citation checks) ----
evaluate: extract
	$(RUN_ENV) $(PY) -m evaluation.decision_bench --config $(CFG) --stage evaluate

test:
	$(RUN_ENV) $(PY) -m unittest discover -s tests -v
	$(RUN_ENV) $(PY) -m agents.test_agents

all: agents evaluate

# UTC historical research has its own input schema and outputs.
research-fetch:
	$(RUN_ENV) $(PY) -m research.pipeline --config $(RCFG) --fetch

research:
	$(RUN_ENV) $(PY) -m research.pipeline --config $(RCFG)

research-demo:
	$(RUN_ENV) $(PY) -m research.pipeline --config $(RCFG) --demo

counterfactuals:
	$(RUN_ENV) $(PY) -m agents.counterfactuals --config $(CFG)

clean:
	rm -rf results
