# quant_signal_game — Makefile
# Run from the quant_signal_game/ directory:  make all  (or make graph, etc.)
# Each recipe cd's to the repo root so modules resolve as quant_signal_game.*.
PY := $(abspath ../.venv/bin/python)
CFG := quant_signal_game/config/config.yaml

.PHONY: venv graph signaling extract evaluate all clean

venv:
	python3 -m venv ../.venv
	$(PY) -m pip install -r quant_signal_game/requirements.txt

# ---- stage 1: knowledge graph (players + companies + signal-access) ----
graph:
	cd .. && $(PY) -m quant_signal_game.graph.build_graph --config $(CFG) --stage graph

# ---- stage 2: signaling game (sender-receiver, equilibria, belief update) ----
signaling:
	cd .. && $(PY) -m quant_signal_game.signaling.signal_game --config $(CFG) --stage signaling
	cd .. && $(PY) -m quant_signal_game.signaling.belief_update --config $(CFG) --stage signaling

# ---- stage 3: extraction & synthesis (retrieve + cited decision brief) ----
extract:
	cd .. && $(PY) -m quant_signal_game.extraction.synthesis --config $(CFG) --stage extract

# ---- stage 4: evaluation (no-lookahead bench + citation checks) ----
evaluate:
	cd .. && $(PY) -m quant_signal_game.evaluation.decision_bench --config $(CFG) --stage evaluate

all: graph signaling extract evaluate

clean:
	rm -rf ../quant_signal_game/results
