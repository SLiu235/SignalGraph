"""
agents — multi-agent signaling-game simulation layer.

The static sender->receiver game (signaling/*) is promoted into an ACTIVE simulation:
the graph's players become agents that analyze each other's strategies and make their
own reactions each round. Agents differ in role, information access, response time,
and learning strategy (Bayes, Inverse Reinforcement Learning, fictitious play, heuristic).

Run as a stage:  python -m agents.run --config config/config.yaml --stage agents
Outputs -> results/agents/ {agents.json, simulation.json, learning.json, strategy_analysis.json}
"""
