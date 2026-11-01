"""
agents.strategies — the learning-strategy library for the agent layer.

Each strategy answers three questions:
  * learn(round_signals)          — update the opponent model from what the agent saw,
  * react(belief, round_signals)  — choose a reaction action given its model + belief,
  * analyze_opponents()           — the "analyze each other's strategies" deliverable.

Implemented strategies (config knob agents.role_defaults.<role>.strategy):
  bayes           : Bayesian posterior over opponent type (H/L) from costly-signal
                    credibility, and over the market hypothesis from signal directions;
                    reacts on the posterior sign. Opponent prediction by inferred type.
  irl             : Inverse Reinforcement Learning. Infers each opponent's payoff
                    (theta_truth / theta_contrarian / theta_stay) by maximum-likelihood
                    over observed (belief, action) pairs under a logit choice model,
                    then predicts that opponent's next action from the inferred thetas
                    and reacts to the predicted majority.
  fictitious_play : Empirical-frequency best response: tracks each opponent's action
                    histogram and reacts to the aggregate empirical mode (no preference
                    inference — context-free).
  heuristic       : Role-prior baseline: reacts by role rules to the most credible
                    observed signal (no learning).

Shared generative model (what IRL recovers) — logit choice over a in {+1,-1,0}:
    utility(a | belief) = theta_truth*[a==sign(b)] + theta_contrarian*[a==-sign(b)]
                          + theta_stay*[a==0]
    P(a) = softmax(utility / temperature)
"""
from __future__ import annotations

import numpy as np

from . import types as T

# reaction channel per role (reactions are cheap-talk by default; journalist uses media)
REACTION_CHANNEL = {
    "insider": "commentary", "analyst": "commentary", "competitor": "commentary",
    "supplier": "commentary", "regulator": "commentary", "journalist": "media",
    "fund": "commentary",
}


def _sign(x: float) -> int:
    if x > 1e-9:
        return T.ACTION_SUPPORT
    if x < -1e-9:
        return T.ACTION_COUNTER
    return T.ACTION_STAY

# probs are ordered [support(+1), counter(-1), stay(0)] — see choice_utility.
ACTION_INDEX = {T.ACTION_SUPPORT: 0, T.ACTION_COUNTER: 1, T.ACTION_STAY: 2}
INDEX_ACTION = {0: T.ACTION_SUPPORT, 1: T.ACTION_COUNTER, 2: T.ACTION_STAY}


def aggregate_belief(signals: list[T.ObservedSignal]) -> tuple[int, float]:
    """Signed credibility-weighted direction -> (belief_sign, conviction).
    belief_sign in {+1,-1,0}; conviction = min(1, |weighted sum|). Mirrors the
    Bayesian accumulation in signaling.belief_update (not a calibrated probability).
    If the weighted sum cancels out but a credible signal exists, anchor on the
    most credible observed signal so agents still act on strong evidence."""
    raw = sum(s.direction * s.credibility for s in signals)
    if abs(raw) < 1e-6 and signals:
        best = max(signals, key=lambda s: s.credibility)
        if best.direction != 0:
            return best.direction, best.credibility
    return _sign(raw), min(1.0, abs(raw))


def logit_probs(utility: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(utility / temperature) over actions [support, counter, stay]."""
    u = np.asarray(utility, dtype=float) / max(temperature, 1e-6)
    e = np.exp(u - u.max())
    return e / e.sum()


def choice_utility(theta: dict, belief_sign: int) -> np.ndarray:
    """Utility of each action [support, counter, stay] under the generative model."""
    b = belief_sign
    u = np.array([
        theta.get("truth", 0.0) * (1 if 1 == b else 0) + theta.get("contrarian", 0.0) * (1 if -1 == b else 0),
        theta.get("truth", 0.0) * (1 if -1 == b else 0) + theta.get("contrarian", 0.0) * (1 if 1 == b else 0),
        theta.get("stay", 0.0) * 1,
    ], dtype=float)
    return u


def _theta_key(theta: dict) -> tuple:
    return (theta.get("truth", 0.0), theta.get("contrarian", 0.0), theta.get("stay", 0.0))


# --------------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------------
class LearningStrategy:
    name: str = "base"

    def __init__(self, spec: T.AgentSpec, cfg: dict):
        self.spec = spec
        self.cfg = cfg
        self.observations: list[T.ObservedSignal] = []
        self.reactions: list[T.Reaction] = []
        self._observed_ids: set[str] = set()
        # per-opponent model: player_id -> dict
        self.opponent_models: dict[str, dict] = {}

    def learn(self, round_signals: list[T.ObservedSignal]) -> list[T.ObservedSignal]:
        """Store only newly visible observations and return that incremental batch.

        The simulation supplies each agent with its full visible history every round.
        Without this guard, Bayesian evidence and opponent actions were replayed on
        every round, producing artificial certainty and distorted action frequencies.
        """
        new = [s for s in round_signals if s.key() not in self._observed_ids]
        self._observed_ids.update(s.key() for s in new)
        self.observations.extend(new)
        return new

    def react(self, belief_sign: int, conviction: float,
              round_signals: list[T.ObservedSignal], r: int) -> T.Reaction:
        """Choose a reaction action. Subclasses override; base falls back to stay."""
        return _make_reaction(self, T.ACTION_STAY, 0.0, r)

    def analyze_opponents(self) -> dict:
        """Per-opponent strategy analysis (the 'analyze each other' deliverable)."""
        return {pid: _base_read(self, pid) for pid in self.opponent_models}


# --------------------------------------------------------------------------------
# bayes
# --------------------------------------------------------------------------------
class BayesStrategy(LearningStrategy):
    """Bayesian opponent modeling + market-hypothesis posterior, reacts on posterior sign."""
    name = T.STRATEGY_BAYES

    def __init__(self, spec: T.AgentSpec, cfg: dict):
        super().__init__(spec, cfg)
        self.market_posterior = 0.5                 # P(company outperforms | seen signals)
        self.threshold = float(cfg["signaling"]["receiver"]["belief_threshold"])

    def _update_market_posterior(self, new_signals):
        posterior = self.market_posterior
        for s in new_signals:
            if s.direction == 0:
                continue
            # noisy-signal Bayes: like_H = cred if support else 1-cred (same as belief_update)
            like_H = s.credibility if s.direction > 0 else (1 - s.credibility)
            like_notH = (1 - s.credibility) if s.direction > 0 else s.credibility
            num = posterior * like_H
            den = num + (1 - posterior) * like_notH
            posterior = num / den if den > 0 else posterior
        self.market_posterior = posterior

    def learn(self, round_signals):
        new_signals = super().learn(round_signals)
        self._update_market_posterior(new_signals)
        # opponent type posterior: costly signal -> credibility is P(H|signal); else keep prior
        for s in new_signals:
            if s.sender == self.spec.player_id:
                continue   # do not model yourself as an opponent
            if s.sender not in self.opponent_models:
                self.opponent_models[s.sender] = {"type_posterior": 0.5, "n": 0}
            m = self.opponent_models[s.sender]
            if s.kind == "costly":
                m["type_posterior"] = s.credibility
            m["n"] = m.get("n", 0) + 1

    def react(self, belief_sign, conviction, round_signals, r):
        p = self.market_posterior
        if p > self.threshold:
            return _make_reaction(self, T.ACTION_SUPPORT, p, r)
        if p < (1 - self.threshold):
            return _make_reaction(self, T.ACTION_COUNTER, 1 - p, r)
        return _make_reaction(self, T.ACTION_STAY, 0.0, r)

    def analyze_opponents(self):
        out = {}
        for pid, m in self.opponent_models.items():
            ph = m.get("type_posterior", 0.5)
            pred = T.ACTION_SUPPORT if ph >= 0.5 else T.ACTION_COUNTER
            out[pid] = {
                "model": "bayes_type",
                "inferred_type": {"H": round(ph, 4), "L": round(1 - ph, 4)},
                "predicted_action": pred,
                "confidence": round(max(ph, 1 - ph), 4),
                "read": (f"Bayes type posterior P(H)={ph:.3f} from {m.get('n', 0)} observed "
                         f"signal(s); predicts {'confirm' if pred > 0 else 'counter'}."),
            }
        return out


# --------------------------------------------------------------------------------
# irl
# --------------------------------------------------------------------------------
class IRLStrategy(LearningStrategy):
    """Inverse Reinforcement Learning: MLE-infer each opponent's payoff (theta) params
    from their observed (belief, action) pairs, then predict + best-respond."""
    name = T.STRATEGY_IRL

    def __init__(self, spec: T.AgentSpec, cfg: dict):
        super().__init__(spec, cfg)
        g = cfg["agents"]["irl"]
        self.grid = {
            "truth": list(g["grid_truth"]),
            "contrarian": list(g["grid_contrarian"]),
            "stay": list(g["grid_stay"]),
        }
        self.temperature = float(cfg["agents"].get("temperature", 0.5))

    def learn(self, round_signals):
        # Reactions in this batch were made against the previously visible public
        # context. Record that context once, then add the new observations.
        belief_sign, _ = aggregate_belief(self.observations)
        new_signals = super().learn(round_signals)
        # context proxy: the observer's own aggregate belief (full-access agents share
        # the market context with their opponents). Record (belief, action) pairs per
        # opponent from observed reactions — this is what IRL MLE fits.
        for s in new_signals:
            if not s.is_reaction or s.sender == self.spec.player_id:
                continue
            m = self.opponent_models.setdefault(s.sender, {
                "pairs": [], "theta": None, "n": 0, "last_belief": 0,
            })
            # Simulated reactions disclose the public belief context that generated
            # the action. For ordinary signals (which do not have a choice context),
            # retain the observer's public-information proxy.
            action_context = s.belief_context if s.belief_context is not None else belief_sign
            m["pairs"].append({"belief": action_context, "action": s.direction})
            m["n"] += 1
            m["last_belief"] = action_context
            m["theta"] = _irl_mle(m["pairs"], self.grid, self.temperature)

    def _predict(self, opponent: str, belief_sign: int) -> tuple[int, float]:
        m = self.opponent_models.get(opponent)
        if not m or not m.get("theta"):
            return T.ACTION_STAY, 0.5
        b = belief_sign or m.get("last_belief", 0)
        probs = logit_probs(choice_utility(m["theta"], b), self.temperature)
        return INDEX_ACTION[int(np.argmax(probs))], float(np.max(probs))

    def react(self, belief_sign, conviction, round_signals, r):
        # predict each opponent's next action from their inferred thetas; react as a
        # best response: act on own belief, but counter if the inferred crowd opposes it,
        # and follow the crowd when the agent has no strong belief of its own.
        votes = {}
        for pid, m in self.opponent_models.items():
            b = self._last_observed_belief(pid)
            pred, conf = self._predict(pid, b)
            votes[pred] = votes.get(pred, 0) + conf
        if belief_sign != 0:
            best = max(votes, key=lambda k: votes[k]) if votes else belief_sign
            if best == -belief_sign:
                return _make_reaction(self, T.ACTION_COUNTER, conviction, r)
            return _make_reaction(self, belief_sign, conviction, r)
        if not votes:
            return _make_reaction(self, T.ACTION_STAY, 0.0, r)
        best = max(votes, key=lambda k: votes[k])
        return _make_reaction(self, best, min(1.0, votes[best]), r)

    def _last_observed_belief(self, opponent: str) -> int:
        m = self.opponent_models.get(opponent)
        return m.get("last_belief", 0) if m else 0

    def analyze_opponents(self):
        out = {}
        for pid, m in self.opponent_models.items():
            th = m.get("theta") or {"truth": 0.0, "contrarian": 0.0, "stay": 0.0}
            pred, conf = self._predict(pid, self._last_observed_belief(pid))
            # IRL fidelity vs the opponent's true latent theta
            true = self.true_theta_of(pid)
            err = _theta_error(th, true)
            out[pid] = {
                "model": "irl_payoff",
                "inferred_theta": {k: round(v, 3) for k, v in th.items()},
                "true_theta": {k: round(v, 3) for k, v in true.items()},
                "theta_error": round(err, 3),
                "predicted_action": pred,
                "confidence": round(conf, 4),
                "read": (f"IRL inferred payoff from {m['n']} observed reactions: "
                         f"truth={th['truth']:.2f} (favor {'confirm' if th['truth'] > th.get('contrarian',0) else 'contrarian'}), "
                         f"stay={th['stay']:.2f}; predicts {'confirm' if pred > 0 else 'counter' if pred < 0 else 'stay'} "
                         f"(theta_error {err:.3f} vs true)."),
            }
        return out

    def true_theta_of(self, opponent: str) -> dict:
        """Ground-truth theta of an opponent for IRL-fidelity evaluation."""
        for a in self._all_specs():
            if a.player_id == opponent:
                return a.true_theta
        return {"truth": 0.0, "contrarian": 0.0, "stay": 0.0}

    def _all_specs(self):
        return getattr(self, "_specs", [])


# --------------------------------------------------------------------------------
# fictitious play
# --------------------------------------------------------------------------------
class FictitiousPlayStrategy(LearningStrategy):
    """Empirical-frequency best response to opponents' action distributions."""
    name = T.STRATEGY_FP

    def __init__(self, spec: T.AgentSpec, cfg: dict):
        super().__init__(spec, cfg)

    def learn(self, round_signals):
        new_signals = super().learn(round_signals)
        for s in new_signals:
            if not s.is_reaction or s.sender == self.spec.player_id:
                continue
            m = self.opponent_models.setdefault(s.sender, {"dist": {1: 0, -1: 0, 0: 0}, "n": 0})
            m["dist"][s.direction] = m["dist"].get(s.direction, 0) + 1
            m["n"] += 1

    def react(self, belief_sign, conviction, round_signals, r):
        votes = {}
        for pid, m in self.opponent_models.items():
            d = m["dist"]
            if m["n"] == 0:
                continue
            mode = max(d, key=lambda k: d[k])
            votes[mode] = votes.get(mode, 0) + d[mode] / m["n"]
        if not votes:
            return _make_reaction(self, T.ACTION_STAY, 0.0, r)
        best = max(votes, key=lambda k: votes[k])
        return _make_reaction(self, best, min(1.0, votes[best]), r)

    def analyze_opponents(self):
        out = {}
        for pid, m in self.opponent_models.items():
            d = m.get("dist", {})
            tot = m.get("n", 0) or 1
            emp = {k: round(v / tot, 3) for k, v in d.items()}
            pred = max(d, key=lambda k: d[k]) if d else 0
            out[pid] = {
                "model": "fictitious_play",
                "empirical_dist": emp,
                "predicted_action": pred,
                "confidence": round(max(emp.values(), default=0.5), 3),
                "read": (f"Empirical action distribution over {m.get('n', 0)} observed reactions "
                         f"confirm/counter/stay = {emp.get(1,0):.2f}/{emp.get(-1,0):.2f}/{emp.get(0,0):.2f}; "
                         f"best-responds to mode {'confirm' if pred > 0 else 'counter' if pred < 0 else 'stay'}."),
            }
        return out


# --------------------------------------------------------------------------------
# heuristic
# --------------------------------------------------------------------------------
class HeuristicStrategy(LearningStrategy):
    """Role-prior baseline: reacts to the most credible observed signal by role rules."""
    name = T.STRATEGY_HEURISTIC

    def __init__(self, spec: T.AgentSpec, cfg: dict):
        super().__init__(spec, cfg)

    def learn(self, round_signals):
        super().learn(round_signals)

    def react(self, belief_sign, conviction, round_signals, r):
        role = self.spec.role
        # role rules: high-credibility roles confirm the most credible costly signal;
        # journalist follows media; competitor/supplier stay (noise traders)
        if role in ("insider", "regulator", "fund"):
            costly = [s for s in self.observations if s.kind == "costly"]
            if costly:
                best = max(costly, key=lambda s: s.credibility)
                return _make_reaction(self, best.direction, best.credibility, r)
        if role == "journalist":
            media = [s for s in self.observations if s.signal_type == "media"]
            if media:
                best = max(media, key=lambda s: s.credibility)
                return _make_reaction(self, best.direction, best.credibility, r)
        if role == "analyst":
            commentary = [s for s in self.observations if s.signal_type == "commentary"]
            if commentary:
                best = max(commentary, key=lambda s: s.credibility)
                return _make_reaction(self, best.direction, best.credibility, r)
        return _make_reaction(self, T.ACTION_STAY, 0.0, r)

    def analyze_opponents(self):
        return {}


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------
def _make_reaction(strat: LearningStrategy, direction: int, conviction: float, r: int) -> T.Reaction:
    spec = strat.spec
    word = "confirms" if direction > 0 else "counters" if direction < 0 else "stays"
    target = "the signal"
    channel = REACTION_CHANNEL.get(spec.role, "commentary")
    content = (f"{spec.role}@{spec.company_ref} reaction (round {r}): {word} {target}; "
               f"conviction {conviction * 100:.0f}%.")
    return T.Reaction(
        round=r, agent=spec.player_id, role=spec.role, company_ref=spec.company_ref,
        action_type="react" if direction else "stay", direction=direction, conviction=conviction,
        signal_type=channel, content=content, credibility=spec.p_high)


def _theta_error(a: dict, b: dict) -> float:
    keys = ("truth", "contrarian", "stay")
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / len(keys)


def _base_read(strat: LearningStrategy, pid: str) -> dict:
    return {"model": "none", "read": "no observations yet"}


def _irl_mle(pairs: list, grid: dict, temperature: float) -> dict:
    """Maximum-likelihood estimate of (truth, contrarian, stay) from observed
    (belief, action) pairs under the logit choice model. Grid search over the
    configured theta ranges; deterministic, no external deps."""
    best = None
    best_ll = -np.inf
    for th in grid["truth"]:
        for ct in grid["contrarian"]:
            for st in grid["stay"]:
                theta = {"truth": th, "contrarian": ct, "stay": st}
                ll = 0.0
                for pair in pairs:
                    b = pair["belief"]
                    a = pair["action"]
                    probs = logit_probs(choice_utility(theta, b), temperature)
                    ll += np.log(probs[ACTION_INDEX[a]])   # probs order [support,counter,stay]
                if ll > best_ll:
                    best_ll = ll
                    best = theta
    return best or {"truth": 0.0, "contrarian": 0.0, "stay": 0.0}


def make_strategy(name: str, spec: T.AgentSpec, cfg: dict) -> LearningStrategy:
    """Factory: instantiate a learning strategy from a strategy name."""
    cls = {
        T.STRATEGY_BAYES: BayesStrategy,
        T.STRATEGY_IRL: IRLStrategy,
        T.STRATEGY_FP: FictitiousPlayStrategy,
        T.STRATEGY_HEURISTIC: HeuristicStrategy,
    }.get(name)
    if cls is None:
        raise ValueError(f"unknown strategy '{name}'; choose from {T.STRATEGIES}")
    strat = cls(spec, cfg)
    strat._specs = []   # filled by simulation for true_theta_of lookups
    return strat
