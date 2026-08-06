"""
signaling.types — sender types, signal types (costly vs cheap talk), and the
signal-cost function. Grounded in Spence (1973) and Crawford-Sobel (1982).

Sender types are drawn from a 2D space:
    informed    : does the sender actually know the truth (observe it)?
    aligned     : are the sender's preferences aligned with the receiver's objective?
A sender is "high type" (H) if informed AND aligned; otherwise "low type" (L).
The cost function is the Spence lever: H bears lower marginal cost to send a costly
signal (they know the truth, so certifying it is cheap); L bears higher cost. That
cost differential is what lets a costly signal *separate* types.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..graph import ontology as O

# signal kinds
COSTLY = "costly"
CHEAP_TALK = "cheap_talk"


@dataclass(frozen=True)
class SignalTypeSpec:
    name: str
    kind: str                      # costly | cheap_talk
    channel: str
    base_cost: float               # raw per-signal cost before type scaling

    def is_costly(self) -> bool:
        return self.kind == COSTLY


@dataclass(frozen=True)
class SenderType:
    informed: bool
    aligned: bool

    @property
    def is_high(self) -> bool:
        """H = informed AND aligned (truthful + aligned preferences)."""
        return self.informed and self.aligned

    @property
    def label(self) -> str:
        if self.is_high:
            return "H"
        return "L"


def high_type_prior(p_informed: float, p_aligned: float) -> float:
    """P(H) = P(informed) * P(aligned), assuming independence."""
    return p_informed * p_aligned


def signal_cost(stype: SenderType, spec: SignalTypeSpec, cfg: dict) -> float:
    """Cost function: base_cost scaled by sender type.
    Costly signals: H bears a lower marginal cost (knows the truth); L bears more.
    Cheap talk: near-zero cost for both -> no Spence separation (Crawford-Sobel)."""
    cost = cfg["signaling"]["cost"]
    if spec.kind == CHEAP_TALK:
        return cost["cheap_talk_base"]  # ~0 for everyone
    mult = cost["misaligned_cost_mult"] if not stype.is_high else cost["informed_cost_mult"]
    return spec.base_cost * mult


def type_prior(p_informed: float, p_aligned: float) -> dict:
    """Return P(H), P(L) as a dict for Bayesian updates."""
    ph = high_type_prior(p_informed, p_aligned)
    return {"H": ph, "L": 1.0 - ph}


def make_signal_type_specs() -> dict[str, SignalTypeSpec]:
    return {name: SignalTypeSpec(name=name, kind=d["kind"], channel=d["channel"],
                                 base_cost=d["base_cost"])
            for name, d in O.SIGNAL_TYPES.items()}
