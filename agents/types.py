"""
agents.types — dataclasses for the multi-agent signaling-game simulation.

An agent is a graph player promoted to an active decision-maker. Beyond the static
sender->receiver game, each agent:

  * holds a role, an information-access tier, a response-time tier, and a learning strategy,
  * analyzes OTHER agents' strategies (opponent modeling: inferred type + inferred payoff
    params + predicted action + a plain-language "read"),
  * makes its own reactions each round, which feed back as observable signals for others.

Information access (what an agent sees):
  full    -> all signals + all other agents' reactions
  partial -> only signals it can observe (its signal_access.can_observe channels)
             + reactions of agents linked to the same company
  limited -> only signals it can observe; no other agents' reactions

Response time (when an agent reacts):
  fast -> reacts in the current round (sees this round's incoming signals)
  slow -> reacts on a one-round lag (sees only prior-round signals)

Learning strategies (how an agent updates its opponent model and reacts):
  bayes            -> Bayesian posterior over opponent type and over the market hypothesis
  irl              -> Inverse Reinforcement Learning: infer opponents' payoff (theta)
                       parameters by maximum-likelihood from observed (belief, action) pairs
  fictitious_play  -> empirical-frequency best response to opponents' actions
  heuristic        -> role-prior baseline (no learning)

Opponent generative model (what IRL recovers) — a logit choice over actions a in {+1,-1,0}:
    utility(a | belief) = theta_truth*[a==sign(b)] + theta_contrarian*[a==-sign(b)]
                          + theta_stay*[a==0]
    P(a) = softmax(utility / temperature)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- info-access tiers ----
ACCESS_FULL = "full"
ACCESS_PARTIAL = "partial"
ACCESS_LIMITED = "limited"
INFO_ACCESS = (ACCESS_FULL, ACCESS_PARTIAL, ACCESS_LIMITED)

# ---- response-time tiers ----
RESPONSE_FAST = "fast"
RESPONSE_SLOW = "slow"
RESPONSE_TIME = (RESPONSE_FAST, RESPONSE_SLOW)

# ---- learning strategies ----
STRATEGY_BAYES = "bayes"
STRATEGY_IRL = "irl"
STRATEGY_FP = "fictitious_play"
STRATEGY_HEURISTIC = "heuristic"
STRATEGIES = (STRATEGY_BAYES, STRATEGY_IRL, STRATEGY_FP, STRATEGY_HEURISTIC)

# ---- action codes ----
ACTION_SUPPORT = 1    # confirm / support the belief direction
ACTION_COUNTER = -1   # counter / oppose the belief direction
ACTION_STAY = 0       # no reaction


@dataclass
class AgentSpec:
    """Static specification of one agent (derived from a graph player node)."""
    player_id: str
    role: str
    company_ref: str
    strategy: str                 # one of STRATEGIES
    info_access: str              # one of INFO_ACCESS
    response_time: str            # one of RESPONSE_TIME
    credibility_prior: float = 0.5
    informed_prior: float = 0.5
    aligned_prior: float = 0.5
    # signal types this agent can observe (from signal_access.can_observe edges)
    observe_channels: list = field(default_factory=list)
    # player_ids of agents linked to the same company (partial-access reaction visibility)
    linked_agents: list = field(default_factory=list)
    # true latent payoff params (ground truth for IRL recovery)
    true_theta: dict = field(default_factory=dict)   # {"truth","contrarian","stay"}

    @property
    def p_high(self) -> float:
        """P(H) for this sender = P(informed) * P(aligned)."""
        return self.informed_prior * self.aligned_prior


@dataclass
class ObservedSignal:
    """A signal/reaction an agent observed, with the direction the agent reads."""
    round: int
    signal_id: str
    sender: str
    role: str
    signal_type: str
    kind: str                     # costly | cheap_talk
    content: str
    direction: int                # -1 | 0 | +1 (read from content)
    credibility: float            # game-derived credibility (P(H|signal))
    is_reaction: bool = False
    # Public belief context stated with a simulated reaction. Seed signals do not
    # have one; it lets IRL condition payoff inference on the decision state that
    # generated an opponent action.
    belief_context: int | None = None
    timestamp: str = ""

    def key(self) -> str:
        return self.signal_id


@dataclass
class Reaction:
    """An agent's own reaction action in a round."""
    round: int
    agent: str
    role: str
    company_ref: str
    action_type: str              # react | act | stay
    direction: int                # -1 | 0 | +1
    conviction: float             # strength of the reaction, 0..1
    signal_type: str              # reaction channel (cheap talk by default)
    content: str
    credibility: float            # pooling credibility = sender p_H (cheap talk)
    is_reaction: bool = True
    # The strategy's public belief sign at the point of choice. The payoff policy
    # transforms this context into the realized action through a logit draw.
    belief_context: int = 0
    timestamp: str = ""

    def as_observed(self) -> ObservedSignal:
        return ObservedSignal(
            round=self.round, signal_id=f"rx_{self.agent}_r{self.round}",
            sender=self.agent, role=self.role, signal_type=self.signal_type,
            kind="cheap_talk", content=self.content, direction=self.direction,
            credibility=self.credibility, is_reaction=True,
            belief_context=self.belief_context, timestamp=self.timestamp)
