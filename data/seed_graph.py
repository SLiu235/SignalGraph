"""
data.seed_graph — deterministic seed dataset: companies + players + signal-access.

Produces raw seed *records* (Company / Player / Signal objects). build_graph.py
assembles these into the networkx graph. Everything is seeded from config so offline
runs reproduce exactly. No external API required (synthetic fallback by design).

Per-player variation is applied here (jittered priors, informed/aligned draws, access
cost jitter) so the signaling layer produces non-degenerate credibility scores.
"""
from __future__ import annotations

import numpy as np

from ..graph import ontology as O
from ..common import get_rng

# Fixed base for per-ticker outcome-direction RNG so `y` is reproducible but NOT
# computable from the public ticker string alone (the benchmark's ground truth is
# not readable from the brief's evidence — only from the noisy signals).
_OUTCOME_SEED_BASE = 7919

# Plausible seed universe: 12 companies across sectors.
SEED_COMPANIES = [
    ("AAPL", "Apple Inc.", "Technology"),
    ("MSFT", "Microsoft Corp.", "Technology"),
    ("TSLA", "Tesla Inc.", "Automotive"),
    ("AMZN", "Amazon.com Inc.", "Retail"),
    ("NVDA", "NVIDIA Corp.", "Semiconductors"),
    ("JPM", "JPMorgan Chase", "Financials"),
    ("XOM", "Exxon Mobil", "Energy"),
    ("UNH", "UnitedHealth Group", "Healthcare"),
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("BA", "Boeing Co.", "Aerospace"),
    ("T", "AT&T Inc.", "Telecom"),
    ("KO", "Coca-Cola Co.", "Consumer Staples"),
]

# Per-role default relation to the company.
ROLE_RELATION = {
    "insider": "insider_of",
    "analyst": "covers",
    "competitor": "competitor_of",
    "supplier": "supplier_to",
    "regulator": "regulates",
    "journalist": "covers_news",
    "fund": "invested_in",
}


def _gen_price_history(rng, length: int, start: float = 100.0,
                       drift: float = 0.0, drift_start: int = 0) -> list:
    """Random-walk price path. When `drift != 0`, days >= drift_start move toward the
    latent outcome direction, so the realized return over (brief_day, brief_day+window]
    is predictable from the transmitted signals the brief reads (meaningful benchmark,
    still no-lookahead: the drift is the OUTCOME realized strictly after the brief)."""
    prices = []
    p = start
    for i in range(length):
        daily = rng.normal(0.0, 0.02) + (drift if i >= drift_start else 0.0)
        p = p * (1.0 + daily)
        prices.append({"date": f"D{i:04d}", "price": round(p, 2)})
    return prices


def _outcome_direction(ticker: str) -> int:
    """Latent ground-truth outcome direction for a company: a deterministic sign from a
    per-ticker salted RNG (no shared state, so seed and signaling stages agree). The
    synthetic market moves this way after the brief day; costly (credible) transmitted
    signals are noisy observations of it. Cheap talk stays independent noise —
    uninformative, per Crawford-Sobel. Not derivable from the ticker alone, so a brief
    cannot read the ground truth except through the noisy signals."""
    rng = np.random.default_rng(_OUTCOME_SEED_BASE + sum(ord(c) for c in ticker.lower()))
    return 1 if rng.uniform() < 0.5 else -1


def _gen_fundamentals(rng) -> dict:
    return {
        "revenue": round(rng.uniform(1e9, 4e9), 0),
        "net_income": round(rng.uniform(1e8, 5e8), 0),
        "eps": round(rng.uniform(1.0, 8.0), 2),
        "debt_ratio": round(rng.uniform(0.1, 0.6), 2),
        "mkt_cap": round(rng.uniform(1e10, 3e10), 0),
    }


def _gen_filing_content(rng, ticker, doc_type, period) -> str:
    eps = rng.uniform(1.0, 6.0)
    return (f"{doc_type} filing for {ticker} covering {period}: "
            f"reported revenue guidance revision, EPS {eps:.2f}, "
            f"cost guidance tightened by {rng.uniform(0.5, 4.0):.1f}%.")


def _gen_news_content(rng, ticker, kind) -> str:
    if kind == "media":
        return (f"Media headline on {ticker}: analyst "
                f"flags {rng.choice(['margin compression', 'growth slowdown', 'margin compression'])}.")
    return (f"Analyst commentary on {ticker}: {rng.choice(['overweight', 'overweight', 'neutral', 'underweight'])} "
            f"on {rng.choice(['guidance', 'demand', 'guidance'])}; sees {rng.uniform(5, 30):.0f}% upside.")


def _gen_transmitted_content(rng, ticker, stype, y: int, agree_p: float) -> str:
    """Transmitted signal content. Costly signals (certified_filing/disclosure) are
    NOISY observations of the latent outcome direction `y`: with probability `agree_p`
    the direction word agrees with y, else it opposes it. Cheap talk stays independent
    (uninformative noise), so only credible signals are predictive of the outcome."""
    if stype == "certified_filing":
        agree = rng.uniform() < agree_p
        word = "raised" if (y > 0) == agree else "lowered"
        return (f"{ticker} insider certified filing revision: guidance {word} "
                f"by {rng.uniform(2, 8):.1f}%; "
                f"costs {rng.choice(['under control', 'rising', 'under control'])}.")
    if stype == "disclosure":
        agree = rng.uniform() < agree_p
        word = "accumulating" if (y > 0) == agree else "trimming"
        return (f"{ticker} position disclosure: {word} "
                f"{rng.uniform(0.5, 3.0):.1f}% of float; conviction {rng.uniform(0.6, 0.9):.2f}.")
    if stype == "commentary":
        return (f"{ticker} commentary: {rng.choice(['bullish', 'cautious', 'bullish'])} on "
                f"{rng.choice(['guidance', 'demand', 'margin'])}; sees {rng.uniform(2, 12):.0f}% move.")
    return (f"{ticker} media: {rng.choice(['positive', 'negative', 'neutral'])} framing "
            f"of {rng.choice(['quarter', 'guidance', 'quarter'])} results.")


def build_transmitted_signals(cfg: dict, players: list) -> list:
    """Each player transmits one signal through a channel their signal-access grants.
    Prefer a costly channel if accessible (insider/regulator/fund), else cheap talk.
    Costly signals are coupled to the company's latent outcome direction (noisy
    observation); cheap talk is independent. Returns Signal objects with a real sender
    -> feeds the signaling game."""
    rng = get_rng(cfg, salt="transmitted")
    agree_p = float(cfg["data"].get("signal_outcome_agree", 0.75))
    out = []
    for p in players:
        transmitable = [st for st, spec in p.access_profile.items() if spec["can_transmit"]]
        if not transmitable:
            continue
        # order: costly types first, then cheap talk
        transmitable.sort(key=lambda st: 0 if O.SIGNAL_TYPES[st]["kind"] == "costly" else 1)
        stype = transmitable[0]
        y = _outcome_direction(p.company_ref)
        sig = O.Signal(
            signal_id=f"tx_{p.player_id}", signal_type=stype,
            kind=O.SIGNAL_TYPES[stype]["kind"],
            content=_gen_transmitted_content(rng, p.company_ref, stype, y, agree_p),
            sender=p.player_id, company=p.company_ref, timestamp="D35",
            source="player", source_id=f"tx-{p.player_id}",
        )
        out.append(sig)
    return out


def build_seed_records(cfg: dict):
    """Return (companies, players, signals) deterministic seed records."""
    rng = get_rng(cfg, salt="seed_graph")
    n = int(cfg["data"]["num_companies"])
    n_players = int(cfg["data"]["num_players_per_company"])
    price_len = int(cfg["data"]["price_history_len"])
    filings_n = int(cfg["data"]["filings_per_company"])
    news_n = int(cfg["data"]["news_per_company"])
    brief_day = int(cfg["evaluation"]["brief_day"])
    outcome_drift = float(cfg["data"].get("outcome_drift", 0.01))

    companies = []
    players = []
    signals = []
    for i in range(n):
        ticker, name, sector = SEED_COMPANIES[i % len(SEED_COMPANIES)]
        # latent outcome direction drives the post-brief price drift (meaningful outcome);
        # drift starts strictly after brief_day so the day-t base price carries no signal
        y = _outcome_direction(ticker)
        c = O.Company(
            ticker=ticker, name=name, sector=sector,
            fundamentals=_gen_fundamentals(rng),
            price_history=_gen_price_history(rng, price_len, drift=y * outcome_drift,
                                             drift_start=brief_day + 1),
            created_at="seed",
        )
        companies.append(c)

        # players per company, one per role
        for role in O.ROLES:
            pid = f"{ticker.lower()}_{role}"
            cred = O.ROLE_CREDIBILITY_PRIOR[role] + rng.uniform(-0.08, 0.08)
            cred = max(0.05, min(0.98, cred))
            p = O.Player(
                player_id=pid, name=f"{role}@{ticker}", role=role,
                credibility_prior=round(cred, 3),
                informed_prior=round(rng.uniform(0.3, 0.8), 3),
                aligned_prior=round(rng.uniform(0.25, 0.75), 3),
                company_ref=ticker, relation=ROLE_RELATION[role],
            )
            players.append(p)

            # instantiate signal-access edges for this player with per-player cost jitter
            for spec in O.ROLE_ACCESS[role]:
                # store access profile on the player for build_graph to turn into edges
                # (jittered transmit_cost; cheap talk stays ~0, costly varies)
                jitter = 1.0 + rng.uniform(-0.25, 0.25)
                spec = dict(spec)
                spec["transmit_cost"] = round(spec["transmit_cost"] * jitter, 4)
                p.access_profile[spec["signal_type"]] = spec

        # filings (costly certified_filing) — source of record
        for f_i in range(filings_n):
            doc_type = "10-K" if f_i % 2 == 0 else "8-K"
            content = _gen_filing_content(rng, ticker, doc_type, f"FY{i}")
            sig = O.Signal(
                signal_id=f"{ticker.lower()}_f{f_i}", signal_type="certified_filing",
                kind="costly", content=content, sender="", company=ticker,
                timestamp=f"D{f_i * 10:04d}", source="filing", source_id=f"{doc_type}-{i}-{f_i}",
            )
            signals.append(sig)

        # news (cheap talk: media + commentary)
        for n_i in range(news_n):
            kind = "media" if n_i % 2 == 0 else "commentary"
            content = _gen_news_content(rng, ticker, kind)
            sig = O.Signal(
                signal_id=f"{ticker.lower()}_n{n_i}", signal_type=kind,
                kind="cheap_talk", content=content, sender="", company=ticker,
                timestamp=f"D{20 + n_i * 5:04d}", source="news", source_id=f"news-{i}-{n_i}",
            )
            signals.append(sig)

    return companies, players, signals
