"""The tournament field: 10 contestants (mix of models and strategies).

Each contestant is the *same* everything — prompt scaffold, risk rules,
watchlist, starting capital — except its `model` and its `strategy` persona,
and whether it may do its own web research. That isolation is the point: any
performance difference is attributable to the model + strategy, not the harness.

Notes
-----
* Only models that support structured outputs are used for decisions
  (Claude: Opus 5 / 4.8, Sonnet 5, Haiku 4.5, Fable 5; OpenAI: GPT-5.x).
* `research=True` is enabled only on Claude models with first-class web search.
  OpenAI web research is not wired up yet, so OpenAI contestants trade on price
  data + strategy only (see brains.py).
* OpenAI model IDs below are best-effort for August 2026 — verify against
  platform.openai.com/docs/models and edit as needed. A contestant whose model
  or provider key is unavailable is reported at run time, not silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contestant:
    id: str
    provider: str  # "claude" | "openai"
    model: str
    strategy: str
    research: bool = False


DEFAULT_ROSTER: list[Contestant] = [
    # NOTE: the priciest models are intentionally left out to control token
    # spend — Opus 5 and Sonnet-5-style thinking-on-by-default burn tokens on
    # simple decisions (thinking is disabled on decisions in brains.py), Fable 5
    # can't disable thinking at all ($10/$50), and GPT-5.6 Sol is $5/$30. The two
    # OpenAI entries use the cheap tiers (Terra $2/$12, Luna $0.20/$1.20). Re-add
    # any of them here once you're happy with the cost profile.
    Contestant(
        id="opus48-macro",
        provider="claude",
        model="claude-opus-4-8",
        research=True,
        strategy=(
            "Macro / news-driven trader. Weight recent catalysts, earnings, and "
            "macro data heavily; position with the prevailing regime and step "
            "aside when the news is genuinely uncertain."
        ),
    ),
    Contestant(
        id="sonnet5-momentum",
        provider="claude",
        model="claude-sonnet-5",
        research=True,
        strategy=(
            "Fast momentum trader. Ride strength while it persists, rotate "
            "toward whichever watchlist name has the strongest recent tape."
        ),
    ),
    Contestant(
        id="sonnet5-contrarian",
        provider="claude",
        model="claude-sonnet-5",
        research=False,
        strategy=(
            "Mean-reversion trader. Fade overextended moves: lighten up after "
            "sharp rallies well above the moving averages, accumulate after "
            "sharp drops toward the period low."
        ),
    ),
    Contestant(
        id="haiku45-momentum",
        provider="claude",
        model="claude-haiku-4-5",
        research=False,
        strategy=(
            "Simple momentum trader. Buy what is going up and above its "
            "averages; sell what is going down. Keep it decisive and cheap."
        ),
    ),
    Contestant(
        id="haiku45-buyhold",
        provider="claude",
        model="claude-haiku-4-5",
        research=False,
        strategy=(
            "Buy-and-hold baseline. Put cash to work early in the broad-market "
            "name (SPY) and mostly hold; only sell on a clear, sustained "
            "breakdown. You are the benchmark the others must beat."
        ),
    ),
    Contestant(
        id="gpt56luna-momentum",
        provider="openai",
        model="gpt-5.6-luna",
        research=False,
        strategy=(
            "Trend/momentum trader. Favor strength and moving-average alignment; "
            "cut weakness. Decisive but disciplined."
        ),
    ),
    Contestant(
        id="gpt56terra-value",
        provider="openai",
        model="gpt-5.6-terra",
        research=False,
        strategy=(
            "Value / contrarian trader. Buy pullbacks in quality names toward "
            "support; trim into strength. Prefer patience to churn."
        ),
    ),
]


def default_roster() -> list[Contestant]:
    return list(DEFAULT_ROSTER)
