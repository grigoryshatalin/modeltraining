"""Per-contestant model calls: optional web research, then a structured decision.

Each function returns (result, cost_usd) so the engine can charge real API spend
against the contestant's net score. Provider clients are created once by the
engine and passed in.
"""

from __future__ import annotations

from ..ai.prompt import (
    RESEARCH_SYSTEM,
    build_system_prompt,
    competitors_section,
    render_context,
    research_section,
    research_user,
    web_search_tool_for,
)
from ..ai.schema import TradeDecision
from ..context import MarketContext
from . import pricing

_DECISION_MAX_TOKENS = 1024
_RESEARCH_MAX_TOKENS = 1500
_RESEARCH_MAX_STEPS = 4
_RESEARCH_MAX_SEARCHES = 3

# A buy/sell/hold decision does not need extended reasoning. On models where
# thinking is on by default (Opus 5, Sonnet 5) that quietly burns output tokens
# on every call, so we disable it for the decision. Models that can't disable
# (Fable) or don't think by default (Haiku) are left to their default.
_CAN_DISABLE_THINKING = {
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
}


def _decision_thinking(model: str) -> dict | None:
    return {"type": "disabled"} if model in _CAN_DISABLE_THINKING else None


# --------------------------------------------------------------------------- #
# Research (Claude web search)                                                 #
# --------------------------------------------------------------------------- #

def research_claude(client, model: str, symbols: list[str], today: str) -> tuple[str, float]:
    """Let a Claude model search the web and return a short briefing + cost."""
    tool = {"type": web_search_tool_for(model), "name": "web_search", "max_uses": _RESEARCH_MAX_SEARCHES}
    messages = [{"role": "user", "content": research_user(symbols, today)}]
    cost = 0.0
    final = None

    for _ in range(_RESEARCH_MAX_STEPS):
        resp = client.messages.create(
            model=model,
            max_tokens=_RESEARCH_MAX_TOKENS,
            system=RESEARCH_SYSTEM,
            tools=[tool],
            messages=messages,
        )
        cost += pricing.anthropic_call_cost(model, resp.usage)
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            final = resp
            continue
        final = resp
        break

    note = ""
    if final is not None:
        note = "\n".join(b.text for b in final.content if getattr(b, "type", "") == "text").strip()
    return note[:2000], cost


# --------------------------------------------------------------------------- #
# Decision (structured output)                                                 #
# --------------------------------------------------------------------------- #

def _decision_user(context: MarketContext, note: str, competitors: str) -> str:
    return render_context(context) + research_section(note) + competitors_section(competitors)


def decide_claude(
    client, model: str, strategy: str, note: str, context: MarketContext, competitors: str = ""
) -> tuple[TradeDecision, float]:
    params = dict(
        model=model,
        max_tokens=_DECISION_MAX_TOKENS,
        system=build_system_prompt(strategy),
        messages=[{"role": "user", "content": _decision_user(context, note, competitors)}],
        output_format=TradeDecision,
    )
    thinking = _decision_thinking(model)
    if thinking is not None:
        params["thinking"] = thinking
    resp = client.messages.parse(**params)
    cost = pricing.anthropic_call_cost(model, resp.usage)
    decision = resp.parsed_output
    if decision is None:
        raise RuntimeError(f"no parseable decision (stop_reason={resp.stop_reason})")
    return decision, cost


def decide_openai(
    client, model: str, strategy: str, note: str, context: MarketContext, competitors: str = ""
) -> tuple[TradeDecision, float]:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(strategy)},
            {"role": "user", "content": _decision_user(context, note, competitors)},
        ],
        response_format=TradeDecision,
    )
    cost = pricing.openai_call_cost(model, completion.usage)
    decision = completion.choices[0].message.parsed
    if decision is None:
        raise RuntimeError("no parseable decision")
    return decision, cost
