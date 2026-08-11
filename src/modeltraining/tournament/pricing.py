"""Estimate the real API cost of each model call, from reported token usage.

Rates are USD per 1M tokens (input, output), current as of August 2026. Anthropic
figures are first-party list prices; OpenAI figures are best-effort for the
GPT-5.6 family — edit if they drift. A model with no entry costs $0 here (and a
warning is surfaced), so a missing price never crashes a run.
"""

from __future__ import annotations

# model_id -> (input_per_1M, output_per_1M)
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI (verify IDs/rates)
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.20, 1.20),
}

WEB_SEARCH_COST = 0.01  # per search, ~ $10 / 1,000


def has_price(model: str) -> bool:
    return model in PRICES


def anthropic_call_cost(model: str, usage: object) -> float:
    """Cost of one Anthropic call from its `usage` object (tokens + web searches)."""
    in_rate, out_rate = PRICES.get(model, (0.0, 0.0))
    inp = getattr(usage, "input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0

    searches = 0
    stu = getattr(usage, "server_tool_use", None)
    if stu is not None:
        searches = getattr(stu, "web_search_requests", 0) or 0

    cost = (inp + 1.25 * cache_create + 0.1 * cache_read) / 1e6 * in_rate
    cost += out / 1e6 * out_rate
    cost += searches * WEB_SEARCH_COST
    return cost


def openai_call_cost(model: str, usage: object) -> float:
    """Cost of one OpenAI call from its `usage` object."""
    in_rate, out_rate = PRICES.get(model, (0.0, 0.0))
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return prompt_tokens / 1e6 * in_rate + completion_tokens / 1e6 * out_rate
