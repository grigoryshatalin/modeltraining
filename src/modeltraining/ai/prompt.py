"""Shared, provider-neutral prompt used by every model adapter."""

from __future__ import annotations

from ..context import MarketContext

SYSTEM_PROMPT = (
    "You are a disciplined systematic trading assistant for a single retail "
    "brokerage account. For the given symbol and market context, decide whether "
    "to BUY, SELL, or HOLD.\n\n"
    "Guidelines:\n"
    "- Recommend a trade only when the provided data gives a clear, specific "
    "reason; otherwise HOLD.\n"
    "- Never recommend selling more shares than are currently held. Short "
    "selling is not allowed.\n"
    "- Keep quantities small relative to available buying power.\n"
    "- Base your reasoning only on the data provided; do not assume information "
    "you were not given.\n"
    "- Return confidence between 0.0 and 1.0 and a one- or two-sentence rationale.\n\n"
    "This is not financial advice; you are one component of a risk-managed system "
    "that applies its own hard limits after your decision."
)


def render_context(ctx: MarketContext) -> str:
    """Render a MarketContext into a compact text block for the model."""
    stats = ctx.summary_stats()

    if ctx.position and ctx.position.qty:
        pos = (
            f"{ctx.position.qty:g} shares @ avg ${ctx.position.avg_entry_price:,.2f} "
            f"(unrealized P/L ${ctx.position.unrealized_pl:,.2f})"
        )
    else:
        pos = "none"

    acct = ctx.account
    acct_line = (
        f"buying_power=${acct.buying_power:,.2f}, equity=${acct.equity:,.2f}, "
        f"day_pnl=${acct.day_pnl:,.2f}"
        if acct
        else "unavailable"
    )

    recent = ", ".join(f"{b.close:g}" for b in ctx.bars[-10:]) or "none"

    return (
        f"Symbol: {ctx.symbol}\n"
        f"As of: {ctx.as_of.isoformat()}\n"
        f"Latest price: ${ctx.latest_price:,.4f} (bid ${ctx.bid:,.4f} / ask ${ctx.ask:,.4f})\n"
        f"Current position: {pos}\n"
        f"Account: {acct_line}\n\n"
        f"Recent daily technicals ({stats['bar_count']} bars):\n"
        f"  last_close={stats['last_close']}  "
        f"sma5={stats['sma_5']}  sma10={stats['sma_10']}  sma20={stats['sma_20']}\n"
        f"  period_high={stats['period_high']}  period_low={stats['period_low']}  "
        f"pct_change_period={stats['pct_change_period']}%\n"
        f"  last_10_closes=[{recent}]\n"
    )


# ---------------------------------------------------------------------------
# Tournament prompting: per-contestant strategy personas + web research
# ---------------------------------------------------------------------------

def build_system_prompt(strategy: str) -> str:
    """Base trading rules plus this contestant's own strategy mandate."""
    return (
        SYSTEM_PROMPT
        + "\n\nYour trading mandate (this is the style you compete under — "
        "stay true to it):\n"
        + strategy.strip()
    )


def research_section(note: str) -> str:
    """Render a research note to append to the decision prompt, if any."""
    note = (note or "").strip()
    if not note:
        return ""
    return (
        "\n\nYour own research notes from today (gathered via web search):\n"
        + note
    )


def competitors_section(standings: str) -> str:
    """Render the live rival-standings block to append to the decision prompt."""
    standings = (standings or "").strip()
    if not standings:
        return ""
    return (
        "\n\nYou are competing against other trading models for the highest "
        "return, and can see everyone's current book:\n"
        + standings
        + "\nFactor this in however you judge best (e.g. how much risk you need "
        "to take to catch up or protect a lead), but stay true to your mandate."
    )


RESEARCH_SYSTEM = (
    "You are a market research analyst preparing a same-day briefing for a "
    "trader. Use web search to find recent, decision-relevant information "
    "(breaking news, earnings, guidance, analyst moves, macro data, notable "
    "price action) for the given tickers. Return a concise brief: 3-6 bullet "
    "points, each tagged with the ticker and a specific, sourced fact. If you "
    "find nothing material, say so briefly. Do not give a trade recommendation "
    "— just the facts."
)


def research_user(symbols: list[str], today: str) -> str:
    return (
        f"Date: {today}\n"
        f"Tickers: {', '.join(symbols)}\n\n"
        "Research anything from the last few days that could affect a short-term "
        "trading decision on these tickers today."
    )


def web_search_tool_for(model: str) -> str:
    """Pick the web-search tool version the given model supports."""
    dynamic = {
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    }
    return "web_search_20260209" if model in dynamic else "web_search_20250305"
