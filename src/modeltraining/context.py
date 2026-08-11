"""Broker/provider-neutral domain types shared across the app.

These dataclasses decouple the rest of the code from the Alpaca SDK's own
models, so the risk engine, AI adapters, and tests never import `alpaca-py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean


@dataclass(frozen=True)
class Bar:
    """A single OHLCV price bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class AccountSnapshot:
    """A point-in-time view of the brokerage account."""

    buying_power: float
    cash: float
    equity: float
    last_equity: float
    portfolio_value: float

    @property
    def day_pnl(self) -> float:
        """Profit/loss so far today (equity minus prior close equity)."""
        return self.equity - self.last_equity


@dataclass(frozen=True)
class PositionSnapshot:
    """An open position in a single symbol."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float


@dataclass
class MarketContext:
    """Everything the AI model needs to reason about one symbol."""

    symbol: str
    as_of: datetime
    latest_price: float
    bid: float
    ask: float
    bars: list[Bar] = field(default_factory=list)
    position: PositionSnapshot | None = None
    account: AccountSnapshot | None = None

    def summary_stats(self) -> dict[str, float | int | None]:
        """Compact technical summary derived from the recent bars."""
        closes = [b.close for b in self.bars]
        highs = [b.high for b in self.bars]
        lows = [b.low for b in self.bars]

        def sma(n: int) -> float | None:
            return round(mean(closes[-n:]), 4) if len(closes) >= n else None

        pct_change = None
        if len(closes) >= 2 and closes[0]:
            pct_change = round((closes[-1] / closes[0] - 1) * 100, 2)

        return {
            "bar_count": len(self.bars),
            "last_close": round(closes[-1], 4) if closes else None,
            "sma_5": sma(5),
            "sma_10": sma(10),
            "sma_20": sma(20),
            "period_high": round(max(highs), 4) if highs else None,
            "period_low": round(min(lows), 4) if lows else None,
            "pct_change_period": pct_change,
        }
