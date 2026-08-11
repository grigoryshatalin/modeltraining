"""Fetch quotes and bars from Alpaca and assemble a MarketContext."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from ..config import Settings
from ..context import (
    AccountSnapshot,
    Bar,
    MarketContext,
    PositionSnapshot,
)


class MarketDataClient:
    def __init__(self, settings: Settings, client: StockHistoricalDataClient | None = None) -> None:
        settings.require_alpaca_keys()
        self._settings = settings
        self._client = client or StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def get_latest_quote(self, symbol: str) -> tuple[float, float, float]:
        """Return (bid, ask, mid_price). Falls back to 0 when unavailable."""
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._client.get_stock_latest_quote(req)
        quote = quotes.get(symbol)
        if quote is None:
            return 0.0, 0.0, 0.0
        bid = float(getattr(quote, "bid_price", 0) or 0)
        ask = float(getattr(quote, "ask_price", 0) or 0)
        mid = (bid + ask) / 2 if bid and ask else (ask or bid)
        return bid, ask, mid

    def get_recent_bars(self, symbol: str, days: int) -> list[Bar]:
        # Pad the calendar window so weekends/holidays still yield ~`days` bars.
        start = datetime.now(timezone.utc) - timedelta(days=days * 2 + 5)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
        )
        bar_set = self._client.get_stock_bars(req)
        rows = bar_set.data.get(symbol, [])
        bars = [
            Bar(
                timestamp=b.timestamp,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in rows
        ]
        return bars[-days:]

    def build_context(
        self,
        symbol: str,
        account: AccountSnapshot | None = None,
        position: PositionSnapshot | None = None,
    ) -> MarketContext:
        bid, ask, mid = self.get_latest_quote(symbol)
        bars = self.get_recent_bars(symbol, self._settings.bar_lookback_days)

        latest_price = mid
        if latest_price <= 0 and bars:
            latest_price = bars[-1].close

        return MarketContext(
            symbol=symbol,
            as_of=datetime.now(timezone.utc),
            latest_price=latest_price,
            bid=bid,
            ask=ask,
            bars=bars,
            position=position,
            account=account,
        )
