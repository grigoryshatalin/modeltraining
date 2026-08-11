"""Thin wrapper over alpaca-py's TradingClient.

Exposes only the operations this bot needs, mapping Alpaca's models onto the
neutral dataclasses in `modeltraining.context`. Alpaca returns most numeric
fields as strings, so we cast defensively.
"""

from __future__ import annotations

import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from ..config import Settings
from ..context import AccountSnapshot, PositionSnapshot

log = logging.getLogger(__name__)


def _f(value: object, default: float = 0.0) -> float:
    """Best-effort float cast for Alpaca's string-typed numeric fields."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class AlpacaBroker:
    def __init__(self, settings: Settings, client: TradingClient | None = None) -> None:
        settings.require_alpaca_keys()
        self._paper = settings.alpaca_paper
        self._client = client or TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper,
        )

    @property
    def is_paper(self) -> bool:
        return self._paper

    def get_account(self) -> AccountSnapshot:
        a = self._client.get_account()
        return AccountSnapshot(
            buying_power=_f(a.buying_power),
            cash=_f(a.cash),
            equity=_f(a.equity),
            last_equity=_f(a.last_equity),
            portfolio_value=_f(a.portfolio_value),
        )

    def list_positions(self) -> list[PositionSnapshot]:
        return [self._to_position(p) for p in self._client.get_all_positions()]

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        try:
            p = self._client.get_open_position(symbol)
        except Exception:
            # alpaca-py raises when there is no open position for the symbol.
            return None
        return self._to_position(p)

    def submit_market_order(self, symbol: str, qty: int, side: str) -> str:
        """Submit a day market order. Returns the broker order id."""
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(request)
        log.info("submitted %s %s x%d -> order %s", side, symbol, qty, order.id)
        return str(order.id)

    @staticmethod
    def _to_position(p: object) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=str(getattr(p, "symbol")),
            qty=_f(getattr(p, "qty", 0)),
            avg_entry_price=_f(getattr(p, "avg_entry_price", 0)),
            current_price=_f(getattr(p, "current_price", 0)),
            market_value=_f(getattr(p, "market_value", 0)),
            unrealized_pl=_f(getattr(p, "unrealized_pl", 0)),
        )
