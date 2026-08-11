"""Deterministic risk checks that gate and size every order.

The AI model proposes; the RiskManager disposes. Nothing reaches the broker
without passing through here, and quantities are clamped down (never up) to
respect the configured limits. This layer is intentionally free of any AI or
broker SDK imports so it is easy to reason about and unit-test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Settings
from ..context import AccountSnapshot, MarketContext, PositionSnapshot
from ..ai.schema import TradeAction, TradeDecision


@dataclass(frozen=True)
class RiskResult:
    """Outcome of evaluating one decision against the guardrails."""

    approved: bool
    action: TradeAction
    quantity: int
    reason: str


class RiskManager:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def evaluate(
        self,
        decision: TradeDecision,
        context: MarketContext,
        account: AccountSnapshot,
        position: PositionSnapshot | None,
    ) -> RiskResult:
        s = self._s
        action = decision.action
        price = context.latest_price
        pos_qty = position.qty if position else 0.0

        # HOLD is never an order.
        if action == TradeAction.HOLD:
            return RiskResult(False, action, 0, "model chose HOLD")

        if price <= 0:
            return RiskResult(False, action, 0, "no valid price available")

        if decision.confidence < s.min_confidence:
            return RiskResult(
                False,
                action,
                0,
                f"confidence {decision.confidence:.2f} < min {s.min_confidence:.2f}",
            )

        if decision.quantity <= 0:
            return RiskResult(False, action, 0, "model proposed zero shares")

        if action == TradeAction.SELL:
            return self._evaluate_sell(decision, pos_qty)

        return self._evaluate_buy(decision, price, account, position)

    def _evaluate_sell(self, decision: TradeDecision, pos_qty: float) -> RiskResult:
        # No short selling: cannot sell more than currently held.
        if pos_qty <= 0:
            return RiskResult(False, TradeAction.SELL, 0, "no shares held (shorting disabled)")

        qty = min(decision.quantity, int(pos_qty))
        if qty <= 0:
            return RiskResult(False, TradeAction.SELL, 0, "nothing to sell after limits")

        note = ""
        if qty < decision.quantity:
            note = f" (clamped from {decision.quantity} to held qty)"
        return RiskResult(True, TradeAction.SELL, qty, f"approved sell of {qty}{note}")

    def _evaluate_buy(
        self,
        decision: TradeDecision,
        price: float,
        account: AccountSnapshot,
        position: PositionSnapshot | None,
    ) -> RiskResult:
        s = self._s

        # Kill switch: stop opening/adding risk once the day is down too much.
        if account.day_pnl <= -abs(s.max_daily_loss):
            return RiskResult(
                False,
                TradeAction.BUY,
                0,
                f"daily loss ${account.day_pnl:,.2f} hit limit ${s.max_daily_loss:,.2f}",
            )

        # Per-order notional cap.
        max_qty_trade = math.floor(s.max_trade_notional / price)

        # Per-symbol position notional cap (headroom above what's already held).
        held_notional = position.market_value if position else 0.0
        room = s.max_position_notional - held_notional
        max_qty_room = math.floor(room / price) if room > 0 else 0

        # Buying-power cap.
        max_qty_bp = math.floor(account.buying_power / price)

        qty = min(decision.quantity, max_qty_trade, max_qty_room, max_qty_bp)

        if qty <= 0:
            binding = self._binding_constraint(
                max_qty_trade, max_qty_room, max_qty_bp
            )
            return RiskResult(False, TradeAction.BUY, 0, f"buy blocked by {binding}")

        note = "" if qty == decision.quantity else f" (clamped from {decision.quantity})"
        return RiskResult(True, TradeAction.BUY, qty, f"approved buy of {qty}{note}")

    @staticmethod
    def _binding_constraint(trade: int, room: int, bp: int) -> str:
        smallest = min(trade, room, bp)
        if smallest == room:
            return "position notional limit"
        if smallest == bp:
            return "buying power"
        return "per-trade notional limit"
