"""Orchestrates one decision-and-execution cycle over the watchlist."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .ai.base import ModelDecisionError, TradingModel
from .ai.schema import TradeAction, TradeDecision
from .broker.alpaca import AlpacaBroker
from .config import Settings
from .data.market_data import MarketDataClient
from .risk.guardrails import RiskManager, RiskResult

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """What happened for one symbol during a cycle."""

    symbol: str
    decision: TradeDecision | None
    risk: RiskResult | None
    executed: str  # human-readable outcome
    error: str | None = None


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        broker: AlpacaBroker,
        data: MarketDataClient,
        model: TradingModel,
        risk: RiskManager,
    ) -> None:
        self._s = settings
        self._broker = broker
        self._data = data
        self._model = model
        self._risk = risk

    def run_cycle(self) -> list[CycleResult]:
        """Evaluate every watchlist symbol once and (optionally) place orders."""
        account = self._broker.get_account()
        results: list[CycleResult] = []

        for symbol in self._s.symbol_list:
            results.append(self._run_symbol(symbol, account))
        return results

    def _run_symbol(self, symbol, account) -> CycleResult:  # noqa: ANN001
        try:
            position = self._broker.get_position(symbol)
            context = self._data.build_context(symbol, account, position)
            decision = self._model.decide(context)
        except ModelDecisionError as exc:
            log.warning("%s: model error: %s", symbol, exc)
            return CycleResult(symbol, None, None, "skipped", error=str(exc))
        except Exception as exc:  # pragma: no cover - network path
            log.warning("%s: data/broker error: %s", symbol, exc)
            return CycleResult(symbol, None, None, "skipped", error=str(exc))

        risk = self._risk.evaluate(decision, context, account, position)

        if not risk.approved or risk.quantity <= 0 or decision.action == TradeAction.HOLD:
            return CycleResult(symbol, decision, risk, "no order")

        side = "buy" if risk.action == TradeAction.BUY else "sell"

        if self._s.dry_run:
            return CycleResult(
                symbol, decision, risk, f"DRY-RUN: would {side} {risk.quantity}"
            )

        try:
            order_id = self._broker.submit_market_order(symbol, risk.quantity, side)
        except Exception as exc:  # pragma: no cover - network path
            log.warning("%s: order failed: %s", symbol, exc)
            return CycleResult(symbol, decision, risk, "order failed", error=str(exc))

        return CycleResult(symbol, decision, risk, f"submitted {side} {risk.quantity} ({order_id})")
