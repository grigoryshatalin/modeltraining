"""The provider-agnostic decision-model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import MarketContext
from .schema import TradeDecision


class ModelDecisionError(RuntimeError):
    """Raised when a model fails to produce a valid structured decision."""


class TradingModel(ABC):
    """Anything that can turn a MarketContext into a TradeDecision.

    Implement this once per provider. `ClaudeModel` is the reference
    implementation; `OpenAIModel` is a working second adapter.
    """

    @abstractmethod
    def decide(self, context: MarketContext) -> TradeDecision:
        """Return a structured trade decision for `context.symbol`."""
        raise NotImplementedError
