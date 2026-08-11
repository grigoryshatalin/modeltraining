"""AI decision layer: a provider-agnostic model interface plus adapters.

Kept import-light on purpose — importing this package must NOT pull in the
Anthropic or OpenAI SDKs. Import the concrete adapters directly when needed
(see `modeltraining.factory.build_model`).
"""

from .base import ModelDecisionError, TradingModel
from .schema import TradeAction, TradeDecision

__all__ = ["TradingModel", "ModelDecisionError", "TradeAction", "TradeDecision"]
