"""Wire the components together from Settings.

Adapters that pull in heavy SDKs (Anthropic / OpenAI) are imported lazily so
that, e.g., running the OpenAI path never requires the Anthropic package.
"""

from __future__ import annotations

from .ai.base import TradingModel
from .broker.alpaca import AlpacaBroker
from .config import Settings, get_settings
from .data.market_data import MarketDataClient
from .engine import TradingEngine
from .risk.guardrails import RiskManager


def build_model(settings: Settings) -> TradingModel:
    provider = settings.ai_provider.strip().lower()
    if provider == "claude":
        from .ai.claude import ClaudeModel

        return ClaudeModel(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key or None,
        )
    if provider == "openai":
        from .ai.openai_model import OpenAIModel

        return OpenAIModel(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
        )
    raise ValueError(f"Unknown AI_PROVIDER: {settings.ai_provider!r} (use 'claude' or 'openai')")


def build_engine(settings: Settings | None = None) -> TradingEngine:
    settings = settings or get_settings()
    broker = AlpacaBroker(settings)
    data = MarketDataClient(settings)
    model = build_model(settings)
    risk = RiskManager(settings)
    return TradingEngine(settings, broker, data, model, risk)
