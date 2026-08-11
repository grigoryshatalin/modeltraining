"""Application configuration, loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Alpaca (personal account, API-key auth) ---
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # --- AI decision model ---
    ai_provider: str = "claude"  # "claude" | "openai"
    anthropic_api_key: str = ""  # optional; SDK also reads ANTHROPIC_API_KEY
    claude_model: str = "claude-opus-5"
    openai_api_key: str = ""
    openai_model: str = ""

    # --- Watchlist ---
    symbols: str = "AAPL,MSFT,SPY"
    bar_lookback_days: int = 30

    # --- Risk guardrails (USD) ---
    max_trade_notional: float = 1000.0
    max_position_notional: float = 5000.0
    max_daily_loss: float = 500.0
    min_confidence: float = 0.6

    # --- Safety ---
    dry_run: bool = True

    # --- Tournament (multi-model competition) ---
    tournament_capital: float = 100.0         # starting virtual $ per contestant
    tournament_slippage_bps: float = 5.0      # simulated cost per trade, basis points
    tournament_elimination_days: int = 7      # eliminate the worst every N days (a "week")
    tournament_grace_days: int = 21           # no elimination for the first N days (3 weeks)
    tournament_safe_weekly_return: float = 5.0  # weekly return above this % = safe that week
    tournament_max_daily_spend: float = 5.0   # halt a run once real API spend hits this
    tournament_min_confidence: float = 0.55   # confidence needed to act (overrides min_confidence)

    @property
    def symbol_list(self) -> list[str]:
        """Watchlist parsed from the comma-separated `SYMBOLS` value."""
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    def require_alpaca_keys(self) -> None:
        """Raise a clear error if Alpaca credentials are missing."""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise RuntimeError(
                "Alpaca credentials are not set. Copy .env.example to .env and set "
                "ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys from app.alpaca.markets)."
            )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
