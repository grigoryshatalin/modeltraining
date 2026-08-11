# modeltraining

An AI-model-driven trading bot for a **personal Alpaca account**. An LLM looks at
market context for each symbol on your watchlist and returns a structured
`buy` / `sell` / `hold` decision; a deterministic risk layer then sizes and gates
every order before it can reach the broker.

> ⚠️ **Not financial advice.** This is an educational scaffold. It defaults to
> **paper trading** and **dry-run** (decide-and-log, never submit). Read the
> [safety model](#safety-model) before changing those defaults. Automated trading
> can lose money quickly — you are responsible for anything it does.

---

## Safety model

Three independent layers have to agree before real money moves. All default to safe:

| Layer | Setting | Safe default | What it does |
|---|---|---|---|
| Broker environment | `ALPACA_PAPER` | `true` | Trades against Alpaca's paper account, not your real one. |
| Execution switch | `DRY_RUN` | `true` | The engine decides and logs but **never submits orders**. |
| Risk guardrails | `MAX_*`, `MIN_CONFIDENCE` | conservative | Hard caps on order size, position size, daily loss, and confidence — applied in code, after the model, and impossible for the model to override. |

To place a real order you must flip **all** of: `ALPACA_PAPER=false`, `DRY_RUN=false`
(or `run --execute`), and pass the risk checks. Even then, start on paper.

---

## How a decision flows

```
watchlist symbol
      │
      ▼
 MarketDataClient ──► latest quote + recent daily bars ─┐
      │                                                 │
 AlpacaBroker ─────► account snapshot + open position ──┤
                                                        ▼
                                                 MarketContext
                                                        │
                                                        ▼
                                          TradingModel.decide()      (Claude / OpenAI)
                                                        │
                                                        ▼
                                          TradeDecision {action, qty, confidence, rationale}
                                                        │
                                                        ▼
                                          RiskManager.evaluate()     ← hard caps, sizing, kill-switch
                                                        │
                                    approved & not dry-run ─► AlpacaBroker.submit_market_order()
```

The AI proposes; the risk layer disposes. Quantities are only ever **clamped down**,
never up.

## Project layout

```
src/modeltraining/
├── config.py            # env / .env settings (pydantic-settings)
├── context.py           # neutral domain types (Bar, MarketContext, Account/Position snapshots)
├── ai/
│   ├── base.py          # TradingModel interface (provider-agnostic)
│   ├── schema.py        # TradeDecision structured-output schema
│   ├── prompt.py        # shared, provider-neutral prompt
│   ├── claude.py        # Anthropic Claude adapter (reference implementation)
│   └── openai_model.py  # OpenAI adapter (optional extra)
├── risk/guardrails.py   # deterministic order gating + sizing  ← the safety core
├── broker/alpaca.py     # thin alpaca-py TradingClient wrapper
├── data/market_data.py  # quotes + bars → MarketContext
├── engine.py            # one decision-and-execution cycle
├── factory.py           # wires everything from Settings
├── tournament/          # multi-model competition (see "Model tournament" below)
│   ├── roster.py        #   the 10 contestants (models × strategies)
│   ├── ledger.py        #   simulated long-only paper book
│   ├── brains.py        #   per-model web research + structured decision
│   ├── pricing.py       #   real API cost from token usage
│   ├── state.py         #   persistent tournament state (JSON)
│   └── engine.py        #   daily cycle + weekly elimination
└── cli.py               # `modeltraining` command
tests/                   # risk engine, schema, ledger, and an offline engine cycle
```

---

## Setup

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # add '.[openai]' for the OpenAI adapter, '.[dev]' for pytest
```

Requires Python 3.9+.

### 2. Configure

```bash
cp .env.example .env
```

Then edit `.env`:

- **Alpaca keys** — create free **paper** keys at
  [app.alpaca.markets](https://app.alpaca.markets) → *Paper Trading* → *API Keys*.
  Put them in `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Keep `ALPACA_PAPER=true`.
- **AI keys** — for the default Claude provider, set `ANTHROPIC_API_KEY`
  (from [console.anthropic.com](https://console.anthropic.com)). The default model is
  `claude-opus-5`. For OpenAI, set `AI_PROVIDER=openai`, `OPENAI_API_KEY`, and
  `OPENAI_MODEL` (a structured-output-capable model).
- **Watchlist & risk** — tune `SYMBOLS`, `MAX_TRADE_NOTIONAL`,
  `MAX_POSITION_NOTIONAL`, `MAX_DAILY_LOSS`, `MIN_CONFIDENCE`.

### 3. Run

```bash
modeltraining account            # sanity-check credentials + show account
modeltraining positions          # list open positions
modeltraining quote AAPL         # latest quote for a symbol
modeltraining run                # one decision cycle (DRY-RUN by default)
modeltraining run --loop 300     # repeat every 5 minutes
modeltraining run --execute      # actually submit orders (still paper unless ALPACA_PAPER=false)
```

`run` prints, per symbol, the model's action, the risk-approved quantity, the
confidence, the outcome, and the model's one-line rationale.

---

## Risk configuration

Set in `.env` (all USD except confidence):

| Var | Meaning |
|---|---|
| `MAX_TRADE_NOTIONAL` | Largest dollar value of a single order. |
| `MAX_POSITION_NOTIONAL` | Largest dollar value held in any one symbol. |
| `MAX_DAILY_LOSS` | Once today's P/L is down this much, **new buys are halted** (kill switch). |
| `MIN_CONFIDENCE` | Minimum model confidence (0–1) required to act on a decision. |

Additional hard rules in `risk/guardrails.py`: **no short selling** (can't sell more
than held), buys are capped by available buying power, and `HOLD` never places an
order. These are the most important lines in the project — they have unit tests, and
you should read them before trusting the bot with anything.

---

## Using / swapping the AI model

The model is behind a small interface (`ai/base.py`):

```python
class TradingModel(ABC):
    def decide(self, context: MarketContext) -> TradeDecision: ...
```

- **Claude** (`ai/claude.py`) is the reference implementation — it uses the Anthropic
  Messages API structured-output helper so the model returns a validated
  `TradeDecision`, with adaptive thinking enabled.
- **OpenAI** (`ai/openai_model.py`) is a second, working adapter using OpenAI's
  structured outputs (install with `pip install -e '.[openai]'`).

Both share the same prompt (`ai/prompt.py`). To add another provider, implement
`TradingModel.decide` and register it in `factory.build_model`.

---

## Model tournament (experiment)

A built-in competition where **10 contestants** each trade a simulated **$100
book**, and the worst performer is eliminated weekly until one remains. Every
contestant runs the *identical* setup — same watchlist, risk rules, starting
capital, and prompt scaffold — so the only variables are the **model** and the
**strategy persona**. No real orders are ever placed; ledgers are marked to real
daily closes.

**The field** (`src/modeltraining/tournament/roster.py`, fully editable): a mix
of 8 Claude contestants (Opus 5, Opus 4.8, Sonnet 5, Haiku 4.5, Fable 5) and 2
OpenAI contestants, spread across strategy personas — momentum, value/contrarian,
macro/news, mean-reversion, systematic, and a buy-and-hold baseline.

**Research:** contestants flagged `research=yes` (the Opus/Sonnet Claude models)
do their **own web research** each day via Claude's web-search tool — pulling
recent news/catalysts into their decision — before the price-based call. Others
trade on price + strategy only.

**Everyone sees everyone.** Before deciding, each model is shown the **live
standings** — every rival's current book and return, with its own row flagged —
so it can play the competition (press an edge, take more risk to catch up).

**Scoring is net of API cost.** Each model's real token spend is charged against
its book, so a pricey model has to *earn* its cost. Elimination drops the lowest
**net** return.

### Elimination rules

- **3-week grace period:** no eliminations for the first 21 days. The first cut
  happens in **week 4**, then at most one **per week** after that.
- **Safe at +5%:** any contestant whose **trailing-week return exceeds 5%** is
  safe and cannot be cut that week. Elimination targets the worst contestant
  *among the non-safe*. If everyone is safe, nobody goes that week.
- `tournament run --force-eliminate` overrides both (manual "cut the worst now").

All four knobs — grace length, weekly cadence, safe threshold, starting capital —
are `TOURNAMENT_*` variables in `.env`.

### Commands

```bash
modeltraining tournament roster       # show the 10 contestants (no API calls)
modeltraining tournament init         # start a fresh tournament ($100 each)
modeltraining tournament run          # one daily cycle: research → decide → simulate
modeltraining tournament standings    # leaderboard (no API calls)
modeltraining tournament run --force-eliminate   # eliminate the worst now (else weekly)
```

State persists in `state/tournament.json` (git-ignored), so run it once a day
(cron/launchd, or by hand) and it accumulates over weeks.

### Keys, cost, and running Claude-only

- **Keys:** Claude contestants need `ANTHROPIC_API_KEY`; the 2 OpenAI contestants
  need `OPENAI_API_KEY` **and** `pip install -e '.[openai]'`. `run` refuses to
  start (before spending anything) if a needed key is missing.
- **Claude-only:** if you only have an Anthropic key, start an 8-model field with
  `modeltraining tournament init --providers claude`.
- **Cost:** roughly **~$1 per daily run** for the full field (research calls
  dominate), i.e. a few dollars to ~$20/month — and it shrinks as models are
  eliminated. `TOURNAMENT_MAX_DAILY_SPEND` (default $5) hard-stops a run if it
  ever exceeds that.

### Honest caveats

Over a few weeks this is **dominated by luck and market regime, not skill** — in
a rising market whoever holds the most winners looks brilliant. Treat the
"winner" as entertainment plus a look at *behavior* (trade frequency, sizing,
reaction to drawdowns), not proof any model can predict prices. The net-of-cost
scoring is the part that asks a real question: *is this model worth running?*

Tune it via the `TOURNAMENT_*` variables in `.env` (starting capital, slippage,
elimination cadence, spend cap, confidence threshold).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

The tests cover the risk engine (clamping, kill switch, no-shorting, confidence
gating) and the decision schema. They run offline — no API keys or network required.

---

## Notes on authentication (OAuth)

This bot uses **Alpaca API keys**, which is the standard path for trading your own
account. You linked Alpaca's
[OAuth2 token issuance endpoint](https://docs.alpaca.markets/us/reference/issuetokens)
(`https://authx.alpaca.markets/v1/oauth2/token`). That flow is for building an
**app that trades on behalf of other Alpaca users** — you'd exchange client
credentials / authorization grants for a short-lived bearer token and call the
trading API with it instead of static keys. If you go that direction later, the
natural place to plug it in is a new broker adapter alongside `broker/alpaca.py`
that acquires and refreshes an OAuth token; the rest of the app (engine, risk, AI)
is unchanged because it only depends on the broker interface.

---

## Roadmap ideas

- Limit / bracket orders and stop-losses (currently market orders only).
- Backtesting harness over historical bars using the same model + risk path.
- Richer market context (intraday bars, fundamentals, news) in the prompt.
- Persisting decisions and fills for later review.
