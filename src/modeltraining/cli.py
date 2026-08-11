"""Command-line entry point.

Usage:
    modeltraining account            Show account snapshot
    modeltraining positions          List open positions
    modeltraining quote SYMBOL       Show latest quote for a symbol
    modeltraining run                Run one decision cycle over the watchlist
    modeltraining run --loop 300     Repeat every 300 seconds
    modeltraining run --execute      Actually submit orders (overrides DRY_RUN)
"""

from __future__ import annotations

import argparse
import sys
import time

from .broker.alpaca import AlpacaBroker
from .config import get_settings
from .data.market_data import MarketDataClient
from .engine import CycleResult
from .factory import build_engine
from .logging_config import configure_logging


def _cmd_account(_: argparse.Namespace) -> int:
    settings = get_settings()
    broker = AlpacaBroker(settings)
    a = broker.get_account()
    mode = "PAPER" if broker.is_paper else "LIVE"
    print(f"Account ({mode})")
    print(f"  equity          ${a.equity:,.2f}")
    print(f"  cash            ${a.cash:,.2f}")
    print(f"  buying_power    ${a.buying_power:,.2f}")
    print(f"  portfolio_value ${a.portfolio_value:,.2f}")
    print(f"  day_pnl         ${a.day_pnl:,.2f}")
    return 0


def _cmd_positions(_: argparse.Namespace) -> int:
    broker = AlpacaBroker(get_settings())
    positions = broker.list_positions()
    if not positions:
        print("No open positions.")
        return 0
    print(f"{'SYMBOL':<8}{'QTY':>10}{'AVG':>12}{'PRICE':>12}{'MKT VAL':>14}{'UNREAL P/L':>14}")
    for p in positions:
        print(
            f"{p.symbol:<8}{p.qty:>10g}{p.avg_entry_price:>12,.2f}"
            f"{p.current_price:>12,.2f}{p.market_value:>14,.2f}{p.unrealized_pl:>14,.2f}"
        )
    return 0


def _cmd_quote(args: argparse.Namespace) -> int:
    data = MarketDataClient(get_settings())
    symbol = args.symbol.upper()
    bid, ask, mid = data.get_latest_quote(symbol)
    print(f"{symbol}: bid ${bid:,.4f}  ask ${ask:,.4f}  mid ${mid:,.4f}")
    return 0


def _print_results(results: list[CycleResult]) -> None:
    print(f"\n{'SYMBOL':<8}{'ACTION':<8}{'QTY':>6}{'CONF':>7}  OUTCOME")
    for r in results:
        if r.error and r.decision is None:
            print(f"{r.symbol:<8}{'-':<8}{'-':>6}{'-':>7}  error: {r.error}")
            continue
        d = r.decision
        action = d.action.value if d else "-"
        qty = r.risk.quantity if r.risk else 0
        conf = f"{d.confidence:.2f}" if d else "-"
        print(f"{r.symbol:<8}{action:<8}{qty:>6}{conf:>7}  {r.executed}")
        if d and d.rationale:
            print(f"         └ {d.rationale}")


def _cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.execute:
        settings.dry_run = False

    engine = build_engine(settings)
    mode = "EXECUTE" if not settings.dry_run else "DRY-RUN"
    broker_mode = "PAPER" if AlpacaBroker(settings).is_paper else "LIVE"
    print(f"Provider={settings.ai_provider}  Broker={broker_mode}  Mode={mode}  "
          f"Symbols={','.join(settings.symbol_list)}")

    while True:
        results = engine.run_cycle()
        _print_results(results)
        if not args.loop:
            return 0
        print(f"\nSleeping {args.loop}s...  (Ctrl-C to stop)")
        try:
            time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


def _fmt_rows(state) -> None:
    from datetime import date

    start = state.starting_capital
    today = date.today().isoformat()
    alive_n = len(state.alive())
    print(f"\nDay {state.day_index}  |  {alive_n}/{len(state.contestants)} alive  |  "
          f"start ${start:,.0f} each")
    print(f"{'':2}{'CONTESTANT':<20}{'MODEL':<18}{'EQUITY':>9}{'RET%':>7}"
          f"{'WK%':>7}{'API$':>7}{'NET%':>7}  LAST")
    ranked = sorted(state.contestants, key=lambda c: (c.alive, c.net_equity()), reverse=True)
    for c in ranked:
        eq, net = c.equity(), c.net_equity()
        ret = (eq / start - 1) * 100 if start else 0.0
        net_ret = (net / start - 1) * 100 if start else 0.0
        wk = c.weekly_return(today, start)
        flag = " " if c.alive else "x"
        action = c.last_action if c.alive else f"eliminated {c.eliminated_on}"
        print(f"{flag:2}{c.id:<20}{c.model:<18}{eq:>9,.2f}{ret:>7.2f}"
              f"{wk:>7.2f}{c.api_cost:>7.2f}{net_ret:>7.2f}  {action[:28]}")


def _cmd_tournament(args: argparse.Namespace) -> int:
    from .tournament.roster import default_roster
    from .tournament.state import TournamentState

    settings = get_settings()

    # roster / standings need no keys; only init / run build the engine.
    if args.tcmd == "roster":
        roster = default_roster()
        print(f"{len(roster)} contestants (start ${settings.tournament_capital:,.0f} each, "
              f"symbols {','.join(settings.symbol_list)}):\n")
        print(f"{'CONTESTANT':<20}{'PROVIDER':<9}{'MODEL':<20}{'RESEARCH':<9}STRATEGY")
        for c in roster:
            print(f"{c.id:<20}{c.provider:<9}{c.model:<20}"
                  f"{'yes' if c.research else 'no':<9}{c.strategy[:48]}")
        providers = {c.provider for c in roster}
        print("\nProviders needed:", ", ".join(sorted(providers)))
        return 0

    if args.tcmd == "standings":
        if not TournamentState.exists():
            print("No tournament yet. Start one with:  modeltraining tournament init")
            return 1
        _fmt_rows(TournamentState.load())
        return 0

    # init / run need market data (Alpaca keys) and, for run, model keys.
    from .tournament.engine import TournamentEngine

    engine = TournamentEngine(settings)

    if args.tcmd == "init":
        roster = default_roster()
        if args.providers:
            wanted = {p.strip().lower() for p in args.providers.split(",") if p.strip()}
            roster = [c for c in roster if c.provider in wanted]
            if not roster:
                print(f"No contestants match providers {sorted(wanted)}.")
                return 1
        state = engine.init(roster=roster, force=args.force)
        print(f"Initialized tournament: {len(state.contestants)} contestants, "
              f"${state.starting_capital:,.0f} each, symbols {','.join(state.symbols)}.")
        print("Run a daily cycle with:  modeltraining tournament run")
        return 0

    if args.tcmd == "run":
        if not TournamentState.exists():
            print("No tournament yet. Start one with:  modeltraining tournament init")
            return 1
        result = engine.run_day(force_eliminate=args.force_eliminate)
        _fmt_rows(engine.standings())
        print(f"\nCycle complete. Real API spend this run: ${result['spend']:.3f}")
        if result.get("capped"):
            print("(daily spend cap was hit — some contestants were skipped)")
        if result["eliminated"]:
            print(f"Eliminated this week: {result['eliminated']}")
        return 0

    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modeltraining", description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ...")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("account", help="Show account snapshot").set_defaults(func=_cmd_account)
    sub.add_parser("positions", help="List open positions").set_defaults(func=_cmd_positions)

    p_quote = sub.add_parser("quote", help="Show latest quote for a symbol")
    p_quote.add_argument("symbol")
    p_quote.set_defaults(func=_cmd_quote)

    p_run = sub.add_parser("run", help="Run one (or looping) decision cycle")
    p_run.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                       help="Repeat every N seconds instead of running once")
    p_run.add_argument("--execute", action="store_true",
                       help="Actually submit orders (overrides DRY_RUN for this run)")
    p_run.set_defaults(func=_cmd_run)

    p_tourn = sub.add_parser("tournament", help="Multi-model competition (simulated ledgers)")
    tsub = p_tourn.add_subparsers(dest="tcmd", required=True)
    tsub.add_parser("roster", help="Show the configured contestants (no API calls)")
    t_init = tsub.add_parser("init", help="Start a fresh tournament")
    t_init.add_argument("--force", action="store_true", help="Reset an existing tournament")
    t_init.add_argument("--providers", default="",
                        help="Comma list to include only some providers, e.g. 'claude'")
    tsub.add_parser("standings", help="Show the leaderboard (no API calls)")
    t_run = tsub.add_parser("run", help="Run one daily decision cycle for all alive contestants")
    t_run.add_argument("--force-eliminate", action="store_true",
                       help="Eliminate the worst contestant now, ignoring the weekly schedule")
    p_tourn.set_defaults(func=_cmd_tournament)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
