# ============================================================
#  main.py  — CrowdWisdomTrading Crypto Prediction Agent
#
#  Entry point. Orchestrates the 5-agent pipeline:
#    Agent 1: Market Scout    → market odds from Polymarket
#    Agent 2: Data Fetcher    → OHLCV from Binance
#    Agent 3: Kronos Predictor → price direction forecast
#    Agent 4: Kelly Risk Mgr  → trade sizing
#    Agent 5: Feedback Loop   → outcome tracking + accuracy
#
#  Usage:
#    python main.py
#    python main.py --asset BTC
#    python main.py --asset ETH
#    python main.py --asset ALL --mode stacked_1min --cycles 3
#
#  All errors are caught at the top level; a clean rich panel is
#  shown on failure and the full traceback is written to logs/error.log.
# ============================================================

import argparse
import sys
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import box

# -----------------------------------------------------------------------
# Configure logging FIRST (before any agent imports that trigger logging)
# -----------------------------------------------------------------------
from utils.logger import configure_logging, get_logger
from utils.config import LOG_LEVEL, PREDICTION_MODE, _mask_key

configure_logging(level=LOG_LEVEL)
log = get_logger(__name__)
console = Console()

# -----------------------------------------------------------------------
# Agent imports (these trigger env validation in utils/config.py)
# -----------------------------------------------------------------------
from agents.market_scout import MarketScoutAgent
from agents.data_fetcher import DataFetcherAgent
from agents.market_analyst import MarketAnalystAgent
from agents.kronos_predictor import KronosPredictorAgent
from agents.kelly_risk_manager import KellyRiskManagerAgent
from agents.feedback_loop import FeedbackLoopAgent, _load_results, _compute_accuracy_stats

# Error log path for reference in the error panel
LOGS_DIR = Path(__file__).parent / "logs"
ERROR_LOG = LOGS_DIR / "error.log"


# -----------------------------------------------------------------------
# CLI Argument Parser
# -----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading — Crypto Prediction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --asset BTC --mode direct_5min\n"
            "  python main.py --asset ALL --mode stacked_1min --cycles 3\n"
        ),
    )
    parser.add_argument(
        "--asset",
        choices=["BTC", "ETH", "ALL"],
        default="ALL",
        help="Which asset(s) to analyse (default: ALL)",
    )
    parser.add_argument(
        "--mode",
        choices=["direct_5min", "stacked_1min"],
        default=PREDICTION_MODE,
        help=f"Prediction mode (default: {PREDICTION_MODE})",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of prediction → feedback cycles to run (default: 1)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Score feedback immediately (skip 5-min wait). Useful for testing.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Welcome Banner
# -----------------------------------------------------------------------
def _print_welcome(args: argparse.Namespace) -> None:
    from utils.config import OPENROUTER_API_KEY, APIFY_API_TOKEN
    console.print(
        Panel(
            "[bold cyan]CrowdWisdomTrading — Crypto Prediction Agent[/bold cyan]\n"
            "[dim]Multi-agent pipeline: Market Scout → Data Fetcher → Kronos → Kelly → Feedback[/dim]\n\n"
            f"  Asset(s) : [yellow]{args.asset}[/yellow]\n"
            f"  Mode     : [yellow]{args.mode}[/yellow]  "
            f"[dim]({'5 stacked 1-min predictions' if args.mode == 'stacked_1min' else 'single 5-bar prediction'})[/dim]\n"
            f"  Cycles   : [yellow]{args.cycles}[/yellow]\n"
            f"  OpenRouter key : [dim]{_mask_key(OPENROUTER_API_KEY)}[/dim]\n"
            f"  Apify token    : [dim]{_mask_key(APIFY_API_TOKEN)}[/dim]\n",
            title="🚀 Starting up",
            border_style="cyan",
            expand=False,
        )
    )


# -----------------------------------------------------------------------
# Final Summary Panel
# -----------------------------------------------------------------------
def _print_final_summary(
    assets: list[str],
    predictions_made: int,
    stats: dict,
    all_warnings: list[str],
    all_errors: list[str],
    run_start: datetime,
) -> None:
    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()

    status_color = "green" if not all_errors else "yellow"
    status_text = "✅ Run completed" if not all_errors else "⚠️  Run completed with warnings"

    lines = [
        f"[bold {status_color}]{status_text}[/bold {status_color}]\n",
        f"[bold]Assets analysed[/bold]    : {', '.join(assets)}",
        f"[bold]Predictions made[/bold]  : {predictions_made}",
        f"[bold]Rolling accuracy[/bold]   : {stats.get('accuracy_pct', 0):.1f}% "
        f"({stats.get('correct', 0)}/{stats.get('total', 0)} scored)",
        f"[bold]Avg Kelly fraction[/bold] : {stats.get('avg_kelly', 0):.4f}",
        f"[bold]Elapsed time[/bold]       : {elapsed:.1f}s",
    ]

    if all_warnings:
        lines.append("\n[bold yellow]⚠  Warnings:[/bold yellow]")
        for w in all_warnings[:5]:  # cap display to 5
            lines.append(f"  • {w}")

    if all_errors:
        lines.append("\n[bold red]✖  Errors:[/bold red]")
        for e in all_errors[:5]:
            lines.append(f"  • {e}")
        lines.append(f"\n[dim]Full details in: {ERROR_LOG}[/dim]")

    console.print(
        Panel(
            "\n".join(lines),
            title="📋 Run Summary",
            border_style=status_color,
            expand=False,
        )
    )


# -----------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> None:
    run_start = datetime.now(timezone.utc)
    assets = ["BTC", "ETH"] if args.asset == "ALL" else [args.asset]
    all_warnings: list[str] = []
    all_errors: list[str] = []
    predictions_made = 0

    # ---- Initialise agents ----
    log.info("[Main] Initialising agents…")
    market_scout = MarketScoutAgent()
    data_fetcher = DataFetcherAgent()
    market_analyst = MarketAnalystAgent()
    kronos = KronosPredictorAgent()
    kelly = KellyRiskManagerAgent()
    feedback = FeedbackLoopAgent()

    # ===== OUTER CYCLE LOOP =====
    for cycle in range(args.cycles):
        if args.cycles > 1:
            console.print(
                Panel(
                    f"Cycle [bold cyan]{cycle + 1}[/bold cyan] of {args.cycles}",
                    border_style="blue",
                    expand=False,
                )
            )

        # ------------------------------------------------------------------
        # AGENT 1 — Market Scout
        # ------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Agent 1:[/bold cyan] Scraping prediction markets…"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("scraping", total=None)
            market_result = market_scout.step(assets=assets)

        log.info(str(market_result))
        all_warnings.extend(market_result.warnings)
        all_errors.extend(market_result.errors)

        # ------------------------------------------------------------------
        # AGENT 2 — Data Fetcher
        # ------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Agent 2:[/bold cyan] Fetching OHLCV data from Binance…"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            progress.add_task("fetching", total=None)
            data_result = data_fetcher.step(assets=assets)

        log.info(str(data_result))
        all_errors.extend(data_result.errors)

        if not data_result.success:
            log.error("[Main] Data fetch failed for all assets. Cannot proceed to prediction.")
            all_errors.append("Data fetch failed — skipping cycle")
            continue

        # Build entry price map (last known close per asset)
        entry_prices = {
            asset: float(df["close"].iloc[-1])
            for asset, df in data_result.data.items()
        }

        # ------------------------------------------------------------------
        # AGENT 2.5 — Market Analyst
        # ------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Agent 2.5:[/bold cyan] Running LLM Market Analysis…"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            progress.add_task("analysing", total=None)
            analyst_result = market_analyst.step(
                assets=assets,
                data=data_result.data,
                predictions=market_result.data,
            )

        log.info(str(analyst_result))
        all_errors.extend(analyst_result.errors)

        # ------------------------------------------------------------------
        # AGENT 3 — Kronos Predictor
        # ------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn(
                f"[bold cyan]Agent 3:[/bold cyan] Running Kronos ({args.mode}) predictions…"
            ),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            progress.add_task("predicting", total=None)
            kronos_result = kronos.step(
                data=data_result.data,
                mode=args.mode,
            )

        log.info(str(kronos_result))
        all_errors.extend(kronos_result.errors)

        if not kronos_result.success:
            log.error("[Main] Kronos prediction failed. Skipping Kelly sizing.")
            continue

        # ------------------------------------------------------------------
        # AGENT 4 — Kelly Risk Manager
        # ------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Agent 4:[/bold cyan] Computing Kelly position sizes…"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            progress.add_task("kelly", total=None)
            kelly_result = kelly.step(
                predictions=kronos_result.data,
                market_odds=market_result.data,
                analyst_insights=analyst_result.data,
                data=data_result.data,
            )

        log.info(str(kelly_result))

        # ---- Print per-asset prediction table ----
        pred_table = Table(
            title="🔮 Predictions & Sizing",
            show_header=True,
            header_style="bold magenta",
            border_style="magenta",
            box=box.ROUNDED,
        )
        pred_table.add_column("Asset", style="bold")
        pred_table.add_column("Direction", justify="center")
        pred_table.add_column("Probability", justify="right")
        pred_table.add_column("Pred Close", justify="right")
        pred_table.add_column("Market Odds", justify="right")
        pred_table.add_column("Kelly Frac", justify="right")
        pred_table.add_column("Bet (USD)", justify="right")
        pred_table.add_column("Mode")

        for asset in assets:
            pred = kronos_result.data.get(asset, {})
            sizing = kelly_result.data.get(asset, {})
            if not pred:
                continue

            direction = pred.get("direction", "?")
            dir_color = "green" if direction == "UP" else "red"

            pred_table.add_row(
                asset,
                f"[{dir_color}]{direction}[/{dir_color}]",
                f"{pred.get('probability', 0):.3f}",
                f"{pred.get('predicted_close', 0):.2f}",
                f"{market_result.data.get(asset, {}).get('net_odds', 0):.3f}",
                f"{sizing.get('kelly_fraction', 0):.4f}",
                f"${sizing.get('hypothetical_bet_usd', 0):.2f}",
                pred.get("mode", args.mode),
            )

            # Register prediction with feedback loop agent
            feedback.add_prediction(
                asset=asset,
                predicted_direction=direction,
                predicted_probability=pred.get("probability", 0.5),
                predicted_close=pred.get("predicted_close", 0),
                market_odds=market_result.data.get(asset, {}).get("net_odds", 1.0),
                kelly_fraction=sizing.get("kelly_fraction", 0),
                mode=pred.get("mode", args.mode),
                entry_price=entry_prices.get(asset, 0),
            )
            predictions_made += 1

        console.print(pred_table)

        # ------------------------------------------------------------------
        # CONSENSUS GAUGE (UI Visibility)
        # ------------------------------------------------------------------
        console.print()
        for asset in assets:
            k_pred = kronos_result.data.get(asset, {})
            if not k_pred:
                continue
            kronos_dir = k_pred.get("direction", "UNKNOWN")
            
            a_insights = analyst_result.data.get(asset, {})
            analyst_score = float(a_insights.get("sentiment_score", 0.0))
            
            m_odds = market_result.data.get(asset, {})
            market_prob = float(m_odds.get("probability", 0.5))
            
            feedback.print_consensus_gauge(
                asset, kronos_dir, analyst_score, market_prob
            )
        console.print()

        # ------------------------------------------------------------------
        # AGENT 5 — Feedback Loop
        # ------------------------------------------------------------------
        wait_seconds = 305 if not args.no_wait else 0  # 5 min + 5s buffer

        if wait_seconds > 0:
            console.print(
                Panel(
                    f"[cyan]Agent 5 (Feedback Loop)[/cyan] waiting [bold]{wait_seconds}s[/bold] "
                    f"for the 5-min window to elapse before scoring…\n"
                    f"[dim]Press Ctrl+C to skip waiting and exit.[/dim]",
                    border_style="blue",
                    expand=False,
                )
            )
            try:
                time.sleep(wait_seconds)
            except KeyboardInterrupt:
                log.info("[Main] Wait interrupted by user. Scoring predictions anyway…")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Agent 5:[/bold cyan] Scoring predictions & updating accuracy…"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            progress.add_task("scoring", total=None)
            feedback_result = feedback.step(wait=False)

        log.info(str(feedback_result))
        all_errors.extend(feedback_result.errors)
        
        # Render table explicitly exactly once per cycle outside of Progress context
        if feedback_result.data and "stats" in feedback_result.data:
            from agents.feedback_loop import _print_accuracy_table
            _print_accuracy_table(feedback_result.data["stats"])

    # ===== FINAL SUMMARY =====
    all_results = _load_results()
    final_stats = _compute_accuracy_stats(all_results)
    _print_final_summary(
        assets=assets,
        predictions_made=predictions_made,
        stats=final_stats,
        all_warnings=all_warnings,
        all_errors=all_errors,
        run_start=run_start,
    )


# -----------------------------------------------------------------------
# Top-level entry point with top-level error handler
# -----------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    _print_welcome(args)

    try:
        run_pipeline(args)

    except KeyboardInterrupt:
        console.print(
            Panel(
                "⚡ [yellow]Run interrupted by user (Ctrl+C).[/yellow]",
                border_style="yellow",
                expand=False,
            )
        )
        sys.exit(0)

    except Exception:
        # Log full traceback to file — show only clean panel to user
        tb = traceback.format_exc()
        LOGS_DIR.mkdir(exist_ok=True)
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{datetime.now(timezone.utc).isoformat()}\n{tb}\n")

        console.print(
            Panel(
                "[bold red]An unexpected error occurred.[/bold red]\n\n"
                f"[dim]Full traceback written to: {ERROR_LOG}[/dim]\n\n"
                f"[yellow]Error summary:[/yellow] {tb.splitlines()[-1]}",
                title="❌ Fatal Error",
                border_style="red",
                expand=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
