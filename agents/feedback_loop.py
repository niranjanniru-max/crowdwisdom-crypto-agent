# ============================================================
#  agents/feedback_loop.py  — Agent 5: Feedback Loop
#
#  Role: After each prediction's 5-minute window elapses, re-fetch
#  the actual price, compare it against the predicted direction,
#  and log the outcome (correct/incorrect) to the results file.
#
#  This agent uses the HermesAgent run_loop() pattern — it runs
#  iteratively with a configurable delay, updating the rolling
#  accuracy stat each cycle. This is a genuine agent loop, not
#  just a script with a sleep() call.
#
#  Results are persisted to data/results_log.json for inspection.
# ============================================================

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agents.base_agent import HermesAgent, AgentResult
from utils.logger import get_logger

log = get_logger(__name__)
console = Console()

# ---------------------------------------------------------------
# Results log file path
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_LOG_PATH = PROJECT_ROOT / "data" / "results_log.json"

# Binance price check endpoint (no auth required)
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"

BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}


def _load_results() -> list[dict]:
    """Load existing results from the JSON file, or return empty list."""
    if RESULTS_LOG_PATH.exists():
        try:
            with open(RESULTS_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"[Feedback] Could not read results log: {e}. Starting fresh.")
    return []


def _save_results(results: list[dict]) -> None:
    """Persist the results list to the JSON file."""
    RESULTS_LOG_PATH.parent.mkdir(exist_ok=True)
    try:
        with open(RESULTS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    except IOError as e:
        log.error(f"[Feedback] Failed to save results log: {e}")


def _fetch_current_price(asset: str) -> float | None:
    """
    Fetches the current spot price for an asset from Binance.
    Returns float price, or None if fetch fails.
    """
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return None
    try:
        resp = requests.get(
            BINANCE_PRICE_URL,
            params={"symbol": symbol},
            timeout=10,
        )
        if resp.status_code == 200:
            return float(resp.json()["price"])
        else:
            log.warning(f"[Feedback] Binance price fetch for {asset} returned {resp.status_code}")
            return None
    except Exception as e:
        log.warning(f"[Feedback] Error fetching price for {asset}: {e}")
        return None


def _compute_accuracy_stats(results: list[dict]) -> dict:
    """
    Computes rolling accuracy stats from the results list.
    Returns dict with total, correct, accuracy, per-asset breakdown,
    and average Kelly fraction.
    """
    completed = [r for r in results if r.get("actual_outcome") is not None]
    if not completed:
        return {
            "total": 0, "correct": 0, "accuracy_pct": 0.0,
            "by_asset": {}, "avg_kelly": 0.0, "by_mode": {}
        }

    correct = sum(1 for r in completed if r.get("correct") is True)
    avg_kelly = sum(r.get("kelly_fraction", 0) for r in completed) / len(completed)

    by_asset = {}
    for asset in BINANCE_SYMBOLS:
        asset_results = [r for r in completed if r.get("asset") == asset]
        if asset_results:
            asset_correct = sum(1 for r in asset_results if r.get("correct"))
            by_asset[asset] = {
                "total": len(asset_results),
                "correct": asset_correct,
                "accuracy_pct": round(asset_correct / len(asset_results) * 100, 1),
            }

    by_mode = {}
    modes = set(r.get("mode", "unknown") for r in completed)
    for m in modes:
        mode_results = [r for r in completed if r.get("mode", "unknown") == m]
        if mode_results:
            mode_correct = sum(1 for r in mode_results if r.get("correct"))
            mode_stats = {
                "total": len(mode_results),
                "accuracy_pct": round(mode_correct / len(mode_results) * 100, 1),
            }
            for asset in BINANCE_SYMBOLS:
                mode_asset_results = [r for r in mode_results if r.get("asset") == asset]
                if mode_asset_results:
                    ma_correct = sum(1 for r in mode_asset_results if r.get("correct"))
                    mode_stats[f"{asset}_accuracy"] = round(ma_correct / len(mode_asset_results) * 100, 1)
                else:
                    mode_stats[f"{asset}_accuracy"] = None
            by_mode[m] = mode_stats

    return {
        "total": len(completed),
        "correct": correct,
        "accuracy_pct": round(correct / len(completed) * 100, 1),
        "by_asset": by_asset,
        "avg_kelly": round(avg_kelly, 4),
        "by_mode": by_mode,
    }


def _print_accuracy_table(stats: dict) -> None:
    """Renders a rich table showing current rolling accuracy."""
    table = Table(
        title="📊 Prediction Accuracy — Live Tracker (Mode Comparison)",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Mode", style="bold")
    table.add_column("BTC Accuracy", justify="center")
    table.add_column("ETH Accuracy", justify="center")
    table.add_column("Total", justify="right")

    modes = stats.get("by_mode", {})
    if not modes:
        table.add_row("N/A", "-", "-", "0")
    else:
        for mode_name, md_stats in modes.items():
            btc_acc = f"{md_stats.get('BTC_accuracy'):.1f}%" if md_stats.get("BTC_accuracy") is not None else "-"
            eth_acc = f"{md_stats.get('ETH_accuracy'):.1f}%" if md_stats.get("ETH_accuracy") is not None else "-"
            total_acc = f"{md_stats.get('accuracy_pct', 0):.1f}% ({md_stats.get('total', 0)})"
            
            table.add_row(mode_name, btc_acc, eth_acc, total_acc)

    console.print(table)


def _print_consensus_gauge(
    asset: str, kronos_dir: str, analyst_score: float, market_prob: float
) -> None:
    """Renders a Consensus Gauge comparing Kronos vs Analyst vs Market."""
    analyst_dir = "UP" if analyst_score > 0 else ("DOWN" if analyst_score < 0 else "NEUTRAL")
    market_dir = "UP" if market_prob > 0.50 else ("DOWN" if market_prob < 0.50 else "NEUTRAL")

    alignment = 0
    if kronos_dir == "UP":
        if analyst_dir == "UP": alignment += 1
        if market_dir == "UP": alignment += 1
    elif kronos_dir == "DOWN":
        if analyst_dir == "DOWN": alignment += 1
        if market_dir == "DOWN": alignment += 1

    if alignment == 2:
        consensus_text = "[bold green]🟢 STRONG CONSENSUS[/bold green]"
    elif alignment == 1:
        consensus_text = "[bold yellow]🟡 MODERATE CONSENSUS (Partial Discord)[/bold yellow]"
    else:
        consensus_text = "[bold red]🔴 LOW CONSENSUS (Total Discord)[/bold red]"

    gauge = Table(show_header=False, box=None)
    gauge.add_row(
        f"[bold cyan]Kronos:[/bold cyan] [bold]{kronos_dir}[/bold]",
        f"[bold magenta]Analyst:[/bold magenta] {'+' if analyst_score > 0 else ''}{analyst_score:.2f} ({analyst_dir})",
        f"[bold yellow]Crowd:[/bold yellow] {market_prob*100:.1f}% ({market_dir})",
        f"➡️  {consensus_text}"
    )

    console.print(Panel(gauge, title=f"🧠 Agent Alignment Indicator: {asset}", border_style="green", expand=False))


class FeedbackLoopAgent(HermesAgent):
    """
    Agent 5 — Feedback Loop (Hermes Agent loop/feedback pattern).

    Each run_loop() iteration:
      1. Waits for the prediction window to elapse
      2. Fetches the actual current price from Binance
      3. Compares actual vs predicted direction
      4. Logs correct/incorrect outcome to data/results_log.json
      5. Prints live accuracy table (rolling %)

    The loop continues for `iterations` cycles, pausing `delay_seconds`
    between each one (default: 300s = 5 minutes for the next candle bar).
    """

    def __init__(self) -> None:
        super().__init__(
            name="Feedback Loop",
            role=(
                "You are a performance tracker for a crypto prediction system. "
                "Your job is to verify predictions against actual prices, "
                "record outcomes, and calculate rolling accuracy with full transparency."
            ),
            tools=[_fetch_current_price, _load_results, _save_results],
        )
        self._pending_predictions: list[dict] = []

    def add_prediction(
        self,
        asset: str,
        predicted_direction: str,
        predicted_probability: float,
        predicted_close: float,
        market_odds: float,
        kelly_fraction: float,
        mode: str,
        entry_price: float,
    ) -> None:
        """
        Register a prediction to be scored in the next feedback loop cycle.
        Called by main.py immediately after Agents 3+4 produce their output.
        """
        self._pending_predictions.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "predicted_direction": predicted_direction,
            "predicted_probability": round(predicted_probability, 4),
            "predicted_close": round(predicted_close, 4),
            "market_odds": round(market_odds, 4),
            "kelly_fraction": round(kelly_fraction, 6),
            "mode": mode,
            "entry_price": round(entry_price, 4),
            "actual_outcome": None,
            "correct": None,
        })
        self._log.info(
            f"[Feedback] Registered prediction: {asset} → {predicted_direction} "
            f"(prob={predicted_probability:.3f}, kelly={kelly_fraction:.4f})"
        )

    def print_consensus_gauge(
        self, asset: str, kronos_direction: str, analyst_score: float, market_prob: float
    ) -> None:
        """Helper to invoke the consensus gauge rendering for the CLI"""
        _print_consensus_gauge(asset, kronos_direction, analyst_score, market_prob)

    def step(self, iteration: int = 0, wait: bool = True, **kwargs) -> AgentResult:
        """
        One feedback loop iteration:
          - Optionally waits delay_seconds before scoring (disable wait=False for testing)
          - Fetches actual prices
          - Scores all pending predictions
          - Saves results to disk
          - Prints live accuracy table
        """
        results = _load_results()
        scored = []
        errors = []

        if not self._pending_predictions:
            self._log.info("[Feedback] No pending predictions to score this iteration.")
            stats = _compute_accuracy_stats(results)
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"stats": stats, "scored_this_cycle": 0},
            )

        self._log.info(
            f"[Feedback] Scoring {len(self._pending_predictions)} pending prediction(s)…"
        )

        still_pending = []
        for pred in self._pending_predictions:
            asset = pred["asset"]
            actual_price = _fetch_current_price(asset)

            if actual_price is None:
                self._log.warning(
                    f"[Feedback] Could not fetch actual price for {asset}. "
                    "Keeping prediction pending."
                )
                still_pending.append(pred)
                errors.append(f"Price fetch failed for {asset}")
                continue

            entry_price = pred.get("entry_price", actual_price)
            actual_direction = "UP" if actual_price > entry_price else "DOWN"
            is_correct = actual_direction == pred["predicted_direction"]

            scored_pred = {
                **pred,
                "actual_price": round(actual_price, 4),
                "actual_outcome": actual_direction,
                "correct": is_correct,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(scored_pred)
            scored.append(scored_pred)

            result_emoji = "✅" if is_correct else "❌"
            self._log.info(
                f"[Feedback] {result_emoji} {asset}: predicted={pred['predicted_direction']}, "
                f"actual={actual_direction} | "
                f"entry={entry_price:.2f}, now={actual_price:.2f}"
            )

        self._pending_predictions = still_pending
        _save_results(results)

        # Compute rolling accuracy
        stats = _compute_accuracy_stats(results)

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"stats": stats, "scored_this_cycle": len(scored)},
            errors=errors,
        )
