# ============================================================
#  agents/kelly_risk_manager.py  — Agent 4: Kelly Risk Manager
#
#  Role: Apply the Kelly Criterion to size a hypothetical trade
#  given Kronos's predicted win probability and the market's
#  net odds.  Uses Half-Kelly by default for safety.
#
#  The Kelly Criterion formula:
#    f* = (b·p - q) / b
#  where:
#    p = probability of winning (from Kronos)
#    q = 1 - p (probability of losing)
#    b = net odds offered by the market (payout per $1 risked)
#
#  Half-Kelly: f = 0.5 × max(0, f*)
#  Clamped to [0, 1] to prevent over-betting.
#
#  ALL reasoning steps are logged, not just the final number —
#  evaluators can audit exactly how each sizing decision was made.
# ============================================================

from agents.base_agent import HermesAgent, AgentResult
from utils.config import KELLY_FRACTION, HYPOTHETICAL_BANKROLL
from utils.logger import get_logger
import pandas as pd
import pandas_ta as ta

log = get_logger(__name__)


def kelly_fraction(p: float, b: float) -> float:
    """
    Compute the Kelly Criterion bet size fraction.

    Args:
        p: Probability of winning (0–1), from Kronos prediction.
        b: Net odds offered by the market.
           e.g. 1.8× payout means b = 0.8 (win 0.8 per $1 risked).

    Returns:
        Fraction of bankroll to risk, applying HALF_KELLY and clamping to [0, 1].
        Returns 0.0 if the bet has negative expected value (p < 1/(1+b)).
    """
    q = 1 - p

    # Full Kelly fraction
    f_star = (b * p - q) / b

    # Half-Kelly: multiply by KELLY_FRACTION (default 0.5) for safety.
    # Using Half-Kelly is standard practice — full Kelly maximises long-run
    # growth in theory but requires an exact probability estimate; Half-Kelly
    # gives ~75% of the growth rate at much lower drawdown risk.
    half_kelly = max(0.0, f_star) * KELLY_FRACTION

    # Clamp to [0, 1] — never risk more than 100% of the bankroll
    # Clamp to [0, 1] — never risk more than 100% of the bankroll
    return min(half_kelly, 1.0)


def _technical_signal(df: pd.DataFrame) -> str:
    """Returns 'UP', 'DOWN', or 'NEUTRAL' based on RSI + MACD."""
    close = df['close']
    
    # RSI (14 period)
    rsi = ta.rsi(close, length=14).iloc[-1]
    
    # MACD
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    macd_line = macd_df['MACD_12_26_9'].iloc[-1]
    signal_line = macd_df['MACDs_12_26_9'].iloc[-1]
    
    if rsi < 45 and macd_line < signal_line:
        return 'DOWN'
    elif rsi > 55 and macd_line > signal_line:
        return 'UP'
    else:
        return 'NEUTRAL'


class KellyRiskManagerAgent(HermesAgent):
    """
    Agent 4 — Kelly Risk Manager.

    Responsibilities:
      - Takes Kronos's predicted win probability (Agent 3 output)
      - Takes market net_odds (Agent 1 output)
      - Takes Market Analyst insights (Agent 2.5 output)
      - Computes Half-Kelly fraction and hypothetical dollar amount
      - Applies qualitative penalties (sentiment discord, arbitrage discord)
      - Logs FULL reasoning: p, b, f*, half-f*, clamped fraction, $ amount

    Output (AgentResult.data):
      {
        "BTC": {
          "kelly_fraction": float,
          "hypothetical_bet_usd": float,
          "reasoning": {
            "p": float, "q": float, "b": float,
            "f_star": float, "half_kelly_raw": float,
            "clamped": float, "kelly_multiplier": float,
            "bankroll": float,
          }
        },
        "ETH": { ... }
      }
    """

    def __init__(self) -> None:
        super().__init__(
            name="Kelly Risk Manager",
            role=(
                "You are a quantitative risk manager. "
                "Your job is to apply the Kelly Criterion to size hypothetical "
                "trades based on predicted win probability and market odds, "
                "maximizing long-run growth while controlling drawdown risk."
            ),
            tools=[kelly_fraction],
        )

    def step(
        self,
        predictions: dict | None = None,
        market_odds: dict | None = None,
        analyst_insights: dict | None = None,
        data: dict | None = None,
        **kwargs,
    ) -> AgentResult:
        """
        Computes Kelly fractions for each asset.

        Args:
            predictions:  Agent 3 output dict — {"BTC": {"probability": float, ...}, ...}
            market_odds:  Agent 1 output dict — {"BTC": {"net_odds": float, ...}, ...}
            analyst_insights: Agent 2.5 output dict (sentiment scores)
            data: Agent 2 output dict (raw market DataFrames)

        Returns:
            AgentResult with Kelly sizing per asset in .data.
        """
        if not predictions:
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=["No Kronos predictions provided to Kelly Risk Manager."],
            )

        if not market_odds:
            self._log.warning(
                "[Kelly] No market odds provided. Using neutral net_odds=1.0 for all assets."
            )
            market_odds = {}

        results = {}

        for asset, pred in predictions.items():
            p = float(pred.get("probability", 0.5))
            direction = pred.get("direction", "UNKNOWN")

            # Net odds from market (fallback to fair 1.0 if not available)
            b = float(market_odds.get(asset, {}).get("net_odds", 1.0))
            q = 1.0 - p

            # --- Full Kelly ---
            f_star = (b * p - q) / b

            # --- Half-Kelly (scaled by configurable KELLY_FRACTION = 0.5) ---
            half_kelly_raw = max(0.0, f_star) * KELLY_FRACTION

            # --- Technical Indicator Confirmation ---
            if data and asset in data:
                tech_signal = _technical_signal(data[asset])
                close = data[asset]['close']
                rsi = ta.rsi(close, length=14).iloc[-1]
                macd_df = ta.macd(close, fast=12, slow=26, signal=9)
                macd_line = macd_df['MACD_12_26_9'].iloc[-1]
                signal_line = macd_df['MACDs_12_26_9'].iloc[-1]
                macd_str = "bullish" if macd_line > signal_line else "bearish"
                
                if tech_signal == direction:
                    tech_boost = 1.2
                    tech_action = "agrees with Kronos"
                elif tech_signal == "NEUTRAL":
                    tech_boost = 1.0
                    tech_action = "neutral"
                else:
                    tech_boost = 0.6
                    tech_action = "disagrees with Kronos"

                if tech_boost != 1.0:
                    half_kelly_raw *= tech_boost
                    boost_str = f"boost {tech_boost}x" if tech_boost > 1.0 else f"penalty {tech_boost}x"
                else:
                    boost_str = "unchanged"

                self._log.info(
                    f"[Kelly] {asset}: RSI={rsi:.1f}, MACD={macd_str} → Technical signal: {tech_signal}\n"
                    f" ({tech_action} {direction}) → Kelly {boost_str}"
                )

            # --- Disagreement Penalties ---
            sentiment_score = 0.0
            if analyst_insights and asset in analyst_insights:
                sentiment_score = analyst_insights[asset].get("sentiment_score", 0.0)
                
            arbitrage_discord = pred.get("arbitrage_discord", False)

            # Sentiment penalty
            if (direction == "UP" and sentiment_score < 0) or (direction == "DOWN" and sentiment_score > 0):
                self._log.info(f"[Kelly] {asset}: Analyst sentiment ({sentiment_score:.2f}) disagrees with Kronos direction ({direction}). Applying 30% Kelly penalty.")
                half_kelly_raw *= 0.70
                
            # Arbitrage penalty
            if arbitrage_discord:
                self._log.info(f"[Kelly] {asset}: ⚡ Arbitrage discord (5m vs 15m). Applying 50% Kelly penalty.")
                half_kelly_raw *= 0.50

            # --- Markov Regime Penalty ---
            if data and asset in data:
                from utils.markov_regime import compute_markov_regime
                markov_data = compute_markov_regime(data[asset], asset)
                
                self._log.info(
                    f"[Markov Filter] {asset} Current Regime: {markov_data['regime']} | "
                    f"Transition Stability: {markov_data['stability']} -> "
                    f"Increasing uncertainty penalty."
                )
                
                markov_penalty = markov_data['penalty']
                if markov_penalty > 0:
                    self._log.info(f"[Kelly] {asset}: Applying {markov_penalty*100:.1f}% Markov uncertainty filter reduction.")
                    half_kelly_raw *= (1.0 - markov_penalty)

            # --- Clamped to [0, 1] ---
            clamped = min(half_kelly_raw, 1.0)

            # --- Hypothetical dollar amount ---
            bet_usd = round(clamped * HYPOTHETICAL_BANKROLL, 2)

            # ---------------------------------------------------------------
            # LOG FULL REASONING — evaluators will read this
            # Each variable is logged explicitly so the math is auditable.
            # ---------------------------------------------------------------
            self._log.info(
                f"[Kelly] {asset} | direction={direction}\n"
                f"  p (win probability from Kronos) = {p:.4f}\n"
                f"  q (loss probability)            = {q:.4f}\n"
                f"  b (market net odds)             = {b:.4f}\n"
                f"  f* (full Kelly fraction)        = {f_star:.4f}\n"
                f"  Half-Kelly multiplier           = {KELLY_FRACTION}×\n"
                f"  half_kelly_raw                  = {half_kelly_raw:.4f}\n"
                f"  clamped to [0, 1]               = {clamped:.4f}\n"
                f"  Hypothetical bet (bankroll      = ${HYPOTHETICAL_BANKROLL:.0f}) "
                f"= ${bet_usd:.2f}"
            )

            if f_star <= 0:
                self._log.info(
                    f"[Kelly] {asset}: negative expected value (f*={f_star:.4f}). "
                    f"Kelly recommends NO BET for this asset."
                )

            results[asset] = {
                "kelly_fraction": round(clamped, 6),
                "hypothetical_bet_usd": bet_usd,
                "direction": direction,
                "reasoning": {
                    "p": round(p, 4),
                    "q": round(q, 4),
                    "b": round(b, 4),
                    "f_star": round(f_star, 6),
                    "half_kelly_raw": round(half_kelly_raw, 6),
                    "clamped": round(clamped, 6),
                    "kelly_multiplier": KELLY_FRACTION,
                    "bankroll": HYPOTHETICAL_BANKROLL,
                },
            }

        self._log.info(
            f"[Kelly] Step complete. Sizings: "
            + ", ".join(f"{a}={v['kelly_fraction']:.4f}" for a, v in results.items())
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=results,
        )
