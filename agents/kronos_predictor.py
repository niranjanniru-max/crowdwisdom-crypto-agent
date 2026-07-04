# ============================================================
#  agents/kronos_predictor.py  — Agent 3: Kronos Predictor
#
#  Role: Load the Kronos-mini time-series model from HuggingFace
#  and use it to forecast BTC/ETH price direction for the next
#  5 minutes. Supports two prediction modes:
#
#  "direct_5min":   Single predict() call with pred_len=5
#  "stacked_1min":  5 sequential 1-min predictions, each window
#                   shifted forward; majority-vote for direction
#                   and compounded probability for confidence.
#
#  The model and tokenizer are loaded once and cached in memory
#  across all calls within a run (module-level singleton).
#  Model loading is slow (~30-60s on CPU); inference is fast.
#
#  IMPORTANT: Kronos is cloned from GitHub (not PyPI) because
#  the `model` module is a local file import, not a package.
#  Auto-clone happens on first use if kronos_src/ is missing.
# ============================================================

import sys
import subprocess
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import HermesAgent, AgentResult
from utils.config import PREDICTION_MODE
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------
# Kronos GitHub repository — cloned into kronos_src/ at project root
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_SRC_DIR = PROJECT_ROOT / "kronos_src"
KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"

# HuggingFace Hub identifiers
KRONOS_MODEL_ID = "NeoQuasar/Kronos-mini"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"

# Context window for Kronos-mini (from official docs)
KRONOS_MAX_CONTEXT = 2048


# ---------------------------------------------------------------
# Module-level singleton — loaded once per process
# ---------------------------------------------------------------
_predictor = None   # KronosPredictor instance; None until first use


def _ensure_kronos_available() -> None:
    """
    Checks that kronos_src/ contains the Kronos model code.
    If not, clones the repo. Adds the src dir to sys.path so
    `from model import Kronos, ...` works as the repo expects.
    """
    model_file = KRONOS_SRC_DIR / "model" / "__init__.py"
    if not model_file.exists():
        log.info(
            f"[Kronos] kronos_src/ not found or incomplete. "
            f"Cloning {KRONOS_REPO_URL} → {KRONOS_SRC_DIR}"
        )
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", KRONOS_REPO_URL, str(KRONOS_SRC_DIR)],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info("[Kronos] Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to clone Kronos repo: {e.stderr}. "
                "Check internet connectivity and git installation."
            ) from e

    # Add to sys.path so `from model import ...` works
    src_str = str(KRONOS_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        log.debug(f"[Kronos] Added to sys.path: {src_str}")


def _load_predictor():
    """
    Loads Kronos-mini + Kronos-Tokenizer-2k from HuggingFace Hub.
    Returns a KronosPredictor instance (cached after first call).

    The model is loaded on CPU (device="cpu") — no GPU required.
    First load takes 30-90s to download weights; subsequent calls
    use the cached _predictor singleton and are near-instant.
    """
    global _predictor
    if _predictor is not None:
        log.debug("[Kronos] Using cached predictor (already loaded).")
        return _predictor

    _ensure_kronos_available()

    log.info(
        f"[Kronos] Loading model {KRONOS_MODEL_ID} + tokenizer {KRONOS_TOKENIZER_ID} "
        f"from HuggingFace Hub (CPU). First load may take 1-2 minutes…"
    )

    try:
        # These imports work because we added kronos_src/ to sys.path above
        from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
        import os
        import logging
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            # Suppress unauthenticated warning from HuggingFace Hub
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

        tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID, token=hf_token)
        model = Kronos.from_pretrained(KRONOS_MODEL_ID, token=hf_token)
        _predictor = KronosPredictor(
            model,
            tokenizer,
            device="cpu",
            max_context=KRONOS_MAX_CONTEXT,
        )
        log.info("[Kronos] ✅ Model loaded and cached in memory.")
        return _predictor

    except Exception as e:
        raise RuntimeError(
            f"Failed to load Kronos model: {e}. "
            f"Ensure HuggingFace Hub access and the kronos_src/ repo is intact."
        ) from e


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Average True Range (ATR) over the last `period` bars.
    Used as a volatility normaliser for the probability proxy below.
    """
    high = df["high"].values[-period:]
    low = df["low"].values[-period:]
    close = df["close"].values[-period:]
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    return float(np.mean(tr[1:]))  # skip first NaN row


def _direction_and_probability(
    last_close: float,
    pred_df: pd.DataFrame,
    df_history: pd.DataFrame,
) -> dict:
    """
    Derives direction and a probability proxy from a single Kronos forecast.

    METHODOLOGY (sample_count=1, deterministic path):
      - direction: "UP" if forecasted final close > last actual close, else "DOWN"
      - probability proxy: we map the normalised price move to a confidence score
        using sigmoid-like rescaling on the move's magnitude relative to ATR:

          move_size = (pred_close - last_close) / last_close
          normalised = |move_size| / ATR_pct   # how "large" is this move vs volatility
          probability = tanh(normalised * 2) * 0.4 + 0.5

        This gives:
          - Tiny move  → ~0.50-0.55  (low confidence, close to 50/50)
          - Normal move → ~0.60-0.70 (moderate confidence)
          - Big move   → ~0.85-0.90  (high confidence, capped by tanh)
        Documented here because this is a judgment call — if sample_count > 1,
        replace this with the fraction of samples predicting "UP".
    """
    pred_close = float(pred_df["close"].iloc[-1])
    direction = "UP" if pred_close > last_close else "DOWN"

    atr = _compute_atr(df_history)
    price_pct = last_close * 0.01  # 1% of price as ATR floor to avoid /0
    atr = max(atr, price_pct)

    move_size = abs(pred_close - last_close)
    normalised = move_size / atr
    raw_prob = math.tanh(normalised * 2) * 0.4 + 0.5  # range [0.50, 0.90]

    # If direction is DOWN we invert: P(DOWN) = raw_prob
    # P(UP) is what we store for Kelly — same confidence for DOWN is fine
    probability = round(raw_prob, 4)

    return {
        "direction": direction,
        "probability": probability,
        "predicted_close": round(pred_close, 4),
    }


def _predict_direct_5min(
    predictor, df: pd.DataFrame, asset: str
) -> dict:
    """
    Mode: direct_5min
    Single predict() call with pred_len=5 (5 one-minute bars).
    The final forecasted close vs current close gives direction.
    """
    lookback = min(len(df), KRONOS_MAX_CONTEXT)
    x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = x_df.index.to_series().reset_index(drop=True)

    # Generate 5 future timestamps continuing from the last bar (1-min freq)
    last_ts = df.index[-1]
    y_timestamps = pd.date_range(start=last_ts, periods=6, freq="1min")[1:]
    y_timestamp = pd.Series(y_timestamps)

    log.info(f"[Kronos] {asset} direct_5min: predicting 5 bars ahead from {last_ts}")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=5,
        T=1.0,
        top_p=0.9,
        sample_count=1,
    )

    last_close = float(df["close"].iloc[-1])
    result = _direction_and_probability(last_close, pred_df, df)
    result["mode"] = "direct_5min"
    return result


def _predict_stacked_1min(
    predictor, df: pd.DataFrame, asset: str
) -> dict:
    """
    Mode: stacked_1min  (Scale Idea — Extra Credit)
    Runs 5 sequential 1-minute predictions. After each prediction,
    the forecasted bar is appended to the context window so each
    subsequent prediction sees the prior predicted bar (compounding).

    Direction: majority vote (>=3 of 5 bars predict UP → "UP")
    Probability: geometric mean of per-bar confidence scores
    This is a stronger signal than a single 5-bar call because it:
      a) gives an explicit view of how each minute is expected to move
      b) compounds uncertainty — if early bars disagree the probability
         converges toward 0.5 (low confidence), which is honest.
    """
    log.info(f"[Kronos] {asset} stacked_1min: running 5 sequential 1-min predictions")

    current_df = df.copy()
    per_bar_results = []

    for bar_idx in range(5):
        lookback = min(len(current_df), KRONOS_MAX_CONTEXT)
        x_df = current_df.tail(lookback)[["open", "high", "low", "close", "volume"]].copy()
        x_timestamp = x_df.index.to_series().reset_index(drop=True)

        last_ts = current_df.index[-1]
        next_ts = last_ts + pd.Timedelta(minutes=1)
        y_timestamp = pd.Series([next_ts])

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=1,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )

        last_close = float(current_df["close"].iloc[-1])
        bar_result = _direction_and_probability(last_close, pred_df, current_df)
        per_bar_results.append(bar_result)

        # Append the predicted bar to the context for the next iteration
        pred_row = pred_df.iloc[0]
        new_bar = pd.DataFrame(
            {
                "open": [float(pred_row.get("open", last_close))],
                "high": [float(pred_row.get("high", last_close))],
                "low": [float(pred_row.get("low", last_close))],
                "close": [float(pred_row["close"])],
                "volume": [float(pred_row.get("volume", 0))],
            },
            index=[next_ts],
        )
        current_df = pd.concat([current_df, new_bar])

        log.debug(
            f"[Kronos] {asset} bar {bar_idx+1}/5: "
            f"direction={bar_result['direction']}, "
            f"prob={bar_result['probability']:.3f}, "
            f"pred_close={bar_result['predicted_close']:.4f}"
        )

    # Aggregate: majority vote for direction
    up_votes = sum(1 for r in per_bar_results if r["direction"] == "UP")
    final_direction = "UP" if up_votes >= 3 else "DOWN"

    # Compound probability: geometric mean of per-bar confidence scores
    probs = [r["probability"] for r in per_bar_results]
    compound_prob = float(np.prod(np.array(probs)) ** (1.0 / len(probs)))
    # If direction flips from minority bars, reduce confidence toward 0.5
    minority_correction = min(up_votes, 5 - up_votes) / 5  # 0 = unanimous, 0.4 = 3-2 split
    final_prob = round(compound_prob * (1 - minority_correction * 0.2) + 0.5 * minority_correction * 0.2, 4)

    final_close = per_bar_results[-1]["predicted_close"]

    log.info(
        f"[Kronos] {asset} stacked_1min result: "
        f"direction={final_direction} ({up_votes}/5 UP votes), "
        f"probability={final_prob:.3f}, "
        f"predicted_close={final_close:.4f}"
    )
    
    arbitrage_discord = False
    try:
        log.info(f"[Kronos] {asset} running secondary 15-min direct prediction for arbitrage signal...")
        lookback = min(len(df), KRONOS_MAX_CONTEXT)
        x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]].copy()
        x_timestamp = x_df.index.to_series().reset_index(drop=True)
        
        last_ts = df.index[-1]
        y_timestamps = pd.date_range(start=last_ts, periods=16, freq="1min")[1:]
        y_timestamp = pd.Series(y_timestamps)
        
        pred_15 = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=15,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )
        
        last_close_original = float(df["close"].iloc[-1])
        result_15 = _direction_and_probability(last_close_original, pred_15, df)
        dir_15 = result_15["direction"]
        
        arbitrage_discord = (final_direction != dir_15)
        
        if arbitrage_discord:
            log.warning(f"[yellow]⚡ ARBITRAGE: 5-min model says {final_direction} but 15-min model says {dir_15} — potential mean-reversion opportunity[/yellow]")
        else:
            log.info(f"[green]✅ CONFLUENCE: both timeframes agree → normal Kelly sizing[/green]")
    except Exception as e:
        log.warning(f"[Kronos] Failed to run 15-min arbitrage check for {asset}: {e}")

    return {
        "direction": final_direction,
        "probability": final_prob,
        "predicted_close": final_close,
        "mode": "stacked_1min",
        "per_bar": per_bar_results,
        "arbitrage_discord": arbitrage_discord,
    }


class KronosPredictorAgent(HermesAgent):
    """
    Agent 3 — Kronos Predictor.

    Responsibilities:
      - Load Kronos-mini + Kronos-Tokenizer-2k (once, then cached)
      - Run price predictions in direct_5min or stacked_1min mode
      - Output direction (UP/DOWN), probability, and predicted close

    Output (AgentResult.data):
      {
        "BTC": {"direction": str, "probability": float, "predicted_close": float, "mode": str},
        "ETH": {"direction": str, "probability": float, "predicted_close": float, "mode": str},
      }
    """

    def __init__(self) -> None:
        super().__init__(
            name="Kronos Predictor",
            role=(
                "You are a quantitative analyst specializing in time-series forecasting. "
                "Your job is to use the Kronos foundation model to predict short-term "
                "BTC and ETH price direction with an honest confidence estimate."
            ),
            tools=[_predict_direct_5min, _predict_stacked_1min],
        )
        self._mode = PREDICTION_MODE

    def step(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        mode: Optional[str] = None,
        **kwargs,
    ) -> AgentResult:
        """
        Runs Kronos predictions for each asset in `data`.

        Args:
            data: Dict mapping asset ticker → OHLCV DataFrame (from Agent 2).
            mode: Override prediction mode ("direct_5min" | "stacked_1min").

        Returns:
            AgentResult with prediction dicts per asset in .data.
        """
        if data is None or not data:
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=["No OHLCV data provided to Kronos Predictor."],
            )

        active_mode = mode or self._mode
        self._log.info(
            f"[Kronos] Starting predictions | mode={active_mode} | "
            f"assets={list(data.keys())}"
        )

        # Load model (fast if already cached)
        try:
            predictor = _load_predictor()
        except RuntimeError as e:
            self._log.error(f"[Kronos] Model load failed: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=[str(e)],
            )

        results = {}
        errors = []

        for asset, df in data.items():
            if df is None or df.empty:
                self._log.warning(f"[Kronos] No data for {asset}. Skipping.")
                errors.append(f"Empty DataFrame for {asset}")
                continue

            try:
                if active_mode == "direct_5min":
                    pred = _predict_direct_5min(predictor, df, asset)
                else:
                    pred = _predict_stacked_1min(predictor, df, asset)

                results[asset] = pred
                self._log.info(
                    f"[Kronos] {asset} → direction={pred['direction']}, "
                    f"probability={pred['probability']:.3f}, "
                    f"predicted_close={pred['predicted_close']}"
                )

            except Exception as e:
                self._log.error(f"[Kronos] Prediction failed for {asset}: {e}")
                errors.append(f"Prediction error for {asset}: {e}")

        success = len(results) > 0
        return AgentResult(
            agent_name=self.name,
            success=success,
            data=results,
            errors=errors,
        )
