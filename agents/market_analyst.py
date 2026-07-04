# ============================================================
#  agents/market_analyst.py  — Agent 3: Market Analyst
#
#  Role: Apply LLM analysis (OpenRouter) on recent price 
#  action and scouted odds to determine a sentiment score 
#  which acts as a qualitative penalty layer in risk sizing.
# ============================================================

import json
import re
import pandas as pd

from agents.base_agent import HermesAgent, AgentResult
from llm.openrouter_client import call_llm
from utils.logger import get_logger

log = get_logger(__name__)


def _extract_json(raw: str) -> dict:
    """
    Extract JSON object from LLM response, handling common formatting issues.

    Uses brace-depth scanning so it correctly handles:
    - Thinking / reasoning text before the JSON block
    - Markdown code fences (```json ... ```)
    - Nested braces inside string values
    """
    if not raw:
        raise ValueError("Empty response")

    # Step 1: strip markdown fences
    clean = raw.strip()
    clean = re.sub(r"^```json\s*", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"^```\s*", "", clean, flags=re.MULTILINE)
    clean = clean.strip()

    # Step 2: try direct parse first (fastest path)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Step 3: brace-depth scan — find the first complete {...} block,
    # even when the model prepended thinking text or used nested braces.
    start = clean.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(clean[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = clean[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Step 4: handle truncated JSON — try appending closing brace
    if clean.count("{") > clean.count("}"):
        try:
            return json.loads(clean + "}")
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from: {raw[:200]}")


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> and similar model reasoning blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class MarketAnalystAgent(HermesAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Market Analyst",
            role=(
                "You are an expert technical market analyst. "
                "Analyze recent price action and external prediction market odds "
                "to provide a concise readout of market sentiment."
            ),
            tools=[],
        )

    def step(self, assets: list[str] | None = None, data: dict = None, predictions: dict = None, **kwargs) -> AgentResult:
        """
        data: Dict of asset -> pandas DataFrame (from data_fetcher)
        predictions: Dict of asset -> Scout output (from market_scout)
        """
        if assets is None:
            assets = ["BTC", "ETH"]

        if not data:
            self._log.error("[Market Analyst] Missing required market data.")
            return AgentResult(agent_name=self.name, success=False, errors=["Missing dataset"])
            
        if not predictions:
            predictions = {}

        self._log.info(f"[Market Analyst] Starting LLM analysis for assets: {assets}")

        results = {}
        errors = []

        system_msg = (
            'Output ONLY a raw JSON object — no other text, no markdown, no preamble. '
            'Schema: {"signal":"UP"|"DOWN"|"NEUTRAL","confidence":0.00,"key_driver":"string"}. '
            "Analyze the last 10 crypto price candles and decide if the trend is UP, DOWN, or NEUTRAL. "
            "confidence is a float from 0.0 to 1.0. key_driver is a single short sentence. "
            "If uncertain, return NEUTRAL with confidence 0.5."
        )

        for asset in assets:
            df = data.get(asset)
            if df is None or df.empty:
                errors.append(f"No data for {asset}")
                continue

            # Take last 10 candles for LLM context, but only close and volume as requested
            recent_data = df.tail(10)[["close", "volume"]].to_dict(orient="records")
            scout_odds = predictions.get(asset, {})
            implied_prob = scout_odds.get("probability", 0.5)

            prompt = f"Candlestick Data (Last 10 bars close prices and volumes): {json.dumps(recent_data)}"

            result_json = None
            parse_error = None

            # Attempt 1: concise system-message prompt, no response_format
            # (response_format=json_object triggers verbose prose on openrouter/free)
            # Attempt 2: absolute minimal prompt as last resort
            _close_val = df["close"].iloc[-1]
            _attempts = [
                dict(
                    prompt=prompt,
                    system=system_msg,
                    max_tokens=200,
                    temperature=0.1,
                ),
                dict(
                    prompt=(
                        f'{asset} last close={_close_val:.2f}. '
                        f'Output ONLY: {{"signal":"UP"|"DOWN"|"NEUTRAL",'
                        f'"confidence":0.60,"key_driver":"reason"}}'
                    ),
                    system=None,
                    max_tokens=80,
                    temperature=0.0,
                ),
            ]
            for attempt, call_kwargs in enumerate(_attempts):
                try:
                    raw_response = call_llm(**call_kwargs)
                    cleaned = _strip_thinking(raw_response)
                    result_json = _extract_json(cleaned)
                    break  # success — stop retrying

                except Exception as e:
                    parse_error = e
                    self._log.debug(
                        f"[Market Analyst] Parse attempt {attempt + 1} failed for {asset}: {e}. "
                        f"{'Retrying with simpler prompt.' if attempt == 0 else 'Giving up.'}"
                    )

            if result_json is not None:
                signal = result_json.get("signal", "NEUTRAL")
                try:
                    confidence = float(result_json.get("confidence", 0.5))
                except (TypeError, ValueError):
                    confidence = 0.5
                key_driver = result_json.get("key_driver", "No clear driver detected")

                sentiment_score = 0.0
                if signal == "UP":
                    sentiment_score = confidence
                elif signal == "DOWN":
                    sentiment_score = -confidence
            else:
                self._log.warning(
                    f"[Market Analyst] Both LLM attempts failed for {asset}, returning safe fallback: {parse_error}"
                )
                signal = "NEUTRAL"
                confidence = 0.0
                key_driver = "LLM parse failed"
                sentiment_score = 0.0

            results[asset] = {
                "signal": signal,
                "confidence": confidence,
                "key_driver": key_driver,
                "sentiment_score": sentiment_score
            }
            
            self._log.info(
                f"[Market Analyst] {asset}: signal={signal} | confidence={confidence:.2f} | driver={key_driver}"
            )

        success = True  # Always succeeds on agent level even if fallback used
        return AgentResult(
            agent_name=self.name,
            success=success,
            data=results,
            errors=errors,
        )
