# ============================================================
#  agents/market_scout.py  — Agent 1: Market Scout
#
#  Role: Scrape Polymarket and Kalshi APIs via the custom Apify
#  actor "chirpy_uplift/crypto-fetcher" to extract short-term
#  implied probabilities for BTC/ETH.
#  Falls back gracefully to direct REST and then neutral.
# ============================================================

import json
import requests
from typing import Optional

from apify_client import ApifyClient

from agents.base_agent import HermesAgent, AgentResult
from utils.config import APIFY_API_TOKEN, _mask_key
from utils.logger import get_logger

log = get_logger(__name__)

NEUTRAL_ODDS = {"probability": 0.5, "net_odds": 1.0, "direction": "UNKNOWN", "source": "fallback neutral odds"}
CUSTOM_ACTOR_ID = "chirpy_uplift/crypto-fetcher"

# Keywords used to match markets to each asset
ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
}


class MarketScoutAgent(HermesAgent):
    """
    Agent 1 — Market Scout.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Market Scout",
            role=(
                "You are a crypto prediction market analyst. "
                "Your job is to extract current market-implied probabilities "
                "from Polymarket and Kalshi for short-term price prediction markets."
            ),
            tools=[],
        )
        self._client = ApifyClient(APIFY_API_TOKEN)

    def _run_custom_actor(self) -> tuple[list, str | None]:
        """
        Calls chirpy_uplift/crypto-fetcher ONCE and returns (items, run_id).
        Items have fields: {platform, ticker, question, yes_price, no_price, volume}
        """
        try:
            self._log.info(f"[Market Scout] Calling Apify actor '{CUSTOM_ACTOR_ID}'…")
            run = self._client.actor(CUSTOM_ACTOR_ID).call()

            # Support both dict-style and object-style run responses
            if isinstance(run, dict):
                dataset_id = run.get("defaultDatasetId")
                run_id = run.get("id")
            else:
                dataset_id = run.default_dataset_id
                run_id = getattr(run, "id", None)

            self._log.info(f"[Market Scout] Actor run ID: {run_id} | dataset: {dataset_id}")

            items = self._client.dataset(dataset_id).list_items().items
            self._log.info(f"[Market Scout] Actor returned {len(items)} market items")
            return items, run_id

        except Exception as e:
            self._log.warning(f"[Market Scout] Custom actor call failed: {e}")
            return [], None

    def _extract_signal_from_items(self, items: list, asset: str) -> Optional[dict]:
        """
        Filter actor items for the given asset and return the highest-volume match
        as a signal dict, or None if no match found.
        """
        keywords = ASSET_KEYWORDS.get(asset.upper(), [])
        matches = []

        for item in items:
            question = (item.get("question") or "").lower()
            if not any(kw in question for kw in keywords):
                continue

            platform = (item.get("platform") or "").lower()
            if platform not in ("polymarket", "kalshi"):
                continue

            yes_price = item.get("yes_price")
            volume = float(item.get("volume") or 0)

            try:
                yes_price = float(yes_price)
            except (TypeError, ValueError):
                continue

            if not (0 < yes_price < 1):
                continue

            matches.append({**item, "_yes_price": yes_price, "_volume": volume})

        if not matches:
            return None

        # Take the highest-volume match
        matches.sort(key=lambda x: x["_volume"], reverse=True)
        best = matches[0]

        yes_price = best["_yes_price"]
        direction = "UP" if yes_price >= 0.5 else "DOWN"
        net_odds = (1.0 / yes_price) - 1.0

        return {
            "direction": direction,
            "probability": yes_price,
            "net_odds": net_odds,
            "source": "Apify/chirpy_uplift-crypto-fetcher",
        }

    def _fetch_polymarket_fallback(self, asset: str) -> Optional[dict]:
        """Direct REST fallback to gamma-api.polymarket.com."""
        asset_query = "bitcoin" if asset.upper() == "BTC" else "ethereum"
        url = f"https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=50&search={asset_query}"
        try:
            self._log.info(
                f"[Market Scout] No actor data for {asset}; falling back to "
                f"gamma-api.polymarket.com REST call…"
            )
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for market in data:
                if market.get("closed", False):
                    continue
                prices = market.get("outcomePrices")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices.replace("'", '"'))
                    except json.JSONDecodeError:
                        pass

                if prices and isinstance(prices, list) and len(prices) > 0:
                    prob = float(prices[0])
                    if 0 < prob < 1:
                        direction = "UP" if prob >= 0.5 else "DOWN"
                        net_odds = (1.0 / prob) - 1.0
                        return {
                            "direction": direction,
                            "probability": prob,
                            "net_odds": net_odds,
                            "source": "fallback-REST",
                        }
        except Exception as e:
            self._log.warning(f"[Market Scout] Polymarket fallback error for {asset}: {e}")
        return None

    def step(self, assets: list[str] | None = None, **kwargs) -> AgentResult:
        if assets is None:
            assets = ["BTC", "ETH"]

        self._log.info(f"[Market Scout] Starting for assets: {assets}")

        # ── Run the actor ONCE ──────────────────────────────────────────────
        actor_items, run_id = self._run_custom_actor()

        results = {}
        warnings = []

        for asset in assets:
            data = None

            # 1. Try to extract signal from actor items
            if actor_items:
                data = self._extract_signal_from_items(actor_items, asset)

            # 2. Fall back to direct REST if actor returned no matching market
            if not data:
                data = self._fetch_polymarket_fallback(asset)

            # 3. Ultimate fallback: neutral odds
            if not data:
                self._log.warning(
                    f"[Market Scout] No valid market found via any source for {asset}. "
                    f"Using neutral fallback."
                )
                warnings.append(f"No market data for {asset}")
                data = dict(NEUTRAL_ODDS)

            # Structured log line matching the requested format
            source_str = data["source"]
            if run_id and "Apify" in source_str:
                source_str = f"{source_str} (run ID: {run_id})"

            self._log.info(
                f"[Market Scout] {asset}: source={source_str} | "
                f"prob={data['probability']:.3f} | "
                f"direction={data['direction']}"
            )

            results[asset] = data

        success = any(r["source"] != "fallback neutral odds" for r in results.values())
        if not success:
            warnings.append("All markets fell back to neutral odds")

        self._log.info(f"[Market Scout] Step complete. Results: {results}")
        return AgentResult(
            agent_name=self.name,
            success=True,
            data=results,
            warnings=warnings,
        )
