# ============================================================
#  agents/data_fetcher.py  — Agent 2: Data Fetcher
#
#  Role: Fetch the last 1000 1-minute OHLCV candles for BTC
#  and ETH from CryptoCompare via the Apify apify/http-request
#  actor.  Falls back to direct Binance REST on any failure.
# ============================================================

import time
import json
from datetime import datetime, timezone

import pandas as pd
from apify_client import ApifyClient

from agents.base_agent import HermesAgent, AgentResult
from utils.logger import get_logger
from utils.config import APIFY_API_TOKEN, _mask_key

log = get_logger(__name__)

# Binance public REST endpoint (fallback only)
BINANCE_KLINE_URL_TEMPLATE = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

# CryptoCompare symbols
CRYPTOCOMPARE_SYMBOLS = {
    "BTC": "BTC",
    "ETH": "ETH",
}

CANDLE_INTERVAL = "1m"
CANDLE_LIMIT = 1000
APIFY_ACTOR_ID = "apify/http-request"


def _fetch_candles(asset: str, symbol: str, client: ApifyClient) -> pd.DataFrame | None:
    """
    Fetches the last CANDLE_LIMIT 1-minute OHLCV bars via the
    apify/http-request actor → CryptoCompare histominute endpoint.

    Returns:
        pandas DataFrame with columns ['open','high','low','close','volume']
        and a DatetimeIndex (UTC), or None on failure.
    """
    cc_sym = CRYPTOCOMPARE_SYMBOLS.get(asset, asset)
    url = (
        f"https://min-api.cryptocompare.com/data/v2/histominute"
        f"?fsym={cc_sym}&tsym=USD&limit={CANDLE_LIMIT}"
    )
    log.info(f"[Data Fetcher] Fetching {CANDLE_LIMIT}×1m candles for {asset} via Apify/CryptoCompare")

    for attempt in range(2):
        run_id = None
        try:
            # Try with outputFormat="json" first (attempt 0)
            # On retry (attempt 1) omit the flag and parse raw response
            run_input = {"url": url, "method": "GET"}
            if attempt == 0:
                run_input["outputFormat"] = "json"

            run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)

            if isinstance(run, dict):
                dataset_id = run.get("defaultDatasetId")
                run_id = run.get("id")
            else:
                dataset_id = run.default_dataset_id
                run_id = getattr(run, "id", None)

            items = client.dataset(dataset_id).list_items().items

            if not items:
                log.warning(f"[Data Fetcher] {asset}: Apify actor returned no items (attempt {attempt + 1}).")
                if attempt == 0:
                    log.info(f"[Data Fetcher] {asset}: Retrying without outputFormat flag…")
                    continue
                return None

            log.debug(f"[Data Fetcher] {asset} raw item keys: {list(items[0].keys())}")

            raw = items[0]

            # Depending on whether outputFormat="json" worked, the parsed body
            # arrives as a dict directly (items[0]) or nested under a key.
            # Try to locate the CryptoCompare "Data" → "Data" list.
            candles = None

            # Case A: outputFormat="json" returned the full parsed payload as items[0]
            if isinstance(raw, dict) and "Data" in raw:
                inner = raw["Data"]
                if isinstance(inner, dict) and "Data" in inner:
                    candles = inner["Data"]

            # Case B: body is under a "body" key (raw HTTP response)
            if candles is None and isinstance(raw, dict) and "body" in raw:
                body = raw["body"]
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except Exception:
                        pass
                if isinstance(body, dict) and "Data" in body:
                    inner = body["Data"]
                    if isinstance(inner, dict) and "Data" in inner:
                        candles = inner["Data"]

            # Case C: items[0] IS the response dict already (no wrapping)
            if candles is None:
                for key in ("data", "response", "result"):
                    if key in raw:
                        candidate = raw[key]
                        if isinstance(candidate, dict) and "Data" in candidate:
                            inner = candidate["Data"]
                            if isinstance(inner, dict) and "Data" in inner:
                                candles = inner["Data"]
                                break

            if not candles or not isinstance(candles, list) or len(candles) == 0:
                log.warning(
                    f"[Data Fetcher] {asset}: Could not locate candle array in Apify response "
                    f"(attempt {attempt + 1}): {str(raw)[:300]}"
                )
                if attempt == 0:
                    log.info(f"[Data Fetcher] {asset}: Retrying without outputFormat flag…")
                    continue
                return None

            # CryptoCompare histominute fields: time, open, high, low, close, volumefrom, volumeto
            df = pd.DataFrame(candles)
            if "time" not in df.columns:
                log.warning(f"[Data Fetcher] {asset}: Unexpected candle schema: {df.columns.tolist()}")
                return None

            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("timestamp")

            # Rename volumefrom → volume (denominated in base asset)
            col_map = {}
            if "volumefrom" in df.columns:
                col_map["volumefrom"] = "volume"
            df = df.rename(columns=col_map)

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            df = df[["open", "high", "low", "close", "volume"]].dropna().sort_index()

            log.info(
                f"[Data Fetcher] {asset}: fetched via Apify actor (run ID: {run_id}) | {len(df)} candles"
            )
            log.info(f"[Data Fetcher] {asset} range: {df.index[0]} → {df.index[-1]}")
            return df

        except json.JSONDecodeError as e:
            log.error(f"[Data Fetcher] JSON parsing error fetching {asset}: {e}")
            return None

        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "403" in err_str:
                log.error(
                    f"[Data Fetcher] Apify auth error for {asset}. "
                    f"Check APIFY_API_TOKEN (key: {_mask_key(APIFY_API_TOKEN)})"
                )
                return None

            if attempt == 0:
                log.warning(f"[Data Fetcher] Error for {asset} (attempt 1): {e}. Retrying…")
                time.sleep(2)
            else:
                log.warning(
                    f"[Data Fetcher] WARNING: Apify actor failed for {asset} on retry: {e}. "
                    f"Will use Binance REST fallback."
                )
                return None

    return None


def _fetch_candles_binance_fallback(asset: str, symbol: str) -> pd.DataFrame | None:
    """Direct Binance REST fallback — used when Apify actor fails."""
    import requests
    url = BINANCE_KLINE_URL_TEMPLATE.format(symbol=symbol, interval=CANDLE_INTERVAL, limit=CANDLE_LIMIT)
    try:
        log.warning(
            f"[Data Fetcher] WARNING: Falling back to direct Binance REST for {asset} "
            f"(geo-blocking may apply)."
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        log.info(f"[Data Fetcher] {asset}: fetched via direct Binance REST fallback | {len(df)} candles")
        return df
    except Exception as e:
        log.error(f"[Data Fetcher] WARNING: Direct Binance fallback also failed for {asset}: {e}")
        return None


class DataFetcherAgent(HermesAgent):
    """
    Agent 2 — Data Fetcher.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Data Fetcher",
            role=(
                "You are a market data engineer. "
                "Your job is to fetch accurate, clean OHLCV price data "
                "for crypto assets from reliable public APIs."
            ),
            tools=[_fetch_candles],
        )
        self._client = ApifyClient(APIFY_API_TOKEN)

    def step(self, assets: list[str] | None = None, **kwargs) -> AgentResult:
        if assets is None:
            assets = ["BTC", "ETH"]

        self._log.info(f"[Data Fetcher] Starting data fetch for assets: {assets} via Apify/CryptoCompare")

        data = {}
        errors = []

        for asset in assets:
            symbol = BINANCE_SYMBOLS.get(asset)
            if not symbol:
                self._log.warning(f"[Data Fetcher] Unknown asset: {asset}. Skipping.")
                errors.append(f"Unknown asset: {asset}")
                continue

            df = _fetch_candles(asset, symbol, self._client)

            if df is None or df.empty:
                # Apify actor failed — use direct Binance REST fallback
                df = _fetch_candles_binance_fallback(asset, symbol)

            if df is not None and not df.empty:
                data[asset] = df
                self._log.info(
                    f"[Data Fetcher] {asset} ready: {len(df)} rows, "
                    f"last close={df['close'].iloc[-1]:.2f}"
                )
            else:
                errors.append(f"Failed to fetch data for {asset}")
                self._log.error(f"[Data Fetcher] Could not retrieve data for {asset}")

        success = len(data) > 0
        self._log.info(f"[Data Fetcher] Step complete. Assets with data: {list(data.keys())}")

        return AgentResult(
            agent_name=self.name,
            success=success,
            data=data,
            errors=errors,
        )
