# CrowdWisdomTrading — Crypto Prediction Agent

A production-ready Python CLI system that scrapes crypto prediction markets, fetches live price data, forecasts the next 5-minute up/down move using the **Kronos** foundation model, sizes hypothetical trades using the **Kelly Criterion**, and tracks whether each prediction was correct.

---

## Architecture — 5-Agent Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│          CrowdWisdomTrading — Multi-Agent Pipeline               │
│                                                                  │
│  ┌─────────────────┐   odds   ┌─────────────────┐              │
│  │  Agent 1        │ ──────►  │  Agent 4         │             │
│  │  Market Scout   │          │  Kelly Risk Mgr  │             │
│  │  (Polymarket/   │          │  (Half-Kelly     │             │
│  │   Apify)        │          │   fraction)      │             │
│  └─────────────────┘          └────────┬─────────┘             │
│                                         │                        │
│  ┌─────────────────┐  OHLCV  ┌─────────▼─────────┐             │
│  │  Agent 2        │ ──────► │  Agent 3            │            │
│  │  Data Fetcher   │         │  Kronos Predictor   │            │
│  │  (Binance REST) │         │  (UP/DOWN + prob)   │            │
│  └─────────────────┘         └───────────────────╼┘            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Agent 5 — Feedback Loop (Hermes Agent run_loop pattern)    │ │
│  │  Waits 5 min → re-fetches actual price → logs correct/wrong │ │
│  │  Updates rolling accuracy table in real time                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Details

| #   | Agent                  | Library                   | Output                                   |
| --- | ---------------------- | ------------------------- | ---------------------------------------- |
| 1   | **Market Scout**       | Apify web-scraper         | Implied probability + net odds per asset |
| 2   | **Data Fetcher**       | Binance public REST       | 1000×1-min OHLCV DataFrame               |
| 3   | **Kronos Predictor**   | Kronos-mini (HuggingFace) | UP/DOWN direction + confidence           |
| 4   | **Kelly Risk Manager** | Pure Python math          | Half-Kelly fraction + $ size             |
| 5   | **Feedback Loop**      | Hermes Agent loop pattern | Outcome log + rolling accuracy           |

---

## Getting API Keys (both are free)

### OpenRouter (LLM provider)

1. Go to [https://openrouter.ai](https://openrouter.ai) → Sign up for free
2. Navigate to **Keys** → Create a new key
3. Copy the key (starts with `sk-or-v1-`)

### Apify (web scraping)

1. Go to [https://apify.com](https://apify.com) → Sign up for free
2. Navigate to **Settings → Integrations** → copy your **API token** (starts with `apify_api_`)

---

## Setup

### 1. Clone / download the project

```bash
cd "d:\Crypto Prediction Agent"
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux
```

### 3. Install dependencies

```bash
# CPU-only PyTorch (smaller, faster to install)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -r crypto-prediction-agent/requirements.txt
```

### 4. Configure environment variables

```bash
cd crypto-prediction-agent
copy .env.example .env
# Fill in your real keys in .env
```

---

## Running

```bash
# From crypto-prediction-agent/ with venv active:

# Default: BTC + ETH, stacked_1min mode, 1 cycle
python main.py

# Specific asset, specific mode
python main.py --asset BTC --mode direct_5min

# Multiple cycles (runs prediction + waits 5min + scores, repeated N times)
python main.py --asset ALL --mode stacked_1min --cycles 3

# Fast test (skip 5-min wait, score immediately)
python main.py --no-wait
```

### CLI Options

| Flag        | Default        | Description                       |
| ----------- | -------------- | --------------------------------- |
| `--asset`   | `ALL`          | `BTC`, `ETH`, or `ALL`            |
| `--mode`    | `stacked_1min` | `direct_5min` or `stacked_1min`   |
| `--cycles`  | `1`            | Number of predict→score cycles    |
| `--no-wait` | false          | Skip the 5-min wait (for testing) |

---

## Prediction Modes

### `direct_5min`

Makes a single Kronos prediction for 5 bars ahead. Simple, fast (one model call per asset).

### `stacked_1min` (default — extra credit scale idea)

Makes 5 sequential 1-minute predictions, feeding each predicted bar back as context for the next. Results are combined as:

- **Direction**: majority vote (≥3 of 5 bars → UP)
- **Probability**: geometric mean of per-bar confidence scores, adjusted downward for split votes

This mode is slower but produces a richer signal. It also allows seeing **how each minute is expected to evolve**, which is useful for the demo video.

---

## Kelly Criterion — Math Note

The Kelly Criterion determines the optimal fraction of bankroll to bet to maximise long-run growth:

```
f* = (b·p - q) / b
```

Where:

- `p` = probability of winning (from Kronos forecast)
- `q = 1 - p` = probability of losing
- `b` = net odds offered by the market (e.g. 1.8× payout → b = 0.8)

**Half-Kelly** (`f = 0.5 × max(0, f*)`) is used by default — it gives ~75% of the log-growth rate of full Kelly but with much lower variance and drawdown risk. This is appropriate since our probability estimates (`p`) are model outputs with uncertainty.

If `f* ≤ 0`, the bet has negative expected value and Kelly recommends no bet.

All reasoning steps (p, q, b, f*, half-f*, final fraction, $ amount) are logged for every decision.

---

## Output Files

| File                    | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `data/results_log.json` | All predictions + outcomes (one JSON object per entry) |
| `logs/error.log`        | Full tracebacks for any errors during a run            |

### results_log.json schema

```json
{
  "timestamp": "2026-06-30T12:00:00+00:00",
  "asset": "BTC",
  "predicted_direction": "UP",
  "predicted_probability": 0.6347,
  "predicted_close": 61234.5,
  "market_odds": 0.9,
  "kelly_fraction": 0.0924,
  "mode": "stacked_1min",
  "entry_price": 61100.0,
  "actual_price": 61450.0,
  "actual_outcome": "UP",
  "correct": true,
  "scored_at": "2026-06-30T12:05:10+00:00"
}
```

---

## Project Structure

```
crypto-prediction-agent/
├── agents/
│   ├── base_agent.py        # HermesAgent ABC (Hermes Agent design pattern)
│   ├── market_scout.py      # Agent 1
│   ├── data_fetcher.py      # Agent 2
│   ├── kronos_predictor.py  # Agent 3
│   ├── kelly_risk_manager.py# Agent 4
│   └── feedback_loop.py     # Agent 5
├── llm/
│   └── openrouter_client.py # Multi-model fallback LLM wrapper
├── utils/
│   ├── config.py            # Env validation + masking
│   └── logger.py            # Rich-based logging setup
├── data/
│   └── results_log.json     # Created at runtime
├── logs/
│   └── error.log            # Created at runtime
├── kronos_src/              # Kronos repo (auto-cloned on first run)
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Notes on Kronos Model Loading

The first run will:

1. Auto-clone the [Kronos GitHub repo](https://github.com/shiyu-coder/Kronos) into `kronos_src/`
2. Download `NeoQuasar/Kronos-mini` weights from HuggingFace Hub (~150-300MB)
3. Cache the model in memory — subsequent runs within the same process are fast

This can take **2-5 minutes** on first run. A progress spinner is shown.

---

## Project Retrospective

### Problems We Faced

1. **Data Infrastructure Constraints**: The original Binance data fetcher faced geo-blocking issues. We had to pivot to using custom Apify actors targeting CryptoCompare for reliable OHLCV data retrieval. We also implemented robust fallback mechanisms in `market_scout.py` to route Polymarket and Kalshi data pulls securely.
2. **LLM Orchestration Instability**: We encountered deprecation issues with our OpenRouter model fallback chain, requiring us to migrate to the official auto-routing free model in our `openrouter_client.py`.
3. **Parsing Unstructured LLM Output**: Handling free-tier model prose output required significantly hardening our Market Analyst parsing logic to ensure reliable extraction of sentiment and confidence scores for assets like BTC and ETH.

### Final Achievements

1. **Completed 6-Agent Pipeline**: Successfully orchestrated a robust, automated pipeline integrating Hermes, Kronos, Kelly Risk Manager, Apify, Market Analyst, and OpenRouter.
2. **Reliable Data Sourcing**: Modernized the infrastructure to use custom Apify actors (`apify/http-request`, `chirpy_uplift/crypto-fetcher`), successfully bypassing geographical restrictions.
3. **Resilient Architecture**: Replaced fragile LLM chains with robust auto-routing and hardened response parsing. This resulted in stable predictions, accurate sentiment analysis, and continuous tracking through our run_loop pattern feedback loop. We pushed ourselves to construct a fault-tolerant system that can withstand free-tier LLM hallucinations and data blockades, ensuring a production-ready application.
