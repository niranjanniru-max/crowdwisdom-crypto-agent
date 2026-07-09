import argparse
import sys
import time
import json
import csv
from datetime import datetime, timezone
import requests
import pandas as pd
# pyrefly: ignore [missing-import]
import pandas_ta as ta
import numpy as np
import math
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Suppress minor warnings for backtest
import warnings
warnings.filterwarnings('ignore')

console = Console()

def fetch_historical_klines(symbol: str, total_bars: int, interval: str = "1m") -> pd.DataFrame:
    """Fetch previous `total_bars` from Binance."""
    klines = []
    end_time = None
    limit = 1000
    remaining = total_bars
    
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]Fetching {total_bars} bars for {symbol} ({interval}) from Binance...[/cyan]"),
        BarColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("fetching", total=total_bars)
        while remaining > 0:
            fetch_limit = min(remaining, limit)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={fetch_limit}"
            if end_time:
                url += f"&endTime={end_time}"
                
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            klines = data + klines  # prepend since we are going backward in time
            end_time = data[0][0] - 1  # end_time for the next chunk is right before the earliest bar of this chunk
            remaining -= len(data)
            progress.update(task, advance=len(data))
            time.sleep(0.5)

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]].sort_index()

def fetch_funding_history(symbol: str) -> pd.Series:
    """Fetch full funding rate history for symbol via paginated requests."""
    all_entries = []
    end_time = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]Fetching funding rate history for {symbol} from Binance...[/cyan]"),
        transient=True
    ) as progress:
        progress.add_task("fetching", total=None)
        for _ in range(50):  # Max 50 pages * 1000 = 50,000 entries ~= 5.7 years
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000"
            if end_time:
                url += f"&endTime={end_time}"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not fetch funding rate: {e}[/yellow]")
                break
            
            if not data:
                break
            all_entries = data + all_entries
            end_time = data[0]["fundingTime"] - 1
            time.sleep(0.2)
            if len(data) < 1000:
                break  # No more pages
            
    if not all_entries:
        return pd.Series(dtype=float, name="funding_rate")
        
    fdf = pd.DataFrame(all_entries)
    fdf["timestamp"] = pd.to_datetime(fdf["fundingTime"], unit="ms", utc=True)
    fdf["funding_rate"] = fdf["fundingRate"].astype(float)
    fdf = fdf.drop_duplicates("timestamp").set_index("timestamp")
    
    return fdf["funding_rate"].sort_index()

def _technical_signal(df: pd.DataFrame) -> str:
    """Returns 'UP', 'DOWN', or 'NEUTRAL' based on RSI + MACD."""
    close = df['close']
    rsi_series = ta.rsi(close, length=14)
    if rsi_series is None or rsi_series.empty:
        return 'NEUTRAL'
    
    rsi = float(rsi_series.iloc[-1])
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        return 'NEUTRAL'
    macd_line = float(macd_df['MACD_12_26_9'].iloc[-1])
    signal_line = float(macd_df['MACDs_12_26_9'].iloc[-1])
    
    if rsi < 45 and macd_line < signal_line:
        return 'DOWN'
    elif rsi > 55 and macd_line > signal_line:
        return 'UP'
    return 'NEUTRAL'

def compute_kelly_fraction(p: float, b: float = 1.0) -> float:
    """Half Kelly sizing."""
    q = 1.0 - p
    f_star = (b * p - q) / b
    return min(max(0.0, f_star) * 0.5, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Backtest Kronos & Confluence System.")
    parser.add_argument("--asset", choices=["BTC", "ETH", "ALL"], default="BTC", help="Asset to backtest (BTC, ETH, ALL)")
    parser.add_argument("--bars", type=int, default=3000, help="Number of historical 15m bars to fetch (default: 3000)")
    parser.add_argument("--interval", default="15m", help="Candle interval (default: 15m)")
    parser.add_argument("--filter-mode", choices=["asis", "inverted", "off", "all"], default="all", help="Technical filter mode to test")
    parser.add_argument("--strategy", choices=["confluence", "random", "funding_only"], default="confluence", help="Strategy to use")
    parser.add_argument("--direction", choices=["both", "long_only", "short_only"], default="both", help="Only take long or short trades (default: both)")
    parser.add_argument("--debug-trades", type=int, default=0, help="Print details for first N trades")
    args = parser.parse_args()

    # Load predictor safely by importing it from agents
    try:
        from agents.kronos_predictor import _load_predictor, _direction_and_probability
        from agents.strategies import trend_signal, meanrev_signal, breakout_signal, regime_filter, kronos_signal, funding_extreme_signal
        from agents.ensemble import evaluate_confluence
    except ImportError as e:
        console.print(f"[red]Error loading agents: {e}[/red]")
        sys.exit(1)
        
    predictor = _load_predictor()
    
    assets = ["BTC", "ETH"] if args.asset == "ALL" else [args.asset]
    symbols = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
    
    starting_bankroll = 1000.0
    mode_map = {"asis": "as-is", "inverted": "inverted", "off": "removed"}
    modes = ["as-is", "inverted", "removed"] if args.filter_mode == "all" else [mode_map[args.filter_mode]]
    
    metrics = {
        mode: {
            "total_windows": 0,
            "trades_taken": 0,
            "windows_skipped": 0,
            "bankroll": starting_bankroll,
            "correct_trades": 0,
            "btc_correct": 0,
            "btc_total": 0,
            "eth_correct": 0,
            "eth_total": 0,
            "long_correct": 0,
            "long_total": 0,
            "short_correct": 0,
            "short_total": 0,
            "max_consec_correct": 0,
            "max_consec_wrong": 0,
            "current_consec_correct": 0,
            "current_consec_wrong": 0,
            "cm_ll": 0,
            "cm_ls": 0,
            "cm_sl": 0,
            "cm_ss": 0,
            "debug_prints": 0,
            # Split-half tracking: first/second half by trade index
            "h1_correct": 0, "h1_total": 0,
            "h2_correct": 0, "h2_total": 0,
            # Per-direction P&L
            "long_pnl": 0.0,
            "short_pnl": 0.0,
        } for mode in modes
    }
    
    all_results = {mode: [] for mode in modes}

    # Settings
    lookback = 500
    pred_len = 5
    step = 5
    
    if args.strategy == "random":
        np.random.seed(42)
    
    for asset in assets:
        df = fetch_historical_klines(symbols[asset], args.bars, interval=args.interval)
        funding_series = fetch_funding_history(symbols[asset])
        
        # Merge funding rate into main dataframe with ffill
        df = df.join(funding_series.rename("funding_rate"), how="left")
        df["funding_rate"] = df["funding_rate"].ffill().bfill().fillna(0.0)
        
        if len(df) < lookback + pred_len:
            console.print(f"[red]Not enough bars fetched for {asset}. Expected > {lookback+pred_len}, got {len(df)}.[/red]")
            continue
            
        console.print(f"[bold green]Starting backtest for {asset} on {args.interval} timeframe...[/bold green]")
        
        # Walk-forward simulation
        num_windows = (len(df) - lookback - pred_len) // step + 1
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Simulating {asset} - {{task.percentage:>3.0f}}%[/cyan]"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total} windows[/cyan]"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("simulating", total=num_windows)
            
            for i in range(lookback, len(df) - pred_len, step):
                # 1. Slide window
                x_df = df.iloc[i - lookback : i].copy()
                x_timestamp = x_df.index.to_series().reset_index(drop=True)
                
                last_ts = x_df.index[-1]
                # Assuming simple numeric range since exact timestamps don't matter that much to the mock
                y_timestamp = pd.Series(pd.date_range(start=last_ts, periods=pred_len+1, freq="15min")[1:])
                
                current_close = float(x_df["close"].iloc[-1])
                
                # 2. Kronos Prediction
                try:
                    pred_df = predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=1.0,
                        top_p=0.9,
                        sample_count=1,
                    )
                except Exception as e:
                    console.print(f"[red]Predictor failed at index {i}: {e}[/red]")
                    continue
                
                kronos_result = _direction_and_probability(current_close, pred_df, x_df)
                predicted_direction = kronos_result["direction"]
                probability = kronos_result["probability"]
                
                # 3. Evaluate new strategies
                regime = regime_filter(x_df)
                t_dir, t_conf = trend_signal(x_df)
                m_dir, m_conf = meanrev_signal(x_df)
                b_dir, b_conf = breakout_signal(x_df)
                k_dir, k_conf = kronos_signal(predicted_direction, probability)
                f_dir, f_conf = funding_extreme_signal(x_df)
                
                signals = {
                    "trend": (t_dir, t_conf),
                    "meanrev": (m_dir, m_conf),
                    "breakout": (b_dir, b_conf),
                    "kronos": (k_dir, k_conf)
                }
                
                ens_dir, ens_conf = evaluate_confluence(regime, signals)
                tech_signal = _technical_signal(x_df)
                
                if args.strategy == "random":
                    ens_dir = np.random.choice(["long", "short"])
                    ens_conf = 1.0
                    tech_signal = "NEUTRAL"
                elif args.strategy == "funding_only":
                    ens_dir = f_dir
                    ens_conf = f_conf if f_dir != "flat" else 0.0
                    tech_signal = "NEUTRAL"
                
                # ATR for Risk Management
                atr_series = ta.atr(x_df['high'], x_df['low'], x_df['close'], length=14)
                atr = float(atr_series.iloc[-1]) if (atr_series is not None and not atr_series.empty and not pd.isna(atr_series.iloc[-1])) else (current_close * 0.005)

                future_df = df.iloc[i : i + pred_len]
                
                # 4. Filter Modes and Trade Simulation
                for mode in modes:
                    m = metrics[mode]
                    m["total_windows"] += 1
                    
                    if ens_conf <= 0.6 or ens_dir == "flat":
                        m["windows_skipped"] += 1
                        continue
                        
                    # Filter evaluation
                    skip = False
                    if tech_signal != "NEUTRAL":
                        if mode == "as-is":
                            if ens_dir == "long" and tech_signal == "DOWN": skip = True
                            if ens_dir == "short" and tech_signal == "UP": skip = True
                        elif mode == "inverted":
                            if ens_dir == "long" and tech_signal == "UP": skip = True
                            if ens_dir == "short" and tech_signal == "DOWN": skip = True
                            
                    if skip:
                        m["windows_skipped"] += 1
                        continue
                        
                    m["trades_taken"] += 1
                    
                    # Direction filter for --direction long_only / short_only
                    if args.direction == "long_only" and ens_dir != "long":
                        m["windows_skipped"] += 1
                        m["trades_taken"] -= 1
                        continue
                    if args.direction == "short_only" and ens_dir != "short":
                        m["windows_skipped"] += 1
                        m["trades_taken"] -= 1
                        continue
                    
                    # Position Sizing
                    kf = compute_kelly_fraction(ens_conf)
                    pos_pct = min(kf, 0.05) # 5% hard cap
                    position_size = m["bankroll"] * pos_pct
                    
                    entry_price = current_close
                    if ens_dir == "long":
                        sl_price = entry_price - 2 * atr
                        tp_price = entry_price + 3 * atr
                    else:
                        sl_price = entry_price + 2 * atr
                        tp_price = entry_price - 3 * atr
                        
                    # Simulate Price Path for Trade Output
                    exit_price = float(future_df['close'].iloc[-1])
                    exit_timestamp = future_df.index[-1]
                    exit_idx = i + pred_len - 1
                    
                    for idx_offset, (ts, row) in enumerate(future_df.iterrows()):
                        if ens_dir == "long":
                            if row['low'] <= sl_price:
                                exit_price = sl_price
                                exit_timestamp = ts
                                exit_idx = i + idx_offset
                                break
                            elif row['high'] >= tp_price:
                                exit_price = tp_price
                                exit_timestamp = ts
                                exit_idx = i + idx_offset
                                break
                        else:
                            if row['high'] >= sl_price:
                                exit_price = sl_price
                                exit_timestamp = ts
                                exit_idx = i + idx_offset
                                break
                            elif row['low'] <= tp_price:
                                exit_price = tp_price
                                exit_timestamp = ts
                                exit_idx = i + idx_offset
                                break
                                
                    if ens_dir == "long":
                        pct_return = (exit_price - entry_price) / entry_price
                    else:
                        pct_return = (entry_price - exit_price) / entry_price
                        
                    # 0.06% fee (entry+exit) + 0.02% slippage (entry+exit) = 0.16%
                    total_cost_rate = (0.0006 * 2) + (0.0002 * 2)
                    trade_pnl = (position_size * pct_return) - (position_size * total_cost_rate)
                    
                    m["bankroll"] += trade_pnl
                    actual_direction = "long" if exit_price > entry_price else "short"
                    correct_trade = (ens_dir == actual_direction)
                    
                    if m["debug_prints"] < args.debug_trades:
                        if m["debug_prints"] == 0:
                            console.print("trade_num | entry_idx | entry_timestamp | entry_close_price | exit_idx | exit_timestamp | exit_close_price | predicted_direction | actual_direction | correct")
                        
                        entry_idx = i - 1
                        console.print(f"{m['trades_taken']} | {entry_idx} | {last_ts} | {entry_price:.2f} | {exit_idx} | {exit_timestamp} | {exit_price:.2f} | {ens_dir} | {actual_direction} | {correct_trade}")
                        m["debug_prints"] += 1
                        
                    if ens_dir == "long" and actual_direction == "long": m["cm_ll"] += 1
                    elif ens_dir == "long" and actual_direction == "short": m["cm_ls"] += 1
                    elif ens_dir == "short" and actual_direction == "long": m["cm_sl"] += 1
                    elif ens_dir == "short" and actual_direction == "short": m["cm_ss"] += 1
                    
                    if correct_trade:
                        m["correct_trades"] += 1
                        m["current_consec_correct"] += 1
                        m["current_consec_wrong"] = 0
                        m["max_consec_correct"] = max(m["max_consec_correct"], m["current_consec_correct"])
                        if asset == "BTC": m["btc_correct"] += 1
                        if asset == "ETH": m["eth_correct"] += 1
                        if ens_dir == "long": m["long_correct"] += 1
                        if ens_dir == "short": m["short_correct"] += 1
                    else:
                        m["current_consec_wrong"] += 1
                        m["current_consec_correct"] = 0
                        m["max_consec_wrong"] = max(m["max_consec_wrong"], m["current_consec_wrong"])
                        
                    if asset == "BTC": m["btc_total"] += 1
                    if asset == "ETH": m["eth_total"] += 1
                    if ens_dir == "long":
                        m["long_total"] += 1
                        m["long_pnl"] += trade_pnl
                    if ens_dir == "short":
                        m["short_total"] += 1
                        m["short_pnl"] += trade_pnl
                    
                    # Split-half tracking (by trade sequence index within this mode)
                    half_boundary = num_windows // 2  # rough midpoint of windows
                    window_idx = (i - lookback) // step
                    if window_idx < half_boundary:
                        m["h1_total"] += 1
                        if correct_trade: m["h1_correct"] += 1
                    else:
                        m["h2_total"] += 1
                        if correct_trade: m["h2_correct"] += 1
                    
                    record = {
                        "timestamp": str(last_ts),
                        "asset": asset,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "predicted_direction": ens_dir,
                        "confidence": ens_conf,
                        "trade_pnl": trade_pnl,
                        "correct": correct_trade,
                        "running_bankroll": round(m["bankroll"], 2)
                    }
                    all_results[mode].append(record)
                    
                progress.update(task, advance=1)
                
                if (i // step) % 50 == 0:
                    mode_print = modes[0]
                    console.print(f"[dim]Step {i // step}: {asset} bankroll ({mode_print}) ${metrics[mode_print]['bankroll']:.2f}[/dim]")
    
    # 5. Output Results
    Path("data").mkdir(exist_ok=True)
    
    for mode in modes:
        if not all_results[mode]:
            console.print(f"[red]No predictions were made for mode {mode}.[/red]")
            continue
            
        with open(f"data/backtest_results_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(all_results[mode], f, indent=2)
            
        keys = all_results[mode][0].keys()
        with open(f"data/backtest_trades_{mode}.csv", "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results[mode])
            
        m = metrics[mode]
        table = Table(title=f"📋 Backtest Report ({mode})", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        acc = (m["correct_trades"] / m["trades_taken"] * 100) if m["trades_taken"] else 0.0
        
        total_cm = m['cm_ll'] + m['cm_ls'] + m['cm_sl'] + m['cm_ss']
        if total_cm > 0:
            cm_acc = (m['cm_ll'] + m['cm_ss']) / total_cm * 100
            assert abs(acc - cm_acc) < 0.1, f"Accuracy mismatch! Reported {acc} vs CM {cm_acc}"
        
        btc_acc = (m["btc_correct"] / m["btc_total"] * 100) if m["btc_total"] else 0.0
        eth_acc = (m["eth_correct"] / m["eth_total"] * 100) if m["eth_total"] else 0.0
        long_acc = (m["long_correct"] / m["long_total"] * 100) if m["long_total"] else 0.0
        short_acc = (m["short_correct"] / m["short_total"] * 100) if m["short_total"] else 0.0
        
        roi = (m["bankroll"] - starting_bankroll) / starting_bankroll * 100
        
        table.add_row("Total Windows", str(m["total_windows"]))
        table.add_row("Trades Taken vs Skipped", f"{m['trades_taken']} taken / {m['windows_skipped']} skipped")
        table.add_row("Accuracy on Trades Taken", f"{acc:.1f}%")
        
        if m["btc_total"] > 0:
            table.add_row("BTC Accuracy", f"{btc_acc:.1f}%")
        if m["eth_total"] > 0:
            table.add_row("ETH Accuracy", f"{eth_acc:.1f}%")
            
        table.add_row("Long Accuracy", f"{long_acc:.1f}% (n={m['long_total']}, PnL: ${m['long_pnl']:.2f})")
        table.add_row("Short Accuracy", f"{short_acc:.1f}% (n={m['short_total']}, PnL: ${m['short_pnl']:.2f})")
        
        h1_acc = (m["h1_correct"] / m["h1_total"] * 100) if m["h1_total"] else 0.0
        h2_acc = (m["h2_correct"] / m["h2_total"] * 100) if m["h2_total"] else 0.0
        table.add_row("1st Half Accuracy", f"{h1_acc:.1f}% (n={m['h1_total']})")
        table.add_row("2nd Half Accuracy", f"{h2_acc:.1f}% (n={m['h2_total']})")
        
        table.add_row("Starting Bankroll", f"${starting_bankroll:,.2f}")
        table.add_row("Final Simulated Bankroll", f"${m['bankroll']:,.2f}")
        
        prefix = "+" if roi > 0 else ""
        table.add_row("Return (includes fees/slippage)", f"{prefix}{roi:.2f}%")
        
        table.add_row("Max Consecutive Correct", str(m["max_consec_correct"]))
        table.add_row("Max Consecutive Wrong", str(m["max_consec_wrong"]))

        console.print(Panel(table, border_style="green", expand=False))
        
        console.print(f"\n[cyan]Confusion Matrix ({mode}):[/cyan]")
        console.print(f"{'':<15} | {'Actual Long':<15} | {'Actual Short':<15}")
        console.print("-" * 50)
        console.print(f"{'Pred Long':<15} | {m['cm_ll']:<15} | {m['cm_ls']:<15}")
        console.print(f"{'Pred Short':<15} | {m['cm_sl']:<15} | {m['cm_ss']:<15}\n")

if __name__ == "__main__":
    main()
