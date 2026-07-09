import pandas as pd
import pandas_ta as ta

def trend_signal(df: pd.DataFrame) -> tuple[str, float]:
    """
    EMA9 vs EMA21 crossover.
    Long if EMA9 > EMA21 and rising, short if inverse, else flat.
    """
    if len(df) < 21:
        return "flat", 0.0
    
    close = df['close']
    ema9 = ta.ema(close, length=9)
    ema21 = ta.ema(close, length=21)
    
    if ema9 is None or ema21 is None:
        return "flat", 0.0
        
    if ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-1] > ema9.iloc[-2]:
        return "long", 1.0
    elif ema9.iloc[-1] < ema21.iloc[-1] and ema9.iloc[-1] < ema9.iloc[-2]:
        return "short", 1.0
    return "flat", 0.0

def meanrev_signal(df: pd.DataFrame) -> tuple[str, float]:
    """
    Bollinger Bands (20, 2 std) + RSI(14). 
    Long if price < lower band AND RSI < 30. Short if price > upper band AND RSI > 70. Else flat.
    """
    if len(df) < 20:
        return "flat", 0.0
        
    close = df['close']
    bbands = ta.bbands(close, length=20, std=2)
    rsi = ta.rsi(close, length=14)
    
    if bbands is None or rsi is None:
        return "flat", 0.0
        
    lower_band = bbands['BBL_20_2.0_2.0'].iloc[-1]
    upper_band = bbands['BBU_20_2.0_2.0'].iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_price = close.iloc[-1]
    
    if current_price < lower_band and current_rsi < 30:
        return "long", 1.0
    elif current_price > upper_band and current_rsi > 70:
        return "short", 1.0
    return "flat", 0.0

def breakout_signal(df: pd.DataFrame) -> tuple[str, float]:
    """
    Donchian channel (20-period high/low) + volume.
    Long if price breaks above 20-period high with volume > 1.5x avg.
    Short if breaks below with same volume condition.
    """
    if len(df) < 21: # Need at least 21 to get 20 prior periods + current
        return "flat", 0.0
        
    highs = df['high']
    lows = df['low']
    volumes = df['volume']
    
    # Donchian channel is usually highest high of N periods.
    # Exclude current candle to check for breakout
    dc_upper = highs.iloc[-21:-1].max()
    dc_lower = lows.iloc[-21:-1].min()
    
    avg_vol = volumes.iloc[-21:-1].mean()
    current_vol = volumes.iloc[-1]
    current_close = df['close'].iloc[-1]
    
    if current_close > dc_upper and current_vol > 1.5 * avg_vol:
        return "long", 1.0
    elif current_close < dc_lower and current_vol > 1.5 * avg_vol:
        return "short", 1.0
    return "flat", 0.0

def regime_filter(df: pd.DataFrame) -> str:
    """
    ADX(14). Return "trending" if ADX > 25, else "ranging".
    """
    if len(df) < 28: # ADX needs at least 2 * length data
        return "ranging"
        
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is None:
        return "ranging"
        
    current_adx = adx_df['ADX_14'].iloc[-1]
    if pd.isna(current_adx):
        return "ranging"
        
    if current_adx > 25:
        return "trending"
    return "ranging"

def kronos_signal(predicted_direction: str, probability: float) -> tuple[str, float]:
    """
    Adapt Kronos prediction to signals format.
    """
    if predicted_direction == "UP":
        return "long", probability
    elif predicted_direction == "DOWN":
        return "short", probability
    return "flat", 0.0

def funding_extreme_signal(df: pd.DataFrame) -> tuple[str, float]:
    """
    Computes a 30-day (120 * 8h or 720 * 1h) rolling z-score of the funding rate.
    Uses df["funding_rate"].
    Returns contrarian long if z-score < -2, contrarian short if z-score > 2, else flat.
    """
    if 'funding_rate' not in df.columns:
        return "flat", 0.0
        
    funding = df['funding_rate']
    mean_30d = funding.rolling('30d').mean()
    std_30d = funding.rolling('30d').std()
    
    current_funding = funding.iloc[-1]
    current_mean = mean_30d.iloc[-1]
    current_std = std_30d.iloc[-1]
    
    if pd.isna(current_std) or current_std == 0:
        return "flat", 0.0
        
    z_score = (current_funding - current_mean) / current_std
    
    if z_score > 2.0:
        return "short", 1.0  # High funding -> crowded longs -> contrarian short
    elif z_score < -2.0:
        return "long", 1.0   # Low funding -> crowded shorts -> contrarian long
    else:
        return "flat", 0.0

