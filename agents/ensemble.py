from typing import Dict, Tuple

def evaluate_confluence(
    regime: str,
    signals: Dict[str, Tuple[str, float]]
) -> Tuple[str, float]:
    """
    Confluence engine:
    - Weights signals by regime.
    - Requires at least 2 of the 4 directional signals to agree.
    - Final confidence = weighted average of agreeing signals' confidence.
    """
    weights = {
        "trend": 1.0,
        "meanrev": 1.0,
        "breakout": 1.0,
        "kronos": 1.0
    }
    
    if regime == "trending":
        weights["trend"] = 1.5
        weights["breakout"] = 1.5
        weights["meanrev"] = 0.5
    elif regime == "ranging":
        weights["meanrev"] = 1.5
        weights["trend"] = 0.5
        weights["breakout"] = 0.5

    long_count = 0
    short_count = 0
    
    for name, (direction, conf) in signals.items():
        if direction == "long":
            long_count += 1
        elif direction == "short":
            short_count += 1
            
    final_dir = "flat"
    if long_count >= 2 and long_count > short_count:
        final_dir = "long"
    elif short_count >= 2 and short_count > long_count:
        final_dir = "short"
        
    if final_dir == "flat":
        return "flat", 0.0
        
    total_conf = 0.0
    total_weight = 0.0
    
    for name, (direction, conf) in signals.items():
        if direction == final_dir:
            weight = weights.get(name, 1.0)
            total_conf += conf * weight
            total_weight += weight
            
    final_conf = total_conf / total_weight if total_weight > 0 else 0.0
    final_conf = min(max(final_conf, 0.0), 1.0)
    
    return final_dir, final_conf
