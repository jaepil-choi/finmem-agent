import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List

# 1. Indicator Registry
INDICATOR_REGISTRY: Dict[str, Callable[[pd.Series], Any]] = {}

def register_indicator(name: str):
    """Decorator to register new indicators dynamically."""
    def decorator(func: Callable):
        INDICATOR_REGISTRY[name] = func
        return func
    return decorator

# 2. Performance & Volatility (Annualized)
@register_indicator("weekly_returns")
def calculate_weekly_returns(returns: pd.Series) -> List[float]:
    """Returns the last 5 daily returns."""
    return [float(r) for r in returns.tail(5).tolist()]

@register_indicator("monthly_stats")
def calculate_monthly_stats(returns: pd.Series) -> Dict[str, float]:
    """Annualized monthly return and volatility (last 21 trading days)."""
    m_returns = returns.tail(21)
    if m_returns.empty:
        return {"annualized_return_monthly": 0.0, "annualized_vol_monthly": 0.0}
    
    ann_return = float(m_returns.mean() * 252)
    ann_vol = float(m_returns.std() * np.sqrt(252))
    
    return {"annualized_return_monthly": ann_return, "annualized_vol_monthly": ann_vol}

@register_indicator("yearly_stats")
def calculate_yearly_stats(returns: pd.Series) -> Dict[str, float]:
    """Annualized yearly return and volatility (last 252 trading days)."""
    y_returns = returns.tail(252)
    if y_returns.empty:
        return {"annualized_return_yearly": 0.0, "annualized_vol_yearly": 0.0}
    
    ann_return = float(y_returns.mean() * 252)
    ann_vol = float(y_returns.std() * np.sqrt(252))
    
    return {"annualized_return_yearly": ann_return, "annualized_vol_yearly": ann_vol}

# 3. Higher Moments
@register_indicator("moments")
def calculate_moments(returns: pd.Series) -> Dict[str, float]:
    """Skewness and Kurtosis for Monthly vs Yearly data."""
    m_returns = returns.tail(21)
    y_returns = returns.tail(252)
    
    return {
        "skewness_monthly": float(m_returns.skew()) if not m_returns.empty else 0.0,
        "kurtosis_monthly": float(m_returns.kurt()) if not m_returns.empty else 0.0,
        "skewness_yearly": float(y_returns.skew()) if not y_returns.empty else 0.0,
        "kurtosis_yearly": float(y_returns.kurt()) if not y_returns.empty else 0.0
    }

# 4. Stability & Regime Metrics
@register_indicator("vol_of_vol")
def calculate_vol_of_vol(returns: pd.Series) -> float:
    """Standard deviation of the rolling 21-day standard deviation (annualized)."""
    if len(returns) < 42: # Need enough for rolling std and then std of that
        return 0.0
    rolling_std = returns.rolling(window=21).std() * np.sqrt(252)
    return float(rolling_std.std())

@register_indicator("efficiency_ratio")
def calculate_efficiency_ratio(returns: pd.Series) -> float:
    """Kaufman's Efficiency Ratio (last 21 days)."""
    m_returns = returns.tail(21)
    if len(m_returns) < 2:
        return 0.0
    
    # Cumulative change vs sum of absolute changes
    total_change = abs(m_returns.sum()) # Roughly equivalent to Price(T) - Price(T-n) in log returns
    volatility = m_returns.abs().sum()
    
    if volatility == 0:
        return 0.0
    return float(total_change / volatility)

@register_indicator("hurst_exponent")
def calculate_hurst_exponent(returns: pd.Series) -> float:
    """
    Simplified Hurst Exponent calculation (last 252 days).
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    y_returns = returns.tail(252).values
    if len(y_returns) < 100:
        return 0.5 # Default to random walk if not enough data
    
    # Simplified R/S analysis
    lags = range(10, 50)
    tau = [np.sqrt(np.std(np.subtract(y_returns[lag:], y_returns[:-lag]))) for lag in lags]
    
    # Slope of log(tau) vs log(lags)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)

# 5. Core Engine
class RegimeCalculator:
    """Orchestrates modular indicator calculation."""
    def __init__(self, indicators: List[str] = None):
        self.indicators = indicators or list(INDICATOR_REGISTRY.keys())

    def calculate(self, history: pd.Series) -> Dict[str, Any]:
        """Calculates all registered indicators for the given history."""
        results = {}
        for name in self.indicators:
            if name in INDICATOR_REGISTRY:
                try:
                    res = INDICATOR_REGISTRY[name](history)
                    if isinstance(res, dict):
                        results.update(res)
                    else:
                        results[name] = res
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Error calculating {name}: {e}")
                    results[name] = None
        return results
