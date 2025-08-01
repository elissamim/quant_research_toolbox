"""
Functions and classes for risk adjusted metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional
from risk import DownsideRisk


def sharpe_ratio(returns: pd.Series, risk_free_rate: Optional[float] = 0.0) -> float:
    """ """
    excess_returns = returns - risk_free_rate
    std_returns = np.std(excess_returns)
    return float(np.mean(excess_returns) / std_returns)


def sortino_ratio(returns: pd.Series, risk_free_rate: Optional[float] = 0.0) -> float:
    """ """
    downside = DownsideRisk.downside_standard_deviation(
        returns, threshold=risk_free_rate
    )
    excess_returns = np.mean(returns) - risk_free_rate
    return float(excess_returns / downside) if downside > 0 else 0.0


def omega_ratio(returns: pd.Series, threshold: Optional[float] = 0.0) -> float:
    """ """
    excess_gains = returns[returns > threshold] - threshold
    excess_losses = threshold - returns[returns < threshold]
    return (
        float(excess_gains.sum() / excess_losses.sum())
        if excess_losses.sum() > 0
        else np.inf
    )
