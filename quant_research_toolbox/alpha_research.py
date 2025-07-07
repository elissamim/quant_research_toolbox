import pandas as pd
import numpy as np
from typing import Optional

def sma_crossover(df_stock:pd.DataFrame,
                  col_price:Optional[str]="Close",
                  fast_sma_window:Optional[int]=5,
                  slow_sma_window:Optional[int]=30) -> pd.DataFrame:
    """
    Generate signals and orders based on a SMA crossover strategy: buy signal (1) generated when fast SMA is higher
    and sell signal (-1) when slow SMA is higher.

    Args:
        df_stock (pd.DataFrame): Time series of a stock (index is datetime).
        col_price (Optional, str): Column name of price series. Defaults to `Close`.
        fast_sma_window (Optional, int): Window for fast SMA. Defaults to `5`.
        slow_sma_window (Optional, int): Window for slow SMA. Defaults to `30`.

    Returns:
        pd.DataFrame: 
    """

    df_signals = pd.DataFrame(index=df_stock.index)

    for sma, sma_window in zip(["fast_sma", "slow_sma"],[fast_sma_window, slow_wma_window]):
        df_signals[sma] = (
            df_stock[col_price].rolling(window=sma_window,
                                        min_periods=1,
                                       center=False).mean()
        )

    df_signals["signal"] = (
        np.where(df_signals["fast_sma"] > df_signals["slow_sma"],
                1,
                0)
    )

    df_signals["orders"] = (
        df_signals["signam"].diff()
    )

    df_signals.loc[df_signals["orders"] == 0, "orders"] = None

    return df_signals