"""
Functions and classes for tradinf strategies including: momentum, mean-reversion, pair trading and oscillators.
"""

import pandas as pd
import numpy as np
from typing import Optional

class Momentum:
    """ 
    """

    @staticmethod
    def sma_crossover(
        df_stock: pd.DataFrame,
        col_price: Optional[str] = "close",
        fast_sma_window: Optional[int] = 5,
        slow_sma_window: Optional[int] = 30,
    ) -> pd.DataFrame:
        """
        Generate signals and orders based on a SMA crossover strategy: buy signal (1) generated when fast SMA is higher
        and sell signal (-1) when slow SMA is higher.

        Args:
            df_stock (pd.DataFrame): Time series of a stock (index is datetime).
            col_price (Optional, str): Column name of price series. Defaults to `close`.
            fast_sma_window (Optional, int): Window for fast SMA. Defaults to `5`.
            slow_sma_window (Optional, int): Window for slow SMA. Defaults to `30`.

        Returns:
            pd.DataFrame: A dataframe containing trading signals and Long/Short orders.
        """

        df_signals = pd.DataFrame(index=df_stock.index)

        for sma, sma_window in zip(
            ["fast_sma", "slow_sma"], [fast_sma_window, slow_sma_window]
        ):
            df_signals[sma] = (
                df_stock[col_price]
                .rolling(window=sma_window, min_periods=1, center=False)
                .mean()
            )

        df_signals["signal"] = np.where(
            df_signals["fast_sma"] > df_signals["slow_sma"], 1, 0
        )

        df_signals["orders"] = df_signals["signal"].diff()

        df_signals.loc[df_signals["orders"] == 0, "orders"] = None

        return df_signals

    @staticmethod
    def naive_momentum(
        df_stock: pd.DataFrame,
        col_price: Optional[str] = "close",
        nb_consecutive_days: Optional[int] = 2
    ) -> pd.DataFrame:
        """

        """

        df_signals = pd.DataFrame(index=df_stock.index)
        df_signals["orders"] = 0

        df_price_diff = df_stock[col_price].diff()

        signal = 0
        count_consecutive_days = 0

        for i in range(1, len(df_stock.index)):
            if df_price_diff.iloc[i] > 0:
                count_consecutive_days += 1 if count_consecutive_days >= 0 else 1
                if count_consecutive_days == nb_consecutive_days and signal != 1:
                    df_signals.loc[df_signals.index[i],"orders"] = 1
                    signal = 1
            elif df_price_diff.iloc[i] < 0:
                count_consecutive_days -= 1 if count_consecutive_days <= 0 else -1
                if count_consecutive_days == -nb_consecutive_days and signal !=-1:
                    df_signals.loc[df_signals.index[i], "orders"] = -1
                    signal = -1
            else:
                count_consecutive_days = 0
                signal = 0

        return df_signals
            

        
        
class MeanReversion:

    pass