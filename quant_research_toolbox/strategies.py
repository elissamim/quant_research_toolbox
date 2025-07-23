"""
Functions and classes for trading strategies including: momentum, mean-reversion, pair trading and oscillators.
"""

import pandas as pd
import numpy as np
from typing import Optional


class Momentum:
    """
    Class for Momentum strategies.
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
        after cross over and sell signal (-1) when slow SMA is higher before cross over.

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

        df_signals["orders"] = df_signals["signal"].diff().fillna(0)

        return df_signals

    @staticmethod
    def naive_momentum(
        df_stock: pd.DataFrame,
        col_price: Optional[str] = "close",
        nb_consecutive_days: Optional[int] = 2,
    ) -> pd.DataFrame:
        """
        Generate signals and orders following naive momentum strategy:
        buy signal (1) if the price keeps growing for at least a certain number of days,
        and sell signal (-1) if the price keeps diminishing for the same number of days.

        Args:
            df_stock (pd.DataFrame): Time series of a stock (index is datetime).
            col_price (Optional, str): Column name of price series. Defaults to `close`.
            nb_consecutive_days (Optional, int): Threshold of days for momentum. Defaults to `2`.

        Returns:
            pd.DataFrame: A dataframe containing trading signals and Long/Short orders.
        """

        df_signals = pd.DataFrame(index=df_stock.index)
        df_signals["nb_consecutive_days"] = 0
        df_signals["signal"] = 0
        df_signals["price_diff"] = df_stock[col_price].diff()
        df_signals["orders"] = 0

        signal = 0
        count_consecutive_days = 0

        for i in range(1, len(df_signals.index)):
            if df_signals.loc[df_signals.index[i], "price_diff"] > 0:
                count_consecutive_days += 1 if count_consecutive_days >= 0 else 1
                df_signals.loc[df_signals.index[i], "nb_consecutive_days"] = (
                    count_consecutive_days
                )

                if count_consecutive_days == nb_consecutive_days and signal != 1:
                    df_signals.loc[df_signals.index[i], "orders"] = 1
                    signal = 1
                    df_signals.loc[df_signals.index[i], "signal"] = signal

            elif df_signals.loc[df_signals.index[i], "price_diff"] < 0:
                count_consecutive_days -= 1 if count_consecutive_days <= 0 else -1
                df_signals.loc[df_signals.index[i], "nb_consecutive_days"] = (
                    count_consecutive_days
                )

                if count_consecutive_days == -nb_consecutive_days and signal != -1:
                    df_signals.loc[df_signals.index[i], "orders"] = -1
                    signal = -1
                    df_signals.loc[df_signals.index[i], "orders"] = signal

            else:
                count_consecutive_days = 0
                signal = 0

        return df_signals


class MeanReversion:
    """
    Class for Mean Reversion strategies.
    """

    @staticmethod
    def sma_mean_reversion(
        df_stock: pd.DataFrame,
        col_price: Optional[str] = "close",
        entry_threshold: Optional[float] = 1.0,
        exit_threshold: Optional[float] = 0.5,
        window: Optional[int] = 20,
    ) -> pd.DataFrame:
        """
        Generate signals for SMA mean reversion: buy (1) when the price is under
        the SMA minus a given number of standard deviations, sell (-1) when the price is over
        the SMA plus a given number of standard deviations.

        Args:
            df_stock (pd.DataFrame): Table containing stock prices indexed by time.
            col_price (Optional, str): Name of the column containing prices. Defaults to `close`.
            entry_threshold (Optional, float): Number of standard deviations above the SMA. Defaults to `1`.
            exit_threshold (Optional, float): Number of standard deviations under the SMA. Defaults to `0.5`.
            window (Optional, int): Window for the SMA.

        Returns:
            pd.DataFrame: A dataframe containing trading signals and Long/Short orders.
        """

        df_signals = pd.DataFrame(index=df_stock.index)

        df_signals["sma"] = (
            df_stock[col_price]
            .rolling(window=window, center=False, min_periods=1)
            .mean()
        )

        df_signals["std"] = (
            df_stock[col_price]
            .rolling(window=window, center=False, min_periods=1)
            .std()
        )

        conditions = [
            df_stock[col_price]
            > df_signals["sma"] + entry_threshold * df_signals["std"],
            df_stock[col_price]
            < df_signals["sma"] - exit_threshold * df_signals["std"],
        ]

        choices = [-1, 1]

        df_signals["signal"] = np.select(conditions, choices, default=0)

        df_signals["orders"] = df_signals["signal"].diff().fillna(0)

        return df_signals
