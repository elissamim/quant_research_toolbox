"""
Functions and classes for auxiliary tasks.
"""

import pandas as pd
import yfinance as yf

def load_ticker_data(ticker_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ 
    Load data for a given ticker from yahoo finance.

    Args:
        ticker_name (str):Name of the ticker.
        start_date (str):Start date of the time series (YYYY-MM-DD).
        end_date (str):End date of the time series (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Time series with data for the given ticker including columns for
                      OHCL, typical, median, open, close, high and low prices for each day.
    """

    df_ticker = yf.download(ticker_name, start=start_date, end=end_date)
    df_ticker.columns = [col[0].lower() for col in df_ticker.columns]
    df_ticker["ohlc_price"] = df_ticker[["close", "high", "low", "open"]].mean(1)
    df_ticker["typical_price"] = df_ticker[["close", "high", "low"]].mean(1)
    df_ticker["median_price"] = df_ticker[["high", "low"]].mean(1)

    return df_ticker

def compute_cumulative_returns(
    df_signals: pd.DataFrame, col_prices: str, col_orders: str
) -> pd.Series:
    """
    Compute cumulative returns from a strategy given prices and orders.

    Args:
        df_signals (pd.DataFrame): DataFrame containing orders and prices.
        col_prices (str): Name of the columns containing the prices.
        col_orders (str): Name of the column containing the orders.

    Returns:
        pd.DataFrame: DataFrame with a column containing the strategy cumulative returns.
    """

    returns = pd.Series(0.0, index=df_signals.index)

    buy_dates = df_signals[df_signals[col_orders] == 1].index
    sell_dates = df_signals[df_signals[col_orders] == -1].index

    for buy_date in buy_dates:

        future_sells = sell_dates[sell_dates > buy_date]

        if not future_sells.empty:

            sell_date = future_sells[0]
            returns[sell_date] = (
                df_signals.loc[sell_date, col_prices] / 
                df_signals.loc[buy_date, col_prices] - 1
            )

        else:
            # If not futur sell dates after buy date, close the position at the last date
            returns.iloc[-1]=(
                df_signals.loc[df_signals.index[-1], col_prices] / 
                df_signals.loc[buy_date, col_prices] - 1
            )
            break

    return (1+returns).cumprod()-1
