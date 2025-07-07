import pandas as pd


def daily_drawdown(df_cumulative_returns:pd.Series)->pd.Series:
    """

    """

    df_running_max = df_cumulative_returns.cummax()
    df_daily_drawdown = df_cumulative_returns/df_running_max -1
    return df_daily_drawdown

def max_drawdown(df_cumulative_returns:pd.Series)->pd.Series:
    """

    """

    df_daily_drawdown = daily_drawdown(df_cumulative_returns)
    mdd = df_daily_drawdown.min()
    return mdd