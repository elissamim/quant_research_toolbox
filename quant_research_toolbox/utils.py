import pandas as pd
import yfinance as yf


def load_ticker_data(ticker_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """ 
    
    """

    df_ticker = yf.download(ticker_name, start=start_date, end=end_date)
    df_ticker.columns = [col[0].lower() for col in df_ticker.columns]
    df_ticker["ohlc_price"] = df_ticker[["close", "high", "low", "open"]].mean(1)
    df_ticker["typical_price"] = df_ticker[["close", "high", "low"]].mean(1)
    df_ticker["median_price"] = df_ticker[["high", "low"]].mean(1)

    return df_ticker

def compute_returns(df_signals:pd.DataFrame,
                   col_prices:str,
                   col_orders:str) -> pd.DataFrame:
    """

    Args:

    Returns:
        
    """

    df_returns = pd.DataFrame(df_signals.index)

    buy_indices = df_signals[df_signals[col_orders]==1].index
    sell_indices = df_signals[df_signals[col_orders]==-1].index
