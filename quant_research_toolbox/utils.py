import pandas as pd
import yfinance as yf

def load_ticker_data(ticker_name:str,
                    start_date:str,
                    end_date:str) -> pd.DataFrame:
    """

    """

    df_ticker = yfinance.download(ticker_name, 
                                  start=start_date, 
                                  end=end_date)
    df_ticker.columns = [col[0] for col in df_ticker.columns]
    df_ticker["ohlc_price"] = df_ticker[["Close", "High", "Low", "Open"]].mean(1)
    df_ticker["typical_price"] = df_ticker[["Close", "High", "Low"]].mean(1)
    df_ticker["median_price"] = df_ticker[["High", "Low"]].mean(1)

    return df_ticker

def compute_returns():
    pass