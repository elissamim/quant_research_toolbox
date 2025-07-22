import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def bollinger_bands(price: pd.Series, 
                    window:Optional[int]=20,
                    num_std:Optional[int]=2) -> None:
    """
    Return a plot of prices with corresponding Bollinger Bands.

    Args:
        price (pd.Series): Series of prices indexed by time.
        window (Optional, int): Window period for computing the SMA. Defaults to `20`.
        num_std (Optional, int): Number of standard deviations for the bands. Defaults to `2`.

    Returns:
        None.
    """

    sma = price.rolling(
        window=window,
        center=False,
        min_periods=1
    ).mean()

    std = price.rolling(
        window=window,
        center=False,
        min_periods=1
    ).std()

    upper_bound = sma + num_std*std
    lower_bound = sma - num_std*std

    plt.figure(figsize=(12,6))
    plt.plot(price, label="Price")
    plt.plot(sma, label = f"SMA {window} periods")
    plt.plot(upper_bound, linestyle="--", label = "Upper Bollinger Band")
    plt.plot(lower_bound, linestyle="--", label="Lower Bollinger Band")
    plt.fill_between(price.index,
                    lower_bound,
                    upper_bound,
                    alpha=.2)
    plt.legend()
    plt.title("Bollinger bands")
    plt.show()