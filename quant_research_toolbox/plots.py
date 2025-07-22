import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def bollinger_bands(price, 
                    window:Optionel[int]=20,
                    num_std:Optional[int]=2) -> None:
    """

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
    plt.plot(upper_bound, linestyle="--")
    plt.plot(lower_bound, linestyle="--")
    plt.fill_between(price.index,
                    lower_bound,
                    upper_bound,
                    alpha=.2)
    plt.legend()
    plt.title("Bollinger bands")
    plt.show()