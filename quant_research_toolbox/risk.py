import pandas as pd
from dataclasses import dataclass

@dataclass
class ValueAtRisk:

    df_returns: pd.Series
    confidence_level: float = .99

    def historical_var(self) -> float:
        """
        Return the potential loss for a given confidence interval, given
        the distribution of historical returns. It involves sorting the historical returns
        and finding the percentile that corresponds to the desired confidence level.
        """
        pass

    def parametric_var(self) -> float:
        """
        Return the potential loss for a given confidence interval, given
        that the distribution of the returns is assumed normal. Calculate VaR
        using the mean and standard deviation of the portfolio's returns.
        """
        pass

    def monte_carlo_var(self) -> float:
        """
        Return the potential loss for a given confidence interval, using
        random sampling to simulate a range of potential outcomes based on historical data.
        """
        pass
    