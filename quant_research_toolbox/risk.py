import pandas as pd
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ValueAtRisk:

    confidence_level: float = .99
    alpha : float = field(init=False)

    def __post_init__(self):
        if not (0<=self.confidence_level<=1):
            raise ValueError("`confidence_level` should be between 0 and 1")
        self.alpha = 1 - self.confidence_level

    def historical_var(self,
                      df_returns:pd.Series) -> float:
        """
        Return the potential loss for a given confidence interval, given
        the distribution of historical returns. It involves sorting the historical returns
        and finding the percentile that corresponds to the desired confidence level.

        Args:
            df_returns (pd.Series): Returns series.

        Returns:
            float: Historical VaR.
        """

        var = np.percentile(df_returns,
                                100*self.alpha)
        
        return var
        
    def parametric_var(self,
                      df_returns:pd.Series,
                      n_periods:Optional[float]=None) -> float:
        """
        Return the potential loss for a given confidence interval, given
        that the distribution of the returns is assumed normal. Calculate VaR
        using the mean and standard deviation of the portfolio's returns.

        Args:
            df_returns (pd.Series): Returns series.

        Returns:
            float : Parametric VaR.
        """

        z_score = norm.ppf(self.alpha)
        mean_return = np.mean(df_returns)
        std_dev_return = np.std(df_returns)
        var = mean_return + z_score*std_dev_return

        if n_periods is None:
            return var

        return np.sqrt(n_periods) * var

    def monte_carlo_var(self,
                       df_returns:pd.Series,
                       num_simulations:Optional[int]=1000,
                       simulation_horizon:Optional[int]=252,
                       initial_investment:Optional[float]=1e6) -> float:
        """
        Return the potential loss for a given confidence interval, using
        random sampling to simulate a range of potential outcomes based on historical data.

        Args:
            df_returns (pd.Series): Returns series.

        Returns:
            float: Monte-Carlo VaR.
        """

        simulated_returns = (
            np.random.normal(
                np.mean(df_returns),
                np.std(df_returns),
                (simulation_horizon, num_simulations)
            )
        )

        portfolio_values = (
            initial_investment*np.exp(np.cumsum(simulated_returns,
                                               axis=0))
        )

        portfolio_returns = (
            portfolio_values[-1]/portfolio_values[0]-1
        )

        var = np.percentile(portfolio_returns,
                            100*self.alpha)

        return var