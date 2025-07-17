import pandas as pd
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field

@dataclass
class ValueAtRisk:

    confidence_level: float = .99
    alpha : float = field(init=False)

    def __post_init__(self):
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

        return np.percentile(df_returns,
                            100*self.alpha)
        

    def parametric_var(self,
                      df_returns:pd.Series) -> float:
        """
        Return the potential loss for a given confidence interval, given
        that the distribution of the returns is assumed normal. Calculate VaR
        using the mean and standard deviation of the portfolio's returns.

        Args:
            df_returns (pd.Series): Returns series.

        Returns:
            float : Parametric VaR.
        """

        z_score = nrom.ppf(self.alpha)
        mean_return = np.mean(df_returns)
        std_dev_return = np.std(df_returns)

        return mean_return + z_score*std_dev_return

    def monte_carlo_var(self,
                       df_returns:pd.Series,
                       num_simulations:int=1000,
                       simulation_horizon:int=252,
                       intial_investment:float=1e6) -> float:
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
            initial_investment *
        )
        
        
    