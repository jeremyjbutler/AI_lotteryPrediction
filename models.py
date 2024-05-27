import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import beta
from dataframes import LottoDataFrame


class BayseianModel(LottoDataFrame):
    
    def __init__(self):
        pass
    def predict_powerball(self):
        self.df = LottoDataFrame.get_powerball(self)
        
        print(self.df.head())
        recent_results = self.df.nlargest(10, 'date')
        current_range = {}
        for col in self.ball_cols:
            current_range[col] = {
                'min': recent_results[col].min(),
                'max': recent_results[col].max()
            }

        recent_data = self.df[self.df['date'] >= '2019-01-01']

        # Step 1: Use a prior probability distribution based on historical frequency
        frequency_distributions = {}
        for col in self.df.ball_cols:
            frequency_distributions[col] = Counter(recent_data[col])

        # Step 2 & 3: Use Bayesian updating to generate a posterior distribution
        # Here, we use a Beta distribution for simplicity
        posterior_distributions = {}
        for col in self.ball_cols:
            alpha_prior = 1
            beta_prior = 1
            alpha_posterior = alpha_prior + sum(frequency_distributions[col].values())
            beta_posterior = beta_prior + len(recent_data) - sum(frequency_distributions[col].values())
            posterior_distributions[col] = beta(alpha_posterior, beta_posterior)

        # Step 4: Sample the most probable numbers from the posterior distributions
        np.random.seed(42)  # for reproducibility
        next_predicted_numbers = {}
        for col in self.ball_cols:
            next_predicted_numbers[col] = int(posterior_distributions[col].mean() * current_range[col]['max'])

        # Ensure the numbers are within the current game's parameters
        for col, val in next_predicted_numbers.items():
            min_val, max_val = current_range[col]['min'], current_range[col]['max']
            next_predicted_numbers[col] = min(max_val, max(min_val, val))
        return next_predicted_numbers
        #print(f"Baysian Interface Model Prediction: {next_predicted_numbers}")
        
new_bay = BayseianModel()
get_numbers = new_bay.predict_powerball()
print(get_numbers)