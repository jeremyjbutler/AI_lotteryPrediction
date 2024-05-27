
import pandas as pd
import ssl
from collections import Counter
from scipy.stats import beta
import numpy as np
#powerball
ssl._create_default_https_context = ssl._create_unverified_context

class LottoDataFrame():
    def __init__(self):
      pass
    def get_powerball(self):
        self.url = 'https://data.ny.gov/resource/d6yy-54nr.csv'

        # 1. Data Reading and Cleaning
        self.df = pd.read_csv(self.url)
        self.df['winning_numbers'] = self.df['winning_numbers'].str.split()
        self.df['pb'] = self.df['multiplier'].astype(int)

        for i in range(1, 6):  # Assuming there are 5 numbers in each winning set
            self.df[f'b{i}'] = self.df['winning_numbers'].str[i - 1]
        for i in range(1, 6):
            self.df[f'b{i}'] = pd.to_numeric(self.df[f'b{i}'], errors='coerce')
        print(self.df.head())
        self.df['date'] = pd.to_datetime(self.df['draw_date'])

        #Powerball
        ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'pb']
        self.df[ball_cols] = self.df[ball_cols].astype(int)
        recent_results = self.df.nlargest(10, 'date')
        current_range = {}
        for col in ball_cols:
            current_range[col] = {
                'min': recent_results[col].min(),
                'max': recent_results[col].max()
            }

        recent_data = self.df[self.df['date'] >= '2019-01-01']

        # Step 1: Use a prior probability distribution based on historical frequency
        frequency_distributions = {}
        for col in ball_cols:
            frequency_distributions[col] = Counter(recent_data[col])

        # Step 2 & 3: Use Bayesian updating to generate a posterior distribution
        # Here, we use a Beta distribution for simplicity
        posterior_distributions = {}
        for col in ball_cols:
            alpha_prior = 1
            beta_prior = 1
            alpha_posterior = alpha_prior + sum(frequency_distributions[col].values())
            beta_posterior = beta_prior + len(recent_data) - sum(frequency_distributions[col].values())
            posterior_distributions[col] = beta(alpha_posterior, beta_posterior)

        # Step 4: Sample the most probable numbers from the posterior distributions
        np.random.seed(42)  # for reproducibility
        next_predicted_numbers = {}
        for col in ball_cols:
            next_predicted_numbers[col] = int(posterior_distributions[col].mean() * current_range[col]['max'])

        # Ensure the numbers are within the current game's parameters
        for col, val in next_predicted_numbers.items():
            min_val, max_val = current_range[col]['min'], current_range[col]['max']
            next_predicted_numbers[col] = min(max_val, max(min_val, val))
        return next_predicted_numbers
        