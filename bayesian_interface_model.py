import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import beta

#url = 'https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD'

#powerball
url = 'https://data.ny.gov/resource/d6yy-54nr.csv'

# 1. Data Reading and Cleaning
df = pd.read_csv(url)
print(df.head())

df['winning_numbers'] = df['winning_numbers'].str.split()

for i in range(1, 6):  # Assuming there are 5 numbers in each winning set
    df[f'b{i}'] = df['winning_numbers'].str[i - 1]
for i in range(1, 6):
    df[f'b{i}'] = pd.to_numeric(df[f'b{i}'], errors='coerce')
print(df.head())

#df.drop(columns=['Unnamed: 0', 'dp'], inplace=True)
df['date'] = pd.to_datetime(df['draw_date'])

#print(df.head())
#Powerball
ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'multiplier']
df[ball_cols] = df[ball_cols].astype(int)

# 2. Identify Current Range Based on Most Recent Draws
recent_results = df.nlargest(10, 'date')
current_range = {}
for col in ball_cols:
    current_range[col] = {
        'min': recent_results[col].min(),
        'max': recent_results[col].max()
    }

# 3. Bayesian Inference Model

# Filter data to include only the most recent results (assuming current rules started from 2019 onwards)
recent_data = df[df['date'] >= '2019-01-01']

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

print(f"Baysian Interface Model Prediction: {next_predicted_numbers}")