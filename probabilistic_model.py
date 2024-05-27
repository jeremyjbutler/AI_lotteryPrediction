import pandas as pd
import numpy as np
from collections import Counter

# 1. Data Reading and Cleaning
df = pd.read_csv('historical_data.csv')
df.drop(columns=['Unnamed: 0', 'dp'], inplace=True)
df['date'] = pd.to_datetime(df['date'])
ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'pb']
df[ball_cols] = df[ball_cols].astype(int)

# 2. Identify Current Range Based on Most Recent Draws
recent_results = df.nlargest(10, 'date')
current_range = {}
for col in ball_cols:
    current_range[col] = {
        'min': recent_results[col].min(),
        'max': recent_results[col].max()
    }

# 3. Custom Probabilistic Model

# Filter data to include only the most recent results (assuming current rules started from 2019 onwards)
recent_data = df[df['date'] >= '2019-01-01']

# Calculate the frequency distribution for each ball
frequency_distributions = {}
for col in ball_cols:
    frequency_distributions[col] = Counter(recent_data[col])

# Normalize the frequencies to convert them into probabilities
probability_distributions = {}
for col, freq_dist in frequency_distributions.items():
    total_count = sum(freq_dist.values())
    probability_distributions[col] = {k: v / total_count for k, v in freq_dist.items()}

# Use Monte Carlo simulation to sample from this distribution to predict the next numbers
np.random.seed(42)  # for reproducibility

def sample_next_number(prob_dist):
    numbers = list(prob_dist.keys())
    probabilities = list(prob_dist.values())
    return np.random.choice(numbers, p=probabilities)

next_predicted_numbers = {}
for col in ball_cols:
    next_predicted_numbers[col] = sample_next_number(probability_distributions[col])

# Ensure the numbers are within the current game's parameters
for col, val in next_predicted_numbers.items():
    min_val, max_val = current_range[col]['min'], current_range[col]['max']
    next_predicted_numbers[col] = min(max_val, max(min_val, val))

print(f"Here are the next winning lottery numbers: {next_predicted_numbers}")