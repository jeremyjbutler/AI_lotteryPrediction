# Full code with feature engineering and ensemble learning

# Import required libraries
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import beta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from tensorflow.python.keras.layers import LSTMV1, Dense

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class FullLottoDataFrame():
    def __init__(self):
        pass
    # Function to sample next number based on calculated probabilities (for Monte Carlo)
    def sample_next_number(self, prob_dist):
        numbers = list(prob_dist.keys())
        probabilities = list(prob_dist.values())
        return np.random.choice(numbers, p=probabilities)
    def get_powerball(self):
        # Initialize DataFrame to store ensemble predictions
        ensemble_predictions = pd.DataFrame()

        # 1. Data Reading and Cleaning
        #powerball
        url = 'https://data.ny.gov/resource/d6yy-54nr.csv'

        # 1. Data Reading and Cleaning
        df = pd.read_csv(url)
        df['winning_numbers'] = df['winning_numbers'].str.split()
        df['pb'] = df['multiplier'].astype(int)

        for i in range(1, 6):  # Assuming there are 5 numbers in each winning set
            df[f'b{i}'] = df['winning_numbers'].str[i - 1]
        for i in range(1, 6):
            df[f'b{i}'] = pd.to_numeric(df[f'b{i}'], errors='coerce')
        print(df.head())
        df['date'] = pd.to_datetime(df['draw_date'])

        #Powerball
        ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'pb']
        df[ball_cols] = df[ball_cols].astype(int)


        # 1.5 Feature Engineering: Create lag variables
        lags = [1, 4, 11, 15]
        for col in ball_cols:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Drop rows with NaN values generated due to lag
        df.dropna(inplace=True)
        print(df.head())

        # 2. Identify Current Range Based on Most Recent Draws
        recent_results = df.nlargest(10, 'date')
        current_range = {}
        for col in ball_cols:
            current_range[col] = {
                'min': recent_results[col].min(),
                'max': recent_results[col].max()
            }

        # 3. Random Forest Predictions
        # (For demonstration, using limited data and model parameters)
        selected_lags = ['b1_lag_11', 'b2_lag_11', 'b3_lag_11', 'b4_lag_11', 'b5_lag_11', 'pb_lag_11']
        X = df[selected_lags][-10:]
        y = df[ball_cols][-10:]
        model_rf = RandomForestRegressor(n_estimators=10)
        model_rf.fit(X, y)
        rf_predictions = model_rf.predict(X.iloc[-1].values.reshape(1, -1))[0]

        # 4. Monte Carlo Predictions
        # Calculate the frequency distribution for each ball
        frequency_distributions = Counter(df['b1'][-10:])
        # Normalize the frequencies to convert them into probabilities
        total_count = sum(frequency_distributions.values())
        probability_distribution = {k: v / total_count for k, v in frequency_distributions.items()}
        mc_predictions = np.array([self.sample_next_number(probability_distribution) for _ in range(6)])

        # 5. Bayesian Inference Predictions
        # Calculate the frequency distribution for each ball
        frequency_distributions = Counter(df['b1'][-10:])
        # Use Bayesian updating to generate a posterior distribution
        alpha_prior = 1
        beta_prior = 1
        alpha_posterior = alpha_prior + sum(frequency_distributions.values())
        beta_posterior = beta_prior + len(df[-10:]) - sum(frequency_distributions.values())
        posterior_distribution = beta(alpha_posterior, beta_posterior)
        bayesian_predictions = np.array([int(posterior_distribution.mean() * current_range['b1']['max']) for _ in range(6)])

        # 6. LSTM Predictions
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[ball_cols][-10:])
        X, y = [], []
        for i in range(1, len(df[-10:])):
            X.append(scaled_data[i-1:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X, y[:, 0], epochs=50, verbose=0)
        next_num_scaled = model_lstm.predict(X[-1].reshape(1, 1, X.shape[2]))
        next_num = scaler.inverse_transform(np.hstack([next_num_scaled] + [np.zeros((1, 1)) for _ in range(5)]))[0][0]
        lstm_predictions = np.array([int(round(next_num)) for _ in range(6)])

        # 7. Ensemble Learning: Combine Predictions
        ensemble_predictions = pd.DataFrame({
            'RandomForest': rf_predictions,
            'MonteCarlo': mc_predictions,
            'Bayesian': bayesian_predictions,
            'LSTM': lstm_predictions
        })
        ensemble_predictions['Final'] = ensemble_predictions.mean(axis=1).astype(int)

        # 8. Ensure the numbers are within the current game's parameters
        for idx, col in enumerate(ball_cols):
            min_val, max_val = current_range[col]['min'], current_range[col]['max']
            ensemble_predictions.loc[idx, 'Final'] = min(max_val, max(min_val, ensemble_predictions.loc[idx, 'Final']))

        return ensemble_predictions['Final'].values

        #print(f"Ensemble Machine Learning Final Prediction: {ensemble_predictions['Final'].values}")

new_lotto = FullLottoDataFrame()

print(new_lotto.get_powerball())