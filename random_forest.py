import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Data Reading and Cleaning
df = pd.read_csv('historical_data.csv')
df.drop(columns=['Unnamed: 0', 'dp'], inplace=True)
df['date'] = pd.to_datetime(df['date'])
ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'pb']
df[ball_cols] = df[ball_cols].astype(int)

# 2. Feature Engineering - Lagged Variables
max_lag = 20  # Max number of lags to create for each ball
lagged_df = df.copy()
for col in ball_cols:
    for lag in range(1, max_lag + 1):
        lagged_df[f'{col}_lag_{lag}'] = lagged_df[col].shift(lag)
lagged_df.dropna(inplace=True)

# Select the most predictive lags based on our earlier analysis
selected_lags = ['b1_lag_11', 'b1_lag_11', 'b5_lag_11', 'b5_lag_4', 'b5_lag_11', 'b5_lag_15']

# 3. Model Training
X = lagged_df[selected_lags]
y = lagged_df[ball_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Random Forest as the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
test_score = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {test_score}")

# 5. Prediction of Next Winning Numbers
next_X = lagged_df[selected_lags].iloc[-1].values.reshape(1, -1)
next_pred = model.predict(next_X)
next_pred = np.round(next_pred).astype(int)

# Ensure the predicted numbers are within the current game's parameters
next_pred = np.clip(next_pred, [1, 7, 20, 33, 42, 1], [47, 54, 59, 63, 67, 23])

print(f"Here are the next winning lottery numbers: b1={next_pred[0][0]}, b2={next_pred[0][1]}, b3={next_pred[0][2]}, b4={next_pred[0][3]}, b5={next_pred[0][4]}, pb={next_pred[0][5]}")