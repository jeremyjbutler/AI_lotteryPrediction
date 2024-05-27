import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

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
# 2. Identify Current Range Based on Most Recent Draws
recent_results = df.nlargest(10, 'date')
current_range = {}
for col in ball_cols:
    current_range[col] = {
        'min': recent_results[col].min(),
        'max': recent_results[col].max()
    }

# 3. LSTM Model

# Step 1: Preprocess the data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[ball_cols])
X, y = [], []
for i in range(1, len(df)):
    X.append(scaled_data[i-1:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Step 2: Train LSTM model for each ball
next_predicted_numbers = {}
for idx, col in enumerate(ball_cols):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y[:, idx], epochs=50, verbose=0)
    
    # Step 3: Predict the next number for each ball
    next_num_scaled = model.predict(X[-1].reshape(1, 1, X.shape[2]))
    next_num = scaler.inverse_transform(np.hstack([next_num_scaled if i == idx else np.zeros((1, 1)) for i in range(len(ball_cols))]))[0][idx]
    next_predicted_numbers[col] = int(round(next_num))

# Step 4: Ensure the numbers are within the current game's parameters
for col, val in next_predicted_numbers.items():
    min_val, max_val = current_range[col]['min'], current_range[col]['max']
    next_predicted_numbers[col] = min(max_val, max(min_val, val))

print(f"Here are the next winning lottery numbers: {next_predicted_numbers}")