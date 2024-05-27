from ensemble_learning import IsolationForest

# Function to identify anomalies (lucky numbers) using Isolation Forest
def find_anomalies(data):
    model = IsolationForest(contamination=0.1)
    model.fit(data.reshape(-1, 1))
    anomaly_pred = model.predict(data.reshape(-1, 1))
    anomalies = data[anomaly_pred == -1]
    return anomalies

# Initialize dictionary to store lucky numbers
lucky_numbers = {}

# Calculate the frequency distribution for each ball and find anomalies
for col in ball_cols:
    frequency_distributions = df[col].value_counts().reset_index()
    frequency_distributions.columns = ['Number', 'Frequency']
    lucky_nums = find_anomalies(frequency_distributions['Frequency'].values)
    lucky_numbers[col] = frequency_distributions[frequency_distributions['Frequency'].isin(lucky_nums)]['Number'].values

# Ensuring the lucky numbers are within the current game's parameters
for col in lucky_numbers.keys():
    lucky_numbers[col] = np.clip(lucky_numbers[col], current_range[col]['min'], current_range[col]['max'])

print(f"Lucky numbers based on anomaly detection: {lucky_numbers}")