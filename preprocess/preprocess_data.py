import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load the data
data = pd.read_csv('data/stock_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

sequence_length = 60  # e.g., use 60 days of data to predict the next day's price
sequences = create_sequences(scaled_data, sequence_length)
X, y = sequences[:, :-1], sequences[:, -1]

# Split into training and testing sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save preprocessed data
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)
np.save('data/scaler.npy', scaler)

print("Data preprocessed and saved.")
