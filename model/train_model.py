import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os
import time

# Load preprocessed data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Start timing
start_time = time.time()

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=10)

# End timing
end_time = time.time()

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model in the native Keras format
model.save('model/stock_lstm_model.keras')

print("Model trained and saved in Keras format.")
print(f"Training time: {end_time - start_time} seconds")
