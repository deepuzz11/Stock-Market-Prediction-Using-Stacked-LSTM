import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the model
model = load_model('model/stock_lstm_model.keras')
print("Model loaded successfully.")

# Load preprocessed test data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
scaler = np.load('data/scaler.npy', allow_pickle=True).item()

# Reshape data for LSTM [samples, time steps, features]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f"Root Mean Squared Error: {rmse}")

# Load original data for plotting
data = pd.read_csv('data/stock_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Split the data into training and validation sets
train_size = int(len(data) * 0.8)
train = data[:train_size]
valid = data[train_size:]

# Diagnostic outputs
print(f"Total data length: {len(data)}")
print(f"Training data length: {len(train)}")
print(f"Validation data length: {len(valid)}")
print(f"Predictions length: {len(predictions)}")

# Ensure lengths match
if len(predictions) < len(valid):
    valid = valid.iloc[:len(predictions)]
elif len(predictions) > len(valid):
    predictions = predictions[:len(valid)]

valid['Predictions'] = predictions

# Plot the results
plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'], label='Train')
plt.plot(valid[['Close', 'Predictions']], label=['Close', 'Predictions'])
plt.legend(loc='lower right')
plt.show()
