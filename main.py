import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import load_model

# Load preprocessed data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
scaler = np.load('data/scaler.npy', allow_pickle=True).item()

# Reshape data for LSTM [samples, time steps, features]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Load the trained model
model = load_model('model/stock_lstm_model.h5')

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f"Root Mean Squared Error: {rmse}")

# Plot the predictions
data = pd.read_csv('data/stock_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

train_size = int(len(data) * 0.8)
train = data[:train_size]
valid = data[train_size:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
