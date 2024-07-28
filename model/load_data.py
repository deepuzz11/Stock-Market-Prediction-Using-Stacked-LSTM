from keras.models import load_model
import numpy as np

# Load the model
model = load_model('model/stock_lstm_model.keras')
print("Model loaded successfully.")

# Create dummy data for testing the model
dummy_input = np.random.rand(1, 60, 1)  # Replace 60 with your sequence length
dummy_input = dummy_input.reshape((dummy_input.shape[0], dummy_input.shape[1], 1))

# Make a prediction
prediction = model.predict(dummy_input)
print(f"Prediction for dummy input: {prediction}")
