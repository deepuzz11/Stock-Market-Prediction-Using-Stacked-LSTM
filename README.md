# StockNet : Stock Market Prediction Using Stacked LSTM
Objective: Develop a model to predict stock prices using a stacked LSTM network.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Project Overview
This project investigates the application of stacked Long Short-Term Memory (LSTM) networks for stock price prediction. LSTMs are a powerful type of recurrent neural network (RNN) adept at learning complex relationships in sequential data, making them well-suited for analyzing historical stock price movements.

## Installation
1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/Stock-Market-Prediction-Using-Stacked-LSTM.git
    ```
2. **Navigate to the project directory:**
    ```sh
    cd Stock-Market-Prediction-Using-Stacked-LSTM
    ```
3. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```
4. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Training the Model
1. **Ensure the preprocessed data is in the `data/` directory.**
2. **Run the training script:**
    ```sh
    python train_model.py
    ```
3. **The trained model will be saved in the `model/` directory as `stock_lstm_model.keras`.**

### Evaluating the Model
1. **Run the evaluation script:**
    ```sh
    python evaluate_model.py
    ```
2. **The script will load the trained model, make predictions, and display the results.**

## Results
The `evaluate_model.py` script will output the Root Mean Squared Error (RMSE) of the model's predictions and display a plot comparing the actual and predicted stock prices.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This project uses TensorFlow and Keras for building and training the LSTM model.
