# Stock Market Prediction Using Stacked LSTM
Objective: Develop a model to predict stock prices using a stacked LSTM network.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Project Overview
This project aims to predict stock market prices using a Stacked Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock prices to predict future stock prices.

## Project Structure
Stock-Market-Prediction-Using-Stacked-LSTM/
│
├── data/
│ ├── X_train.npy
│ ├── X_test.npy
│ ├── y_train.npy
│ └── y_test.npy
│
├── model/
│ └── stock_lstm_model.keras
│
├── train_model.py
├── evaluate_model.py
├── requirements.txt
└── README.md


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
