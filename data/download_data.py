import yfinance as yf
import os

# Define the stock ticker symbol
ticker_symbol = 'AAPL'  # Example: Apple Inc.

# Fetch historical data
data = yf.download(ticker_symbol, start='2010-01-01', end='2024-07-28')

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save data to CSV
data.to_csv('data/stock_prices.csv')

print("Data downloaded and saved to 'stock_prices.csv'.")
