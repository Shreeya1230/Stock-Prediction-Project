import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_stock_data(api_endpoint):
    response = requests.get(api_endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error: Couldn't retrieve stock data")
        return None

def format_stock_data(raw_data):
    if 'Time Series (5min)' in raw_data:
        stock_df = pd.DataFrame(raw_data['Time Series (5min)']).T
        stock_df = stock_df.rename(columns={
            "1. open": "opening_price",
            "2. high": "highest_price",
            "3. low": "lowest_price",
            "4. close": "closing_price",
            "5. volume": "trade_volume"
        })
        stock_df = stock_df.astype(float)
        stock_df.dropna(inplace=True)
        stock_df.sort_index(inplace=True)
        return stock_df
    else:
        print("Error: Unexpected stock data format")
        return None

def build_price_model(train_features, train_target):
    price_model = LinearRegression()
    price_model.fit(train_features, train_target)
    return price_model

def check_model_accuracy(price_model, test_features, test_target):
    predicted_prices = price_model.predict(test_features)
    error_value = mean_squared_error(test_target, predicted_prices)
    return error_value

def predict_future_prices(price_model, input_features):
    return price_model.predict(input_features)

def plot_predictions(real_prices, estimated_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(real_prices, label='Real Prices', color='blue')
    plt.plot(estimated_prices, label='Estimated Prices', color='red')
    plt.legend(['Real Prices', 'Estimated Prices'])
    plt.title("Stock Market Predictions")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.show()

def save_analysis_report(report_data):
    with open("stock_report.json", "w") as file:
        json.dump(report_data, file, indent=4)

def forecast_latest_price(price_model, most_recent_data):
    recent_data_df = pd.DataFrame([most_recent_data])
    estimated_price = price_model.predict(recent_data_df)
    print("Estimated Closing Price:", estimated_price[0])
    return estimated_price[0]

API_KEY = "8G46Y51D4GNWHSTD"
API_URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={API_KEY}"

stock_data = get_stock_data(API_URL)
if stock_data:
    stock_df = format_stock_data(stock_data)
    if stock_df is not None:
        feature_data = stock_df[['opening_price', 'highest_price', 'lowest_price', 'trade_volume']]
        target_data = stock_df['closing_price']
        train_features, test_features, train_target, test_target = train_test_split(feature_data, target_data, test_size=0.2, random_state=42)
        price_model = build_price_model(train_features, train_target)
        model_error = check_model_accuracy(price_model, test_features, test_target)
        estimated_prices = predict_future_prices(price_model, test_features)
        plot_predictions(test_target.values, estimated_prices)
        save_analysis_report({"Error Value": model_error, "Predicted Prices": estimated_prices.tolist()})
        latest_stock_info = feature_data.iloc[-1]
        forecast_latest_price(price_model, latest_stock_info)
