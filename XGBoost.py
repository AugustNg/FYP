import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
def load_data():
    df = pd.read_csv('retail_store_inventory.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert boolean columns to integers (1 for True, 0 for False)
    boolean_columns = ['Holiday/Promotion']
    df[boolean_columns] = df[boolean_columns].astype(int)

    # Handle categorical columns (use One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'], drop_first=True)

    # Drop unnecessary columns
    df = df.drop(columns=['Date', 'Units Ordered', 'Demand Forecast'])

    # Define Features (X) and Target (y)
    X = df.drop(columns=['Units Sold'])
    y = df['Units Sold']

    # Scale numerical columns
    scaler = MinMaxScaler()
    numerical_columns = ['Inventory Level', 'Price', 'Discount', 'Competitor Pricing']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Create Sliding Window for Time Series (window_size = 14)
    window_size = 14
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X.iloc[i - window_size:i].values)
        y_seq.append(y.iloc[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq, scaler, X, y

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for predicting 'Units Sold' for the next day
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create the line chart for past sales and predicted demand
def plot_past_sales_and_predictions(actual_sales, predicted_sales):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_sales, label='Actual Sales')
    plt.plot(predicted_sales, label='Predicted Sales')
    plt.title('Past Sales and Predicted Demand for each SKU')
    plt.xlabel('Index')
    plt.ylabel('Units Sold')
    plt.legend()
    st.pyplot()

# Create the demand prediction for the next 7 days
def plot_next_7_days_demand(predictions_7_days):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 8), predictions_7_days)
    plt.title('Predicted Demand for Next 7 Days')
    plt.xlabel('Days')
    plt.ylabel('Units Sold')
    st.pyplot()

# Main function to display Streamlit UI
def main():
    st.title("Retail Store Demand Prediction Dashboard")

    # Load and preprocess data
    df = load_data()
    X_seq, y_seq, scaler, X, y = preprocess_data(df)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Flatten predictions and actual values for easier comparison
    y_pred_original = y_pred_original.flatten()
    y_test_original = y_test_original.flatten()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)

    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Plot Actual vs Predicted Units Sold
    st.header("Actual vs Predicted Units Sold")
    plot_past_sales_and_predictions(y_test_original, y_pred_original)

    # Predict demand for next 7 days (example using the last day from the test set)
    next_7_days_demand = np.array([y_pred_original[-1]] * 7)  # Simplified, replace with actual LSTM prediction logic
    st.header("Predicted Demand for Next 7 Days")
    plot_next_7_days_demand(next_7_days_demand)

    # Additional dashboard features can be added here (e.g., store-wise breakdown, SKU details, etc.)

if __name__ == "__main__":
    main()