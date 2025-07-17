#!/usr/bin/env python
# coding: utf-8

# # Data Undetstanding

# In[143]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the dataset (replace with your actual data file path)
df = pd.read_csv('retail_store_inventory.csv')

# Step 2: Convert Date Column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Step 3: Convert boolean columns to integers (1 for True, 0 for False)
boolean_columns = ['Holiday/Promotion']
df[boolean_columns] = df[boolean_columns].astype(int)

# Step 4: Handle categorical columns (use One-Hot Encoding or Label Encoding)
df = pd.get_dummies(df, columns=['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'], drop_first=True)

# Step 5: Drop any irrelevant or unnecessary columns
df = df.drop(columns=['Date', 'Units Ordered', 'Demand Forecast'])  # Dropping columns that aren't directly used for prediction

# Step 6: Define Features (X) and Target (y)
X = df.drop(columns=['Units Sold'])  # Features (all columns except target)
y = df['Units Sold']  # Target variable (Units Sold)

# Step 7: Scale the Numerical Data (using MinMaxScaler)
scaler = MinMaxScaler()
numerical_columns = ['Inventory Level', 'Price', 'Discount', 'Competitor Pricing']  # Example numerical columns

# Scale only the numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 8: Create Sliding Window for Time Series (window_size = 14)
window_size = 14
X_seq, y_seq = [], []

for i in range(window_size, len(X)):
    X_seq.append(X.iloc[i - window_size:i].values)  # Last 14 days as input for each sample
    y_seq.append(y.iloc[i])  # Target variable (Units Sold) for the next day

# Convert to numpy arrays
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Step 9: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Reshape the data for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))  # [samples, timesteps, features]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))  # [samples, timesteps, features]

# Step 10: Ensure the columns are numeric (handle potential non-numeric data)
# Convert to numeric and replace non-numeric values with NaN
X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)

# Handle NaN values by replacing them with 0 (or any strategy you prefer)
X_train = np.nan_to_num(X_train, nan=0)  # Replace NaN with 0
X_test = np.nan_to_num(X_test, nan=0)    # Replace NaN with 0

# Step 11: Ensure the target is numeric (convert to float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Step 12: Build the LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for predicting 'Units Sold' for the next day
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build the model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))  # Define input shape based on the number of timesteps and features

# Step 13: Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Step 14: Predict and Inverse Transform the Results (for evaluating the actual prediction amount)
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values (Units Sold)
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))  # Inverse transform predicted target
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))    # Inverse transform actual target

# Flatten to 1D arrays for easier comparison
y_pred_original = y_pred_original.flatten()
y_test_original = y_test_original.flatten()

# Step 15: Calculate RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)

# Print RMSE and MAE
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Step 16: Plot the Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 17: Plot Actual vs Predicted Units Sold
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual Units Sold')
plt.plot(y_pred_original, label='Predicted Units Sold')
plt.title('Actual vs Predicted Units Sold')
plt.xlabel('Test Sample Index')
plt.ylabel('Units Sold')
plt.legend()
plt.show()

