import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Load model
@st.cache_resource
def load_model():
    try:
        # Load model from pickle file
        with open('best_xgb_model.pkl', 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_model()

# Streamlit App
st.title("📈 Sales Forecast Dashboard")

# Display dashboard info if file is uploaded
if 'df' in st.session_state:
    # Retrieve data from session state if it's already processed
    df = st.session_state.df

    # Ensure the 'sales_amount' column is created before any grouping or operations
    df['sales_amount'] = df['Units Sold'] * df['Price']

    # Create temporal features from the 'Date' column
    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday

    # 🔮 7-Day Demand Forecast Per SKU (Separate by Store ID)
    st.subheader("🔮 7-Day Demand Forecast Per SKU")

    future_days = 7
    latest_date = df['Date'].max()

    # Get latest data per SKU (for each store)
    latest_rows = df.sort_values('Date').groupby(['Product ID', 'Store ID']).tail(1)

    future_forecasts = []
    for _, row in latest_rows.iterrows():
        for i in range(1, future_days + 1):
            future_date = latest_date + pd.Timedelta(days=i)
            future_row = row.copy()
            future_row['Date'] = future_date
            future_forecasts.append(future_row)

    future_df = pd.DataFrame(future_forecasts)

    # Filter the forecast data by selected stores
    future_df_filtered = future_df[future_df['Store ID'].isin(st.session_state.selected_stores)]

    # Model input columns expected by the model
    categorical_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
    numerical_cols = ['Price', 'Units Ordered', 'Demand Forecast', 'Holiday/Promotion', 'Discount', 'Competitor Pricing', 'Inventory Level', 'Hour', 'Day', 'Month', 'Weekday']

    # Ensure the data passed to the model's preprocessor matches the required format
    input_data = future_df_filtered[categorical_cols + numerical_cols]

    # Apply the preprocessing pipeline to the input data
    future_df_transformed = model.named_steps['preprocessor'].transform(input_data)

    # Now, ensure the model uses the correct input data
    future_df_filtered['Predicted Units Sold'] = model.predict(future_df_transformed).astype(int)

    # Display the forecast results
    forecast_output = future_df_filtered[['Product ID', 'Store ID', 'Date', 'Predicted Units Sold']].sort_values(['Product ID', 'Store ID', 'Date'])
    st.dataframe(forecast_output)

else:
    # File upload section moved to the bottom
    uploaded_file = st.file_uploader("Upload CSV with sales data", type=['csv'], key='file_uploader')

    if uploaded_file is not None:
        # Load and preprocess
        df = pd.read_csv(uploaded_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df.dropna(subset=['Units Sold', 'Price'], inplace=True)

        # Create temporal features from the 'Date' column
        df['Hour'] = df['Date'].dt.hour
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.weekday

        # Save the dataframe to session state for later use
        st.session_state.df = df

        # Notify user that the file has been uploaded successfully
        st.success("File uploaded successfully! Now, the dashboard is updated with your data.")
    else:
        st.info("Please upload a CSV file to get started.")
