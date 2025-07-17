import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

# Load model
@st.cache_resource
def load_model():
    with open('best_lstm_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Streamlit App
st.title("📈 Sales Forecast Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload CSV with sales data", type=['csv'])

if uploaded_file is not None:
    # Load and preprocess
    df = pd.read_csv(uploaded_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Units Sold', 'Price'], inplace=True)

    # 🔹 Historical Sales Line Chart (Separate by Store ID)
    st.subheader("🔹 Historical Sales")
    df['sales_amount'] = df['Units Sold'] * df['Price']
    sales_over_time = df.groupby(['Store ID', 'Date'])['sales_amount'].sum().reset_index()

    # Create a list of store IDs for user to select
    store_ids = sales_over_time['Store ID'].unique()
    selected_stores = st.multiselect("Select Stores to view", options=store_ids, default=store_ids)

    # Plot sales for the selected stores
    filtered_sales = sales_over_time[sales_over_time['Store ID'].isin(selected_stores)]
    store_sales = filtered_sales.pivot(index='Date', columns='Store ID', values='sales_amount')
    st.line_chart(store_sales)

    # 🔮 7-Day Demand Forecast per SKU (Separate by Store ID)
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

    # Model input columns expected by the model
    model_input_cols = ['Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount']

    # Check if all required columns are in the future_df
    missing_cols = [col for col in model_input_cols if col not in future_df.columns]

    # If there are missing columns, handle them
    if missing_cols:
        st.warning(f"The following required columns are missing in the data: {missing_cols}")
        # Optionally, you can fill missing columns with default values (e.g., 0 or mean)
        for col in missing_cols:
            if col == 'Inventory Level' or col == 'Units Ordered':  # Example: fill with 0 if necessary
                future_df[col] = 0
            elif col == 'Price' or col == 'Discount':  # Example: fill with mean or any default value
                future_df[col] = future_df['Price'].mean() if 'Price' in future_df.columns else 0
            elif col == 'Demand Forecast':  # Example: fill with a constant or median
                future_df[col] = future_df['Demand Forecast'].mean() if 'Demand Forecast' in future_df.columns else 0

    # Now predict only if the necessary columns exist
    if all(col in future_df.columns for col in model_input_cols):
        future_df['Predicted Units Sold'] = model.predict(future_df[model_input_cols]).astype(int)
        forecast_output = future_df[['Product ID', 'Store ID', 'Date', 'Predicted Units Sold']].sort_values(['Product ID', 'Store ID', 'Date'])
        st.dataframe(forecast_output)
    else:
        st.error("Missing necessary columns for prediction.")

    # 🔹 Sales (YTD, MTD, Today's Sales) KPIs
    today = pd.to_datetime(datetime.date.today())
    ytd_sales = df[df['Date'].dt.year == today.year]['sales_amount'].sum()
    mtd_sales = df[(df['Date'].dt.year == today.year) & (df['Date'].dt.month == today.month)]['sales_amount'].sum()
    today_sales = df[df['Date'].dt.date == today.date()]['sales_amount'].sum()

    st.subheader("🔹 Sales")
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Year-to-Date", f"${ytd_sales:,.2f}")
    col2.metric("📆 Month-to-Date", f"${mtd_sales:,.2f}")
    col3.metric("🕒 Today's Sales", f"${today_sales:,.2f}")

    # Display the raw data at the bottom
    st.subheader("Raw Data")
    st.dataframe(df.head())

else:
    st.info("Please upload a CSV file to get started.")