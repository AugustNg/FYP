import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Streamlit App
st.title("📈 Sales Forecast Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload CSV with sales data", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Visualize Line Chart of Past Sales
    st.subheader("🔹 Historical Sales Line Chart")
    sales_over_time = df.groupby('date')['sales'].sum().reset_index()
    st.line_chart(sales_over_time.rename(columns={'sales': 'Total Sales'}).set_index('date'))

    # Predict sales using the model (assume you predict per SKU)
    st.subheader("🔹 Predicted Sales per SKU")
    
    # Customize depending on your model input
    model_input_cols = ['feature1', 'feature2']  # Update based on your model
    pred_df = df.dropna(subset=model_input_cols).copy()
    pred_df['predicted_sales'] = model.predict(pred_df[model_input_cols])
    st.dataframe(pred_df[['SKU', 'predicted_sales']].groupby('SKU').sum())

    # Calculate YTD, MTD, Today
    today = pd.to_datetime(datetime.date.today())
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    df = df.dropna(subset=['sales'])

    ytd_sales = df[df['date'].dt.year == today.year]['sales'].sum()
    mtd_sales = df[(df['date'].dt.year == today.year) & (df['date'].dt.month == today.month)]['sales'].sum()
    today_sales = df[df['date'].dt.date == today.date()]['sales'].sum()

    # Display KPIs
    st.subheader("🔹 Sales KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Year-to-Date", f"${ytd_sales:,.2f}")
    col2.metric("📆 Month-to-Date", f"${mtd_sales:,.2f}")
    col3.metric("🕒 Today's Sales", f"${today_sales:,.2f}")

else:
    st.info("Please upload a CSV file to get started.")
