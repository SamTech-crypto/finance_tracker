import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Mock Data Generator
# -----------------------------
@st.cache_data
def generate_mock_data(start_date="2023-01-01", end_date="2025-12-31"):
    dates = pd.date_range(start_date, end_date, freq="D")
    np.random.seed(42)

    data = {
        "Date": dates,
        "Income": np.where(np.random.rand(len(dates)) < 0.1,
                           np.random.normal(2000, 500, len(dates)), 0),
        "Expenses": np.random.normal(1000, 200, len(dates)),
        "Investment_Contribution": np.where(np.random.rand(len(dates)) < 0.05,
                                            np.random.normal(500, 100, len(dates)), 0),
        "Investment_Value": np.cumsum(np.random.normal(10000, 1000, len(dates)))
    }

    df = pd.DataFrame(data)
    df["Net_Balance"] = df["Income"] - df["Expenses"] + df["Investment_Value"].diff().fillna(0)
    df["Expenses_Category"] = np.random.choice(["Rent", "Groceries", "Entertainment"], len(dates))
    return df

# -----------------------------
# Data Validation
# -----------------------------
def validate_data(df):
    df["Expenses"] = df["Expenses"].clip(lower=0)
    df["Income"] = df["Income"].clip(lower=0)
    z_scores = (df["Expenses"] - df["Expenses"].mean()) / df["Expenses"].std()
    df.loc[z_scores.abs() > 3, "Expenses"] = df["Expenses"].mean()
    return df

# -----------------------------
# Forecasting
# -----------------------------
def forecast_balance(df, months=12):
    monthly_balance = df.resample("M", on="Date")["Net_Balance"].sum()
    model = ARIMA(monthly_balance, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(steps=months)

    forecast_dates = pd.date_range(monthly_balance.index[-1], periods=months+1, freq="M")[1:]
    return pd.Series(forecast, index=forecast_dates), monthly_balance

# -----------------------------
# Visualization
# -----------------------------
def plot_forecast(monthly_balance, forecast):
    forecast_df = pd.DataFrame({
        "Date": forecast.index,
        "Forecasted Balance": forecast.values
    })

    fig = px.line(x=monthly_balance.index, y=monthly_balance, title="Net Balance Forecast")
    fig.add_scatter(x=forecast_df["Date"], y=forecast_df["Forecasted Balance"],
                    mode="lines", name="Forecast")
    fig.update_layout(xaxis_title="Date", yaxis_title="Balance")
    return fig

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("📊 Household Finance Tracker & Forecaster")
st.markdown("This app generates mock financial data, validates it, and forecasts net balance.")

# Sidebar options
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))
forecast_months = st.sidebar.slider("Forecast Months", 3, 24, 12)

# Load data
df = generate_mock_data(start_date, end_date)
df = validate_data(df)

# Show raw data
with st.expander("🔍 Show Raw Data"):
    st.write(df.head(20))

# Forecast
forecast, monthly_balance = forecast_balance(df, forecast_months)

# Plot
st.plotly_chart(plot_forecast(monthly_balance, forecast), use_container_width=True)

# Optional: Show category spending
with st.expander("📂 Expense Category Breakdown"):
    category_summary = df.groupby("Expenses_Category")["Expenses"].sum()
    st.bar_chart(category_summary)

# -----------------------------
# Made with ❤️ Section at the bottom
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Made with ❤️ by [SamTech](https://github.com/SamTech-crypto)", unsafe_allow_html=True)
