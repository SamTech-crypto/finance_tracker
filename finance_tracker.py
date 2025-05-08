# finance_tracker.py

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

    categories = [
        "Rent/Mortgage", "Groceries", "Utilities", "Transportation", "Insurance",
        "Healthcare", "Subscriptions", "Education", "Childcare", "Clothing",
        "Dining Out", "Entertainment", "Savings", "Donations", "Miscellaneous"
    ]

    data = {
        "Date": dates,
        "Income": np.where(np.random.rand(len(dates)) < 0.1,
                           np.random.normal(2000, 500, len(dates)), 0),
        "Expenses": np.random.normal(1000, 200, len(dates)),
        "Investment_Contribution": np.where(np.random.rand(len(dates)) < 0.05,
                                            np.random.normal(500, 100, len(dates)), 0),
        "Investment_Value": np.cumsum(np.random.normal(10000, 1000, len(dates))),
        "Expenses_Category": np.random.choice(categories, len(dates)),
        "Currency": "USD"
    }

    df = pd.DataFrame(data)
    df["Net_Balance"] = df["Income"] - df["Expenses"] + df["Investment_Value"].diff().fillna(0)
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
# Forecast Plot
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
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Finance Tracker", layout="wide")
st.title("📊 Household Finance Tracker & Forecaster")

# Sidebar controls
st.sidebar.markdown("### Data Source")
use_uploaded = st.sidebar.toggle("Use uploaded CSV file", False)

template = {
    "Date": "2024-01-01",
    "Income": 2000,
    "Expenses": 1200,
    "Investment_Contribution": 300,
    "Investment_Value": 10500,
    "Expenses_Category": "Groceries",
    "Currency": "USD"
}

if st.sidebar.button("📥 Download CSV Template"):
    template_df = pd.DataFrame([template])
    st.download_button("Download Template CSV", template_df.to_csv(index=False), file_name="finance_template.csv")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))

if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload Your Finance CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
        st.success("Uploaded file loaded successfully.")
    else:
        st.warning("Please upload a valid CSV file.")
        st.stop()
else:
    df = generate_mock_data(start_date, end_date)

df = validate_data(df)

# -----------------------------
# Currency Conversion
# -----------------------------
currency_rates = {
    "USD": 1, "EUR": 1.07, "GBP": 1.24,
    "KES": 0.0068, "UGX": 0.00026, "RWF": 0.00083, "NGN": 0.00066
}
default_currency = st.sidebar.selectbox("Convert to Currency", list(currency_rates.keys()), index=0)

if "Currency" in df.columns:
    df["Rate"] = df["Currency"].map(currency_rates).fillna(1)
else:
    df["Currency"] = "USD"
    df["Rate"] = 1

df[["Income", "Expenses", "Investment_Value"]] *= df["Rate"]
df["Net_Balance"] = df["Income"] - df["Expenses"] + df["Investment_Value"].diff().fillna(0)

# -----------------------------
# Forecasting
# -----------------------------
forecast_months = st.sidebar.slider("Forecast Months", 3, 24, 12)
forecast, monthly_balance = forecast_balance(df, forecast_months)
st.plotly_chart(plot_forecast(monthly_balance, forecast), use_container_width=True)

# -----------------------------
# Budget Comparison
# -----------------------------
with st.expander("📉 Budget vs Actual Spending"):
    budgets = {
        "Rent/Mortgage": 1200, "Groceries": 500, "Utilities": 150,
        "Transportation": 200, "Insurance": 250, "Healthcare": 100,
        "Subscriptions": 50, "Education": 300, "Childcare": 400,
        "Clothing": 100, "Dining Out": 150, "Entertainment": 100,
        "Savings": 500, "Donations": 75, "Miscellaneous": 100
    }

    monthly_cat = df.groupby([pd.Grouper(key="Date", freq="M"), "Expenses_Category"])["Expenses"].sum().unstack().fillna(0)
    st.write("### Actual Monthly Spending")
    st.line_chart(monthly_cat)

    latest_month = monthly_cat.tail(1).T
    latest_month.columns = ["Actual"]
    latest_month["Budget"] = latest_month.index.map(budgets).fillna(0)
    st.write("### Latest Month: Budget vs Actual")
    st.bar_chart(latest_month)

# -----------------------------
# Export Summary
# -----------------------------
with st.expander("📤 Export Summary"):
    summary = df.resample("M", on="Date")[["Income", "Expenses", "Net_Balance"]].sum()
    csv = summary.to_csv()
    st.download_button("Download Monthly Summary (CSV)", csv, "monthly_summary.csv", "text/csv")

    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Monthly Finance Summary", ln=1, align="C")
        for idx, row in summary.iterrows():
            line = f"{idx.strftime('%Y-%m')}: Income: {row['Income']:.2f}, Expenses: {row['Expenses']:.2f}, Net: {row['Net_Balance']:.2f}"
            pdf.cell(200, 10, txt=line, ln=1)
        pdf.output("finance_summary.pdf")
        with open("finance_summary.pdf", "rb") as f:
            st.download_button("Download PDF Summary", f, file_name="finance_summary.pdf")
    except ImportError:
        st.warning("PDF export requires installing `fpdf`. Run `pip install fpdf`.")
