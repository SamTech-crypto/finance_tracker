import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
import warnings
from fpdf import FPDF
from io import StringIO
warnings.filterwarnings("ignore")

# -----------------------------
# Custom CSS for App Styling
# -----------------------------
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background-color: #2e3d49;
            color: white;
        }
        .sidebar .sidebar-header {
            background-color: #2c3e50;
        }
        .sidebar .sidebar-menu a {
            color: #ecf0f1;
        }
        .streamlit-expanderHeader {
            font-weight: bold;
        }
        .streamlit-expanderContent {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #27ae60;
        }
        h1 {
            color: #2980b9;
        }
        h2, h3 {
            color: #34495e;
        }
    </style>
""", unsafe_allow_html=True)

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
# Forecasting Functions
# -----------------------------
def forecast_balance(df, model_choice='ARIMA', months=12):
    monthly_balance = df.resample("M", on="Date")["Net_Balance"].sum()

    if model_choice == 'ARIMA':
        model = ARIMA(monthly_balance, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=months)

    elif model_choice == 'Exponential Smoothing':
        model = ExponentialSmoothing(monthly_balance, trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(steps=months)

    elif model_choice == 'Prophet':
        prophet_df = monthly_balance.reset_index()
        prophet_df.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(prophet_df, periods=months, freq='M')
        forecast = model.predict(future)['yhat'][-months:]

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
forecast_model = st.sidebar.selectbox("Choose Forecast Model", ["ARIMA", "Exponential Smoothing", "Prophet"])

# Load data: Upload CSV or Generate Mock
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
    "USD": 1, "EUR": 1.07, "KES": 0.0068, "UGX": 0.00027, 
    "NGN": 0.0024, "GBP": 1.35, "RWF": 0.00091
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
# Display Data
# -----------------------------
with st.expander("🔍 Show Raw Data"):
    st.write(df.head(20))

# -----------------------------
# Forecast
# -----------------------------
forecast, monthly_balance = forecast_balance(df, forecast_model, forecast_months)

# Plot
st.plotly_chart(plot_forecast(monthly_balance, forecast), use_container_width=True)

# -----------------------------
# Expense Category Breakdown
# -----------------------------
with st.expander("📂 Expense Category Breakdown"):
    category_summary = df.groupby("Expenses_Category")["Expenses"].sum()
    st.bar_chart(category_summary)

# -----------------------------
# Budget vs Actual Spending
# -----------------------------
with st.expander("📉 Budget vs Actual Spending"):
    budgets = {"Rent": 1500, "Groceries": 600, "Entertainment": 400}
    monthly_cat = df.groupby([pd.Grouper(key="Date", freq="M"), "Expenses_Category"])["Expenses"].sum().unstack().fillna(0)
    st.write("### Actual Monthly Spending")
    st.line_chart(monthly_cat)

    # Compare with budgets
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
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Monthly Finance Summary", ln=1, align="C")

        for idx, row in summary.iterrows():
            line = f"{idx.strftime('%Y-%m')}: Income: {row['Income']:.2f}, Expenses: {row['Expenses']:.2f}, Net: {row['Net_Balance']:.2f}"
            pdf.cell(200, 10, txt=line, ln=1)

        pdf_output = StringIO()
        pdf.output(pdf_output)
        st.download_button("Download PDF Summary", pdf_output.getvalue(), file_name="finance_summary.pdf")

    except ImportError:
        st.warning("PDF export requires installing `fpdf` package. Run `pip install fpdf`.")

# -----------------------------
# Made with ❤️ Section at the bottom
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Made with ❤️ by [SamTech](https://github.com/SamTech-crypto)", unsafe_allow_html=True)
