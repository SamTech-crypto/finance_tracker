import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import warnings
from fpdf import FPDF
from io import BytesIO

warnings.filterwarnings("ignore")

# -----------------------------
# Custom Styling
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
        .streamlit-expanderHeader {
            font-weight: bold;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
        }
        h1 {
            color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Realistic Household Mock Data Generator
# -----------------------------
@st.cache_data
def generate_mock_data(start_date="2023-01-01", end_date="2025-12-31"):
    dates = pd.date_range(start_date, end_date, freq="D")
    np.random.seed(42)

    # Income: Monthly + occasional freelance
    base_income = 3000
    income = [base_income + np.random.normal(100, 50) if d.day == 1 else 0 for d in dates]
    freelance_income = [np.random.normal(300, 100) if np.random.rand() < 0.1 else 0 for _ in dates]
    total_income = np.array(income) + np.array(freelance_income)

    # Expenses across multiple categories
    categories = ["Rent", "Groceries", "Utilities", "Transport", "Healthcare",
                  "Insurance", "Education", "Subscriptions", "Childcare", "Loan Repayment"]
    expenses_data = {cat: np.random.normal(2 if cat != "Rent" else 45, 1, len(dates)) for cat in categories}
    expenses_df = pd.DataFrame(expenses_data)
    expenses_df["Total_Expenses"] = expenses_df.sum(axis=1)

    # Investment: 15% of income
    investment_contribution = np.array([0.15 * inc if inc > 0 else 0 for inc in total_income])

    # Investment Value: Compound growth
    investment_value = []
    value = 5000
    for contrib in investment_contribution:
        value += value * 0.0005 + contrib
        investment_value.append(value)

    # Build dataframe
    df = pd.DataFrame({
        "Date": dates,
        "Income": total_income,
        "Investment_Contribution": investment_contribution,
        "Investment_Value": investment_value,
        "Expenses": expenses_df["Total_Expenses"],
        "Expenses_Category": np.random.choice(categories, len(dates), p=[0.1]*10)
    })

    df["Net_Balance"] = df["Income"] - df["Expenses"] + pd.Series(investment_value).diff().fillna(0)
    return df

# -----------------------------
# Data Validation
# -----------------------------
def validate_data(df):
    df["Expenses"] = df["Expenses"].clip(lower=0)
    df["Income"] = df["Income"].clip(lower=0)
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
    fig = px.line(x=monthly_balance.index, y=monthly_balance, title="Net Balance Forecast")
    fig.add_scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast")
    fig.update_layout(xaxis_title="Date", yaxis_title="Net Balance")
    return fig

# -----------------------------
# App Title
# -----------------------------
st.title("🏡 Household Budget Tracker & Forecaster")
st.markdown("Monitor your income, expenses, investments, and forecast your household's financial future.")

# Sidebar Inputs
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))
forecast_months = st.sidebar.slider("Forecast Months", 3, 24, 12)

# Data Loading
df = generate_mock_data(start_date, end_date)
df = validate_data(df)

# Currency Conversion
currency_rates = {"USD": 1, "EUR": 1.07, "KES": 0.0068, "UGX": 0.00027, "NGN": 0.0024, "GBP": 1.35, "RWF": 0.00091}
currency = st.sidebar.selectbox("Display Currency", list(currency_rates.keys()), index=0)
rate = currency_rates[currency]

df[["Income", "Expenses", "Investment_Value", "Investment_Contribution", "Net_Balance"]] *= rate

# -----------------------------
# Raw Data Preview
# -----------------------------
with st.expander("📄 Show Raw Data"):
    st.dataframe(df.head(20))

# -----------------------------
# Forecasting Section
# -----------------------------
forecast, monthly_balance = forecast_balance(df, forecast_months)
st.plotly_chart(plot_forecast(monthly_balance, forecast), use_container_width=True)

# -----------------------------
# Category Breakdown
# -----------------------------
with st.expander("📊 Expense Category Breakdown"):
    daily_expenses = df.groupby("Date")["Expenses"].sum()
    category_total = df["Expenses_Category"].value_counts().sort_values(ascending=False)
    st.subheader("Top Expense Categories")
    st.bar_chart(category_total)

    category_spending = df.groupby(["Date", "Expenses_Category"])["Expenses"].sum().unstack().fillna(0)
    st.subheader("Daily Category Spending")
    st.line_chart(category_spending)

# -----------------------------
# Budget Comparison
# -----------------------------
with st.expander("📉 Budget vs Actual Spending"):
    budgets = {
        "Rent": 1350, "Groceries": 600, "Utilities": 300, "Transport": 250,
        "Healthcare": 200, "Insurance": 150, "Education": 200,
        "Subscriptions": 80, "Childcare": 400, "Loan Repayment": 300
    }

    monthly_actual = df.groupby([pd.Grouper(key="Date", freq="M"), "Expenses_Category"])["Expenses"].sum().unstack().fillna(0)
    st.line_chart(monthly_actual)

    # Compare with budgets (latest month only)
    latest = monthly_actual.tail(1).T
    latest.columns = ["Actual"]
    latest["Budget"] = latest.index.map(budgets)
    latest["% Over/Under"] = ((latest["Actual"] - latest["Budget"]) / latest["Budget"]) * 100
    st.write("### Latest Month Budget Comparison")
    st.dataframe(latest.style.format({"Actual": "{:.2f}", "Budget": "{:.2f}", "% Over/Under": "{:.1f}%"}))

# -----------------------------
# Export Summary
# -----------------------------
with st.expander("📤 Export Financial Summary"):
    summary = df.resample("M", on="Date")[["Income", "Expenses", "Net_Balance"]].sum()
    csv = summary.to_csv()
    st.download_button("⬇️ Download Monthly Summary (CSV)", csv, "monthly_summary.csv", "text/csv")

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Monthly Financial Summary", ln=1, align="C")

        for idx, row in summary.iterrows():
            line = f"{idx.strftime('%Y-%m')}: Income: {row['Income']:.2f}, Expenses: {row['Expenses']:.2f}, Net: {row['Net_Balance']:.2f}"
            pdf.cell(200, 10, txt=line, ln=1)

        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button("⬇️ Download PDF Summary", data=buffer.getvalue(), file_name="finance_summary.pdf")

    except Exception as e:
        st.warning("PDF export failed. Make sure `fpdf` is installed.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Made with ❤️ by [SamTech](https://github.com/SamTech-crypto)", unsafe_allow_html=True)
