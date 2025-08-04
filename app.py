import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="EV Forecast", layout="wide", initial_sidebar_state="expanded")

# Load model
import os


model_path = os.path.join(os.path.dirname(__file__), "forecasting_ev_model.pkl")
model = joblib.load(model_path)


# Style
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            color: #262730;
        }
        .stApp {
            background: linear-gradient(to right, #dae2f8, #d6a4a4);
        }
        h1, h2, h3, h4 {
            color: #222831;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #1f1f1f;'>🔮 EV Forecasting for Washington Counties</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #2d3436;'>Explore EV growth predictions powered by ML models</h4>", unsafe_allow_html=True)

# Banner image
st.image("ev-car-factory.jpg", use_container_width=True, caption="The Future is Electric ⚡")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique().tolist())

# Sidebar Inputs
with st.sidebar:
    st.title("⚙️ Forecast Controls")
    county = st.selectbox("Select a County", county_list)
    compare_mode = st.checkbox("Compare up to 3 Counties")
    if compare_mode:
        multi_counties = st.multiselect("Select up to 3 counties", county_list, max_selections=3)

# Filter for selected county
if county not in df['County'].unique():
    st.error(f"County '{county}' not found in data.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

# Forecast loop
forecast_rows = []
forecast_horizon = 36
for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    ev_growth_slope = np.polyfit(range(len(cumulative_ev[-6:])), cumulative_ev[-6:], 1)[0] if len(cumulative_ev) >= 6 else 0

    input_data = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([input_data]))[0]
    forecast_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
    historical_ev.append(pred)
    historical_ev = historical_ev[-6:]
    cumulative_ev.append(cumulative_ev[-1] + pred)
    cumulative_ev = cumulative_ev[-6:]

# Combine historical + forecast
historical_df = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_df['Cumulative EV'] = historical_df['Electric Vehicle (EV) Total'].cumsum()
historical_df['Source'] = "Historical"

forecast_df = pd.DataFrame(forecast_rows)
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_df['Cumulative EV'].iloc[-1]
forecast_df['Source'] = "Forecast"

combined = pd.concat([
    historical_df[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# Plot
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, group in combined.groupby("Source"):
    ax.plot(group["Date"], group["Cumulative EV"], label=label, marker='o')
ax.set_title(f"EV Growth Forecast - {county}", color="#333")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor('#f8f9fa')
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)

# Summary
historical_total = historical_df['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]
growth = ((forecasted_total - historical_total) / historical_total) * 100 if historical_total > 0 else 0
trend = "increase 📈" if growth > 0 else "decrease 📉"
st.metric(label="📈 Forecast Growth in EV Adoption", value=f"{growth:.2f}%", delta=trend)

# Comparison Mode
if compare_mode and multi_counties:
    st.markdown("---")
    st.subheader("📍 Compare Forecasts for Selected Counties")
    comparison_data = []

    for cty in multi_counties:
        temp_df = df[df['County'] == cty].sort_values("Date")
        hist = list(temp_df['Electric Vehicle (EV) Total'].values[-6:])
        cum = list(np.cumsum(hist))
        months = temp_df['months_since_start'].max()
        last_date = temp_df['Date'].max()
        code = temp_df['county_encoded'].iloc[0]
        future = []

        for i in range(1, forecast_horizon + 1):
            months += 1
            future_date = last_date + pd.DateOffset(months=i)
            roll_mean = np.mean(hist[-3:])
            pct1 = (hist[-1] - hist[-2]) / hist[-2] if hist[-2] != 0 else 0
            pct3 = (hist[-1] - hist[-3]) / hist[-3] if hist[-3] != 0 else 0
            slope = np.polyfit(range(len(cum[-6:])), cum[-6:], 1)[0] if len(cum) >= 6 else 0
            row = {
                'months_since_start': months,
                'county_encoded': code,
                'ev_total_lag1': hist[-1],
                'ev_total_lag2': hist[-2],
                'ev_total_lag3': hist[-3],
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct1,
                'ev_total_pct_change_3': pct3,
                'ev_growth_slope': slope
            }
            pred = model.predict(pd.DataFrame([row]))[0]
            future.append({"Date": future_date, "Predicted EV Total": round(pred)})
            hist.append(pred)
            hist = hist[-6:]
            cum.append(cum[-1] + pred)
            cum = cum[-6:]

        hist_df = temp_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_df['Cumulative EV'] = hist_df['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_df['Cumulative EV'].iloc[-1]
        all_cty = pd.concat([
            hist_df[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ])
        all_cty['County'] = cty
        comparison_data.append(all_cty)

    comp_df = pd.concat(comparison_data, ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby("County"):
        ax.plot(group["Date"], group["Cumulative EV"], marker='o', label=cty)
    ax.set_title("EV Growth Comparison", fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Show growth percentages
    st.subheader("📌 County-wise 3-Year Growth")
    summaries = []
    for cty in multi_counties:
        cdf = comp_df[comp_df["County"] == cty].reset_index(drop=True)
        hist = cdf['Cumulative EV'].iloc[len(cdf) - forecast_horizon - 1]
        fcst = cdf['Cumulative EV'].iloc[-1]
        if hist > 0:
            percent = ((fcst - hist) / hist) * 100
            summaries.append(f"{cty}: {percent:.2f}%")
        else:
            summaries.append(f"{cty}: N/A")
    st.info(" | ".join(summaries))

# Footer
st.markdown("---")
st.caption("🔧 Built with 💙 for **AICTE Internship Cycle 2 by S4F**")

