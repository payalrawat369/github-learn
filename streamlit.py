import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 🎯 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Air Quality Data Explorer", layout="wide")

# -------------------------------
# 🎨 HEADER
# -------------------------------
st.markdown("""
# 🌿 Air Quality Data Explorer
### Milestone 1: Working Application (Weeks 1–2)
""")

# -------------------------------
# 📊 SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🧭 Data Controls")
location = st.sidebar.selectbox("Location", ["Downtown Station", "Suburban Station", "Industrial Zone"])
time_range = st.sidebar.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"])
pollutants = st.sidebar.multiselect("Pollutants", ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"], default=["PM2.5"])

if st.sidebar.button("Apply Filters"):
    st.sidebar.success("Filters applied!")

st.sidebar.header("📈 Data Quality")
st.sidebar.progress(92)
st.sidebar.progress(87)

# -------------------------------
# 📈 MAIN DASHBOARD
# -------------------------------
col1, col2 = st.columns([2, 2])
with col1:
    st.subheader("PM2.5 Time Series")
    times = pd.date_range("2025-10-31", periods=24, freq="H")
    values = np.random.randint(10, 80, size=24)
    fig, ax = plt.subplots()
    ax.plot(times, values, marker="o", color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration (µg/m³)")
    st.pyplot(fig)

with col2:
    st.subheader("Pollutant Correlations")
    corr_data = pd.DataFrame(np.random.rand(4, 4), columns=["PM2.5", "PM10", "NO2", "O3"])
    sns.heatmap(corr_data, annot=True, cmap="Greens")
    st.pyplot(plt)

# -------------------------------
# 📉 STATS + DISTRIBUTION
# -------------------------------
col3, col4 = st.columns([2, 2])

with col3:
    st.subheader("Statistical Summary")
    mean = round(values.mean(), 1)
    max_v = values.max()
    min_v = values.min()
    std = round(values.std(), 1)
    total = len(values)
    st.metric("Mean (µg/m³)", mean)
    st.metric("Max (µg/m³)", max_v)
    st.metric("Min (µg/m³)", min_v)
    st.metric("Std Dev", std)
    st.metric("Data Points", total)

with col4:
    st.subheader("Distribution Analysis")
    fig2, ax2 = plt.subplots()
    ax2.hist(values, bins=[0, 20, 40, 60, 80, 100], color="green", edgecolor="white")
    ax2.set_xlabel("PM2.5 Range (µg/m³)")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

st.success("Dashboard loaded successfully ✅")
