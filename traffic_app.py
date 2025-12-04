import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. INTERNAL DATA GENERATOR
# ==========================================
@st.cache_data
def generate_traffic_data():
    np.random.seed(42)
    n = 50

    # Generate Distances (3km to 20km)
    dist = np.random.normal(10.66, 4.0, n)
    dist = np.clip(dist, 3, 20)

    # Generate Signals & Potholes (Correlated with distance)
    signals = np.random.poisson(14, n) + (dist - 10.66) * 0.3
    signals = np.round(np.clip(signals, 0, 30)).astype(int)

    potholes = np.random.normal(5, 3, n) + (dist - 10.66) * 0.2
    potholes = np.round(np.clip(potholes, 0, 20)).astype(int)

    # Generate Time using Regression Logic
    base_time = 9.538 + 4.524 * dist
    noise = (signals * 0.5) + (potholes * 1.0) + np.random.normal(0, 5, n)
    time_raw = base_time + noise

    # Normalize stats
    target_mean = 57.77
    target_sd = 23.45
    time_final = (time_raw - time_raw.mean()) * (target_sd / time_raw.std()) + target_mean
    time_final = np.maximum(time_final, 15)

    df = pd.DataFrame({
        'Distance_km': dist.round(2),
        'Time_min': time_final.round(2),
        'Signals': signals,
        'Potholes': potholes
    })

    zones = ['Whitefield', 'Marathahalli', 'KR Puram', 'Hebbal',
             'Koramangala', 'Silk Board', 'Bellandur', 'HSR Layout']
    df['Zone'] = np.random.choice(zones, n)

    # Road Quality (Inverse to Potholes)
    df['Road_Quality'] = 10 - (df['Potholes'] / 20 * 9).astype(int)
    df['Road_Quality'] = df['Road_Quality'].clip(1, 10)

    return df


df = generate_traffic_data()

# ==========================================
# 2. APP LAYOUT
# ==========================================
st.set_page_config(page_title="Bangalore Traffic Estimator", layout="wide")
st.title("🚦 Bangalore Traffic Congestion Predictor")

# --- SIDEBAR INPUTS ---
st.sidebar.header("📝 Trip Details")

# 1. Distance (Always Manual)
distance = st.sidebar.number_input("Commute Distance (km)", min_value=1.0, value=10.0, step=0.5)

# 2. Zone Selection (Required for Average lookups)
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location")
available_zones = sorted(df['Zone'].unique().tolist())
available_zones.append("Other / Unknown Route")
selected_zone = st.sidebar.selectbox("Select Zone / Area", available_zones)


# Helper function to get averages
def get_zone_averages(zone_name):
    if zone_name == "Other / Unknown Route":
        return int(df['Signals'].mean()), int(df['Road_Quality'].mean())
    else:
        zone_df = df[df['Zone'] == zone_name]
        return int(zone_df['Signals'].mean()), int(zone_df['Road_Quality'].mean())


avg_sig_val, avg_qual_val = get_zone_averages(selected_zone)

# 3. Traffic Signals Input (Independent Control)
st.sidebar.markdown("---")
st.sidebar.subheader("🚦 Traffic Signals")
use_avg_signals = st.sidebar.checkbox(f"Use average signals for {selected_zone.split('/')[0]}", value=False)

if use_avg_signals:
    # AUTO MODE
    signals = avg_sig_val
    st.sidebar.info(f"Using average: **{signals} signals**")
else:
    # MANUAL MODE
    signals = st.sidebar.slider("Number of Signals", 0, 50, 14)

# 4. Road Condition Input (Independent Control)
st.sidebar.markdown("---")
st.sidebar.subheader("🚧 Road Condition")
knows_road_quality = st.sidebar.checkbox("I have traveled this road before", value=True)

if knows_road_quality:
    # MANUAL MODE
    quality_rating = st.sidebar.slider("Rate Road Quality (1-10)", 1, 10, 5,
                                       help="1 = Many Potholes, 10 = Smooth")
    estimated_potholes = int((10 - quality_rating) * 2.2)
    st.sidebar.caption(f"Manual Estimate: ~{estimated_potholes} potholes.")
else:
    # AUTO MODE
    quality_rating = avg_qual_val
    estimated_potholes = int((10 - quality_rating) * 2.2)
    st.sidebar.info(f"Using average quality: **{quality_rating}/10**")
    st.sidebar.caption(f"Auto Estimate: ~{estimated_potholes} potholes.")

# ==========================================
# 3. CALCULATIONS
# ==========================================
INTERCEPT = 9.538
COEFF_DIST = 4.18
COEFF_SIGNAL = 0.95
COEFF_POTHOLE = 1.8

predicted_time = INTERCEPT + (COEFF_DIST * distance) + (COEFF_SIGNAL * signals) + (COEFF_POTHOLE * estimated_potholes)
fuel_loss = (predicted_time / 60) * 0.3
fuel_cost = fuel_loss * 102

# ==========================================
# 4. OUTPUT DISPLAY
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⏱️ Predicted Time", f"{predicted_time:.0f} min")
with col2:
    st.metric("⛽ Est. Fuel Wasted", f"{fuel_loss:.2f} L")
with col3:
    st.metric("💸 Trip Cost", f"₹{fuel_cost:.0f}")

st.markdown("---")

# DASHBOARD
st.header("📊 Data Visualization")
tab1, tab2 = st.tabs(["Zone Analysis", "Raw Data"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Avg Signals by Zone")
        sig_chart = df.groupby("Zone")["Signals"].mean().sort_values()
        st.bar_chart(sig_chart)

    with col_b:
        st.subheader("Avg Road Quality by Zone")
        qual_chart = df.groupby("Zone")["Road_Quality"].mean().sort_values()
        st.bar_chart(qual_chart)

with tab2:
    st.dataframe(df)