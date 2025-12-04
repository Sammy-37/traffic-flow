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

distance = st.sidebar.number_input("Commute Distance (km)", min_value=1.0, value=10.0, step=0.5)
signals = st.sidebar.slider("Traffic Signals", 0, 50, 14)

st.sidebar.markdown("---")
st.sidebar.subheader("Road Condition")

# Check if user knows the route
knows_route = st.sidebar.checkbox("I have traveled this road before", value=True)

if knows_route:
    # MANUAL INPUT
    quality_rating = st.sidebar.slider("Rate Road Quality (1-10)", 1, 10, 5,
                                       help="1 = Many Potholes, 10 = Smooth")
    estimated_potholes = int((10 - quality_rating) * 2.2)
    st.sidebar.caption(f"Manual input: Approx {estimated_potholes} potholes.")

else:
    # AUTOMATIC LOOKUP
    # Get unique zones from data + an 'Unknown' option
    available_zones = sorted(df['Zone'].unique().tolist())
    available_zones.append("Other / Unknown Route")

    selected_zone = st.sidebar.selectbox("Select Zone / Area", available_zones)

    if selected_zone == "Other / Unknown Route":
        # NO DATA CASE
        st.sidebar.error("⚠️ We don't have data for this route yet.")
        st.sidebar.info("Assuming 'Average' quality (5/10) for prediction.")
        quality_rating = 5
        estimated_potholes = int((10 - quality_rating) * 2.2)

    else:
        # PRE-EXISTING DATA CASE
        # Calculate average quality for that specific zone
        zone_data = df[df['Zone'] == selected_zone]
        avg_quality = int(zone_data['Road_Quality'].mean())

        st.sidebar.success(f"✅ Found data for {selected_zone}!")
        st.sidebar.metric("Historical Road Quality", f"{avg_quality}/10")

        quality_rating = avg_quality
        estimated_potholes = int((10 - quality_rating) * 2.2)
        st.sidebar.caption(f"Using historical avg: {estimated_potholes} potholes.")

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
tab1, tab2 = st.tabs(["Zone Quality Analysis", "Raw Data"])

with tab1:
    # Show which zones have the worst roads
    st.subheader("Average Road Quality by Zone")
    quality_chart = df.groupby("Zone")["Road_Quality"].mean().sort_values()
    st.bar_chart(quality_chart)
    st.caption("Lower score = More potholes.")

with tab2:
    st.dataframe(df)