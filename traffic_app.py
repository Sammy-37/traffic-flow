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

st.sidebar.markdown("---")
st.sidebar.subheader("Route Familiarity")

# Check if user knows the route
knows_route = st.sidebar.checkbox("I have traveled this road before", value=True)

if knows_route:
    # === MANUAL INPUT MODE ===
    st.sidebar.markdown("### Manual Input")

    # 1. Signals Input
    signals = st.sidebar.slider("Number of Signals", 0, 50, 14,
                                help="Count of traffic signals on your route.")

    # 2. Road Quality Input
    quality_rating = st.sidebar.slider("Road Quality (1-10)", 1, 10, 5,
                                       help="1 = Many Potholes, 10 = Smooth")

    # Convert Quality to Potholes
    estimated_potholes = int((10 - quality_rating) * 2.2)
    st.sidebar.caption(f"Manual Estimate: ~{estimated_potholes} potholes.")

else:
    # === AUTO / ZONE LOOKUP MODE ===
    st.sidebar.markdown("### Zone Lookup")

    # Select Zone
    available_zones = sorted(df['Zone'].unique().tolist())
    available_zones.append("Other / Unknown Route")
    selected_zone = st.sidebar.selectbox("Select Zone / Area", available_zones)

    if selected_zone == "Other / Unknown Route":
        # Fallback if no data
        st.sidebar.warning("⚠️ No data. Using city-wide averages.")
        signals = int(df['Signals'].mean())
        avg_quality = 5
        estimated_potholes = int(df['Potholes'].mean())

        st.sidebar.metric("Avg Traffic Signals", f"{signals}")
        st.sidebar.metric("Avg Road Quality", "5/10")

    else:
        # Calculate Averages for the specific zone
        zone_data = df[df['Zone'] == selected_zone]

        # 1. Signals Average
        avg_signals = int(zone_data['Signals'].mean())
        signals = avg_signals

        # 2. Road Quality Average
        avg_quality = int(zone_data['Road_Quality'].mean())
        quality_rating = avg_quality
        estimated_potholes = int((10 - quality_rating) * 2.2)

        st.sidebar.success(f"✅ Loaded data for {selected_zone}")

        # Display the metrics so user knows what's being used
        col_s, col_q = st.sidebar.columns(2)
        with col_s:
            st.metric("Avg Signals", f"{signals}")
        with col_q:
            st.metric("Avg Quality", f"{quality_rating}/10")

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