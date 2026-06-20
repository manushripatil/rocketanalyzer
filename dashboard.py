import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import StringIO

# Import our modules
from constants import G, DEFAULT_DRY_MASS, DEFAULT_FRONTAL_AREA, DEFAULT_CD
from rocket_config import RocketConfig, get_motor_profile
from physics import (
    compute_velocity, 
    estimate_cd, 
    detect_phases, 
    detect_events
)
from thrust import estimate_thrust
from simulator import simulate
from data_cleaning import clean_flight_data

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="RocketAnalyzer", layout="wide")
st.title("🚀 RocketAnalyzer")
st.markdown("**Real Flight Data → Physics Analysis → Simulation**")

# ====================== SIDEBAR ======================
st.sidebar.header("Rocket Configuration")

col1, col2 = st.sidebar.columns(2)
with col1:
    mass = st.slider("Dry Mass (kg)", 0.05, 5.0, DEFAULT_DRY_MASS, step=0.01)
    area = st.slider("Frontal Area (m²)", 0.001, 0.05, DEFAULT_FRONTAL_AREA, step=0.0001)

with col2:
    Cd_base = st.slider("Reference Cd", 0.1, 1.5, DEFAULT_CD, step=0.05)
    motor_type = st.selectbox("Motor Type", ["Low", "Medium", "High"])

# ====================== FILE UPLOAD ======================
uploaded_file = st.file_uploader(
    "Upload Flight CSV",
    type=["csv"],
    help="Required columns: time, acceleration, altitude"
)

if uploaded_file is None:
    st.info("👆 Upload your flight data CSV to begin analysis")
    st.stop()

# ====================== PROCESSING ======================
@st.cache_data(show_spinner=True)
def process_flight_data(uploaded_file, mass, area, Cd_base, motor_type):
    try:
        df_raw = pd.read_csv(uploaded_file)
        
        # Clean and validate data
        df = clean_flight_data(df_raw)
        
        time = df["time"].values
        acc = df["acceleration"].values
        alt = df["altitude"].values

        # Core Physics Calculations
        velocity = compute_velocity(time, acc)
        phases = detect_phases(time, velocity, acc)
        events = detect_events(time, alt, velocity)
        
        # Estimate Cd (now properly using phases)
        Cd = estimate_cd(time, velocity, acc, mass, area, phases=phases)

        # Thrust & Simulation
        thrust_model = get_motor_profile(motor_type, time)
        sim_v, sim_h, sim_x, sim_y = simulate(time, thrust_model, mass, Cd_base, area)
        thrust_est = estimate_thrust(acc, velocity, mass, Cd_base, area)

        # Add results to dataframe
        df["velocity"] = velocity
        df["Cd"] = Cd
        df["phase"] = phases
        df["sim_altitude"] = sim_h
        df["thrust_estimated"] = thrust_est
        df["thrust_model"] = thrust_model

        return df, events, thrust_model, sim_h, sim_x, sim_y, Cd

    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.stop()

# Run analysis
df, events, thrust_model, sim_h, sim_x, sim_y, Cd = process_flight_data(
    uploaded_file, mass, area, Cd_base, motor_type
)

# ====================== SUMMARY METRICS ======================
st.subheader("Flight Summary")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Apogee", f"{events.get('apogee_altitude', 0):.1f} m")
c2.metric("Max Velocity", f"{df['velocity'].max():.1f} m/s")
c3.metric("Avg Cd (Coast)", f"{np.nanmean(Cd):.3f}")
c4.metric("Flight Duration", f"{df['time'].iloc[-1]:.1f} s")
c5.metric("Sim RMSE", f"{np.sqrt(np.mean((df['altitude'] - sim_h)**2)):.2f} m")

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Trajectory", "📈 Physics Graphs", "🔥 Thrust & Cd", "📊 Raw Data"])

with tab1:
    st.subheader("3D Flight Trajectory")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=sim_x, y=sim_y, z=sim_h,
        mode='lines', name='Simulated', line=dict(color='blue', width=6)
    ))
    fig.add_trace(go.Scatter3d(
        x=sim_x, y=sim_y, z=df['altitude'],
        mode='lines', name='Measured', line=dict(color='red', width=4, dash='dash')
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Altitude (m)'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['time'], df['altitude'], label='Measured', linewidth=2.5)
        ax.plot(df['time'], sim_h, label='Simulated', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['time'], df['velocity'], color='green', linewidth=2.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.grid(True)
        st.pyplot(fig)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        thrust_df = pd.DataFrame({
            "Estimated Thrust (N)": df["thrust_estimated"],
            "Model Thrust (N)": df["thrust_model"]
        }, index=df['time'])
        st.line_chart(thrust_df)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['velocity'], df['Cd'], alpha=0.7, s=15)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Estimated Cd')
        ax.grid(True)
        st.pyplot(fig)

with tab4:
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Processed Flight Data",
        data=csv,
        file_name="rocket_flight_processed.csv",
        mime="text/csv"
    )

# ====================== SIDEBAR EVENTS ======================
st.sidebar.subheader("🚀 Key Events")
st.sidebar.write(f"**Launch:** {events.get('launch_time', 0):.2f} s")
st.sidebar.write(f"**Apogee:** {events.get('apogee_altitude', 0):.1f} m")
st.sidebar.write(f"**Landing:** {events.get('landing_time', 0):.2f} s")
st.sidebar.write(f"**Burn Time:** {events.get('burn_time', 0):.2f} s")
