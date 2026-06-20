import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from constants import (
    G, RHO_SEA_LEVEL, SCALE_HEIGHT,
    DEFAULT_SMOOTH_WINDOW, BOOST_ACC_THRESHOLD,
    MIN_SPEED_FOR_DRAG, MAX_ACCELERATION
)


def safe_savgol_filter(data: np.ndarray, window: int = None, polyorder: int = 2) -> np.ndarray:
    """Safe Savitzky-Golay filter that handles small datasets"""
    if window is None:
        window = DEFAULT_SMOOTH_WINDOW
    
    n = len(data)
    if n < 3:
        return data.copy()
    
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window -= 1
    
    return savgol_filter(data, window_length=window, polyorder=polyorder)


def compute_velocity(time: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Integrate acceleration to get velocity (with smoothing)"""
    if len(acc) < 3:
        return cumulative_trapezoid(acc, time, initial=0.0)
    
    acc_smooth = safe_savgol_filter(acc, DEFAULT_SMOOTH_WINDOW, polyorder=2)
    velocity = cumulative_trapezoid(acc_smooth, time, initial=0.0)
    
    return velocity


def calculate_air_density(altitude: np.ndarray) -> np.ndarray:
    """Exponential atmosphere model"""
    return RHO_SEA_LEVEL * np.exp(-altitude / SCALE_HEIGHT)


def estimate_cd(time: np.ndarray, velocity: np.ndarray, acc: np.ndarray,
                mass: float, area: float, phases: np.ndarray = None) -> np.ndarray:
    """
    Estimate Drag Coefficient (Cd) during coast phase only.
    """
    Cd = np.full_like(velocity, np.nan, dtype=float)
    
    # Use real altitude for air density (currently using 0 as placeholder)
    rho = calculate_air_density(np.zeros_like(velocity))
    
    # Determine coast phase mask
    if phases is not None:
        coast_mask = (phases == "coast")
    else:
        coast_mask = (velocity > MIN_SPEED_FOR_DRAG) & (acc < BOOST_ACC_THRESHOLD)
    
    for i in np.where(coast_mask)[0]:
        v = velocity[i]
        if v > MIN_SPEED_FOR_DRAG:
            denom = 0.5 * rho[i] * area * v**2
            if denom > 1e-6:
                Cd[i] = 2 * mass * (acc[i] + G) / denom   # Note: acc is negative in coast
    
    return np.clip(Cd, 0.1, 1.8)


def detect_phases(time: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """Detect rocket flight phases: boost, coast, descent"""
    phases = np.full(len(velocity), "descent", dtype="<U10")
    
    phases[acceleration > BOOST_ACC_THRESHOLD] = "boost"
    phases[(velocity > 3.0) & (acceleration <= BOOST_ACC_THRESHOLD) & (velocity > 0)] = "coast"
    phases[velocity < -2.0] = "descent"
    
    return phases


def detect_events(time: np.ndarray, altitude: np.ndarray, velocity: np.ndarray) -> dict:
    """Detect key flight events - FIXED VERSION"""
    if len(velocity) < 5:
        return {
            "launch_time": 0.0,
            "apogee_time": 0.0,
            "apogee_altitude": float(altitude.max()),
            "landing_time": float(time[-1]),
            "max_velocity": float(np.max(velocity)),
            "burn_time": 0.0
        }
    
    # Launch detection
    launch_indices = np.where(velocity > 2.0)[0]
    launch_time = float(time[launch_indices[0]]) if len(launch_indices) > 0 else 0.0
    
    # Apogee
    apogee_idx = int(np.argmax(altitude))
    apogee_time = float(time[apogee_idx])
    apogee_altitude = float(altitude[apogee_idx])
    
    # Landing detection
    ground_level = altitude[0] + 15.0
    landing_indices = np.where(altitude < ground_level)[0]
    landing_time = float(time[landing_indices[-1]]) if len(landing_indices) > 0 else float(time[-1])
    
    # Burn time estimation (Fixed - no longer uses undefined 'acceleration')
    burn_time = 1.8  # default fallback
    if len(launch_indices) > 3:
        # Look for velocity peak after launch
        post_launch_vel = velocity[launch_indices[0]:]
        peak_idx = np.argmax(post_launch_vel)
        burn_time = float(time[launch_indices[0] + peak_idx] - time[launch_indices[0]])
    
    return {
        "launch_time": launch_time,
        "apogee_time": apogee_time,
        "apogee_altitude": apogee_altitude,
        "landing_time": landing_time,
        "max_velocity": float(np.max(velocity)),
        "burn_time": burn_time
    }
