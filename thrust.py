import numpy as np
from scipy.signal import savgol_filter

from constants import G, RHO_SEA_LEVEL, DEFAULT_SMOOTH_WINDOW


def safe_savgol_filter(data: np.ndarray, window: int = None, polyorder: int = 2) -> np.ndarray:
    """Safe Savitzky-Golay filter"""
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


def estimate_thrust(acc: np.ndarray, velocity: np.ndarray, mass: float,
                   Cd: float, area: float, altitude: np.ndarray = None) -> np.ndarray:
    """
    Estimate thrust from measured acceleration using Newton's Second Law.
    
    Thrust = m * (a + g) + Drag
    """
    if len(acc) < 3:
        acc_smooth = acc.copy()
    else:
        acc_smooth = safe_savgol_filter(acc, DEFAULT_SMOOTH_WINDOW, polyorder=2)

    thrust = np.zeros_like(acc_smooth)
    
    # Use altitude for air density if provided, otherwise sea level
    if altitude is not None and len(altitude) == len(acc):
        rho = RHO_SEA_LEVEL * np.exp(-altitude / 8500.0)
    else:
        rho = np.full_like(acc, RHO_SEA_LEVEL)

    for i in range(len(acc_smooth)):
        v = velocity[i]
        speed = abs(v)
        
        # Drag force (always opposes velocity)
        drag = 0.0
        if speed > 0.5:
            drag_mag = 0.5 * rho[i] * Cd * area * speed**2
            drag = np.sign(v) * drag_mag   # Positive when going up, negative when falling
        
        # Thrust estimation
        thrust[i] = mass * (acc_smooth[i] + G) + drag

    # Thrust should not be negative
    return np.maximum(thrust, 0.0)


def get_average_thrust(thrust: np.ndarray, time: np.ndarray) -> float:
    """Calculate average thrust during burn phase"""
    if len(thrust) < 5:
        return float(np.mean(thrust))
    
    # Simple burn detection: where thrust > 5 N
    burn_mask = thrust > 5.0
    if np.any(burn_mask):
        return float(np.mean(thrust[burn_mask]))
    return float(np.max(thrust))
