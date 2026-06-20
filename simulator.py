import numpy as np
from constants import G, RHO_SEA_LEVEL, SCALE_HEIGHT


def get_air_density(altitude: np.ndarray) -> np.ndarray:
    """Exponential atmosphere model"""
    return RHO_SEA_LEVEL * np.exp(-altitude / SCALE_HEIGHT)


def simulate(time: np.ndarray, thrust: np.ndarray, mass: float, Cd: float, area: float):
    """
    Simple 2D rocket flight simulator (vertical + wind drift)
    Uses improved Euler integration.
    """
    n = len(time)
    dt = np.diff(time, prepend=0.0)   # More stable than gradient
    
    v = np.zeros(n)      # vertical velocity (m/s)
    h = np.zeros(n)      # altitude (m)
    x = np.zeros(n)      # horizontal position (m)
    y = np.zeros(n)      # lateral position (m)
    
    # Light wind (can be made configurable later)
    wind_x = 1.8   # m/s
    wind_y = 0.6   # m/s
    
    current_mass = mass
    
    for i in range(1, n):
        rho = get_air_density(h[i-1])
        vel = v[i-1]
        speed = abs(vel)
        
        # Drag force (opposes motion)
        drag = 0.0
        if speed > 0.5:
            drag_mag = 0.5 * rho * Cd * area * speed**2
            drag = -np.sign(vel) * drag_mag
        
        # Net force and acceleration (vertical)
        net_force = thrust[i] + drag - current_mass * G
        a = net_force / current_mass
        
        # Improved Euler integration
        v[i] = v[i-1] + a * dt[i]
        h[i] = h[i-1] + (v[i-1] + v[i]) * 0.5 * dt[i]   # Trapezoidal rule
        
        # Simple wind drift (stronger at higher altitude)
        wind_factor = 0.3 + 0.7 * (h[i-1] / 500.0)
        x[i] = x[i-1] + wind_x * wind_factor * dt[i]
        y[i] = y[i-1] + wind_y * wind_factor * dt[i]
        
        # Very simple mass reduction during thrust (propellant burn)
        if thrust[i] > 2.0:
            current_mass = max(mass * 0.92, current_mass - 0.015 * dt[i])
    
    return v, h, x, y
