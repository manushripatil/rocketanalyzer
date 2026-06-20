import numpy as np
from dataclasses import dataclass
from constants import G, DEFAULT_DRY_MASS, DEFAULT_FRONTAL_AREA, DEFAULT_CD


@dataclass
class RocketConfig:
    """Rocket configuration parameters"""
    mass: float           # Dry mass (kg)
    area: float           # Frontal area (m²)
    Cd: float             # Drag coefficient
    motor_type: str
    propellant_mass: float = 0.0
    motor_burn_time: float = 0.0
    total_impulse: float = 0.0


def get_motor_profile(motor_type: str, time: np.ndarray) -> np.ndarray:
    """
    Generate realistic thrust curve for different motor classes.
    Uses a simple but smoother approximation of model rocket thrust.
    """
    thrust = np.zeros_like(time, dtype=float)
    
    # Motor parameters by class
    motor_params = {
        "Low":    {"burn_time": 1.4,  "peak_thrust": 12.0, "total_impulse": 12.0},
        "Medium": {"burn_time": 1.8,  "peak_thrust": 22.0, "total_impulse": 25.0},
        "High":   {"burn_time": 2.3,  "peak_thrust": 35.0, "total_impulse": 45.0},
    }
    
    params = motor_params.get(motor_type, motor_params["Medium"])
    burn_time = params["burn_time"]
    
    for i, t in enumerate(time):
        if t > burn_time:
            break
            
        # Realistic thrust curve: high initial peak, then steady burn, then tail-off
        if t < 0.15:  # Initial ignition spike
            factor = 1.0 + 0.6 * (1 - t / 0.15)
        elif t < burn_time * 0.75:  # Main burn
            factor = 0.95 - 0.25 * (t / burn_time)
        else:  # Tail-off
            factor = 0.7 * (1 - (t - burn_time * 0.75) / (burn_time * 0.25))
        
        thrust[i] = params["peak_thrust"] * max(0.0, factor)
    
    return thrust


def create_default_rocket(motor_type: str = "Medium") -> RocketConfig:
    """Helper to create a default rocket configuration"""
    return RocketConfig(
        mass=DEFAULT_DRY_MASS,
        area=DEFAULT_FRONTAL_AREA,
        Cd=DEFAULT_CD,
        motor_type=motor_type,
        propellant_mass=0.08 if motor_type == "High" else 0.05 if motor_type == "Medium" else 0.03
    )
