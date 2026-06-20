import numpy as np

# ====================== PHYSICAL CONSTANTS ======================
G = 9.81                    # m/s² - Standard gravity
RHO_SEA_LEVEL = 1.225       # kg/m³ - Air density at sea level
SCALE_HEIGHT = 8500.0       # m - Approximate atmospheric scale height

# ====================== DATA PROCESSING ======================
DEFAULT_SMOOTH_WINDOW = 11
BOOST_ACC_THRESHOLD = 4.0   # m/s² - Threshold to detect boost phase
MIN_SPEED_FOR_DRAG = 5.0    # m/s - Minimum speed to calculate Cd (avoid division by zero)

# ====================== LIMITS & FILTERS ======================
MAX_ACCELERATION = 200.0    # m/s² (~20g) - Physical limit for model rockets
MAX_ALTITUDE = 10000.0      # m - Safety limit
MIN_FLIGHT_DURATION = 2.0   # seconds

# ====================== DEFAULT ROCKET PARAMETERS ======================
DEFAULT_DRY_MASS = 0.45     # kg
DEFAULT_FRONTAL_AREA = 0.005  # m² (approx 8cm diameter rocket)
DEFAULT_CD = 0.65

# ====================== VISUALIZATION ======================
PLOT_DPI = 150
PDF_REPORT_TITLE = "RocketAnalyzer Flight Report"
