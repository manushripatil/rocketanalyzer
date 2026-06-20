import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from constants import DEFAULT_SMOOTH_WINDOW


def clean_acceleration(acc: np.ndarray) -> np.ndarray:
    """Clean accelerometer data: outlier removal + smoothing"""
    if len(acc) < 5:
        return acc.copy()

    s = pd.Series(acc)

    # Outlier removal using Z-score
    z = (s - s.mean()) / s.std()
    s[z.abs() > 3.5] = np.nan
    s = s.interpolate(method='linear').bfill().ffill()

    # Dynamic odd window size
    window = min(DEFAULT_SMOOTH_WINDOW, len(s))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        window = 3

    cleaned = savgol_filter(s.values, window_length=window, polyorder=2)
    return cleaned


def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main cleaning pipeline for rocket flight data"""
    if df.empty:
        raise ValueError("Uploaded CSV is empty")

    df = df.copy()

    required = ["time", "acceleration", "altitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Sort by time and reset index
    df = df.sort_values("time").reset_index(drop=True)

    # Basic validation
    if (df["time"].diff().dropna() < 0).any():
        st.warning("Time column is not strictly increasing. Data has been sorted.")

    # Clean acceleration
    df["acceleration"] = clean_acceleration(df["acceleration"].values)

    # Clean altitude with higher order (better for position data)
    if len(df) >= 7:
        alt_window = min(15, len(df) if len(df) % 2 == 1 else len(df) - 1)
        if alt_window < 3:
            alt_window = 3
        df["altitude"] = savgol_filter(
            df["altitude"].values,
            window_length=alt_window,
            polyorder=3
        )

    # Remove physically impossible values
    df = df[df["acceleration"].abs() < 200]   # ~20g max (adjustable for HPR)
    df = df[df["altitude"] >= 0]

    return df.reset_index(drop=True)


def detect_sampling_rate(time: np.ndarray) -> float:
    """Estimate sampling rate in Hz"""
    if len(time) < 2:
        return 0.0
    dt = np.mean(np.diff(time))
    return 1.0 / dt if dt > 0 else 0.0
