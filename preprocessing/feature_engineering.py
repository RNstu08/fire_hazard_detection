import pandas as pd

def add_rolling_features(df, window_size=5):
    """
    Adds rolling mean and rolling std features to sensor data.
    """
    sensor_cols = ['temperature_C', 'CO_ppm', 'HCN_ppm', 'heat_release_kw', 'smoke_density']
    
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    return df

def add_derivative_features(df):
    """
    Adds first derivative (rate of change) features to sensor data.
    """
    sensor_cols = ['temperature_C', 'CO_ppm', 'HCN_ppm', 'heat_release_kw', 'smoke_density']
    
    for col in sensor_cols:
        df[f'{col}_derivative'] = df[col].diff().fillna(0)
    
    return df

def compute_fire_risk_score(df):
    """
    Compute a simple Fire Risk Score based on weighted sum of sensors.
    """
    df['fire_risk_score'] = (
        0.4 * df['temperature_C_rolling_mean'] +
        0.3 * df['CO_ppm_rolling_mean'] +
        0.2 * df['HCN_ppm_rolling_mean'] +
        0.1 * df['smoke_density_rolling_mean']
    )
    return df
