import pandas as pd
import os

def compute_pv_power(start_time, end_time):
    """
    Load and process PV and consumer demand data for the specified period.

    Args:
        start_time (pd.Timestamp): Start datetime (inclusive).
        end_time (pd.Timestamp): End datetime (exclusive).

    Returns:
        dict: {'pv_production': np.ndarray, 'consumer_demand': np.ndarray}
    """
    # Cross-platform, dynamic path to data file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'Input Data Files', 'pv_FdM_2024.csv')

    df = pd.read_csv(DATA_PATH)
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S')
    df.set_index('Time', inplace=True)

    cols = ['InverterYield', 'GridFeedIn', 'PurchasedFromNet']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Resample to 15min intervals, forward-fill, and scale energy to 15min chunks
    df_15min = df[cols].resample('15min').ffill()
    df_15min = df_15min  # Convert hourly kWh to 15-min kWh

    # Calculate consumer demand
    df_15min['consumer_demand'] = (
        df_15min['InverterYield'] +
        df_15min['PurchasedFromNet'] -
        df_15min['GridFeedIn']
    )

    # Slice for exactly 7 days (7 x 24 x 4 = 672 steps)
    df_week = df_15min.loc[start_time:end_time - pd.Timedelta(minutes=15)]

    expected_steps = 7 * 24 * 4  # 672 for 7 days of 15-min intervals
    if len(df_week) != expected_steps:
        raise ValueError(f"Data slice from {start_time} to {end_time} has {len(df_week)} entries, expected {expected_steps}.")

    pv_production = df_week['InverterYield'].values
    consumer_demand = df_week['consumer_demand'].values

    return {
        'pv_production': pv_production,
        'consumer_demand': consumer_demand
    }
