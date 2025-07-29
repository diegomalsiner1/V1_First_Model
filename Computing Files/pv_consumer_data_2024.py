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
    df_15min = df_15min * 0.25  # Convert hourly kWh to 15-min kWh

    # Calculate consumer demand
    df_15min['consumer_demand'] = (
        df_15min['InverterYield'] +
        df_15min['PurchasedFromNet'] -
        df_15min['GridFeedIn']
    )

    # Slice for exactly 1 day (24h x 4 = 96 steps)
    df_day = df_15min.loc[start_time:end_time - pd.Timedelta(minutes=15)]

    if len(df_day) != 96:
        raise ValueError(f"Data slice from {start_time} to {end_time} has {len(df_day)} entries, expected 96.")

    pv_production = df_day['InverterYield'].values
    consumer_demand = df_day['consumer_demand'].values

    return {
        'pv_production': pv_production,
        'consumer_demand': consumer_demand
    }
