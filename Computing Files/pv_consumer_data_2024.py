import pandas as pd
import os

def compute_pv_power(week_number):
    # Use relative path for portability; assume CSV in data/ or adjust as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, '..', 'data', 'pv_FdM_2024.csv')  # Adjust if structure changes
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S')
    df.set_index('Time', inplace=True)
    cols = ['InverterYield', 'GridFeedIn', 'PurchasedFromNet']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # Handle missing values
    # Resample to 15min intervals with linear interpolation; this effectively distributes hourly energy into 15min kWh
    df_15min = df[cols].resample('15min').interpolate(method='linear')
    # Calculate consumer demand as energy (kWh per 15min)
    df_15min['consumer_demand'] = df_15min['InverterYield'] + df_15min['PurchasedFromNet'] - df_15min['GridFeedIn']
    # Slice for the specified week
    start_time = df_15min.index[0] + pd.Timedelta(days=(week_number - 1) * 7)
    end_time = start_time + pd.Timedelta(hours=167, minutes=45)  # Exactly 7 days at 15min resolution (672 steps)
    df_week = df_15min.loc[start_time:end_time]
    # Ensure exactly 672 entries
    if len(df_week) != 672:
        raise ValueError(f"Data slice for week {week_number} has {len(df_week)} entries, expected 672.")
    return {
        'pv_production': df_week['InverterYield'].values,  # kWh per 15min
        'consumer_demand': df_week['consumer_demand'].values  # kWh per 15min
    }