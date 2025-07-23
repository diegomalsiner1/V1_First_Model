import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def fetch_prices_from_csv(start_dt, end_dt):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    price_file = os.path.join(base_dir, 'Input Data Files', 'Price_Matrix_ITA.CSV')

    # Read raw file
    df = pd.read_csv(price_file, sep=';', encoding='latin1')
    df.columns = df.columns.str.strip()

    # Parse date column and clean hours
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Melt to long format: 1 row per hour
    df_long = df.melt(id_vars=['Date'], var_name='Hour', value_name='Price')
    df_long['Hour'] = pd.to_numeric(df_long['Hour'], errors='coerce')
    df_long = df_long.dropna(subset=['Hour'])

    # Construct full datetime
    df_long['Datetime'] = df_long['Date'] + pd.to_timedelta(df_long['Hour'], unit='h')
    df_long = df_long.sort_values('Datetime')

    # Filter by period
    mask = (df_long['Datetime'] >= start_dt) & (df_long['Datetime'] < end_dt)
    hourly_prices = df_long.loc[mask, 'Price'].values

    if len(hourly_prices) != 168:
        raise ValueError(f"Expected 168 hourly prices but got {len(hourly_prices)}")

    # Expand to 15-min intervals
    prices_15min = np.repeat(hourly_prices, 4)
    return pd.Series(prices_15min, index=pd.date_range(start=start_dt, periods=672, freq='15min'))

if __name__ == "__main__":
    start = datetime.strptime("202405060000", "%Y%m%d%H%M")
    end = datetime.strptime("202405130000", "%Y%m%d%H%M")
    prices = fetch_prices_from_csv(start, end)
    print(prices.head(20))
    print(f"\nTotal 15-min prices: {len(prices)}")
