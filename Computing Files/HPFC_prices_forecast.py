import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def fetch_prices_from_csv(start_dt, end_dt, base_forecast, peak_forecast, normalisation="mean"):
    # Load hourly prices from CSV
    prices_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Input Data Files', '2024_full_prices_ITA.CSV')
    df = pd.read_csv(prices_path)
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') + pd.to_timedelta(df['Hour'] - 1, unit='h')
    # Filter for selected week
    mask = (df['datetime'] >= start_dt) & (df['datetime'] < end_dt)
    df_week = df.loc[mask].copy()
    # Check length
    if len(df_week) != 168:
        raise ValueError(f"Expected 168 hourly prices for the week, got {len(df_week)}.")
    # Use hourly prices directly (no resampling)
    prices_hourly = pd.Series(df_week['Price (Euro/MWh)'].values, index=df_week['datetime'])
    # Identify peak and base periods
    week_dates = prices_hourly.index
    is_peak = week_dates.weekday < 5  # Monday-Friday
    is_peak = is_peak & (week_dates.hour >= 8) & (week_dates.hour < 20)  # 8:00 to 19:00
    is_base = ~is_peak
    # Compute mean or median price for each period
    if normalisation == "median":
        norm_peak_value = prices_hourly[is_peak].median()
        norm_base_value = prices_hourly[is_base].median()
    else:  # default to mean
        norm_peak_value = prices_hourly[is_peak].mean()
        norm_base_value = prices_hourly[is_base].mean()
    # Normalize
    norm_peak = np.zeros(len(prices_hourly))
    norm_base = np.zeros(len(prices_hourly))
    norm_peak[is_peak] = prices_hourly[is_peak] / norm_peak_value if norm_peak_value > 0 else 1.0
    norm_base[is_base] = prices_hourly[is_base] / norm_base_value if norm_base_value > 0 else 1.0
    # Build final price forecast
    price_forecast = np.zeros(len(prices_hourly))
    price_forecast[is_peak] = norm_peak[is_peak] * peak_forecast
    price_forecast[is_base] = norm_base[is_base] * base_forecast
    # Convert Euro/MWh to Euro/kWh
    price_forecast = price_forecast / 1000.0
    # Sell price is buy price minus 0.01
    sell_forecast = price_forecast - 0.01
    return price_forecast, sell_forecast

if __name__ == "__main__":
    # Example usage: load start/end from Constants_Plant.csv
    constants_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#', header=None, names=['Parameter', 'Value'])
    params = {str(row['Parameter']).strip(): row['Value'] for _, row in constants_data.iterrows()}
    start_dt = pd.to_datetime(str(params['PERIOD_START']), format='%Y%m%d%H%M')
    end_dt = pd.to_datetime(str(params['PERIOD_END']), format='%Y%m%d%H%M')
    # Manually set base and peak forecast values (Euro/MWh)
    base_forecast = 100.0
    peak_forecast = 120.0
    buy, sell = fetch_prices_from_csv(start_dt, end_dt, base_forecast, peak_forecast)
    print("HPFC Buy Price (Euro/kWh) for full week:")
    print(buy)
    print("HPFC Sell Price (Euro/kWh) for full week:")
    print(sell)
