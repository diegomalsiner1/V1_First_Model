import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def fetch_prices_from_csv(start_dt, end_dt, base_forecast, peak_forecast, normalisation="mean", years=[2019, 2020, 2021, 2022, 2023]):
    # Folder containing yearly price files
    prices_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Input Data Files', 'Full_ITA_NORD_19-24')
    historical_dfs = []
    for year in years:
        file_name = f"{year}0101_{year}1231_MGP_PrezziZonali_Nord.CSV"
        file_path = os.path.join(prices_folder, file_name)
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') + pd.to_timedelta(df['Hour'] - 1, unit='h')
        year_start = start_dt.replace(year=year)
        year_end = end_dt.replace(year=year)
        mask = (df['datetime'] >= year_start) & (df['datetime'] < year_end)
        historical_dfs.append(df.loc[mask])
    # Average across years
    df_hist = pd.concat(historical_dfs)
    # Assign hour of week (0-167) for each row
    df_hist = df_hist.copy()
    df_hist['hour_of_week'] = ((df_hist['datetime'].dt.dayofweek * 24) + (df_hist['datetime'].dt.hour))
    df_hist_avg = df_hist.groupby('hour_of_week')['Price (Euro/MWh)'].mean().reset_index()
    # Check length
    if len(df_hist_avg) != 168:
        raise ValueError(f"Expected 168 hourly prices for the week, got {len(df_hist_avg)}.")
    prices_hourly = pd.Series(df_hist_avg['Price (Euro/MWh)'].values)
    # Identify peak and base periods
    week_hours = np.arange(168)
    is_weekday = (week_hours // 24) < 5  # Monday-Friday
    is_peak_hour = (week_hours % 24 >= 8) & (week_hours % 24 < 20)  # 8:00 to 19:00
    is_peak = is_weekday & is_peak_hour
    is_base = ~is_peak
    # Compute mean or median price for each period
    if normalisation == "median":
        norm_peak_value = prices_hourly[is_peak].median()
        norm_base_value = prices_hourly[is_base].median()
    else:
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
    price_forecast = price_forecast / 1000.0
    sell_forecast = price_forecast - 0.01
    return price_forecast, sell_forecast

if __name__ == "__main__":
    # Example usage: load start/end from Constants_Plant.csv
    constants_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#', header=None, names=['Parameter', 'Value'])
    params = {str(row['Parameter']).strip(): row['Value'] for _, row in constants_data.iterrows()}
    start_dt = pd.to_datetime(str(params['PERIOD_START']), format='%Y%m%d%H%M')
    end_dt = pd.to_datetime(str(params['PERIOD_END']), format='%Y%m%d%H%M')
    base_forecast = 100.0
    peak_forecast = 120.0
    buy, sell = fetch_prices_from_csv(start_dt, end_dt, base_forecast, peak_forecast)
    print("HPFC Buy Price (Euro/kWh) for full week:")
    print(buy)
    print("HPFC Sell Price (Euro/kWh) for full week:")
    print(sell)
