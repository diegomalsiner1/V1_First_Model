import pandas as pd
import numpy as np
import os
from datetime import datetime

def get_month_from_constants():
    """Read PERIOD_START from Constants_Plant.csv and return the month as integer."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    constants_file = os.path.join(base_dir, 'Input Data Files', 'Constants_Plant.csv')
    constants = pd.read_csv(constants_file, comment='#')
    period_start = constants[constants['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    dt = datetime.strptime(str(period_start), "%Y%m%d%H%M")
    return dt.month

def fetch_prices_from_csv():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    price_file = os.path.join(base_dir, 'Input Data Files', 'Price_Matrix.csv')
    constants_file = os.path.join(base_dir, 'Input Data Files', 'Constants_Plant.csv')
    constants = pd.read_csv(constants_file, comment='#')
    period_start = constants[constants['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    period_end = constants[constants['Parameter'] == 'PERIOD_END']['Value'].iloc[0]
    start_dt = datetime.strptime(str(period_start), "%Y%m%d%H%M")
    end_dt = datetime.strptime(str(period_end), "%Y%m%d%H%M")
    days = (end_dt - start_dt).days
    month = get_month_from_constants()
    # Read raw file
    df = pd.read_csv(price_file, sep=',', encoding='utf-8')
    df.columns = df.columns.str.strip()
    # Map month number to row (case-insensitive, partial match)
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_row = None
    for idx, row in df.iterrows():
        for ref in [month_names[month-1][:3].lower(), month_names[month-1].lower()]:
            if str(row['Mese/Ora']).strip().lower().startswith(ref):
                month_row = row
                break
        if month_row is not None:
            break
    if month_row is None:
        raise ValueError(f"Could not find month row for month {month} in price matrix.")
    # Extract 24 hourly prices, convert to Euro/kWh
    hourly_prices = month_row.iloc[1:25].astype(float) / 1000.0
    hourly_prices = hourly_prices.values
    if len(hourly_prices) != 24:
        raise ValueError(f"Expected 24 hourly prices for month, got {len(hourly_prices)}")
    # Repeat for number of days (1 day for testing)
    hourly_prices_period = np.tile(hourly_prices, days)
    # Expand to 15-min intervals (96 values for 1 day)
    prices_15min = np.repeat(hourly_prices_period, 4)
    # For compatibility with API_prices, return hourly prices as pd.Series (length 24 for 1 day)
    buy_prices_15min = pd.Series(prices_15min)
    sell_prices_15min = pd.Series(prices_15min - 0.005)
    return buy_prices_15min, sell_prices_15min

if __name__ == "__main__":
    buy_prices_15min, sell_prices_15min = fetch_prices_from_csv()
    print("Buy prices (15min intervals):", buy_prices_15min.values)
    print("Sell prices (15min intervals):", sell_prices_15min.values)
    print(f"\nTotal 15min prices: {len(buy_prices_15min)}")