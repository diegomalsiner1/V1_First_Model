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
    # Repeat for 7 days (168 values)
    hourly_prices_7d = np.tile(hourly_prices, 7)
    # Expand to 15-min intervals (672 values)
    prices_15min = np.repeat(hourly_prices_7d, 4)
    # For compatibility with API_prices, return hourly prices as pd.Series (length 168)
    buy_prices = pd.Series(hourly_prices_7d)
    # Sell price: subtract margin (e.g., 0.005) and ensure non-negative
    sell_prices = pd.Series(np.maximum(buy_prices - 0.005, 0))
    return buy_prices, sell_prices

if __name__ == "__main__":
    buy, sell = fetch_prices_from_csv()
    print("Buy prices (hourly, 7 days):", buy.values)
    print("Sell prices (hourly, 7 days):", sell.values)
    print(f"\nTotal hourly prices: {len(buy)}")
