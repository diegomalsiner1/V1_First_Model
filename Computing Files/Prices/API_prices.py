import requests
import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np

# Loads plant constants from CSV file (used for API key, bidding zone, period)
def load_constants():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(csv_path, comment='#')
    return constants_data

# Parses ENTSO-E XML response and returns price list in EUR/kWh
def parse_entsoe_response(xml_data):
    root = ET.fromstring(xml_data)
    prices = []
    for timeseries in root.findall('.//{*}TimeSeries'):
        for period in timeseries.findall('.//{*}Period'):
            for point in period.findall('.//{*}Point'):
                price = float(point.find('./{*}price.amount').text)
                prices.append(price / 1000.0)  # ENTSOE gives in EUR/MWh, convert to EUR/kWh
    return prices

# Requests ENTSO-E API for a specific document type and returns parsed prices
def get_price_document(api_key, domain, start_str, end_str, document_type):
    url = (
        f"https://web-api.tp.entsoe.eu/api?securityToken={api_key}"
        f"&documentType={document_type}"
        f"&in_Domain={domain}&out_Domain={domain}"
        f"&periodStart={start_str}&periodEnd={end_str}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"ENTSOE API request failed: {e}")
    return parse_entsoe_response(response.content)

# Fetches buy and sell prices from ENTSO-E API for the given period and zone
def fetch_prices(period_start=None, period_end=None, bidding_zone=None, api_key=None):
    """
    Fetch buy and sell prices from ENTSO-E API for the given period and zone.
    Returns:
        (pd.Series, pd.Series): buy_prices_15min, sell_prices_15min (length 672 for 7 days of 15-min intervals)
    """
    constants_data = load_constants()
    # Get API key, bidding zone, and period from constants if not provided
    if api_key is None:
        api_key = constants_data[constants_data['Parameter'] == 'ENTSOE_TOKEN']['Value'].iloc[0]
    if bidding_zone is None:
        bidding_zone = constants_data[constants_data['Parameter'] == 'BIDDING_ZONE']['Value'].iloc[0]
    if period_start is None:
        period_start = constants_data[constants_data['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    if period_end is None:
        period_end = constants_data[constants_data['Parameter'] == 'PERIOD_END']['Value'].iloc[0]
    domain = bidding_zone
    # Fetch day-ahead prices from ENTSO-E API
    buy_prices_raw = get_price_document(api_key, domain, period_start, period_end, "A44")
    # Calculate sell prices (subtract 0.005 EUR/kWh)
    sell_prices_raw = [(p - 0.005) for p in buy_prices_raw]
    # Clip to exactly 168 hours (7 days)
    buy_prices = buy_prices_raw[:168]
    sell_prices = sell_prices_raw[:168]
    # Expand to 15-min intervals (672 values for 7 days)
    buy_prices_15min = pd.Series(np.repeat(buy_prices, 4))
    sell_prices_15min = pd.Series(np.repeat(sell_prices, 4))
    return buy_prices_15min, sell_prices_15min

# Debug/test block: fetch and print 15-min prices if run as main script
if __name__ == "__main__":
    buy_prices_15min, sell_prices_15min = fetch_prices()
    print("Buy prices (15min intervals):", buy_prices_15min.values)
    print("Sell prices (15min intervals):", sell_prices_15min.values)
    print(f"\nTotal 15min prices: {len(buy_prices_15min)}")