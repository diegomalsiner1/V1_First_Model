import requests
import pandas as pd
import xml.etree.ElementTree as ET
import os

def load_constants():
    """Load plant constants from CSV file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(csv_path, comment='#')
    return constants_data

def parse_entsoe_response(xml_data):
    """Parse ENTSO-E XML response and return price list in EUR/kWh."""
    root = ET.fromstring(xml_data)
    prices = []
    for timeseries in root.findall('.//{*}TimeSeries'):
        for period in timeseries.findall('.//{*}Period'):
            for point in period.findall('.//{*}Point'):
                price = float(point.find('./{*}price.amount').text)
                prices.append(price / 1000.0)  # ENTSOE gives in EUR/MWh, convert to EUR/kWh
    return prices

def get_price_document(api_key, domain, start_str, end_str, document_type):
    """Request ENTSO-E API for a specific document type and return parsed prices."""
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

def fetch_prices(period_start=None, period_end=None, bidding_zone=None, api_key=None):
    """
    Fetch hourly buy and sell prices from ENTSO-E API for the given period and zone.
    Args:
        period_start (str): YYYYMMDDHHMM, overrides constants if provided.
        period_end (str): YYYYMMDDHHMM, overrides constants if provided.
        bidding_zone (str): ENTSO-E bidding zone, overrides constants if provided.
        api_key (str): ENTSO-E API key, overrides constants if provided.
    Returns:
        (pd.Series, pd.Series): buy_prices, sell_prices (length 168)
    """
    constants_data = load_constants()
    if api_key is None:
        api_key = constants_data[constants_data['Parameter'] == 'ENTSOE_TOKEN']['Value'].iloc[0]
    if bidding_zone is None:
        bidding_zone = constants_data[constants_data['Parameter'] == 'BIDDING_ZONE']['Value'].iloc[0]
    if period_start is None:
        period_start = constants_data[constants_data['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    if period_end is None:
        period_end = constants_data[constants_data['Parameter'] == 'PERIOD_END']['Value'].iloc[0]
    domain = bidding_zone
    buy_prices_raw = get_price_document(api_key, domain, period_start, period_end, "A44")  # Day-ahead prices
    # Adapt sell prices by subtracting 0.01 from buy prices (as per instructions)
    sell_prices_raw = [max(p - 0.005, 0) for p in buy_prices_raw]  # Ensure non-negative
    # Clip to exactly 168 hours (7 days)
    buy_prices = buy_prices_raw[:168]
    sell_prices = sell_prices_raw[:168]
    return pd.Series(buy_prices), pd.Series(sell_prices)