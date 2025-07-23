import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
import os

def load_constants():
    # Use relative path for portability; assume CSV in Input Data Files/ or adjust as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')  # Adjust if structure changes
    constants_data = pd.read_csv(csv_path, comment='#')
    return constants_data

def fetch_prices():
    constants_data = load_constants()
    api_key = constants_data[constants_data['Parameter'] == 'ENTSOE_TOKEN']['Value'].iloc[0]
    bidding_zone = constants_data[constants_data['Parameter'] == 'BIDDING_ZONE']['Value'].iloc[0]
    start_str = constants_data[constants_data['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    end_str = constants_data[constants_data['Parameter'] == 'PERIOD_END']['Value'].iloc[0]
    timezone_offset = int(constants_data[constants_data['Parameter'] == 'TIMEZONE_OFFSET']['Value'].iloc[0])

    domain = bidding_zone

    def parse_entsoe_response(xml_data):
        root = ET.fromstring(xml_data)
        prices = []
        for timeseries in root.findall('.//{*}TimeSeries'):
            for period in timeseries.findall('.//{*}Period'):
                for point in period.findall('.//{*}Point'):
                    price = float(point.find('./{*}price.amount').text)
                    prices.append(price / 1000.0)  # ENTSOE gives in EUR/MWh, convert to EUR/kWh
        return prices

    def get_price_document(document_type):
        url = (
            f"https://web-api.tp.entsoe.eu/api?securityToken={api_key}"
            f"&documentType={document_type}"
            f"&in_Domain={domain}&out_Domain={domain}"
            f"&periodStart={start_str}&periodEnd={end_str}"
        )
        try:
            response = requests.get(url, timeout=10)  # Add timeout for reliability
            response.raise_for_status()  # Raise on non-200 status
        except requests.exceptions.RequestException as e:
            raise ValueError(f"ENTSOE API request failed: {e}")
        return parse_entsoe_response(response.content)

    buy_prices_raw = get_price_document("A44")  # Day-ahead prices

    # Adapt sell prices by subtracting 0.01 from buy prices (as per instructions)
    sell_prices_raw = [max(0, p - 0.005) for p in buy_prices_raw]  # Ensure non-negative, optional

    # Clip to exactly 168 hours (7 days)
    buy_prices = buy_prices_raw[:168]
    sell_prices = sell_prices_raw[:168]

    # Optional: Apply timezone offset if needed (e.g., shift periods), but ENTSO-E uses UTC, so may not be necessary
    # For now, assume periods are in UTC as per API

    return pd.Series(buy_prices), pd.Series(sell_prices)