import requests
from xml.etree import ElementTree
import numpy as np
import warnings
from datetime import datetime, timedelta, timezone

def get_dayahead_prices(api_key: str, area_code: str, start: datetime = None, end: datetime = None):
    """
    Get day-ahead prices from ENTSO-E API.
    Adapted from https://gist.github.com/jpulakka/f866e37dcedeede31e96a34e9f06ed7a
    This function constructs the API URL, sends a GET request, parses the XML response,
    and extracts time-stamped prices into a dictionary.
    """
    # Handling default start time if not provided
    if not start:
        start = datetime.now(timezone.utc)  # Use current UTC time if no start is given
    # Converting start to UTC if it's in another timezone
    elif start.tzinfo and start.tzinfo != timezone.utc:
        start = start.astimezone(timezone.utc)  # Convert to UTC
    # Handling default end time (1 day after start)
    if not end:
        end = start + timedelta(days=1)  # Default to 1 day period
    # Converting end to UTC if needed
    elif end.tzinfo and end.tzinfo != timezone.utc:
        end = end.astimezone(timezone.utc)  # Convert to UTC
    fmt = '%Y%m%d%H00'  # Format for API date parameters (YYYYMMDDHH00)
    # Constructing the API URL with parameters for document type (A44 for day-ahead prices), domains, and period
    url = f'https://web-api.tp.entsoe.eu/api?securityToken={api_key}&documentType=A44&in_Domain={area_code}' \
          f'&out_Domain={area_code}&periodStart={start.strftime(fmt)}&periodEnd={end.strftime(fmt)}'
    response = requests.get(url)  # Sending GET request to the API
    response.raise_for_status()  # Raising an error if the request fails (e.g., HTTP error)
    xml_str = response.text  # Getting the XML response as a string
    result = {}  # Initializing an empty dictionary to store time-price pairs
    # Parsing the XML string into an ElementTree object
    for child in ElementTree.fromstring(xml_str):  # Iterating over root children
        if child.tag.endswith("TimeSeries"):  # Finding TimeSeries elements
            for ts_child in child:  # Iterating over TimeSeries children
                if ts_child.tag.endswith("Period"):  # Finding Period elements
                    for pe_child in ts_child:  # Iterating over Period children
                        if pe_child.tag.endswith("timeInterval"):  # Finding timeInterval
                            for ti_child in pe_child:  # Iterating over timeInterval children
                                if ti_child.tag.endswith("start"):  # Extracting start time
                                    # Parsing the start time string to datetime object
                                    start_time = datetime.strptime(ti_child.text, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc)
                        elif pe_child.tag.endswith("Point"):  # Finding Point elements (hourly data)
                            for po_child in pe_child:  # Iterating over Point children
                                if po_child.tag.endswith("position"):  # Getting position (hour offset)
                                    delta = int(po_child.text) - 1  # Position starts from 1, so subtract 1
                                    time = start_time + timedelta(hours=delta)  # Calculating timestamp
                                elif po_child.tag.endswith("price.amount"):  # Getting price
                                    price = float(po_child.text)  # Converting price to float
                                    result[time] = price  # Storing in result dictionary
    return result  # Returning the dictionary of timestamps to prices

def fetch_prices(api_key, area_code):
    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        prices_dict = get_dayahead_prices(api_key, area_code, start_date, end_date)
        sorted_times = sorted(prices_dict.keys())
        print(f"API debug: Fetched {len(sorted_times)} prices for zone {area_code}")
        if sorted_times:
            print(f"API debug: Date range: {sorted_times[0]} to {sorted_times[-1]}")
        
        filtered_times = [t for t in sorted_times if start_date <= t < end_date]
        print(f"API debug: Filtered to {len(filtered_times)} prices within {start_date} to {end_date}")
        
        current_times = filtered_times
        if len(current_times) != 168:
            warnings.warn(f"Warning: Non-standard data ({len(current_times)} prices). Adjusting to 168.")
            mean_price = np.mean([prices_dict[t] for t in current_times]) if current_times else 0.103 * 1000
            
            while len(current_times) < 168:
                last_t = current_times[-1] if current_times else start_date
                next_t = last_t + timedelta(hours=1)
                prices_dict[next_t] = mean_price
                current_times.append(next_t)
            
            if len(current_times) > 168:
                current_times = current_times[:168]
        
        sorted_times = sorted(current_times)
    except Exception as e:
        print(f"API error: {e}. Falling back to synthetic data.")
        grid_price_hourly = np.full(168, 0.103)
        sorted_times = [start_date + timedelta(hours=i) for i in range(168)]
    else:
        grid_price_hourly = np.array([prices_dict[t] for t in sorted_times]) / 1000

    time_hourly = np.arange(0, 168, 1)
    time_quarter = np.arange(0, 168, 0.25)
    grid_price = np.interp(time_quarter, time_hourly, grid_price_hourly)
    grid_buy_price = grid_price + 0.01
    grid_sell_price = grid_price - 0.01
    
    return grid_buy_price, grid_sell_price