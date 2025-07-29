import numpy as np
import pandas as pd
import os
import API_prices
import Prices_ITA
import pv_consumer_data_2024 as pv_data
import ev_power_profile
from datetime import datetime, timedelta

def load_constants(constants_path=None):
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#', header=None, names=['Parameter', 'Value'])
    params = {}
    for key in ['BESS_Capacity', 'BESS_Power_Limit', 'BESS_Efficiency_Charge', 'BESS_Efficiency_Discharge', 'SOC_Initial', 'BESS_limit']:
        params[key] = float(constants_data[constants_data['Parameter'] == key]['Value'].iloc[0])
    return params

#def load_constants():
#    """Load plant constants from CSV file."""
#    script_dir = os.path.dirname(os.path.abspath(__file__))
#    csv_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
#    constants_data = pd.read_csv(csv_path, comment='#')
#    return constants_data

def build_time_vector(start_dt, end_dt, delta_t):
    """Build time_steps, n_steps, and time_indices based on start/end and delta_t."""
    n_intervals = int(((end_dt - start_dt).total_seconds() / 3600) / delta_t)
    time_steps = np.arange(0, n_intervals * delta_t, delta_t)
    n_steps = len(time_steps)
    time_indices = range(n_steps)
    return time_steps, n_steps, time_indices

def sanity_check(data):
    """Check that all main arrays have the correct length."""
    keys = ['pv_power', 'consumer_demand', 'grid_buy_price', 'grid_sell_price']
    for k in keys:
        if len(data[k]) != data['n_steps']:
            raise ValueError(f"Length mismatch: {k} has {len(data[k])}, expected {data['n_steps']}.")
    return True

def fetch_prices(start_dt, end_dt, use_api=True):
    """Fetch grid prices from API or local CSV, return arrays of hourly prices."""
    if use_api:
        grid_buy_price_raw, grid_sell_price_raw = API_prices.fetch_prices()
    else:
        grid_buy_price_raw = Prices_ITA.fetch_prices_from_csv(start_dt, end_dt)
        grid_sell_price_raw = grid_buy_price_raw - 0.01  # Assumed margin
    if len(grid_buy_price_raw) != 168:
        raise ValueError(f"Expected 168 hourly prices, got {len(grid_buy_price_raw)}.")
    return grid_buy_price_raw, grid_sell_price_raw

def load(reference_case=False, use_api=True):
    """
    Load all input data for the simulation.
    Args:
        reference_case (bool): If True, do not scale PV for new installations.
        use_api (bool): If True, fetch prices from API, else from local CSV.
    Returns:
        dict: All simulation data.
    """
    constants_data = load_constants()
    period_start_str = constants_data[constants_data['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    period_end_str = constants_data[constants_data['Parameter'] == 'PERIOD_END']['Value'].iloc[0]
    start_dt = pd.to_datetime(period_start_str, format='%Y%m%d%H%M')
    end_dt = pd.to_datetime(period_end_str, format='%Y%m%d%H%M')
    delta_t = 0.25
    # Validate period
    if (end_dt - start_dt) != timedelta(days=7):
        raise ValueError("PERIOD_START to PERIOD_END must span exactly 7 days.")
    # Build time vector dynamically
    time_steps, n_steps, time_indices = build_time_vector(start_dt, end_dt, delta_t)
    # Extract constants
    timezone_offset = int(constants_data[constants_data['Parameter'] == 'TIMEZONE_OFFSET']['Value'].iloc[0])
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])
    def get_param(param_name, default=None):
        try:
            return float(constants_data[constants_data['Parameter'] == param_name]['Value'].iloc[0])
        except (IndexError, KeyError):
            if default is None:
                raise ValueError(f"Required parameter {param_name} not found in constants file")
            return default

    pi_consumer = get_param('Consumer_Price')
    # Get EV price
    pi_ev = get_param('EV_PRICE', 0.6)  # Default to 0.6 EUR/kWh if not specified
    lcoe_pv = get_param('LCOE_PV')
    lcoe_bess = float(constants_data[constants_data['Parameter'] == 'LCOE_BESS']['Value'].iloc[0])
    pv_old = float(constants_data[constants_data['Parameter'] == 'PV_OLD']['Value'].iloc[0])
    pv_new = float(constants_data[constants_data['Parameter'] == 'PV_NEW']['Value'].iloc[0])
    bidding_zone = constants_data[constants_data['Parameter'] == 'BIDDING_ZONE']['Value'].iloc[0]
    
    # Fetch prices
    # To switch price source, comment/uncomment the following lines:
    # --- Use API prices ---
    # grid_buy_price_raw, grid_sell_price_raw = fetch_prices(start_dt, end_dt, use_api=True)
    # --- Use ITA price matrix ---
    grid_buy_price_raw, grid_sell_price_raw = Prices_ITA.fetch_prices_from_csv()
    # --- End switch ---
    grid_buy_price = np.repeat(grid_buy_price_raw.values, int(1/delta_t))
    grid_sell_price = np.repeat(grid_sell_price_raw.values, int(1/delta_t))
    
    
    # Load PV and demand
    result = pv_data.compute_pv_power(start_dt, end_dt)
    if reference_case:
        pv_power = result['pv_production'] * (1 / delta_t)  # Reference: no scaling
    else:
        pv_power = ((pv_new + pv_old)/pv_old) * result['pv_production'] * (1 / delta_t)
    consumer_demand = result['consumer_demand'] * (1 / delta_t)
    # Load EV charging profile
    ev_sessions_per_day = int(constants_data[constants_data['Parameter'] == 'EV_NUM_SESSIONS_PER_DAY']['Value'].iloc[0])
    ev_session_energy = float(constants_data[constants_data['Parameter'] == 'EV_SESSION_ENERGY']['Value'].iloc[0])
    ev_load_scale = float(constants_data[constants_data['Parameter'] == 'EV_LOAD_SCALE']['Value'].iloc[0])
    
    ev_profile = ev_power_profile.generate_ev_charging_profile(
        start_dt, end_dt, 
        num_sessions_per_day=ev_sessions_per_day,
        session_energy=ev_session_energy,
        load_scale=ev_load_scale
    )
    ev_demand = ev_profile['ev_demand'] * (1 / delta_t)
    
    # Add EV demand to consumer demand but keep it separate for financial accounting
    total_demand = consumer_demand + ev_demand
    
    # Use actual start_dt for weekday and period string
    start_weekday = start_dt.weekday()
    period_str = f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
    data = {
        'pv_power': pv_power,
        'consumer_demand': consumer_demand,
        'ev_demand': ev_demand,
        'total_demand': total_demand,
        'grid_buy_price': grid_buy_price,
        'grid_sell_price': grid_sell_price,
        'lcoe_pv': lcoe_pv,
        'lcoe_bess': lcoe_bess,
        'pi_ev': pi_ev,  # Add EV price to data dictionary
        'bess_capacity': bess_capacity,
        'bess_power_limit': bess_power_limit,
        'eta_charge': eta_charge,
        'eta_discharge': eta_discharge,
        'soc_initial': soc_initial,
        'pi_consumer': pi_consumer,
        'bidding_zone_desc': f"({bidding_zone})",
        'period_str': period_str,
        'start_weekday': start_weekday,
        'n_steps': n_steps,
        'delta_t': delta_t,
        'time_steps': time_steps,
        'time_indices': time_indices
    }
    sanity_check(data)
    return data