import numpy as np
import pandas as pd
import os
import API_prices
import Prices_ITA
import pv_consumer_data_2024 as pv_data
import ev_power_profile
from datetime import datetime, timedelta

# Loads plant constants from CSV file
def load_constants(constants_path=None):
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#', header=None, names=['Parameter', 'Value'])
    params = {}
    for _, row in constants_data.iterrows():
        params[str(row['Parameter']).strip()] = row['Value']
    return params

# Builds time vector for simulation based on start/end and timestep
def build_time_vector(start_dt, end_dt, delta_t):
    n_intervals = int(((end_dt - start_dt).total_seconds() / 3600) / delta_t)
    time_steps = np.arange(0, n_intervals * delta_t, delta_t)
    n_steps = len(time_steps)
    time_indices = range(n_steps)
    return time_steps, n_steps, time_indices

# Checks that key arrays have expected length
def sanity_check(data):
    keys = ['pv_power', 'consumer_demand', 'grid_buy_price', 'grid_sell_price']
    for k in keys:
        if len(data[k]) != data['n_steps']:
            raise ValueError(f"Length mismatch: {k} has {len(data[k])}, expected {data['n_steps']}.")
    return True

# Fetches grid prices from API or ITA file
def fetch_prices(start_dt, end_dt, use_api=False):
    if use_api:
        grid_buy_price_raw, grid_sell_price_raw = API_prices.fetch_prices()
    else:
        grid_buy_price_raw = Prices_ITA.fetch_prices_from_csv(start_dt, end_dt)
        grid_sell_price_raw = grid_buy_price_raw - 0.01
    if len(grid_buy_price_raw) != 168:
        raise ValueError(f"Expected 168 hourly prices, got {len(grid_buy_price_raw)}.")
    return grid_buy_price_raw, grid_sell_price_raw

# Main loader for all simulation input data
def load(reference_case=False, API_Prices=False):
    """
    Load all input data for the simulation.
    Returns:
        dict: All simulation data.
    """
    constants_data = load_constants()
    # Access parameters directly from the dictionary
    period_start_str = str(constants_data['PERIOD_START'])
    period_end_str = str(constants_data['PERIOD_END'])
    start_dt = pd.to_datetime(period_start_str, format='%Y%m%d%H%M')
    end_dt = pd.to_datetime(period_end_str, format='%Y%m%d%H%M')
    delta_t = 0.25  # Simulation timestep in hours (15 min)
    # Build time vector for simulation
    time_steps, n_steps, time_indices = build_time_vector(start_dt, end_dt, delta_t)
    # Read plant and simulation parameters
    bess_capacity = float(constants_data['BESS_Capacity'])
    bess_power_limit = float(constants_data['BESS_Power_Limit'])
    eta_charge = float(constants_data['BESS_Efficiency_Charge'])
    eta_discharge = float(constants_data['BESS_Efficiency_Discharge'])
    soc_initial = float(constants_data['SOC_Initial'])
    bess_percent_limit = float(constants_data['BESS_limit'])
    pi_consumer = float(constants_data.get('Consumer_Price', 0))
    pi_ev = float(constants_data.get('EV_PRICE'))
    lcoe_pv = float(constants_data.get('LCOE_PV'))
    lcoe_bess = float(constants_data.get('LCOE_BESS'))
    pv_old = float(constants_data.get('PV_OLD'))
    pv_new = float(constants_data.get('PV_NEW'))
    bidding_zone = str(constants_data.get('BIDDING_ZONE'))   
    # Fetch grid prices
    if API_Prices:
        grid_buy_price_raw, grid_sell_price_raw = API_prices.fetch_prices()
    else:
        grid_buy_price_raw, grid_sell_price_raw = Prices_ITA.fetch_prices_from_csv()
    # Convert hourly prices to 15-min intervals if needed
    if len(grid_buy_price_raw) == 168:
        grid_buy_price = np.repeat(grid_buy_price_raw.values, 4)
        grid_sell_price = np.repeat(grid_sell_price_raw.values, 4)
    else:
        grid_buy_price = grid_buy_price_raw.values
        grid_sell_price = grid_sell_price_raw.values
    # Load PV and consumer demand profiles
    result = pv_data.compute_pv_power(start_dt, end_dt)
    if reference_case:
        pv_power = result['pv_production']*(1/delta_t)
        consumer_demand = result['consumer_demand']*(1/delta_t)
    else:
        pv_power = ((pv_new + pv_old)/pv_old) * result['pv_production']*(1/delta_t)
        consumer_demand = result['consumer_demand']*(1/delta_t)
    # Load EV charging profile
    ev_sessions_per_day = int(constants_data.get('EV_NUM_SESSIONS_PER_DAY', 2))
    ev_session_energy = float(constants_data.get('EV_SESSION_ENERGY', 10))
    ev_load_scale = float(constants_data.get('EV_LOAD_SCALE', 1.0))
    ev_profile = ev_power_profile.generate_ev_charging_profile(
        start_dt, end_dt,
        num_sessions_per_day=ev_sessions_per_day,
        session_energy=ev_session_energy,
        load_scale=ev_load_scale
    )
    ev_demand = ev_profile['ev_demand'] * (1 / delta_t)
    total_demand = consumer_demand + ev_demand
    start_weekday = start_dt.weekday()
    period_str = f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
    # Compile all simulation data into a dictionary
    data = {
        'pv_power': pv_power,
        'consumer_demand': consumer_demand,
        'ev_demand': ev_demand,
        'total_demand': total_demand,
        'grid_buy_price': grid_buy_price,
        'grid_sell_price': grid_sell_price,
        'lcoe_pv': lcoe_pv,
        'lcoe_bess': lcoe_bess,
        'pi_ev': pi_ev,
        'bess_capacity': bess_capacity,
        'bess_power_limit': bess_power_limit,
        'eta_charge': eta_charge,
        'eta_discharge': eta_discharge,
        'soc_initial': soc_initial,
        'bess_percent_limit': bess_percent_limit,
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