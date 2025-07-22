import numpy as np
import pandas as pd
import os  # Add this import for relative paths
import API_prices
import pv_consumer_data_2024 as pv_data
from datetime import datetime, timedelta

# Define simulation time granularity
time_steps = np.arange(0, 168, 0.25)  # 15-min intervals for 7 days
n_steps = len(time_steps)
delta_t = 0.25
time_indices = range(n_steps)

def sanity_check(data):
    keys = ['pv_power', 'consumer_demand', 'grid_buy_price', 'grid_sell_price']
    for k in keys:
        if len(data[k]) != n_steps:
            raise ValueError(f"Length mismatch: {k} has {len(data[k])}, expected {n_steps}.")
    return True

def load_constants():
    # Use relative path for portability; assume CSV in Input Data Files/ or adjust as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')  # Adjust if structure changes
    constants_data = pd.read_csv(csv_path, comment='#')
    return constants_data

def load():
    constants_data = load_constants()
    
    timezone_offset = int(constants_data[constants_data['Parameter'] == 'TIMEZONE_OFFSET']['Value'].iloc[0])
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])
    lcoe_pv = float(constants_data[constants_data['Parameter'] == 'LCOE_PV']['Value'].iloc[0])
    lcoe_bess = float(constants_data[constants_data['Parameter'] == 'LCOE_BESS']['Value'].iloc[0])
    pv_old = float(constants_data[constants_data['Parameter'] == 'PV_OLD']['Value'].iloc[0])
    pv_new = float(constants_data[constants_data['Parameter'] == 'PV_NEW']['Value'].iloc[0])
    week_number = int(constants_data[constants_data['Parameter'] == 'WEEK_NUMBER']['Value'].iloc[0])
    bidding_zone = constants_data[constants_data['Parameter'] == 'BIDDING_ZONE']['Value'].iloc[0]

    # Fetch hourly grid prices (now without arguments)
    grid_buy_price_raw, grid_sell_price_raw = API_prices.fetch_prices()
    grid_buy_price_raw = grid_buy_price_raw[:168]
    grid_sell_price_raw = grid_sell_price_raw[:168]
    grid_buy_price = np.repeat(grid_buy_price_raw.values, 4)
    grid_sell_price = np.repeat(grid_sell_price_raw.values, 4)

    # Load PV and demand for specified week
    result = pv_data.compute_pv_power(week_number)
    pv_power = ((pv_new + pv_old)/pv_old) * result['pv_production'] * (1 / delta_t)  # convert kWh per 15 min → kW
    consumer_demand = result['consumer_demand'] * (1 / delta_t)

    local_start_time = datetime(2024, 1, 1) + timedelta(days=(week_number - 1) * 7)
    start_weekday = local_start_time.weekday()
    period_str = f"{local_start_time.strftime('%Y-%m-%d')} to {(local_start_time + timedelta(days=7)).strftime('%Y-%m-%d')}"

    data = {
        'pv_power': pv_power,
        'consumer_demand': consumer_demand,
        'grid_buy_price': grid_buy_price,
        'grid_sell_price': grid_sell_price,
        'lcoe_pv': lcoe_pv,
        'lcoe_bess': lcoe_bess,
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