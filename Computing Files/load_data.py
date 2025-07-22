import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta, timezone
import warnings
import API_prices  # Import for price fetching
import 2024_data_pv_consumer as pv_data  # Import for PV and consumer data from CSV

# Define time framework constants here (moved from main script)
time_steps = np.arange(0, 168, 0.25)
n_steps = len(time_steps)
delta_t = 0.25
time_indices = range(n_steps)

def load():
    constants_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/Constants_Plant.csv', comment='#')
    entsoe_token = constants_data[constants_data['Parameter'] == 'ENTSOE_TOKEN']['Value'].iloc[0]
    bidding_zone = constants_data[constants_data['Parameter'] == 'BIDDING_ZONE']['Value'].iloc[0]
    timezone_offset = int(constants_data[constants_data['Parameter'] == 'TIMEZONE_OFFSET']['Value'].iloc[0])
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])
    lcoe_pv = float(constants_data[constants_data['Parameter'] == 'LCOE_PV']['Value'].iloc[0])
    lcoe_bess = float(constants_data[constants_data['Parameter'] == 'LCOE_BESS']['Value'].iloc[0])
    #pv_peak = float(constants_data[constants_data['Parameter'] == 'PV_PEAK_POWER']['Value'].iloc[0])

    utc_now = datetime.now(timezone.utc)
    start_date = (utc_now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Fetch prices from API_prices.py
    grid_buy_price, grid_sell_price = API_prices.fetch_prices(entsoe_token, bidding_zone)  # Fixed: pass only 2 args

    # Load PV production and consumer demand from CSV for a specific week in 2024
    week_number = 1  # Parameter for choosing a week in 2024 (can be changed as needed)
    result = pv_data.compute_pv_power(week_number=week_number)
    pv_power = result['pv_production'] * (1 / delta_t)  # Convert kWh per 15min to average power in kW
    consumer_demand = result['consumer_demand'] * (1 / delta_t)  # Convert kWh per 15min to average power in kW

    # Set period based on selected week in 2024
    local_start_time = datetime(2024, 1, 1) + timedelta(days=(week_number - 1) * 7)
    start_weekday = local_start_time.weekday()
    period_start = local_start_time.strftime('%Y-%m-%d')
    period_end = (local_start_time + timedelta(days=7)).strftime('%Y-%m-%d')
    period_str = f"{period_start} to {period_end}"

    bidding_zone_desc = f"({bidding_zone})"

    return {
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
        'bidding_zone_desc': bidding_zone_desc,
        'period_str': period_str,
        'start_weekday': start_weekday,
        'n_steps': n_steps,
        'delta_t': delta_t,
        'time_steps': time_steps,
        'time_indices': time_indices
    }