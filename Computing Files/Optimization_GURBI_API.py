import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime, timedelta, timezone
import requests
from xml.etree import ElementTree
import warnings  # Added for warning handling

# TEST

print("--- Running the REAL Optimization Script using GUROBI ---")

# Time framework: 1 week (168 hours), 15-minute intervals (672 steps)
time_steps = np.arange(0, 168, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours
time_indices = range(n_steps)

# ENTSO-E API token and bidding zone
ENTSOE_TOKEN = 'cd4a21d9-d58c-4b68-b233-ae5e0d8707f5'
BIDDING_ZONE = '10YCH-SWISSGRIDZ'  # Switzerland bidding zone
TIMEZONE_OFFSET = 2  # hours for CEST (UTC+2)

# Added function to fetch day-ahead prices from ENTSO-E
def get_dayahead_prices(api_key: str, area_code: str, start: datetime = None, end: datetime = None):
    """
    Get day-ahead prices from ENTSO-E API.
    Adapted from https://gist.github.com/jpulakka/f866e37dcedeede31e96a34e9f06ed7a
    """
    if not start:
        start = datetime.now(timezone.utc)
    elif start.tzinfo and start.tzinfo != timezone.utc:
        start = start.astimezone(timezone.utc)
    if not end:
        end = start + timedelta(days=1)
    elif end.tzinfo and end.tzinfo != timezone.utc:
        end = end.astimezone(timezone.utc)
    fmt = '%Y%m%d%H00'
    url = f'https://web-api.tp.entsoe.eu/api?securityToken={api_key}&documentType=A44&in_Domain={area_code}' \
          f'&out_Domain={area_code}&periodStart={start.strftime(fmt)}&periodEnd={end.strftime(fmt)}'
    response = requests.get(url)
    response.raise_for_status()
    xml_str = response.text
    result = {}
    for child in ElementTree.fromstring(xml_str):
        if child.tag.endswith("TimeSeries"):
            for ts_child in child:
                if ts_child.tag.endswith("Period"):
                    for pe_child in ts_child:
                        if pe_child.tag.endswith("timeInterval"):
                            for ti_child in pe_child:
                                if ti_child.tag.endswith("start"):
                                    start_time = datetime.strptime(ti_child.text, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc)
                        elif pe_child.tag.endswith("Point"):
                            for po_child in pe_child:
                                if po_child.tag.endswith("position"):
                                    delta = int(po_child.text) - 1
                                    time = start_time + timedelta(hours=delta)
                                elif po_child.tag.endswith("price.amount"):
                                    price = float(po_child.text)
                                    result[time] = price
    return result

def load_data():
    # Load LCOE for PV from PV_LCOE.csv, ignoring comment lines
    pv_lcoe_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/PV_LCOE.csv', comment='#')
    lcoe_pv = pv_lcoe_data['LCOE_PV'].iloc[0]

    # Load LCOE for BESS from BESS_LCOE.csv, ignoring comment lines
    bess_lcoe_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/BESS_LCOE.csv', comment='#')
    lcoe_bess = bess_lcoe_data['LCOE_BESS'].iloc[0]

    # Load constants from Constants_Plant.csv, ignoring comment lines
    constants_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/Constants_Plant.csv', comment='#')
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])

    # Fetch grid prices from ENTSO-E for the last 7 days (previous complete week)
    utc_now = datetime.now(timezone.utc)
    start_date = (utc_now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    try:  # Added: Try-except to handle API failures
        prices_dict = get_dayahead_prices(ENTSOE_TOKEN, BIDDING_ZONE, start_date, end_date)
        sorted_times = sorted(prices_dict.keys())
        # Added debug prints
        print(f"API debug: Fetched {len(sorted_times)} prices for zone {BIDDING_ZONE}")
        if sorted_times:
            print(f"API debug: Date range: {sorted_times[0]} to {sorted_times[-1]}")
        
        # Added: Filter times to exactly the requested range [start_date, end_date)
        filtered_times = [t for t in sorted_times if start_date <= t < end_date]
        print(f"API debug: Filtered to {len(filtered_times)} prices within {start_date} to {end_date}")
        
        current_times = filtered_times
        if len(current_times) != 168:
            warnings.warn(f"Warning: Non-standard data ({len(current_times)} prices). Adjusting to 168.")
            mean_price = np.mean([prices_dict[t] for t in current_times]) if current_times else 0.103 * 1000  # Fallback to avg in Eur/MWh
            
            # Pad if fewer
            while len(current_times) < 168:
                last_t = current_times[-1] if current_times else start_date
                next_t = last_t + timedelta(hours=1)
                prices_dict[next_t] = mean_price
                current_times.append(next_t)
            
            # Clip if more
            if len(current_times) > 168:
                current_times = current_times[:168]
        
        sorted_times = sorted(current_times)  # Ensure sorted after adjustments
    except Exception as e:  # Added: Catch and log errors
        print(f"API error: {e}. Falling back to synthetic data.")
        grid_price_hourly = np.full(168, 0.103)  # Fallback to average synthetic hourly prices
        sorted_times = [start_date + timedelta(hours=i) for i in range(168)]
    else:
        grid_price_hourly = np.array([prices_dict[t] for t in sorted_times]) / 1000  # Convert to Eur/kWh

    # Adjust for local timezone to correctly shift consumer demand profile
    local_start_time = sorted_times[0] + timedelta(hours=TIMEZONE_OFFSET)
    start_weekday = local_start_time.weekday()

    # Linear interpolation to 15-min resolution
    time_hourly = np.arange(0, 168, 1)
    time_quarter = np.arange(0, 168, 0.25)
    grid_price = np.interp(time_quarter, time_hourly, grid_price_hourly)
    grid_buy_price = grid_price + 0.01
    grid_sell_price = grid_price - 0.01

    # Sample PV power profile (kW): sinusoidal daytime generation over 7 days, with bad weather and noise
    pv_power = np.zeros(n_steps)
    multipliers = [1.0, 0.9, 0.5, 0.8, 1.0, 0.6, 1.0]
    random.shuffle(multipliers)  # Randomly assign multipliers to days
    for i, t in enumerate(time_steps):
        local_t = t % 24
        day = int(t // 24)
        if 6 <= local_t <= 18:
            amplitude = 2327 * multipliers[day]
            pv_power[i] = amplitude * np.sin(np.pi * (local_t - 6) / 12) + np.random.normal(0, 10)  # Realistic noise
        pv_power[i] = max(0, pv_power[i])  # Ensure non-negative

    # Consumer demand (kW): production days on weekdays, standby on weekends, shifted to match actual days
    consumer_demand = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        local_t = t % 24
        day = int(t // 24)
        actual_weekday = (start_weekday + day) % 7  # 0=Mon ... 6=Sun
        if actual_weekday < 5:  # Weekdays (Mon-Fri): production days
            base = 200.0
            if 6 <= local_t < 8:
                add = 1000.0 * (local_t - 6) / 2
                consumer_demand[i] = base + add
            elif 8 <= local_t <= 16:
                consumer_demand[i] = 1200.0
            elif 16 < local_t <= 18:
                add = 1000.0 * (18 - local_t) / 2
                consumer_demand[i] = base + add
            else:
                consumer_demand[i] = base
        else:  # Weekends (Sat-Sun): standby days
            consumer_demand[i] = 70.0

    # Prepare plot information
    bidding_zone_desc = f"Switzerland ({BIDDING_ZONE})"
    period_start = local_start_time.strftime('%Y-%m-%d')
    period_end = (local_start_time + timedelta(days=7)).strftime('%Y-%m-%d')
    period_str = f"{period_start} to {period_end}"

    return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
            lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
            eta_charge, eta_discharge, soc_initial, pi_consumer,
            bidding_zone_desc, period_str)

# Load data
(pv_power, consumer_demand, grid_buy_price, grid_sell_price,
 lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
 eta_charge, eta_discharge, soc_initial, pi_consumer,
 bidding_zone_desc, period_str) = load_data()

# Added for big-M constants in mutual exclusivity constraints
max_pv = np.max(pv_power)
max_demand = np.max(consumer_demand)
M_bess = bess_power_limit
M_grid = max(max_pv + bess_power_limit, max_demand + bess_power_limit)

# Variables
P_PV_consumer = cp.Variable(n_steps, nonneg=True)
P_PV_BESS = cp.Variable(n_steps, nonneg=True)
P_PV_grid = cp.Variable(n_steps, nonneg=True)
P_BESS_consumer = cp.Variable(n_steps, nonneg=True)
P_BESS_grid = cp.Variable(n_steps, nonneg=True)
P_grid_consumer = cp.Variable(n_steps, nonneg=True)
P_grid_BESS = cp.Variable(n_steps, nonneg=True)
SOC = cp.Variable(n_steps + 1, nonneg=True)
slack = cp.Variable(n_steps, nonneg=True)  # Slack for consumer balance

# Added binary variables for mutual exclusivity
delta_bess = cp.Variable(n_steps, boolean=True)
delta_grid = cp.Variable(n_steps, boolean=True)

# Constraints
constraints = []

# Consumer balance with slack (equality)
constraints += [P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] + slack[t] == consumer_demand[t]
                for t in time_indices]

# PV allocation
constraints += [P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power[t] for t in time_indices]

# BESS constraints
for t in time_indices:
    constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= bess_power_limit,
                    P_BESS_consumer[t] + P_BESS_grid[t] <= bess_power_limit]

# SOC dynamics
constraints += [SOC[0] == soc_initial]
constraints += [SOC[t+1] == SOC[t] + eta_charge * (P_PV_BESS[t] + P_grid_BESS[t]) * delta_t -
                (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t for t in range(n_steps)] #OPTIONAL ADD SELF DISCHARGE OF BATTERY
constraints += [SOC[t] <= bess_capacity for t in range(n_steps + 1)]
constraints += [SOC[t] >= 0.05 * bess_capacity for t in range(n_steps + 1)]  # Minimum SOC constraint 5% of BESS_CAPACITY

# Force SOC at end >= initial for weekly sustainability and arbitrage incentive
constraints += [SOC[n_steps] >= soc_initial]

# Added constraints for mutual exclusivity of BESS charge/discharge (charge when delta_bess=1, discharge when delta_bess=0)
for t in time_indices:
    constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= M_bess * delta_bess[t]]
    constraints += [P_BESS_consumer[t] + P_BESS_grid[t] <= M_bess * (1 - delta_bess[t])]

# Added constraints for mutual exclusivity of Grid buy/sell (buy when delta_grid=1, sell when delta_grid=0)
for t in time_indices:
    constraints += [P_grid_consumer[t] + P_grid_BESS[t] <= M_grid * delta_grid[t]]
    constraints += [P_PV_grid[t] + P_BESS_grid[t] <= M_grid * (1 - delta_grid[t])]

# Objective: Maximize net revenue with slack penalty
revenue = (cp.sum(cp.multiply(P_PV_consumer, grid_buy_price - lcoe_pv) * delta_t) +
           cp.sum(cp.multiply(P_PV_grid + P_BESS_grid, grid_sell_price) * delta_t) -
           cp.sum(cp.multiply(P_grid_consumer + P_grid_BESS, grid_buy_price) * delta_t) +    #CHANGED TO PLUS 15.07
           cp.sum(cp.multiply(P_BESS_consumer + P_BESS_grid, grid_buy_price - lcoe_bess) * delta_t) - #ADDED GRID PRICE like in PV
           1e5 * cp.sum(slack))  # Penalty for unmet demand
objective = cp.Maximize(revenue)

# Problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GUROBI, verbose=True)

# Check status
print("Status:", problem.status)
if problem.status == cp.OPTIMAL:
    # Extract values for plotting
    P_PV_consumer_vals = P_PV_consumer.value
    P_PV_BESS_vals = P_PV_BESS.value
    P_PV_grid_vals = P_PV_grid.value
    P_BESS_consumer_vals = P_BESS_consumer.value
    P_BESS_grid_vals = P_BESS_grid.value
    P_grid_consumer_vals = P_grid_consumer.value
    P_grid_BESS_vals = P_grid_BESS.value
    SOC_vals = SOC.value
    slack_vals = slack.value

    # Compute BESS charge and discharge powers
    P_BESS_charge = P_PV_BESS_vals + P_grid_BESS_vals
    P_BESS_discharge = P_BESS_consumer_vals + P_BESS_grid_vals

    # Compute Grid sold and bought powers
    P_grid_sold = P_PV_grid_vals + P_BESS_grid_vals
    P_grid_bought = P_grid_consumer_vals + P_grid_BESS_vals

    # Compute revenue per step for plotting
    rev_pv_per_step = []
    rev_sell_per_step = []
    cost_grid_per_step = []
    cost_bess_per_step = []
    penalty_per_step = []
    total_net_per_step = []
    bess_rev_per_step = []
    for t in time_indices:
        rev_pv = P_PV_consumer_vals[t] * (grid_buy_price[t] - lcoe_pv) * delta_t
        rev_sell = (P_PV_grid_vals[t] + P_BESS_grid_vals[t]) * grid_sell_price[t] * delta_t
        cost_grid = - (P_grid_consumer_vals[t] + P_grid_BESS_vals[t]) * grid_buy_price[t] * delta_t
        cost_bess = - (P_BESS_consumer_vals[t] + P_BESS_grid_vals[t]) * lcoe_bess * delta_t
        penalty = -1e5 * slack_vals[t]
        net_rev = rev_pv + rev_sell + cost_grid + cost_bess + penalty
        rev_pv_per_step.append(rev_pv)
        rev_sell_per_step.append(rev_sell)
        cost_grid_per_step.append(cost_grid)
        cost_bess_per_step.append(cost_bess)
        penalty_per_step.append(penalty)
        total_net_per_step.append(net_rev)

        # BESS-specific revenue per step
        bess_rev = P_BESS_grid_vals[t] * (grid_sell_price[t] - lcoe_bess) * delta_t + P_BESS_consumer_vals[t] * (grid_buy_price[t] - lcoe_bess) * delta_t
        bess_rev_per_step.append(bess_rev)

    total_revenue = sum(total_net_per_step)
    print(f"Total Revenue: Eur{total_revenue:.2f}")

    # Check for unmet demand
    print("Time steps with unmet demand (kW):")
    for t in time_indices:
        if slack_vals[t] > 1e-6:  # Small tolerance for numerical errors
            print(f"Time {time_steps[t]:.2f}h: Unmet demand = {slack_vals[t]:.2f} kW")

    # Improve plot appearance globally
    plt.rcParams.update({'font.size': 8})

    # Day labels for xticks
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']

    # First Image: Energy Flows
    plt.figure(figsize=(12, 12))  # Smaller size for better fit

    # Plot 1: PV Production with Grid Sold and Bought
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, pv_power, label='PV Gen (kW)', color='orange')
    plt.plot(time_steps, P_grid_sold, label='Grid Sold (kW)', color='blue')
    plt.plot(time_steps, P_grid_bought, label='Grid Bought (kW)', color='magenta')
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title('PV and Grid Flows')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), day_labels)
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Plot 2: BESS Power and SOC
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax1.plot(time_steps, P_BESS_charge, label='BESS Charge (kW)', color='blue')
    ax1.plot(time_steps, P_BESS_discharge, label='BESS Discharge (kW)', color='red')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS Flows and SOC')
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, 169, 24))
    ax1.set_xticklabels(day_labels)
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, 168.25, 0.25), SOC_vals, label='SOC (kWh)', color='green', linestyle='--')
    ax2.set_ylabel('SOC (kWh)')
    ax2.legend(loc='upper right')

    # Plot 3: Consumer Power Flow
    plt.subplot(3, 1, 3)
    plt.stackplot(time_steps, P_PV_consumer_vals, P_BESS_consumer_vals, P_grid_consumer_vals, slack_vals,
                  labels=['PV to Cons', 'BESS to Cons', 'Grid to Cons', 'Unmet'],
                  colors=['orange', 'green', 'blue', 'red'])
    plt.plot(time_steps, consumer_demand, label='Demand', color='black', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title('Consumer Flow Composition')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), day_labels)
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing
    plt.tight_layout(pad=1.5)
    
    # Save first image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Energy_Flows.png'))
    plt.show()

    # Second Image: Financials
    plt.figure(figsize=(12, 12))  # Smaller size for better fit

    # Plot 1: Electricity Price
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, grid_buy_price, label='Grid Price (Eur/kWh)', color='blue')
    plt.plot(time_steps, np.full(n_steps, lcoe_pv), label='PV LCOE', color='orange', linestyle='--')
    plt.plot(time_steps, np.full(n_steps, lcoe_bess), label='BESS LCOE', color='green', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Price (Eur/kWh)')
    plt.title(f'Prices and LCOEs\n{bidding_zone_desc}\n{period_str}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), day_labels)
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Plot 2: Grid sold revenue, buy cost, and BESS cost
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, rev_sell_per_step, label='Grid Sell Rev (Eur)', color='cyan')
    plt.plot(time_steps, cost_grid_per_step, label='Grid Buy Cost (Eur)', color='red')
    plt.plot(time_steps, cost_bess_per_step, label='BESS Cost (Eur)', color='magenta')
    plt.plot(time_steps, rev_pv_per_step, label='PV Avoided Cost (Eur)', color='green')
    plt.xlabel('Time (h)')
    plt.ylabel('Eur/Step')
    plt.title('Revenues and Costs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), day_labels)
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Plot 3: Revenue at each timestep and cumulative revenue
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax1.plot(time_steps, total_net_per_step, label='Rev per Step (Eur)', color='purple')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Rev per Step (Eur)')
    ax1.set_title('Timestep and Cum. Revenue')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, 169, 24))
    ax1.set_xticklabels(day_labels)
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()
    cumulative_revenue = np.cumsum(total_net_per_step)
    ax2.plot(time_steps, cumulative_revenue, label='Cum. Rev (Eur)', color='orange', linestyle='--')
    cumulative_bess_revenue = np.cumsum(bess_rev_per_step)
    ax2.plot(time_steps, cumulative_bess_revenue, label='Cum. BESS Rev (Eur)', color='blue', linestyle='-.')
    ax2.set_ylabel('Cum. Rev (Eur)')
    ax2.legend(loc='upper right')

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(pad=1.5)
    # Save second image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Financials.png'))
    plt.show()