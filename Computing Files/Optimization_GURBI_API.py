import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import requests
from datetime import date, timedelta
from xml.etree import ElementTree as ET

print("--- Running the REAL Optimization Script using GUROBI ---")

# Time framework: 1 week (168 hours), 15-minute intervals (672 steps)
time_steps = np.arange(0, 168, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours
time_indices = range(n_steps)

ENTSOE_TOKEN = "cd4a21d9-d58c-4b68-b233-ae5e0d8707f5"  # Replace with your UUID token from transparency@entsoe.eu

def fetch_prices_last_week():
    # Calculate last full week (Monday to Sunday)
    today = date.today()
    if today.weekday() == 0:  # If today is Monday, previous week
        last_monday = today - timedelta(days=7)
    else:
        last_monday = today - timedelta(days=today.weekday())
    last_sunday = last_monday + timedelta(days=6)

    period_start = last_monday.strftime('%Y%m%d') + '0000'
    period_end = (last_sunday + timedelta(days=1)).strftime('%Y%m%d') + '0000'  # End is exclusive

    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        'securityToken': ENTSOE_TOKEN,
        'documentType': 'A44',  # Day-ahead prices
        'in_Domain': '10Y1001A1001A73I',  # Italy NORD bidding zone
        'out_Domain': '10Y1001A1001A73I',
        'periodStart': period_start,
        'periodEnd': period_end
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        grid_price_hourly = np.zeros(168)  # 7 days * 24 hours
        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0'}  # Namespace
        time_series = root.findall('.//ns:TimeSeries', ns)
        hourly_index = 0
        for ts in time_series:
            points = ts.findall('ns:Period/ns:Point', ns)
            for point in points:
                position = int(point.find('ns:position', ns).text) - 1  # 1-based to 0-based
                price = float(point.find('ns:price.amount', ns).text)
                grid_price_hourly[hourly_index + position] = price / 1000  # €/MWh to €/kWh
            hourly_index += len(points)  # Advance by day hours

        return grid_price_hourly
    except Exception as e:
        print(f"Error fetching ENTSO-E prices: {e}")
        return np.zeros(168)  # Fallback

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

    # Sample PV power profile (kW): sinusoidal daytime generation over 7 days, with bad weather and noise
    pv_power = np.zeros(n_steps)
    multipliers = [1.0, 0.9, 0.5, 0.8, 1.0, 0.6, 1.0]  # Day-specific factors for weather variation (e.g., cloudy days)
    for i, t in enumerate(time_steps):
        local_t = t % 24
        day = int(t // 24)
        if 6 <= local_t <= 18:
            amplitude = 1327 * multipliers[day]
            pv_power[i] = amplitude * np.sin(np.pi * (local_t - 6) / 12) + np.random.normal(0, 10)  # Realistic noise
        pv_power[i] = max(0, pv_power[i])  # Ensure non-negative

    # Consumer demand (kW): 5 production days (Mon-Fri) with ramps, 2 standby days (Sat-Sun) flat 70 kW
    consumer_demand = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        local_t = t % 24
        day = int(t // 24)
        if day < 5:  # Production days
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
        else:  # Standby days
            consumer_demand[i] = 70.0

    # Dynamic fetch of grid prices using ENTSO-E API
    grid_price_hourly = fetch_prices_last_week()

    # Linear interpolation to 15-min resolution
    time_hourly = np.arange(0, 168, 1)
    time_quarter = np.arange(0, 168, 0.25)
    grid_price = np.interp(time_quarter, time_hourly, grid_price_hourly)
    grid_buy_price = grid_price + 0.01
    grid_sell_price = grid_price - 0.01

    return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
            lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
            eta_charge, eta_discharge, soc_initial, pi_consumer)

# Load data
(pv_power, consumer_demand, grid_buy_price, grid_sell_price,
 lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
 eta_charge, eta_discharge, soc_initial, pi_consumer) = load_data()

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
                (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t for t in range(n_steps)]
constraints += [SOC[t] <= bess_capacity for t in range(n_steps + 1)]
constraints += [SOC[t] >= 0.1 * bess_capacity for t in range(n_steps + 1)]  # Minimum SOC constraint

# Force SOC at end >= initial for weekly sustainability and arbitrage incentive
constraints += [SOC[n_steps] >= soc_initial]

# Objective: Maximize net revenue with slack penalty
revenue = (cp.sum(cp.multiply(P_PV_consumer, grid_buy_price - lcoe_pv) * delta_t) +
           cp.sum(cp.multiply(P_PV_grid + P_BESS_grid, grid_sell_price) * delta_t) -
           cp.sum(cp.multiply(P_grid_consumer + P_grid_BESS, grid_buy_price) * delta_t) +
           cp.sum(cp.multiply(P_BESS_consumer, grid_buy_price - lcoe_bess) * delta_t) +
           cp.sum(cp.multiply(P_BESS_grid, grid_sell_price - lcoe_bess) * delta_t) - 
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
    print(f"Total Revenue: Euro{total_revenue:.2f}")

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
    plt.plot(time_steps, P_grid_sold, label='Grid Sold (kW)', color='cyan')
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
    plt.plot(time_steps, grid_buy_price, label='Grid Price (Euro/kWh)', color='blue')
    plt.plot(time_steps, np.full(n_steps, lcoe_pv), label='PV LCOE', color='orange', linestyle='--')
    plt.plot(time_steps, np.full(n_steps, lcoe_bess), label='BESS LCOE', color='green', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Price (Euro/kWh)')
    plt.title('Prices and LCOEs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), day_labels)
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Plot 2: Grid sold revenue, buy cost, and BESS cost
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, rev_sell_per_step, label='Grid Sell Rev (Euro)', color='cyan')
    plt.plot(time_steps, cost_grid_per_step, label='Grid Buy Cost (Euro)', color='red')
    plt.plot(time_steps, cost_bess_per_step, label='BESS Cost (Euro)', color='magenta')
    plt.plot(time_steps, rev_pv_per_step, label='PV Avoided Cost (Euro)', color='green')
    plt.xlabel('Time (h)')
    plt.ylabel('Euro/Step')
    plt.title('Revenues and Costs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), day_labels)
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Plot 3: Revenue at each timestep and cumulative revenue
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax1.plot(time_steps, total_net_per_step, label='Rev per Step (Euro)', color='purple')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Rev per Step (Euro)')
    ax1.set_title('Timestep and Cum. Revenue')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, 169, 24))
    ax1.set_xticklabels(day_labels)
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()
    cumulative_revenue = np.cumsum(total_net_per_step)
    ax2.plot(time_steps, cumulative_revenue, label='Cum. Rev (Euro)', color='orange', linestyle='--')
    cumulative_bess_revenue = np.cumsum(bess_rev_per_step)
    ax2.plot(time_steps, cumulative_bess_revenue, label='Cum. BESS Rev (Euro)', color='blue', linestyle='-.')
    ax2.set_ylabel('Cum. Rev (Euro)')
    ax2.legend(loc='upper right')

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(pad=1.5)
    # Save second image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Financial_Metrics.png'))
    plt.show()