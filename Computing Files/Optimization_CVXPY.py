import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import os

#test

print("--- Running the REAL Optimization Script using CVXPY ---")

# Time framework: 24 hours, 15-minute intervals (96 steps)
time_steps = np.arange(0, 24, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours
time_indices = range(n_steps)

def load_data():
    # Load LCOE for PV from PV_LCOE.csv, ignoring comment lines
    pv_lcoe_data = pd.read_csv('C:/Users\dell\V1_First_Model\Input Data Files\PV_LCOE.csv', comment='#')
    lcoe_pv = pv_lcoe_data['LCOE_PV'].iloc[0]

    # Load LCOE for BESS from BESS_LCOE.csv, ignoring comment lines
    bess_lcoe_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/BESS_LCOE.csv', comment='#')
    lcoe_bess = bess_lcoe_data['LCOE_BESS'].iloc[0]

    # Load constants from Constants_Plant.csv, ignoring comment lines
    constants_data = pd.read_csv('C:/Users\dell\V1_First_Model\Input Data Files/Constants_Plant.csv', comment='#')
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])

    # Sample PV power profile (kW): sinusoidal daytime generation
    pv_power = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        if 6 <= t <=18:
            pv_power[i] = 100 * np.sin(np.pi * (t -6) / 12)

    # Consumer demand (kW): constant 50 kW, 150 kW from 8-18h, with 2h ramps from 6-8h and 18-20
    consumer_demand = np.full(n_steps, 50.0)
    for i, t in enumerate(time_steps):
        if 6 <= t <8:
            consumer_demand[i] += 100.0 * (t -6) /2  # Ramp up from 50 to 150 kW
        elif 8 <= t <=18:
            consumer_demand[i] = 150.0
        elif 18 < t <=20:
            consumer_demand[i] += 100.0 * (20 - t) /2  # Ramp down from 150 to 50 kW

    # Grid prices ($/kWh): Price[t] = 0.1 + rand(x) - 0.02 * sin((t) - y)
    # y is phase shift so min at t=0, max at t=12, min at t=24
    # sin(2πt/24 - y) = -1 at t=0, 1 at t=12, -1 at t=24
    # At t=0: sin(-y) = -1 => -y = -π/2 => y = π/2
    grid_price = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        x = np.random.uniform(-0.05,0.1)
        grid_price[i] = 0.1 + x -0.02 * np.sin(2 * np.pi * t /24 - np.pi /2)
    grid_buy_price = grid_price
    grid_sell_price = grid_price  # Assuming buy and sell prices are the same

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
SOC = cp.Variable(n_steps +1, nonneg=True)
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
constraints += [SOC[t] <= bess_capacity for t in range(n_steps +1)]
constraints += [SOC[t] >= 0.1 * bess_capacity for t in range(n_steps +1)]  # Minimum SOC constraint

# Objective: Maximize net revenue with slack penalty
revenue = (cp.sum(cp.multiply(P_PV_consumer, grid_buy_price - lcoe_pv) * delta_t) +
           cp.sum(cp.multiply(P_PV_grid + P_BESS_grid, grid_sell_price) * delta_t) -
           cp.sum(cp.multiply(P_grid_consumer + P_grid_BESS, grid_buy_price) * delta_t) -
           cp.sum(cp.multiply(P_BESS_consumer + P_BESS_grid, lcoe_bess) * delta_t) -
           1e5 * cp.sum(slack))  # Penalty for unmet demand
objective = cp.Maximize(revenue)

# Problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=True)

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

    total_revenue = sum(total_net_per_step)
    print(f"Total Revenue: ${total_revenue:.2f}")

    # Check for unmet demand
    print("Time steps with unmet demand (kW):")
    for t in time_indices:
        if slack_vals[t] > 1e-6:  # Small tolerance for numerical errors
            print(f"Time {time_steps[t]:.2f}h: Unmet demand = {slack_vals[t]:.2f} kW")

        # First Image: Energy Flows
    plt.figure(figsize=(12, 15))  # Height for 3 subplots

    # Plot 1: PV Production
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, pv_power, label='PV Generation (kW)', color='orange')
    plt.xlabel('Time (hours)')
    plt.ylabel('PV Power (kW)')
    plt.title('PV Generation Profile')
    plt.legend()
    plt.grid(True)

    # Plot 2: BESS Power and SOC
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax1.plot(time_steps, P_BESS_charge, label='BESS Charge In (kW)', color='blue')
    ax1.plot(time_steps, P_BESS_discharge, label='BESS Power Out (kW)', color='red')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS Power In/Out and SOC')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, 24.25, 0.25), SOC_vals, label='SOC (kWh)', color='green', linestyle='--')
    ax2.set_ylabel('SOC (kWh)')
    ax2.legend(loc='upper right')

    # Plot 3: Consumer Power Flow
    plt.subplot(3, 1, 3)
    plt.stackplot(time_steps, P_PV_consumer_vals, P_BESS_consumer_vals, P_grid_consumer_vals, slack_vals,
                  labels=['PV to Consumer', 'BESS to Consumer', 'Grid to Consumer', 'Unmet Demand'],
                  colors=['orange', 'green', 'blue', 'red'])
    plt.plot(time_steps, consumer_demand, label='Demand', color='black', linestyle='--')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Consumer Power Flow Composition')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    # Save first image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Energy_Flows.png'))
    plt.show()

        # Second Image: Financials
    plt.figure(figsize=(12, 10))  # Height for 2 subplots

    # Plot 1: Electricity Price
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, grid_buy_price, label='Electricity Price ($/kWh)', color='blue')
    plt.xlabel('Time (hours)')
    plt.ylabel('Price ($/kWh)')
    plt.title('Electricity Price over the Day')
    plt.legend()
    plt.grid(True)

    # Plot 2: Revenue Components with dual scales
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax1.plot(time_steps, rev_pv_per_step, label='PV Avoided Cost ($)', color='green')
    ax1.plot(time_steps, rev_sell_per_step, label='Grid Sell Revenue ($)', color='cyan')
    ax1.plot(time_steps, cost_grid_per_step, label='Grid Buy Cost ($)', color='red')
    ax1.plot(time_steps, cost_bess_per_step, label='BESS Cost ($)', color='magenta')
    ax1.plot(time_steps, penalty_per_step, label='Penalty ($)', color='black')
    ax1.plot(time_steps, total_net_per_step, label='Total Revenue per Step ($)', color='purple')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Revenue/Cost per Step ($)')
    ax1.set_title('Revenue Components over Time')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    cumulative_revenue = np.cumsum(total_net_per_step)
    ax2.plot(time_steps, cumulative_revenue, label='Cumulative Revenue ($)', color='orange', linestyle='--')
    ax2.set_ylabel('Cumulative Revenue ($)')
    ax2.legend(loc='upper right')

    output_dir = os.path.join(os.path.dirname(os.path.abspath(file)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Financial_Metrics.png'))
    plt.show()
