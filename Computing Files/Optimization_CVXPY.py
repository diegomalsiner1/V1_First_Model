import cvxpy as cp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

print("--- Running the REAL Optimization Script using CVXPY ---")

# Time framework: 24 hours, 15-minute intervals (96 steps)
time_steps = np.arange(0, 24, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours
time_indices = range(n_steps)
M = 1e6  # A large constant for Big-M method

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

    # Sample PV power profile (kW): sinusoidal daytime generation
    pv_power = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        if 5 <= t <= 19:
            pv_power[i] = 100 * np.sin(np.pi * (t - 6) / 12)

    # Sample consumer demand (kW): constant load of 50 kW with 50 kW step from 8 AM to 6 PM
    consumer_demand = np.full(n_steps, 50.0)
    for i, t in enumerate(time_steps):
        if 8 <= t <= 18:
            consumer_demand[i] += 50.0

    # Sample grid prices ($/kWh): constant at 0.12
    grid_buy_price = np.full(n_steps, 0.12)
    grid_sell_price = np.full(n_steps, 0.12)
    
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

b_charge = cp.Variable(n_steps, boolean=True)
b_discharge = cp.Variable(n_steps, boolean=True)
b_grid_buy = cp.Variable(n_steps, boolean=True)
b_grid_sell = cp.Variable(n_steps, boolean=True)

# Constraints
constraints = []
# Consumer balance
constraints += [P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] == consumer_demand[t] for t in time_indices]
# PV allocation
constraints += [P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power[t] for t in time_indices]
# BESS constraints
for t in time_indices:
    constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= bess_power_limit,
                    P_PV_BESS[t] + P_grid_BESS[t] <= M * b_charge[t],
                    P_BESS_consumer[t] + P_BESS_grid[t] <= bess_power_limit,
                    P_BESS_consumer[t] + P_BESS_grid[t] <= M * b_discharge[t],
                    b_charge[t] + b_discharge[t] <= 1]
# Grid constraints
for t in time_indices:
    constraints += [P_PV_grid[t] + P_BESS_grid[t] <= M * b_grid_sell[t],
                    P_grid_consumer[t] + P_grid_BESS[t] <= M * b_grid_buy[t],
                    b_grid_buy[t] + b_grid_sell[t] <= 1]
# SOC dynamics
constraints += [SOC[0] == soc_initial]
constraints += [SOC[t+1] == SOC[t] + eta_charge * (P_PV_BESS[t] + P_grid_BESS[t]) * delta_t -
                (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t for t in range(n_steps)]
constraints += [SOC[t] <= bess_capacity for t in range(n_steps + 1)]

# Objective: Maximize net revenue
revenue = (cp.sum(cp.multiply(P_PV_grid + P_BESS_grid, grid_sell_price) * delta_t) -
           cp.sum(cp.multiply(P_grid_consumer + P_grid_BESS, grid_buy_price) * delta_t) -
           cp.sum(pv_power * lcoe_pv * delta_t) -
           cp.sum(cp.multiply(P_BESS_consumer + P_BESS_grid, lcoe_bess) * delta_t))

objective = cp.Maximize(revenue)

# Problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=True)

# Check status
print("Status:", problem.status)
if problem.status == "optimal":
    print("Optimal Revenue:", problem.value)
    
    # Extract values for plotting
    P_PV_consumer_vals = P_PV_consumer.value
    P_BESS_consumer_vals = P_BESS_consumer.value
    P_grid_consumer_vals = P_grid_consumer.value
    P_BESS_charge = P_PV_BESS.value + P_grid_BESS.value
    P_BESS_discharge = P_BESS_consumer.value + P_BESS_grid.value
    P_grid_sold = P_PV_grid.value + P_BESS_grid.value
    P_grid_bought = P_grid_consumer.value + P_grid_BESS.value
    SOC_vals = SOC.value

    # Compute revenue per step for plotting
    revenue_per_step = []
    for t in time_indices:
        rev_grid = (P_PV_grid.value[t] + P_BESS_grid.value[t]) * grid_sell_price[t] * delta_t
        cost_grid = (P_grid_consumer.value[t] + P_grid_BESS.value[t]) * grid_buy_price[t] * delta_t
        cost_pv = pv_power[t] * lcoe_pv * delta_t
        cost_bess = (P_BESS_consumer.value[t] + P_BESS_grid.value[t]) * lcoe_bess * delta_t
        net_rev = rev_grid - cost_grid - cost_pv - cost_bess
        revenue_per_step.append(net_rev)

    # Plotting critical parameters
    plt.figure(figsize=(12, 20))  # Increased height for 5 subplots

    # Graph 1: PV Production
    plt.subplot(5, 1, 1)
    plt.plot(time_steps, pv_power, label='PV Power (kW)', color='orange')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('PV Power Production Profile')
    plt.legend()
    plt.grid(True)

    # Graph 2: Consumer Energy Composition
    plt.subplot(5, 1, 2)
    plt.stackplot(time_steps, P_PV_consumer_vals, P_BESS_consumer_vals, P_grid_consumer_vals,
                  labels=['PV to Consumer', 'BESS to Consumer', 'Grid to Consumer'],
                  colors=['orange', 'green', 'blue'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Consumer Energy Composition')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Graph 3: BESS Power and SOC
    plt.subplot(5, 1, 3)
    ax1 = plt.gca()
    ax1.plot(time_steps, P_BESS_charge, label='BESS Charge (kW)', color='blue')
    ax1.plot(time_steps, P_BESS_discharge, label='BESS Discharge (kW)', color='red')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS Power Flows and SOC')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, 24.25, 0.25), SOC_vals, label='SOC (kWh)', color='green', linestyle='--')
    ax2.set_ylabel('SOC (kWh)')
    ax2.legend(loc='upper right')

    # Graph 4: Grid Power Flows
    plt.subplot(5, 1, 4)
    plt.plot(time_steps, P_grid_sold, label='Grid Sold (kW)', color='orange')
    plt.plot(time_steps, P_grid_bought, label='Grid Bought (kW)', color='brown')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Grid Power Flows')
    plt.legend()
    plt.grid(True)

    # Graph 5: Financials
    plt.subplot(5, 1, 5)
    plt.plot(time_steps, grid_buy_price, label='Grid Buy Price ($/kWh)', color='blue')
    plt.plot(time_steps, grid_sell_price, label='Grid Sell Price ($/kWh)', color='green')
    plt.plot(time_steps, revenue_per_step, label='Net Revenue ($)', color='purple')
    cumulative_revenue = np.cumsum(revenue_per_step)
    plt.plot(time_steps, cumulative_revenue, label='Cumulative Revenue ($)', color='red')
    plt.xlabel('Time (hours)')
    plt.ylabel('Financials')
    plt.title('Financials over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save plot to Output Files folder
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'optimization_results_plots.png'))
    plt.show()
else:
    print("Check constraints/data for infeasibility.")
