import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
import matplotlib.pyplot as plt
import os

print("--- Running the REAL Optimization Script ---")

# Time framework: 24 hours, 15-minute intervals (96 steps)
time_steps = np.arange(0, 24, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours
time_indices = range(n_steps)
M = 1e6  # A large constant for Big-M method, ensure it's larger than any possible power flow

def load_data():
    # Load LCOE for PV from PV_LCOE.csv, ignoring comment lines
    # Using absolute paths
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

# Set up optimization problem
prob = LpProblem("Energy_Optimization", LpMaximize)

# Decision variables
P_PV_consumer = LpVariable.dicts("P_PV_consumer", time_indices, lowBound=0)
P_PV_BESS = LpVariable.dicts("P_PV_BESS", time_indices, lowBound=0)
P_PV_grid = LpVariable.dicts("P_PV_grid", time_indices, lowBound=0)
P_BESS_consumer = LpVariable.dicts("P_BESS_consumer", time_indices, lowBound=0)
P_BESS_grid = LpVariable.dicts("P_BESS_grid", time_indices, lowBound=0)
P_grid_consumer = LpVariable.dicts("P_grid_consumer", time_indices, lowBound=0)
P_grid_BESS = LpVariable.dicts("P_grid_BESS", time_indices, lowBound=0)
SOC = LpVariable.dicts("SOC", range(n_steps + 1), lowBound=0, upBound=bess_capacity)

# Binary variables for mutual exclusivity
b_charge = LpVariable.dicts("b_charge", time_indices, cat='Binary')
b_discharge = LpVariable.dicts("b_discharge", time_indices, cat='Binary')
b_grid_buy = LpVariable.dicts("b_grid_buy", time_indices, cat='Binary')
b_grid_sell = LpVariable.dicts("b_grid_sell", time_indices, cat='Binary')


# Constraints
# 1. Consumer energy balance
for t in time_indices:
    prob += P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] == consumer_demand[t], f"Consumer_Balance_{t}"

# 2. PV power allocation
for t in time_indices:
    prob += P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power[t], f"PV_Allocation_{t}"

# 3. BESS power limits and mutual exclusivity (Big-M method)
for t in time_indices:
    # Total power into BESS
    prob += P_PV_BESS[t] + P_grid_BESS[t] <= bess_power_limit, f"BESS_Charge_Limit_{t}"
    prob += P_PV_BESS[t] + P_grid_BESS[t] <= M * b_charge[t], f"BESS_Charge_Binary_Link_{t}"

    # Total power out of BESS
    prob += P_BESS_consumer[t] + P_BESS_grid[t] <= bess_power_limit, f"BESS_Discharge_Limit_{t}"
    prob += P_BESS_consumer[t] + P_BESS_grid[t] <= M * b_discharge[t], f"BESS_Discharge_Binary_Link_{t}"

    # Mutual exclusivity: cannot charge and discharge simultaneously
    prob += b_charge[t] + b_discharge[t] <= 1, f"BESS_Mutual_Exclusivity_{t}"

# 4. Grid power limits and mutual exclusivity (Big-M method)
for t in time_indices:
    # Power sold to grid
    prob += P_PV_grid[t] + P_BESS_grid[t] <= M * b_grid_sell[t], f"Grid_Sell_Binary_Link_{t}"
    
    # Power bought from grid
    prob += P_grid_consumer[t] + P_grid_BESS[t] <= M * b_grid_buy[t], f"Grid_Buy_Binary_Link_{t}"

    # Mutual exclusivity: cannot buy and sell from grid simultaneously
    prob += b_grid_buy[t] + b_grid_sell[t] <= 1, f"Grid_Mutual_Exclusivity_{t}"


# 5. BESS SOC dynamics
prob += SOC[0] == soc_initial, "Initial_SOC" # Initial SOC
for t in range(n_steps):
    prob += SOC[t + 1] == SOC[t] + eta_charge * (P_PV_BESS[t] + P_grid_BESS[t]) * delta_t - \
                        (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t, f"SOC_Dynamics_{t}"

# 6. SOC bounds
for t in range(n_steps + 1):
    prob += SOC[t] >= 0, f"SOC_Lower_Bound_{t}"
    prob += SOC[t] <= bess_capacity, f"SOC_Upper_Bound_{t}"

# Objective function: Maximize revenue
# Calculate constant parts of the objective separately
constant_consumer_revenue = sum(consumer_demand[t] * pi_consumer * delta_t for t in time_indices)
constant_pv_cost = sum(pv_power[t] * lcoe_pv * delta_t for t in time_indices)

# Define the variable parts of the objective using lpSum
grid_sell_revenue_expr = lpSum([(P_PV_grid[t] + P_BESS_grid[t]) * grid_sell_price[t] * delta_t for t in time_indices])
grid_buy_cost_expr = lpSum([(P_grid_consumer[t] + P_grid_BESS[t]) * grid_buy_price[t] * delta_t for t in time_indices])
bess_discharge_cost_expr = lpSum([(P_BESS_consumer[t] + P_BESS_grid[t]) * lcoe_bess * delta_t for t in time_indices])

# Combine all parts into the objective function
prob += constant_consumer_revenue + grid_sell_revenue_expr - grid_buy_cost_expr - constant_pv_cost - bess_discharge_cost_expr

# Solve the problem
# PuLP will automatically choose a suitable MILP solver (like CBC or GLPK if installed)
prob.solve()

# Output results
print("Status:", LpStatus[prob.status])

if LpStatus[prob.status] == "Optimal":
    # Extract variable values
    P_PV_consumer_vals = [P_PV_consumer[t].varValue for t in time_indices]
    P_PV_BESS_vals = [P_PV_BESS[t].varValue for t in time_indices]
    P_PV_grid_vals = [P_PV_grid[t].varValue for t in time_indices]
    P_BESS_consumer_vals = [P_BESS_consumer[t].varValue for t in time_indices]
    P_BESS_grid_vals = [P_BESS_grid[t].varValue for t in time_indices]
    P_grid_consumer_vals = [P_grid_consumer[t].varValue for t in time_indices]
    P_grid_BESS_vals = [P_grid_BESS[t].varValue for t in time_indices]
    SOC_vals = [SOC[t].varValue for t in range(n_steps + 1)]

    # Compute BESS charge and discharge powers (now consistent with binary constraints)
    P_BESS_charge = [P_PV_BESS_vals[t] + P_grid_BESS_vals[t] for t in time_indices]
    P_BESS_discharge = [P_BESS_consumer_vals[t] + P_BESS_grid_vals[t] for t in time_indices]

    # Compute Grid sold and bought powers (now consistent with binary constraints)
    P_grid_sold = [P_PV_grid_vals[t] + P_BESS_grid_vals[t] for t in time_indices]
    P_grid_bought = [P_grid_consumer_vals[t] + P_grid_BESS_vals[t] for t in time_indices]

    # Compute revenue per time step
    revenue_per_step = []
    for t in time_indices:
        rev_consumer = consumer_demand[t] * pi_consumer * delta_t
        rev_grid = P_grid_sold[t] * grid_sell_price[t] * delta_t
        cost_grid = P_grid_bought[t] * grid_buy_price[t] * delta_t
        cost_pv = pv_power[t] * lcoe_pv * delta_t
        cost_bess = P_BESS_discharge[t] * lcoe_bess * delta_t
        net_rev = rev_consumer + rev_grid - cost_grid - cost_pv - cost_bess
        revenue_per_step.append(net_rev)

    total_revenue = sum(revenue_per_step)
    print(f"Total Revenue: ${total_revenue:.2f}")

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
    # plt.show()  # Uncomment if you want to display during execution
else:
    print("Optimization did not converge to an optimal solution.")


