import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
import matplotlib.pyplot as plt


print("--- Running the REAL Optimization Script ---") # Add this line


# Time framework: 24 hours, 15-minute intervals (96 steps)
time_steps = np.arange(0, 24, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours
time_indices = range(n_steps)

# Load data from CSV files (placeholders, using sample data since files are empty)
def load_data():
    # Sample PV power profile (kW): sinusoidal daytime generation
    pv_power = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        if 6 <= t <= 18:
            pv_power[i] = 100 * np.sin(np.pi * (t - 6) / 12)

    # Sample consumer demand (kW): constant
    consumer_demand = np.full(n_steps, 50.0)

    # Sample grid prices ($/kWh): higher buy price during peak hours
    grid_buy_price = np.full(n_steps, 0.15)
    grid_sell_price = np.full(n_steps, 0.08)
    for i, t in enumerate(time_steps):
        if 17 <= t <= 21:
            grid_buy_price[i] = 0.20

    # LCOE values ($/kWh)
    lcoe_pv = 0.05
    lcoe_bess = 0.10

    # BESS parameters
    bess_capacity = 200.0  # kWh
    bess_power_limit = 50.0  # kW
    eta_charge = 0.9
    eta_discharge = 0.9
    soc_initial = 0.5 * bess_capacity  # 50% initial SOC

    # Consumer price ($/kWh)
    pi_consumer = 0.12

    return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
            lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
            eta_charge, eta_discharge, soc_initial, pi_consumer)

# Uncomment this when actual data files are available
# def load_data():
#     pv_data = pd.read_csv('Input Data Files/PV_Energy_Profile.csv')
#     market_data = pd.read_csv('Input Data Files/Market_Price.csv')
#     constants = pd.read_csv('Input Data Files/Constants_Plant.csv')
#     lcoe_pv_data = pd.read_csv('Input Data Files/PV_LCOE.csv')
#     lcoe_bess_data = pd.read_csv('Input Data Files/BESS_LCOE.csv')
#     # Extract data (assuming formats as above)
#     pv_power = pv_data['PV_Power'].values
#     consumer_demand = constants[constants['Parameter'] == 'Consumer_Demand']['Value'].values
#     grid_buy_price = market_data['Grid_Buy_Price'].values
#     grid_sell_price = market_data['Grid_Sell_Price'].values
#     lcoe_pv = lcoe_pv_data['LCOE_PV'].iloc[0]
#     lcoe_bess = lcoe_bess_data['LCOE_BESS'].iloc[0]
#     bess_capacity = float(constants[constants['Parameter'] == 'BESS_Capacity']['Value'])
#     bess_power_limit = float(constants[constants['Parameter'] == 'BESS_Power_Limit']['Value'])
#     eta_charge = float(constants[constants['Parameter'] == 'BESS_Efficiency_Charge']['Value'])
#     eta_discharge = float(constants[constants['Parameter'] == 'BESS_Efficiency_Discharge']['Value'])
#     soc_initial = float(constants[constants['Parameter'] == 'SOC_Initial']['Value'])
#     pi_consumer = float(constants[constants['Parameter'] == 'Consumer_Price']['Value'])
#     return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
#             lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
#             eta_charge, eta_discharge, soc_initial, pi_consumer)

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

# Constraints
# 1. Consumer energy balance
for t in time_indices:
    prob += P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] == consumer_demand[t]

# 2. PV power allocation
for t in time_indices:
    prob += P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power[t]

# 3. BESS power limits
for t in time_indices:
    prob += P_PV_BESS[t] + P_grid_BESS[t] <= bess_power_limit  # Charging limit
    prob += P_BESS_consumer[t] + P_BESS_grid[t] <= bess_power_limit  # Discharging limit

# 4. BESS SOC dynamics
prob += SOC[0] == soc_initial  # Initial SOC
for t in range(n_steps):
    prob += SOC[t + 1] == SOC[t] + eta_charge * (P_PV_BESS[t] + P_grid_BESS[t]) * delta_t - \
            (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t

# Objective function: Maximize revenue
prob += (lpSum([consumer_demand[t] * pi_consumer * delta_t for t in time_indices]) +
         lpSum([(P_PV_grid[t] + P_BESS_grid[t]) * grid_sell_price[t] * delta_t for t in time_indices]) -
         lpSum([(P_grid_consumer[t] + P_grid_BESS[t]) * grid_buy_price[t] * delta_t for t in time_indices]) -
         lpSum([pv_power[t] * lcoe_pv * delta_t for t in time_indices]) -
         lpSum([(P_BESS_consumer[t] + P_BESS_grid[t]) * lcoe_bess * delta_t for t in time_indices]))

# Solve the problem
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

    # Compute BESS charge and discharge powers
    P_BESS_charge = [P_PV_BESS_vals[t] + P_grid_BESS_vals[t] for t in time_indices]
    P_BESS_discharge = [P_BESS_consumer_vals[t] + P_BESS_grid_vals[t] for t in time_indices]

    # Compute revenue per time step
    revenue_per_step = []
    for t in time_indices:
        rev_consumer = consumer_demand[t] * pi_consumer * delta_t
        rev_grid = (P_PV_grid_vals[t] + P_BESS_grid_vals[t]) * grid_sell_price[t] * delta_t
        cost_grid = (P_grid_consumer_vals[t] + P_grid_BESS_vals[t]) * grid_buy_price[t] * delta_t
        cost_pv = pv_power[t] * lcoe_pv * delta_t
        cost_bess = P_BESS_discharge[t] * lcoe_bess * delta_t
        net_rev = rev_consumer + rev_grid - cost_grid - cost_pv - cost_bess
        revenue_per_step.append(net_rev)

    total_revenue = sum(revenue_per_step)
    print(f"Total Revenue: ${total_revenue:.2f}")

    # Plotting critical parameters
    plt.figure(figsize=(12, 10))

    # BESS Power Flows
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, P_BESS_charge, label='BESS Charge (kW)', color='blue')
    plt.plot(time_steps, P_BESS_discharge, label='BESS Discharge (kW)', color='red')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('BESS Power Flows')
    plt.legend()
    plt.grid(True)

    # BESS State of Charge
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, 24.25, 0.25), SOC_vals, label='SOC (kWh)', color='green')
    plt.xlabel('Time (hours)')
    plt.ylabel('SOC (kWh)')
    plt.title('BESS State of Charge')
    plt.legend()
    plt.grid(True)

    # Revenue over Time
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, revenue_per_step, label='Net Revenue ($)', color='purple')
    plt.xlabel('Time (hours)')
    plt.ylabel('Revenue ($)')
    plt.title('Net Revenue over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('optimization_results_plots.png') # Or specific names for each subplot
    # plt.show() # Remove or comment this out for server-side execution
else:
    print("Optimization did not converge to an optimal solution.")
