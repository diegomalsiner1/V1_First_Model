import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

print("--- Running the REAL Optimization Script with CVXPY ---")

# Time framework: 24 hours, 15-minute intervals (96 steps)
time_steps = np.arange(0, 24, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours

def load_data():
    # Debug: Print file content
    with open('../Input Data Files/Constants_Plant.csv', 'r') as file:
        print("Constants_Plant.csv content:", file.read())
    # Load LCOE for PV from PV_LCOE.csv, ignoring comment lines
    pv_lcoe_data = pd.read_csv('../Input Data Files/PV_LCOE.csv', comment='#')
    lcoe_pv = pv_lcoe_data['LCOE_PV'].iloc[0]  # 0.055 EUR/kWh

    # Load LCOE for BESS from BESS_LCOE.csv, ignoring comment lines
    bess_lcoe_data = pd.read_csv('../Input Data Files/BESS_LCOE.csv', comment='#')
    lcoe_bess = bess_lcoe_data['LCOE_BESS'].iloc[0]  # 0.08 EUR/kWh

    # Load constants from Constants_Plant.csv, ignoring comment lines
    constants_data = pd.read_csv('../Input Data Files/Constants_Plant.csv', comment='#')
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])  # 4000 kWh
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])  # 100 kW
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])  # 0.984
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])  # 0.984
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])  # 2000 kWh
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])  # 0.12 EUR/kWh

    # Sample PV power profile (kW): sinusoidal daytime generation
    pv_power = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        if 5 <= t <= 19:
            pv_power[i] = 1327 * np.sin(np.pi * (t - 6) / 12)

    # Sample consumer demand (kW): constant load of 200 kW with 100 kW step from 8 AM to 6 PM
    consumer_demand = np.full(n_steps, 200.0)  # Baseline constant load of 200 kW
    for i, t in enumerate(time_steps):
        if 8 <= t <= 18:  # 8:00 to 18:00
            consumer_demand[i] += 100.0  # Reduced to 100 kW step

    # Sample grid prices ($/kWh): higher buy price during peak hours
    grid_buy_price = np.full(n_steps, 0.15)
    grid_sell_price = np.full(n_steps, 0.08)
    for i, t in enumerate(time_steps):
        if 17 <= t <= 21:
            grid_buy_price[i] = 0.12

    return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
            lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
            eta_charge, eta_discharge, soc_initial, pi_consumer)

# Load data
(pv_power, consumer_demand, grid_buy_price, grid_sell_price,
 lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
 eta_charge, eta_discharge, soc_initial, pi_consumer) = load_data()

# Decision variables
P_PV_consumer = cp.Variable(n_steps)
P_PV_BESS = cp.Variable(n_steps)
P_PV_grid = cp.Variable(n_steps)
P_BESS_consumer = cp.Variable(n_steps)
P_BESS_grid = cp.Variable(n_steps)
P_grid_consumer = cp.Variable(n_steps)
P_grid_BESS = cp.Variable(n_steps)  # Allow negative for grid to BESS
SOC = cp.Variable(n_steps + 1)

# Constraints
constraints = []

# 1. Consumer energy balance
for t in range(n_steps):
    constraints.append(P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] == consumer_demand[t])

# 2. PV power allocation
for t in range(n_steps):
    constraints.append(P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power[t])

# 3. BESS power limits (corrected for bidirectional flow)
bess_power_limit_pos = bess_power_limit  # Positive limit for discharge
bess_power_limit_neg = -bess_power_limit  # Negative limit for charge
for t in range(n_steps):
    constraints.append(P_PV_BESS[t] + P_grid_BESS[t] <= bess_power_limit_neg)  # Charging limit (negative power)
    constraints.append(P_BESS_consumer[t] + P_BESS_grid[t] >= bess_power_limit_pos)  # Discharging limit (positive power)
    constraints.append(P_PV_BESS[t] + P_grid_BESS[t] >= 0)  # Ensure charging is non-positive
    constraints.append(P_BESS_consumer[t] + P_BESS_grid[t] <= 0)  # Ensure discharging is non-negative

# 4. BESS SOC dynamics
constraints.append(SOC[0] == soc_initial)  # Initial SOC
for t in range(n_steps):
    constraints.append(SOC[t + 1] == SOC[t] + eta_charge * (P_PV_BESS[t] + P_grid_BESS[t]) * delta_t -
                      (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t)

# 5. SOC bounds
for t in range(n_steps + 1):
    constraints.append(SOC[t] >= 0)
    constraints.append(SOC[t] <= bess_capacity)

# 6. Relaxed non-negativity constraints
for t in range(n_steps):
    constraints.append(P_PV_consumer[t] >= 0)
    constraints.append(P_PV_BESS[t] >= 0)
    constraints.append(P_PV_grid[t] >= 0)
    constraints.append(P_BESS_consumer[t] >= 0)
    # Allow P_BESS_grid and P_grid_BESS to be negative for grid charging
    constraints.append(P_grid_consumer[t] >= 0)

# Objective function: Maximize revenue (corrected)
revenue = cp.sum([(P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t]) * pi_consumer * delta_t for t in range(n_steps)]) + \
          cp.sum([(P_PV_grid[t] + P_BESS_grid[t]) * grid_sell_price[t] * delta_t for t in range(n_steps)]) - \
          cp.sum([(P_grid_consumer[t] + P_grid_BESS[t]) * grid_buy_price[t] * delta_t for t in range(n_steps)]) - \
          cp.sum([pv_power[t] * lcoe_pv * delta_t for t in range(n_steps)]) - \
          cp.sum([(P_BESS_consumer[t] + P_BESS_grid[t]) * lcoe_bess * delta_t for t in range(n_steps)])

# Define and solve the problem
prob = cp.Problem(cp.Maximize(revenue), constraints)
prob.solve(solver=cp.ECOS, verbose=True)

# Output results (same as original)
if prob.status == cp.OPTIMAL:
    # Extract variable values
    P_PV_consumer_vals = P_PV_consumer.value
    P_PV_BESS_vals = P_PV_BESS.value
    P_PV_grid_vals = P_PV_grid.value
    P_BESS_consumer_vals = P_BESS_consumer.value
    P_BESS_grid_vals = P_BESS_grid.value
    P_grid_consumer_vals = P_grid_consumer.value
    P_grid_BESS_vals = P_grid_BESS.value
    SOC_vals = SOC.value

    # Compute BESS charge and discharge powers
    P_BESS_charge = -cp.minimum(P_PV_BESS_vals + P_grid_BESS_vals, 0)  # Negative values indicate charging
    P_BESS_discharge = cp.maximum(P_BESS_consumer_vals + P_BESS_grid_vals, 0)  # Positive values indicate discharging

    # Compute Grid sold and bought powers
    P_grid_sold = cp.maximum(P_PV_grid_vals + P_BESS_grid_vals, 0)
    P_grid_bought = -cp.minimum(P_grid_consumer_vals + P_grid_BESS_vals, 0)

    # Compute revenue per time step
    revenue_per_step = []
    for t in range(n_steps):
        rev_consumer = (P_PV_consumer_vals[t] + P_BESS_consumer_vals[t] + P_grid_consumer_vals[t]) * pi_consumer * delta_t
        rev_grid = P_grid_sold[t] * grid_sell_price[t] * delta_t
        cost_grid = P_grid_bought[t] * grid_buy_price[t] * delta_t
        cost_pv = pv_power[t] * lcoe_pv * delta_t
        cost_bess = P_BESS_discharge[t] * lcoe_bess * delta_t
        net_rev = rev_consumer + rev_grid - cost_grid - cost_pv - cost_bess
        revenue_per_step.append(net_rev)

    total_revenue = sum(revenue_per_step)
    print(f"Total Revenue: ${total_revenue:.2f}")

    # Plotting (same as original, adjusted for new variables)
    plt.figure(figsize=(12, 20))

    plt.subplot(5, 1, 1)
    plt.plot(time_steps, pv_power, label='PV Power (kW)', color='orange')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('PV Power Production Profile')
    plt.legend()
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.stackplot(time_steps, P_PV_consumer_vals, P_BESS_consumer_vals, P_grid_consumer_vals,
                  labels=['PV to Consumer', 'BESS to Consumer', 'Grid to Consumer'],
                  colors=['orange', 'green', 'blue'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Consumer Energy Composition')
    plt.legend(loc='upper left')
    plt.grid(True)

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

    plt.subplot(5, 1, 4)
    plt.plot(time_steps, P_grid_sold, label='Grid Sold (kW)', color='orange')
    plt.plot(time_steps, P_grid_bought, label='Grid Bought (kW)', color='brown')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title('Grid Power Flows')
    plt.legend()
    plt.grid(True)

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
    plt.savefig('optimization_results_plots.png')
else:
    print(f"Optimization failed with status: {prob.status}")
