import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

print("--- Running the REAL Optimization Script with CVXPY ---")

# Time framework: 24 hours, 15-minute intervals (96 steps)
time_steps = np.arange(0, 24, 0.25)
n_steps = len(time_steps)
delta_t = 0.25  # hours

def load_data():
    # Load LCOE for PV from PV_LCOE.csv, ignoring comment lines
    # Using absolute paths as per your request
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
            pv_power[i] = 200 * np.sin(np.pi * (t - 6) / 12)

    # Sample consumer demand (kW): constant load of 200 kW with 100 kW step from 8 AM to 6 PM
    consumer_demand = np.full(n_steps, 100.0)
    for i, t in enumerate(time_steps):
        if 8 <= t <= 18:
            consumer_demand[i] += 100.0

    # Sample grid prices ($/kWh): higher buy price during peak hours
    grid_buy_price = np.full(n_steps, 0.12)
    grid_sell_price = np.full(n_steps, 0.12)
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

# Convert data to CVXPY parameters
pv_power_param = cp.Parameter(n_steps, value=pv_power)
consumer_demand_param = cp.Parameter(n_steps, value=consumer_demand)
grid_buy_price_param = cp.Parameter(n_steps, value=grid_buy_price)
grid_sell_price_param = cp.Parameter(n_steps, value=grid_sell_price)
lcoe_pv_param = cp.Parameter(value=lcoe_pv)
lcoe_bess_param = cp.Parameter(value=lcoe_bess)
bess_capacity_param = cp.Parameter(value=bess_capacity)
bess_power_limit_param = cp.Parameter(value=bess_power_limit)
eta_charge_param = cp.Parameter(value=eta_charge)
eta_discharge_param = cp.Parameter(value=eta_discharge)
soc_initial_param = cp.Parameter(value=soc_initial)
pi_consumer_param = cp.Parameter(value=pi_consumer)

# Decision variables (all non-negative)
P_PV_consumer = cp.Variable(n_steps, nonneg=True)
P_PV_BESS = cp.Variable(n_steps, nonneg=True)
P_PV_grid = cp.Variable(n_steps, nonneg=True)
P_BESS_consumer = cp.Variable(n_steps, nonneg=True)
P_BESS_grid = cp.Variable(n_steps, nonneg=True)
P_grid_consumer = cp.Variable(n_steps, nonneg=True)
P_grid_BESS = cp.Variable(n_steps, nonneg=True) # Power from grid to BESS
SOC = cp.Variable(n_steps + 1, nonneg=True)

# New variables for total BESS charge/discharge to simplify SOC dynamics and enforce limits
P_BESS_charge_total = cp.Variable(n_steps, nonneg=True)
P_BESS_discharge_total = cp.Variable(n_steps, nonneg=True)

# Constraints
constraints = []

# 1. Consumer energy balance
for t in range(n_steps):
    constraints.append(P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] == consumer_demand_param[t])

# 2. PV power allocation
for t in range(n_steps):
    constraints.append(P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power_param[t])

# 3. BESS power limits and definition of total charge/discharge
for t in range(n_steps):
    # Total power flowing INTO BESS
    constraints.append(P_PV_BESS[t] + P_grid_BESS[t] == P_BESS_charge_total[t])
    # Total power flowing OUT OF BESS
    constraints.append(P_BESS_consumer[t] + P_BESS_grid[t] == P_BESS_discharge_total[t])

    # BESS power limits
    constraints.append(P_BESS_charge_total[t] <= bess_power_limit_param)
    constraints.append(P_BESS_discharge_total[t] <= bess_power_limit_param)

# 4. BESS SOC dynamics
constraints.append(SOC[0] == soc_initial_param)  # Initial SOC
for t in range(n_steps):
    constraints.append(SOC[t + 1] == SOC[t] + cp.multiply(eta_charge_param, P_BESS_charge_total[t]) * delta_t - \
                                    cp.multiply(1 / eta_discharge_param, P_BESS_discharge_total[t]) * delta_t)

# 5. SOC bounds
for t in range(n_steps + 1):
    constraints.append(SOC[t] >= 0)
    constraints.append(SOC[t] <= bess_capacity_param)

# Objective function: Maximize revenue
# Revenue from consumer, grid sales, minus costs of grid purchase, PV, and BESS
revenue = cp.sum(cp.multiply(consumer_demand_param, pi_consumer_param)) * delta_t + \
          cp.sum(cp.multiply((P_PV_grid + P_BESS_grid), grid_sell_price_param)) * delta_t - \
          cp.sum(cp.multiply((P_grid_consumer + P_grid_BESS), grid_buy_price_param)) * delta_t - \
          cp.sum(cp.multiply(pv_power_param, lcoe_pv_param)) * delta_t - \
          cp.sum(cp.multiply(P_BESS_discharge_total, lcoe_bess_param)) * delta_t

# Define and solve the problem
prob = cp.Problem(cp.Maximize(revenue), constraints)
prob.solve(solver=cp.ECOS, verbose=True)

# Output results
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
    P_BESS_charge_total_vals = P_BESS_charge_total.value
    P_BESS_discharge_total_vals = P_BESS_discharge_total.value

    # Compute Grid sold and bought powers for plotting
    P_grid_sold = P_PV_grid_vals + P_BESS_grid_vals
    P_grid_bought = P_grid_consumer_vals + P_grid_BESS_vals

    # Compute revenue per time step for plotting
    revenue_per_step = []
    for t in range(n_steps):
        rev_consumer = consumer_demand[t] * pi_consumer_param.value * delta_t
        rev_grid = P_grid_sold[t] * grid_sell_price_param.value[t] * delta_t
        cost_grid = P_grid_bought[t] * grid_buy_price_param.value[t] * delta_t
        cost_pv = pv_power_param.value[t] * lcoe_pv_param.value * delta_t
        cost_bess = P_BESS_discharge_total_vals[t] * lcoe_bess_param.value * delta_t
        net_rev = rev_consumer + rev_grid - cost_grid - cost_pv - cost_bess
        revenue_per_step.append(net_rev)

    total_revenue = sum(revenue_per_step)
    print(f"Total Revenue: ${total_revenue:.2f}")

    # Plotting critical parameters
    plt.figure(figsize=(12, 20))  # Increased height for 5 subplots

    # Graph 1: PV Production
    plt.subplot(5, 1, 1)
    plt.plot(time_steps, pv_power_param.value, label='PV Power (kW)', color='orange')
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
    ax1.plot(time_steps, P_BESS_charge_total_vals, label='BESS Charge (kW)', color='blue')
    ax1.plot(time_steps, P_BESS_discharge_total_vals, label='BESS Discharge (kW)', color='red')
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
    plt.plot(time_steps, grid_buy_price_param.value, label='Grid Buy Price ($/kWh)', color='blue')
    plt.plot(time_steps, grid_sell_price_param.value, label='Grid Sell Price ($/kWh)', color='green')
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
    print(f"Optimization failed with status: {prob.status}")

