import load_data
import model
import solve
import post_process
import plots
import mpc  # Import the MPC module
import numpy as np  # For cumsum and other ops in loop

# Load all input data
data = load_data.load()

# MPC Parameters
horizon = 50 # 24 hours at 15-min intervals; adjust as needed
mpc_controller = mpc.MPC(
    data['bess_capacity'], data['bess_power_limit'], data['eta_charge'],
    data['eta_discharge'], data['lcoe_bess'], data['soc_initial'], data['delta_t']
)

# Initialize tracking arrays for simulation results (to mimic 'results' dict for post-processing/plots)
soc_actual = np.zeros(data['n_steps'] + 1)
soc_actual[0] = data['soc_initial']
P_PV_consumer_vals = np.zeros(data['n_steps'])
P_PV_BESS_vals = np.zeros(data['n_steps'])
P_PV_grid_vals = np.zeros(data['n_steps'])
P_BESS_consumer_vals = np.zeros(data['n_steps'])
P_BESS_grid_vals = np.zeros(data['n_steps'])
P_grid_consumer_vals = np.zeros(data['n_steps'])
P_grid_BESS_vals = np.zeros(data['n_steps'])
slack_vals = np.zeros(data['n_steps'])
total_net_per_step = np.zeros(data['n_steps'])  # For revenues

# MPC Simulation Loop
for t in range(data['n_steps']):
    # Get "forecasts" by slicing remaining data (perfect foresight for now; pad if short)
    remaining = data['n_steps'] - t
    pv_forecast = data['pv_power'][t:t+horizon]
    demand_forecast = data['consumer_demand'][t:t+horizon]
    buy_forecast = data['grid_buy_price'][t:t+horizon]
    sell_forecast = data['grid_sell_price'][t:t+horizon]
    if remaining < horizon:
        pv_forecast = np.pad(pv_forecast, (0, horizon - remaining), mode='edge')
        demand_forecast = np.pad(demand_forecast, (0, horizon - remaining), mode='edge')
        buy_forecast = np.pad(buy_forecast, (0, horizon - remaining), mode='mean')
        sell_forecast = np.pad(sell_forecast, (0, horizon - remaining), mode='mean')

    # Run MPC to get control actions for current step
    control = mpc_controller.predict(soc_actual[t], pv_forecast, demand_forecast, buy_forecast, sell_forecast, data['lcoe_pv'], horizon)

    if control is None:
        print(f"MPC infeasible at t={t}; using fallback (no BESS action).")
        # Fallback: No BESS, direct PV to cons/grid, grid for remaining
        pv_t = data['pv_power'][t]
        demand_t = data['consumer_demand'][t]
        buy_t = data['grid_buy_price'][t]
        sell_t = data['grid_sell_price'][t]
        P_PV_consumer_vals[t] = min(pv_t, demand_t)
        P_PV_BESS_vals[t] = 0
        P_PV_grid_vals[t] = pv_t - P_PV_consumer_vals[t]
        P_BESS_consumer_vals[t] = 0
        P_BESS_grid_vals[t] = 0
        P_grid_consumer_vals[t] = demand_t - P_PV_consumer_vals[t]
        P_grid_BESS_vals[t] = 0
        slack_vals[t] = 0
        soc_actual[t+1] = soc_actual[t]  # No change
    else:
        # Use optimized flows from MPC for first step
        P_PV_consumer_vals[t] = control['P_PV_cons']
        P_PV_BESS_vals[t] = control['P_PV_BESS']
        P_PV_grid_vals[t] = control['P_PV_grid']
        P_BESS_consumer_vals[t] = control['P_BESS_cons']
        P_BESS_grid_vals[t] = control['P_BESS_grid']
        P_grid_consumer_vals[t] = control['P_grid_cons']
        P_grid_BESS_vals[t] = control['P_grid_BESS']
        slack_vals[t] = control['slack']
        soc_actual[t+1] = control['SOC_next']

    # Compute revenue for this step using actual applied flows
    buy_t = data['grid_buy_price'][t]
    sell_t = data['grid_sell_price'][t]
    pv_cons = P_PV_consumer_vals[t] * (buy_t - data['lcoe_pv']) * data['delta_t']
    pv_grid = P_PV_grid_vals[t] * (sell_t - data['lcoe_pv']) * data['delta_t']
    pv_bess = P_PV_BESS_vals[t] * (- data['lcoe_pv']) * data['delta_t']
    bess_cons = P_BESS_consumer_vals[t] * (buy_t - data['lcoe_bess']) * data['delta_t']
    bess_grid = P_BESS_grid_vals[t] * (sell_t - data['lcoe_bess']) * data['delta_t']
    grid_cost = - (P_grid_consumer_vals[t] + P_grid_BESS_vals[t]) * buy_t * data['delta_t']
    penalty = -1e5 * slack_vals[t]
    total_net_per_step[t] = pv_cons + pv_grid + pv_bess + bess_cons + bess_grid + grid_cost + penalty

# After loop: Build results dict for post-processing and plots
results = {
    'P_PV_consumer_vals': P_PV_consumer_vals,
    'P_PV_BESS_vals': P_PV_BESS_vals,
    'P_PV_grid_vals': P_PV_grid_vals,
    'P_BESS_consumer_vals': P_BESS_consumer_vals,
    'P_BESS_grid_vals': P_BESS_grid_vals,
    'P_grid_consumer_vals': P_grid_consumer_vals,
    'P_grid_BESS_vals': P_grid_BESS_vals,
    'SOC_vals': soc_actual,
    'slack_vals': slack_vals,
    'P_BESS_charge': P_PV_BESS_vals + P_grid_BESS_vals,
    'P_BESS_discharge': P_BESS_consumer_vals + P_BESS_grid_vals,
    'P_grid_sold': P_PV_grid_vals + P_BESS_grid_vals,
    'P_grid_bought': P_grid_consumer_vals + P_grid_BESS_vals
}

# Process revenues (using post_process)
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)

# Generate plots
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(8)]
plots.plot_energy_flows(results, data, revenues)
plots.plot_financials(revenues, data)