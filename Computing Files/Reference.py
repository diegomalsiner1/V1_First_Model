import load_data
import post_process
import plots
import numpy as np

# Load input data
data = load_data.load()

# Initialize result arrays (no BESS in this case)
P_PV_consumer_vals = np.zeros(data['n_steps'])
P_PV_grid_vals = np.zeros(data['n_steps'])
P_grid_consumer_vals = np.zeros(data['n_steps'])
slack_vals = np.zeros(data['n_steps'])
SOC_vals = np.zeros(data['n_steps'] + 1)  # remains zero
P_PV_BESS_vals = np.zeros(data['n_steps'])  # all BESS related flows are zero
P_BESS_consumer_vals = np.zeros(data['n_steps'])
P_BESS_grid_vals = np.zeros(data['n_steps'])
P_grid_BESS_vals = np.zeros(data['n_steps'])

total_net_per_step = np.zeros(data['n_steps'])

# Simulate without BESS
for t in range(data['n_steps']):
    pv_t = data['pv_power'][t]
    demand_t = data['consumer_demand'][t]
    buy_t = data['grid_buy_price'][t]
    sell_t = data['grid_sell_price'][t]

    used_pv = min(pv_t, demand_t)
    surplus_pv = max(pv_t - demand_t, 0)
    remaining_demand = max(demand_t - pv_t, 0)

    P_PV_consumer_vals[t] = used_pv
    P_PV_grid_vals[t] = surplus_pv
    P_grid_consumer_vals[t] = remaining_demand

    # Compute revenue for each time step
    pv_cons = used_pv * (buy_t - data['lcoe_pv']) * data['delta_t']
    pv_grid = surplus_pv * (sell_t - data['lcoe_pv']) * data['delta_t']
    grid_cost = - remaining_demand * buy_t * data['delta_t']
    total_net_per_step[t] = pv_cons + pv_grid + grid_cost

# Construct results dictionary
results = {
    'P_PV_consumer_vals': P_PV_consumer_vals,
    'P_PV_BESS_vals': P_PV_BESS_vals,
    'P_PV_grid_vals': P_PV_grid_vals,
    'P_BESS_consumer_vals': P_BESS_consumer_vals,
    'P_BESS_grid_vals': P_BESS_grid_vals,
    'P_grid_consumer_vals': P_grid_consumer_vals,
    'P_grid_BESS_vals': P_grid_BESS_vals,
    'SOC_vals': SOC_vals,
    'slack_vals': slack_vals,
    'P_BESS_charge': P_PV_BESS_vals + P_grid_BESS_vals,
    'P_BESS_discharge': P_BESS_consumer_vals + P_BESS_grid_vals,
    'P_grid_sold': P_PV_grid_vals + P_BESS_grid_vals,
    'P_grid_bought': P_grid_consumer_vals + P_grid_BESS_vals
}

# Compute revenues
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)

# Generate plots
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(8)]

# Add suffix to distinguish plots
data['plot_suffix'] = '_Reference'
plots.plot_energy_flows(results, data, revenues)
plots.plot_financials(revenues, data)