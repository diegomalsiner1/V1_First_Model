import load_data
import post_process
import plots
import numpy as np
import os

# Load input data
data = load_data.load()

# Verify required data fields
required_fields = ['ev_demand', 'pi_ev', 'pv_power', 'consumer_demand', 'grid_buy_price', 'grid_sell_price']
for field in required_fields:
    if field not in data:
        raise KeyError(f"Required field '{field}' missing from data")

# Initialize result arrays (no BESS in this case)
P_PV_consumer_vals = np.zeros(data['n_steps'])
P_PV_ev_vals = np.zeros(data['n_steps'])  # New EV arrays
P_PV_grid_vals = np.zeros(data['n_steps'])
P_grid_consumer_vals = np.zeros(data['n_steps'])
P_grid_ev_vals = np.zeros(data['n_steps'])  # New EV arrays
slack_vals = np.zeros(data['n_steps'])
SOC_vals = np.zeros(data['n_steps'] + 1)  # remains zero
P_PV_BESS_vals = np.zeros(data['n_steps'])
P_BESS_consumer_vals = np.zeros(data['n_steps'])
P_BESS_ev_vals = np.zeros(data['n_steps'])  # New EV arrays
P_BESS_grid_vals = np.zeros(data['n_steps'])
P_grid_BESS_vals = np.zeros(data['n_steps'])

total_net_per_step = np.zeros(data['n_steps'])

# Simulate without BESS
for t in range(data['n_steps']):
    pv_t = data['pv_power'][t]
    demand_t = data['consumer_demand'][t]
    ev_demand_t = data['ev_demand'][t]  # Get EV demand
    buy_t = data['grid_buy_price'][t]
    sell_t = data['grid_sell_price'][t]

    # Handle consumer demand first
    used_pv_consumer = min(pv_t, demand_t)
    remaining_pv = pv_t - used_pv_consumer
    
    # Then handle EV demand with remaining PV
    used_pv_ev = min(remaining_pv, ev_demand_t)
    surplus_pv = max(remaining_pv - used_pv_ev, 0)
    
    remaining_consumer_demand = max(demand_t - used_pv_consumer, 0)
    remaining_ev_demand = max(ev_demand_t - used_pv_ev, 0)

    # Assign values
    P_PV_consumer_vals[t] = used_pv_consumer
    P_PV_ev_vals[t] = used_pv_ev
    P_PV_grid_vals[t] = surplus_pv
    P_grid_consumer_vals[t] = remaining_consumer_demand
    P_grid_ev_vals[t] = remaining_ev_demand

    # Compute revenue for each time step
    pv_cons = used_pv_consumer * (buy_t - data['lcoe_pv']) * data['delta_t']
    pv_ev = used_pv_ev * (data['pi_ev'] - data['lcoe_pv']) * data['delta_t']  # EV charging revenue
    pv_grid = surplus_pv * (sell_t - data['lcoe_pv']) * data['delta_t']
    grid_cost = -(remaining_consumer_demand * buy_t + remaining_ev_demand * buy_t) * data['delta_t']
    ev_revenue = remaining_ev_demand * data['pi_ev'] * data['delta_t']  # Revenue from EV charging from grid
    total_net_per_step[t] = pv_cons + pv_ev + pv_grid + grid_cost + ev_revenue

# Update results dictionary with EV values
results = {
    'P_PV_consumer_vals': P_PV_consumer_vals,
    'P_PV_ev_vals': P_PV_ev_vals,
    'P_PV_BESS_vals': P_PV_BESS_vals,
    'P_PV_grid_vals': P_PV_grid_vals,
    'P_BESS_consumer_vals': P_BESS_consumer_vals,
    'P_BESS_ev_vals': P_BESS_ev_vals,
    'P_BESS_grid_vals': P_BESS_grid_vals,
    'P_grid_consumer_vals': P_grid_consumer_vals,
    'P_grid_ev_vals': P_grid_ev_vals,
    'P_grid_BESS_vals': P_grid_BESS_vals,
    'SOC_vals': SOC_vals,
    'slack_vals': slack_vals,
    'P_BESS_charge': P_PV_BESS_vals + P_grid_BESS_vals,
    'P_BESS_discharge': P_BESS_consumer_vals + P_BESS_ev_vals + P_BESS_grid_vals,
    'P_grid_sold': P_PV_grid_vals + P_BESS_grid_vals,
    'P_grid_bought': P_grid_consumer_vals + P_grid_ev_vals + P_grid_BESS_vals
}

# Compute revenues
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)

# Generate plots
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(7)]

# Add suffix to distinguish plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
os.makedirs(output_dir, exist_ok=True)
data['plot_suffix'] = '_Reference'
plots.plot_energy_flows(results, data, revenues, save_dir=output_dir)
plots.plot_financials(revenues, data, save_dir=output_dir)