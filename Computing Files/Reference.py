import load_data
import post_process
import plots
import numpy as np
import os

# Load input data for the reference case (no scaling, no BESS, no EVs)
data = load_data.load(reference_case=True)

# Ensure no EVs in reference case
data['ev_demand'] = np.zeros(data['n_steps'])

# Verify required data fields
required_fields = ['pv_power', 'consumer_demand', 'grid_buy_price', 'grid_sell_price']
for field in required_fields:
    if field not in data:
        raise KeyError(f"Required field '{field}' missing from data")

pv_power = data['pv_power']
consumer_demand = data['consumer_demand']

# Initialize result arrays (no BESS, no EVs)
P_PV_consumer_vals = np.zeros(data['n_steps'])
P_PV_grid_vals = np.zeros(data['n_steps'])
P_grid_consumer_vals = np.zeros(data['n_steps'])
SOC_vals = np.zeros(data['n_steps'] + 1)  # remains zero

# Additional zero arrays for consistency with other cases
P_PV_EV_vals = np.zeros(data['n_steps'])
P_BESS_EV_vals = np.zeros(data['n_steps'])
P_grid_EV_vals = np.zeros(data['n_steps'])
P_BESS_consumer_vals = np.zeros(data['n_steps'])
P_PV_BESS_vals = np.zeros(data['n_steps'])
P_grid_BESS_vals = np.zeros(data['n_steps'])
P_BESS_grid_vals = np.zeros(data['n_steps'])
P_BESS_charge_vals = np.zeros(data['n_steps'])
P_BESS_discharge_vals = np.zeros(data['n_steps'])

total_net_per_step = np.zeros(data['n_steps'])

# Simulate energy cascade: PV to consumer, excess PV to grid, deficit from grid
for t in range(data['n_steps']):
    pv_t = pv_power[t]
    demand_t = consumer_demand[t]
    buy_t = data['grid_buy_price'][t]
    sell_t = data['grid_sell_price'][t]

    # PV used for consumer demand first
    used_pv_consumer = min(pv_t, demand_t)
    surplus_pv = max(pv_t - used_pv_consumer, 0)
    remaining_consumer_demand = max(demand_t - used_pv_consumer, 0)

    # Assign values
    P_PV_consumer_vals[t] = used_pv_consumer
    P_PV_grid_vals[t] = surplus_pv
    P_grid_consumer_vals[t] = remaining_consumer_demand

    # Compute revenue for each time step
    pv_grid = surplus_pv * sell_t * data['delta_t']
    grid_cost = - remaining_consumer_demand * buy_t * data['delta_t']
    total_net_per_step[t] = pv_grid + grid_cost

# Compile results dictionary for plotting and post-processing
results = {
    'P_PV_gen': pv_power,  # Add this to enable correct flow prioritization in post_process
    'P_PV_consumer_vals': P_PV_consumer_vals,
    'P_PV_grid_vals': P_PV_grid_vals,
    'P_grid_consumer_vals': P_grid_consumer_vals,
    'SOC_vals': SOC_vals,
    'P_grid_sold': P_PV_grid_vals,
    'P_grid_bought': P_grid_consumer_vals,
    'P_PV_EV_vals': P_PV_EV_vals,
    'P_BESS_EV_vals': P_BESS_EV_vals,
    'P_grid_EV_vals': P_grid_EV_vals,
    'P_BESS_consumer_vals': P_BESS_consumer_vals,
    'P_PV_BESS_vals': P_PV_BESS_vals,
    'P_grid_BESS_vals': P_grid_BESS_vals,
    'P_BESS_grid_vals': P_BESS_grid_vals,
    'P_BESS_charge_vals': P_BESS_charge_vals,
    'P_BESS_discharge_vals': P_BESS_discharge_vals
}

# Compute revenues and print summary results
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)

# Generate plots with _Reference suffix
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(7)]
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
os.makedirs(output_dir, exist_ok=True)
data['plot_suffix'] = '_Reference'
plots.plot_energy_flows(results, data, revenues, save_dir=output_dir)
plots.plot_financials(revenues, data, save_dir=output_dir)