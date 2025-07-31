# Main optimization script for energy system simulation.
# Loads input data, runs MPC optimization, collects results, and generates plots.

from mpc import MPC
from load_data import load_constants
import load_data
import numpy as np
import post_process
import plots
import sys
import os

# Print Python executable path for debugging environment issues
print(sys.executable)

# Load all input data (from CSV or API, depending on load_data settings)
data = load_data.load()

# Debug flag for verbose output
DEBUG = True

# Validate input data: ensure all required keys are present
required_data_keys = [
    'pv_power', 'consumer_demand', 'ev_demand', 'grid_buy_price',
    'grid_sell_price', 'pi_ev', 'lcoe_pv', 'n_steps', 'delta_t'
]
missing_keys = [key for key in required_data_keys if key not in data]
if missing_keys:
    raise KeyError(f"Missing required keys in data: {missing_keys}")

if DEBUG:
    print("Data validation passed")
    print(f"Number of timesteps: {data['n_steps']}")
    print(f"EV price loaded: {data.get('pi_ev', 'MISSING')}")

# Assert consistency of data length for all main time series
assert len(data['pv_power']) == data['n_steps']
assert len(data['consumer_demand']) == data['n_steps']
assert len(data['ev_demand']) == data['n_steps']
assert len(data['grid_buy_price']) == data['n_steps']
assert len(data['grid_sell_price']) == data['n_steps']

# MPC Parameters
horizon = 672  # Forecast horizon: 7 days (15-min steps)
mpc_controller = MPC(delta_t=data['delta_t'])

# Initialize arrays for storing results
n_steps = data['n_steps']
data['time_steps'] = np.arange(n_steps)
soc_actual = np.zeros(n_steps + 1)
soc_actual[0] = data['soc_initial']
P_PV_consumer_vals = np.zeros(n_steps)
P_PV_ev_vals = np.zeros(n_steps)
P_PV_grid_vals = np.zeros(n_steps)
P_BESS_discharge_vals = np.zeros(n_steps)
P_BESS_charge_vals = np.zeros(n_steps)
P_grid_consumer_vals = np.zeros(n_steps)
P_grid_ev_vals = np.zeros(n_steps)
P_Grid_to_BESS_vals = np.zeros(n_steps)
P_grid_import_vals = np.zeros(n_steps)
P_grid_export_vals = np.zeros(n_steps)
P_PV_gen = np.zeros(n_steps)  # PV generation profile

# Helper function: pad forecast arrays to match horizon length
def pad_to_horizon(arr, horizon, pad_value=None):
    if len(arr) < horizon:
        if len(arr) > 0:
            return np.pad(arr, (0, horizon - len(arr)), mode='edge')
        else:
            return np.zeros(horizon)
    return arr[:horizon]

# Main MPC loop: run optimization for each timestep
for t in range(n_steps):
    pv_forecast = pad_to_horizon(data['pv_power'][t:t + horizon], horizon)
    demand_forecast = pad_to_horizon(data['consumer_demand'][t:t + horizon], horizon)
    ev_forecast = pad_to_horizon(data['ev_demand'][t:t + horizon], horizon)
    buy_forecast = pad_to_horizon(data['grid_buy_price'][t:t + horizon], horizon)
    sell_forecast = pad_to_horizon(data['grid_sell_price'][t:t + horizon], horizon)

    control = mpc_controller.predict(
        soc_actual[t], pv_forecast, demand_forecast, ev_forecast,
        buy_forecast, sell_forecast, data['lcoe_pv'], data['pi_ev'], data['pi_consumer'], horizon
    )

    # Store results for each timestep
    P_PV_consumer_vals[t] = control['pv_bess_to_consumer']
    P_PV_ev_vals[t] = control['pv_bess_to_ev']
    P_PV_grid_vals[t] = control['pv_bess_to_grid']
    P_BESS_discharge_vals[t] = control['P_BESS_discharge']
    P_BESS_charge_vals[t] = control['P_BESS_charge']
    P_grid_consumer_vals[t] = control['grid_to_consumer']
    P_grid_ev_vals[t] = control['grid_to_ev']
    P_Grid_to_BESS_vals[t] = control['P_grid_to_bess']
    P_grid_import_vals[t] = control['P_grid_import']
    P_grid_export_vals[t] = control['P_grid_export']
    soc_actual[t + 1] = control['SOC_next']
    P_PV_gen[t] = control['P_PV_gen']

# Compile results for post-processing and plotting
results = {
    'P_PV_consumer_vals': P_PV_consumer_vals,
    'P_PV_ev_vals': P_PV_ev_vals,
    'P_PV_grid_vals': P_PV_grid_vals,
    'P_BESS_discharge': P_BESS_discharge_vals,
    'P_BESS_charge': P_BESS_charge_vals,
    'P_grid_consumer_vals': P_grid_consumer_vals,
    'P_grid_ev_vals': P_grid_ev_vals,
    'P_Grid_to_BESS': P_Grid_to_BESS_vals,
    'P_grid_import_vals': P_grid_import_vals,
    'P_grid_export_vals': P_grid_export_vals,
    'SOC_vals': soc_actual,
    'P_PV_gen': P_PV_gen
}

# Compute revenues and print summary results
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)

# Prepare output directory for plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
os.makedirs(output_dir, exist_ok=True)
data['plot_suffix'] = ''  # No suffix for main optimization

# Prepare day labels for plotting (for x-axis ticks)
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(7)]

# Generate plots for energy flows and financials
plots.plot_energy_flows(results, data, revenues, save_dir=output_dir)
plots.plot_financials(revenues, data, save_dir=output_dir)