import load_data
import post_process
import plots
import mpc
import sys
import numpy as np
import os

print(sys.executable)

# Load all input data
data = load_data.load()

# Debug flag
DEBUG = True

# Validate input data
required_data_keys = ['pv_power', 'consumer_demand', 'ev_demand', 'grid_buy_price', 
                     'grid_sell_price', 'pi_ev', 'lcoe_pv', 'n_steps', 'delta_t']
missing_keys = [key for key in required_data_keys if key not in data]
if missing_keys:
    raise KeyError(f"Missing required keys in data: {missing_keys}")

if DEBUG:
    print("Data validation passed")
    print(f"Number of timesteps: {data['n_steps']}")
    print(f"EV price loaded: {data.get('pi_ev', 'MISSING')}")

# Assert consistency of data length
assert len(data['pv_power']) == data['n_steps']
assert len(data['consumer_demand']) == data['n_steps']
assert len(data['ev_demand']) == data['n_steps']
assert len(data['grid_buy_price']) == data['n_steps']
assert len(data['grid_sell_price']) == data['n_steps']

# MPC Parameters
horizon = 30  # 6 hours = 30 x 15-min steps
mpc_controller = mpc.MPC(
    data['bess_capacity'], data['bess_power_limit'], data['eta_charge'],
    data['eta_discharge'], data['lcoe_bess'], data['soc_initial'], data['delta_t']
)

# Initialize arrays including EV-related variables
soc_actual = np.zeros(data['n_steps'] + 1)
soc_actual[0] = data['soc_initial']
P_BESS_vals = np.zeros(data['n_steps'])
P_BESS_discharge_vals = np.zeros(data['n_steps'])
P_BESS_charge_vals = np.zeros(data['n_steps'])
P_PV_gen_vals = np.zeros(data['n_steps'])
P_link_dc_to_ac_vals = np.zeros(data['n_steps'])
P_grid_import_vals = np.zeros(data['n_steps'])
P_grid_export_vals = np.zeros(data['n_steps'])
P_grid_consumer_vals = np.zeros(data['n_steps'])
P_grid_ev_vals = np.zeros(data['n_steps'])
# Old-style arrays for fallback and post-processing
P_PV_consumer_vals = np.zeros(data['n_steps'])
P_PV_ev_vals = np.zeros(data['n_steps'])
P_PV_BESS_vals = np.zeros(data['n_steps'])
P_PV_grid_vals = np.zeros(data['n_steps'])
P_BESS_consumer_vals = np.zeros(data['n_steps'])
P_BESS_ev_vals = np.zeros(data['n_steps'])
P_BESS_grid_vals = np.zeros(data['n_steps'])
P_grid_BESS_vals = np.zeros(data['n_steps'])

# Define forecast padding helper
def pad_to_horizon(arr, horizon, pad_value=None):
    if len(arr) < horizon:
        if pad_value is None:
            pad_value = np.mean(arr) if len(arr) > 0 else 0
        return np.pad(arr, (0, horizon - len(arr)), mode='constant', constant_values=pad_value)
    return arr[:horizon]

# MPC loop
for t in range(data['n_steps']):
    pv_forecast = pad_to_horizon(data['pv_power'][t:t + horizon], horizon)
    demand_forecast = pad_to_horizon(data['consumer_demand'][t:t + horizon], horizon)
    ev_forecast = pad_to_horizon(data['ev_demand'][t:t + horizon], horizon)
    buy_forecast = pad_to_horizon(data['grid_buy_price'][t:t + horizon], horizon)
    sell_forecast = pad_to_horizon(data['grid_sell_price'][t:t + horizon], horizon)

    control = mpc_controller.predict(
        soc_actual[t], pv_forecast, demand_forecast, ev_forecast,
        buy_forecast, sell_forecast, data['lcoe_pv'], data['pi_ev'], horizon
    )

    if control is None:
        print(f"MPC infeasible at t={t}; using fallback (no BESS action).")
        pv_t = data['pv_power'][t]
        demand_t = data['consumer_demand'][t]
        ev_t = data['ev_demand'][t]

        # Simple fallback strategy
        P_PV_consumer_vals[t] = min(pv_t, demand_t)
        P_PV_ev_vals[t] = min(max(0, pv_t - P_PV_consumer_vals[t]), ev_t)
        P_PV_grid_vals[t] = max(0, pv_t - P_PV_consumer_vals[t] - P_PV_ev_vals[t])
        P_BESS_consumer_vals[t] = 0
        P_BESS_ev_vals[t] = 0
        P_BESS_grid_vals[t] = 0
        P_grid_consumer_vals[t] = max(0, demand_t - P_PV_consumer_vals[t])
        P_grid_ev_vals[t] = max(0, ev_t - P_PV_ev_vals[t])
        P_grid_BESS_vals[t] = 0
        soc_actual[t + 1] = soc_actual[t]
    else:
        P_BESS_vals[t] = control['P_BESS']
        P_BESS_discharge_vals[t] = control['P_BESS_discharge']
        P_BESS_charge_vals[t] = control['P_BESS_charge']
        P_PV_gen_vals[t] = control['P_PV_gen']
        P_link_dc_to_ac_vals[t] = control['P_link_dc_to_ac']
        P_grid_import_vals[t] = control['P_grid_import']
        P_grid_export_vals[t] = control['P_grid_export']
        # Use new keys for physical flows
        P_PV_consumer_vals[t] = control['pv_bess_to_consumer']
        P_PV_ev_vals[t] = control['pv_bess_to_ev']
        P_PV_grid_vals[t] = control['pv_bess_to_grid']
        P_grid_consumer_vals[t] = control['grid_to_consumer']
        P_grid_ev_vals[t] = control['grid_to_ev']
        soc_actual[t + 1] = control['SOC_next']

# After MPC computation, map new flows to old-style arrays for post-processing
P_BESS_grid_vals = P_BESS_discharge_vals
P_BESS_consumer_vals = np.zeros(data['n_steps'])
P_BESS_ev_vals = np.zeros(data['n_steps'])

# Compile results
slack_vals = np.zeros(data['n_steps'])
results = {
    'slack_vals': slack_vals,
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
    'SOC_vals': soc_actual,
    'P_BESS_charge': P_BESS_charge_vals,
    'P_BESS_discharge': P_BESS_discharge_vals,
    'P_grid_sold': P_grid_export_vals,
    'P_grid_bought': P_grid_import_vals
}
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)

# Add suffix to distinguish plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
os.makedirs(output_dir, exist_ok=True)
data['plot_suffix'] = ''  # No suffix for main optimization

# Prepare day labels for plotting (must be before plotting calls)
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(7)]

plots.plot_energy_flows(results, data, revenues, save_dir=output_dir)
plots.plot_financials(revenues, data, save_dir=output_dir)