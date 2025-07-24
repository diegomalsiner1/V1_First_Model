import load_data
import post_process
import plots
import mpc
import numpy as np
import os

# Load all input data
data = load_data.load()

# Assert consistency of data length
assert len(data['pv_power']) == data['n_steps']
assert len(data['consumer_demand']) == data['n_steps']
assert len(data['grid_buy_price']) == data['n_steps']
assert len(data['grid_sell_price']) == data['n_steps']

# MPC Parameters
horizon = 30  # 6 hours = 30 x 15-min steps
mpc_controller = mpc.MPC(
    data['bess_capacity'], data['bess_power_limit'], data['eta_charge'],
    data['eta_discharge'], data['lcoe_bess'], data['soc_initial'], data['delta_t']
)

# Initialize arrays
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
    buy_forecast = pad_to_horizon(data['grid_buy_price'][t:t + horizon], horizon)
    sell_forecast = pad_to_horizon(data['grid_sell_price'][t:t + horizon], horizon)

    control = mpc_controller.predict(
        soc_actual[t], pv_forecast, demand_forecast,
        buy_forecast, sell_forecast, data['lcoe_pv'], horizon
    )

    if control is None:
        print(f"MPC infeasible at t={t}; using fallback (no BESS action).")
        pv_t = data['pv_power'][t]
        demand_t = data['consumer_demand'][t]
        buy_t = float(data['grid_buy_price'][t])
        sell_t = float(data['grid_sell_price'][t])

        P_PV_consumer_vals[t] = min(pv_t, demand_t)
        P_PV_BESS_vals[t] = 0
        P_PV_grid_vals[t] = max(pv_t - P_PV_consumer_vals[t], 0)
        P_BESS_consumer_vals[t] = 0
        P_BESS_grid_vals[t] = 0
        P_grid_consumer_vals[t] = max(demand_t - P_PV_consumer_vals[t], 0)
        P_grid_BESS_vals[t] = 0
        slack_vals[t] = max(0, demand_t - (P_PV_consumer_vals[t] + P_grid_consumer_vals[t]))
        soc_actual[t + 1] = soc_actual[t]
    else:
        P_PV_consumer_vals[t] = control['P_PV_cons']
        P_PV_BESS_vals[t] = control['P_PV_BESS']
        P_PV_grid_vals[t] = control['P_PV_grid']
        P_BESS_consumer_vals[t] = control['P_BESS_cons']
        P_BESS_grid_vals[t] = control['P_BESS_grid']
        P_grid_consumer_vals[t] = control['P_grid_cons']
        P_grid_BESS_vals[t] = control['P_grid_BESS']
        slack_vals[t] = control['slack']
        soc_actual[t + 1] = control['SOC_next']

# Prepare day labels for plotting

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data['day_labels'] = [days[(data['start_weekday'] + d) % 7] for d in range(8)]

# Add suffix to distinguish plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
os.makedirs(output_dir, exist_ok=True)
data['plot_suffix'] = ''  # No suffix for main optimization

# Compile results
data['plot_suffix'] = ''
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
revenues = post_process.compute_revenues(results, data)
post_process.print_results(revenues, results, data)
plots.plot_energy_flows(results, data, revenues, save_dir=output_dir)
plots.plot_financials(revenues, data, save_dir=output_dir)