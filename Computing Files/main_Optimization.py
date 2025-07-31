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
import openpyxl
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

#INSERT FROM HERE
# Compute key metrics for LCOE/Financial Model (in kWh, using delta_t in hours)
delta_t = data['delta_t']
total_pv_energy = np.sum(results['P_PV_gen']) * delta_t  # Total PV produced (kWh) over simulation period
total_bess_discharge = np.sum(results['P_BESS_discharge']) * delta_t  # Total BESS provided energy (kWh)
total_grid_sold = np.sum(results['P_grid_sold']) * delta_t  # Grid export (kWh)
total_grid_bought = np.sum(results['P_grid_bought']) * delta_t  # Grid import (kWh)
self_sufficiency = revenues['self_sufficiency']  # % (renewable coverage of consumer demand)
ev_renewable_share = revenues['ev_renewable_share']  # % (renewable coverage of EV demand)
total_revenue = revenues['total_revenue']  # € over simulation period

# Other relevant parameters
bess_capacity = data['bess_capacity']  # kWh
pv_old = float(load_constants()['PV_OLD'])
pv_new = float(load_constants()['PV_NEW'])
pv_scaling_factor = (pv_new + pv_old) / pv_old if pv_old > 0 else 1
simulation_days = 7  # Assuming 7-day simulation; adjust if different for extrapolation in Excel

# Organize data as a dictionary (add more keys if needed, e.g., for CAPEX sensitivity)
export_data = {
    'Total PV Energy Produced (kWh)': total_pv_energy,  # Extrapolate in Excel: =this * (365 / simulation_days)
    'Total BESS Energy Discharged (kWh)': total_bess_discharge,  # Extrapolate similarly
    'Total Grid Sold (kWh)': total_grid_sold,
    'Total Grid Bought (kWh)': total_grid_bought,
    'Self-Sufficiency Ratio (%)': self_sufficiency,
    'EV Renewable Share (%)': ev_renewable_share,
    'Total Revenue (€)': total_revenue,  # Extrapolate if needed
    'BESS Capacity (kWh)': bess_capacity,
    'PV Scaling Factor': pv_scaling_factor,
    'Simulation Period (days)': simulation_days  # For easy extrapolation in Excel
}

# Path to your existing Excel file (hardcoded based on provided path; add file name if not 'Financial_Model.xlsx')
excel_path = r'C:\Users\dell\V1_First_Model\Input Data Files\Financial_Model.xlsx'

# Load existing workbook
wb = openpyxl.load_workbook(excel_path)

# Create or select sheet for inputs (won't overwrite other sheets)
sheet_name = 'Output PyPSA'
if sheet_name not in wb.sheetnames:
    ws = wb.create_sheet(sheet_name)
else:
    ws = wb[sheet_name]
    ws.delete_rows(3, ws.max_row)  # Clear existing data in this sheet only (optional; remove if appending)

# Write data to sheet (Column C: keys, Column D: values; starting at row 1)
row = 3
for key, value in export_data.items():
    ws.cell(row=row, column=1, value=key)
    ws.cell(row=row, column=2, value=value)
    row += 1

# Save updated workbook
wb.save(excel_path)
print(f"Data exported to {excel_path} in sheet '{sheet_name}'")
#END EXCEL SAVE AND EXPORT


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