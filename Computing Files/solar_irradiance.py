import numpy as np
import pandas as pd

def compute_pv_power():
    # Read the CSV file from Input Data Files
    df = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/csv_45.992746_13.361149_fixed_GHI-temp.csv')

    # Fallback for missing values: Interpolate NaNs (linear average of nearby)
    df['ghi'] = df['ghi'].interpolate(method='linear', limit_direction='both')  # Fill gaps with neighbor avg
    df['air_temp'] = df['air_temp'].interpolate(method='linear', limit_direction='both')

    # Extract GHI and air_temp (assume columns are 'ghi' and 'air_temp')
    ghi = df['ghi'].values
    air_temp = df['air_temp'].values

    # Load PV params from Constants_Plant.csv (pv_peak, temp_coeff, module_eff; assume tilt/azimuth not needed for basic model)
    constants_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/Constants_Plant.csv', comment='#')
    pv_peak = float(constants_data[constants_data['Parameter'] == 'PV_PEAK_POWER']['Value'].iloc[0])
    temp_coeff = float(constants_data[constants_data['Parameter'] == 'TEMP_COEFF']['Value'].iloc[0])
    module_eff = float(constants_data[constants_data['Parameter'] == 'MODULE_EFF']['Value'].iloc[0])

    # PV power calculation model (NREL PVWatts simplified: P = pv_peak * (GHI / 1000) * module_eff * (1 - temp_coeff * (temp - 25))
    # Literature: NREL PVWatts Calculator Methodology (pvwatts.nrel.gov/downloads/pvwattsv5.pdf, 2016) for irradiance and temperature correction; assumes GHI is on plane or adjusted.
    # No POA needed if CSV GHI is effective; add noise for realism as in your original.
    pv_power = pv_peak * (ghi / 1000) * module_eff * (1 - temp_coeff * (air_temp - 25))
    pv_power = np.maximum(pv_power, 0)  # Ensure non-negative
    pv_power += np.random.normal(0, 10, len(pv_power))  # Add noise
    pv_power = np.maximum(pv_power, 0)
    pv_power = np.pad(pv_power, (0, 672 - len(pv_power)), mode='edge') if len(pv_power) < 672 else pv_power

    return pv_power  # Returns array matching time steps (assume CSV has 672 rows for 7 days at 15-min)