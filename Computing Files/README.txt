# README.txt

Price source selection (IMPORTANT)
----------------------------------
You can choose which market price data the simulation uses. There are three modes:

1) HPFC (forward curve)
   - Pass price_source="HPFC" to load_data.load(...)
   - Uses Prices/HPFC_prices_forecast.py with Base/Peak params (base_forecast, peak_forecast)

2) API (ENTSO-E Transparency)
   - Pass price_source="API"
   - Uses Prices/API_prices.py

3) ITA (matrix CSV in Input Data Files)
   - Any other value falls back to ITA matrix prices
   - Recommended: price_source="Prices_ITA"
   - Uses Prices/Prices_ITA.py and Input Data Files/Price_Matrix.csv

Where to change it:
- main_Optimization.py: data = load_data.load(reference_case=False, price_source="HPFC", ...)
- Reference.py:        data = load_data.load(reference_case=True,  price_source="HPFC", ...)

Notes:
- HPFC expands hourly prices to 15-min steps and requires exactly 168 hourly points for a 7-day window.
- ITA mode reads Price_Matrix.csv and maps the month from Constants_Plant.csv (PERIOD_START/PERIOD_END).
- If you set price_source="API" or "HPFC" it will use those; otherwise it defaults to ITA.

Energy System Optimization Project - Computing Files
===================================================

This folder contains the main Python scripts for running the energy system simulation and optimization.

Prerequisites
-------------
1. **Python 3.8+** (recommended)
2. **Required Python packages:**
   - numpy
   - pandas
   - matplotlib
   - cvxpy
   - gurobipy (for Gurobi solver)
   - requests

   Install all dependencies with:
   ```
   pip install numpy pandas matplotlib cvxpy gurobipy requests
   ```
   (You need a valid Gurobi license for optimization.)

3. **Input Data Files:**
   - Place `Constants_Plant.csv`, `pv_FdM_2024.csv`, and `Price_Matrix.csv` in the `Input Data Files` folder (one level up from this folder).
   - If using API prices, ensure your ENTSO-E API token is set in `Constants_Plant.csv`.

How the Code Works
------------------
- **main_Optimization.py**: Runs the full Model Predictive Control (MPC) optimization for the energy system, using either API or ITA price data. Results and plots are saved in the `Output Files` folder.
- **Reference.py**: Runs a reference case (no BESS) for comparison.
- **load_data.py**: Loads all input data, constants, and time vectors. You can switch between API and ITA price sources by commenting/uncommenting lines in this file.
- **API_prices.py**: Fetches hourly prices from the ENTSO-E API.
- **Prices_ITA.py**: Loads typical Italian price curves from a matrix CSV, matching the month in the constants file.
- **pv_consumer_data_2024.py**: Loads and processes PV and consumer demand data for the simulation period.
- **mpc.py**: Contains the MPC optimization logic using cvxpy and Gurobi.
- **post_process.py**: Computes revenues, self-sufficiency, and prints summary results.
- **plots.py**: Generates and saves all result plots.

Usage
-----
1. Adjust parameters in `Constants_Plant.csv` as needed.
2. Run `main_Optimization.py` for the main scenario, or `Reference.py` for the reference case.
3. Plots and results will be saved in the `Output Files` folder.

Switching Price Source
----------------------
- In `load_data.py`, comment/uncomment the relevant lines to use either API or ITA price data.


