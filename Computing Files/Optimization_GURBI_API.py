import numpy as np  # Importing NumPy for numerical operations like arrays and math functions
import pandas as pd  # Importing Pandas for data manipulation, especially reading CSV files
import cvxpy as cp  # Importing CVXPY for convex optimization modeling
import matplotlib.pyplot as plt  # Importing Matplotlib for creating plots and visualizations
import os  # Importing OS module for file path operations, like creating directories
import random  # Importing Random for shuffling lists, used in PV multipliers
from datetime import datetime, timedelta, timezone  # Importing datetime tools for handling dates and time zones
import requests  # Importing Requests for making HTTP API calls to ENTSO-E
from xml.etree import ElementTree  # Importing ElementTree for parsing XML responses from API
import warnings  # Importing Warnings to handle non-fatal issues like incomplete data

# TEST - Printing a message to indicate the script is starting
print("--- Running the REAL Optimization Script using GUROBI ---")

# Defining the time framework: Simulating 1 week (168 hours) in 15-minute intervals, resulting in 672 time steps
time_steps = np.arange(0, 168, 0.25)  # Creating an array from 0 to 168 in steps of 0.25 hours
n_steps = len(time_steps)  # Calculating the number of time steps (672)
delta_t = 0.25  # Time interval in hours (15 minutes)
time_indices = range(n_steps)  # Creating a range object for indexing time steps (0 to 671)

# ENTSO-E API token and bidding zone - These are constants for API access
ENTSOE_TOKEN = 'cd4a21d9-d58c-4b68-b233-ae5e0d8707f5'  # Personal API token for ENTSO-E
BIDDING_ZONE = '10YCH-SWISSGRIDZ'  # Bidding zone code for Switzerland
TIMEZONE_OFFSET = 2  # Timezone offset for CEST (UTC+2), used to adjust for local time

# Defining a function to fetch day-ahead prices from ENTSO-E API
def get_dayahead_prices(api_key: str, area_code: str, start: datetime = None, end: datetime = None):
    """
    Get day-ahead prices from ENTSO-E API.
    Adapted from https://gist.github.com/jpulakka/f866e37dcedeede31e96a34e9f06ed7a
    This function constructs the API URL, sends a GET request, parses the XML response,
    and extracts time-stamped prices into a dictionary.
    """
    # Handling default start time if not provided
    if not start:
        start = datetime.now(timezone.utc)  # Use current UTC time if no start is given
    # Converting start to UTC if it's in another timezone
    elif start.tzinfo and start.tzinfo != timezone.utc:
        start = start.astimezone(timezone.utc)  # Convert to UTC
    # Handling default end time (1 day after start)
    if not end:
        end = start + timedelta(days=1)  # Default to 1 day period
    # Converting end to UTC if needed
    elif end.tzinfo and end.tzinfo != timezone.utc:
        end = end.astimezone(timezone.utc)  # Convert to UTC
    fmt = '%Y%m%d%H00'  # Format for API date parameters (YYYYMMDDHH00)
    # Constructing the API URL with parameters for document type (A44 for day-ahead prices), domains, and period
    url = f'https://web-api.tp.entsoe.eu/api?securityToken={api_key}&documentType=A44&in_Domain={area_code}' \
          f'&out_Domain={area_code}&periodStart={start.strftime(fmt)}&periodEnd={end.strftime(fmt)}'
    response = requests.get(url)  # Sending GET request to the API
    response.raise_for_status()  # Raising an error if the request fails (e.g., HTTP error)
    xml_str = response.text  # Getting the XML response as a string
    result = {}  # Initializing an empty dictionary to store time-price pairs
    # Parsing the XML string into an ElementTree object
    for child in ElementTree.fromstring(xml_str):  # Iterating over root children
        if child.tag.endswith("TimeSeries"):  # Finding TimeSeries elements
            for ts_child in child:  # Iterating over TimeSeries children
                if ts_child.tag.endswith("Period"):  # Finding Period elements
                    for pe_child in ts_child:  # Iterating over Period children
                        if pe_child.tag.endswith("timeInterval"):  # Finding timeInterval
                            for ti_child in pe_child:  # Iterating over timeInterval children
                                if ti_child.tag.endswith("start"):  # Extracting start time
                                    # Parsing the start time string to datetime object
                                    start_time = datetime.strptime(ti_child.text, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc)
                        elif pe_child.tag.endswith("Point"):  # Finding Point elements (hourly data)
                            for po_child in pe_child:  # Iterating over Point children
                                if po_child.tag.endswith("position"):  # Getting position (hour offset)
                                    delta = int(po_child.text) - 1  # Position starts from 1, so subtract 1
                                    time = start_time + timedelta(hours=delta)  # Calculating timestamp
                                elif po_child.tag.endswith("price.amount"):  # Getting price
                                    price = float(po_child.text)  # Converting price to float
                                    result[time] = price  # Storing in result dictionary
    return result  # Returning the dictionary of timestamps to prices

# Defining the load_data function to load all input data for the optimization
def load_data():
    # Loading LCOE for PV from CSV, ignoring lines starting with '#'
    pv_lcoe_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/PV_LCOE.csv', comment='#')
    lcoe_pv = pv_lcoe_data['LCOE_PV'].iloc[0]  # Extracting the first LCOE_PV value

    # Loading LCOE for BESS from CSV, ignoring comment lines
    bess_lcoe_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/BESS_LCOE.csv', comment='#')
    lcoe_bess = bess_lcoe_data['LCOE_BESS'].iloc[0]  # Extracting the first LCOE_BESS value

    # Loading constants from CSV, ignoring comment lines
    constants_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/Constants_Plant.csv', comment='#')
    # Extracting specific parameters from the constants DataFrame
    bess_capacity = float(constants_data[constants_data['Parameter'] == 'BESS_Capacity']['Value'].iloc[0])
    bess_power_limit = float(constants_data[constants_data['Parameter'] == 'BESS_Power_Limit']['Value'].iloc[0])
    eta_charge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Charge']['Value'].iloc[0])
    eta_discharge = float(constants_data[constants_data['Parameter'] == 'BESS_Efficiency_Discharge']['Value'].iloc[0])
    soc_initial = float(constants_data[constants_data['Parameter'] == 'SOC_Initial']['Value'].iloc[0])
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])

    # Fetching grid prices from ENTSO-E for the last 7 days (previous complete week)
    utc_now = datetime.now(timezone.utc)  # Getting current UTC time
    start_date = (utc_now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)  # Calculating start of previous week
    end_date = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)  # End of previous week (today midnight)
    try:  # Trying to fetch API data, with exception handling
        prices_dict = get_dayahead_prices(ENTSOE_TOKEN, BIDDING_ZONE, start_date, end_date)  # Calling the API function
        sorted_times = sorted(prices_dict.keys())  # Sorting the timestamps from the dictionary
        # Printing debug information about fetched data
        print(f"API debug: Fetched {len(sorted_times)} prices for zone {BIDDING_ZONE}")
        if sorted_times:
            print(f"API debug: Date range: {sorted_times[0]} to {sorted_times[-1]}")
        
        # Filtering times to exactly the requested range [start_date, end_date)
        filtered_times = [t for t in sorted_times if start_date <= t < end_date]  # List comprehension for filtering
        print(f"API debug: Filtered to {len(filtered_times)} prices within {start_date} to {end_date}")
        
        current_times = filtered_times  # Assigning filtered times
        if len(current_times) != 168:  # Checking if we have exactly 168 hourly points
            # Issuing a warning if data is incomplete
            warnings.warn(f"Warning: Non-standard data ({len(current_times)} prices). Adjusting to 168.")
            # Calculating mean price for padding (in Eur/MWh)
            mean_price = np.mean([prices_dict[t] for t in current_times]) if current_times else 0.103 * 1000  
            
            # Padding if fewer than 168 points
            while len(current_times) < 168:
                last_t = current_times[-1] if current_times else start_date  # Getting last time or start
                next_t = last_t + timedelta(hours=1)  # Adding one hour
                prices_dict[next_t] = mean_price  # Adding mean price
                current_times.append(next_t)  # Appending new time
            
            # Clipping if more than 168 (though filtered should prevent this)
            if len(current_times) > 168:
                current_times = current_times[:168]  # Slicing to 168
        
        sorted_times = sorted(current_times)  # Re-sorting after adjustments
    except Exception as e:  # Catching any errors during API call or processing
        print(f"API error: {e}. Falling back to synthetic data.")  # Printing error message
        grid_price_hourly = np.full(168, 0.103)  # Falling back to constant synthetic prices (Eur/kWh)
        sorted_times = [start_date + timedelta(hours=i) for i in range(168)]  # Generating synthetic timestamps
    else:
        # Converting prices to array and to Eur/kWh
        grid_price_hourly = np.array([prices_dict[t] for t in sorted_times]) / 1000  

    # Adjusting for local timezone to correctly shift consumer demand profile
    local_start_time = sorted_times[0] + timedelta(hours=TIMEZONE_OFFSET)  # Adding offset for local time
    start_weekday = local_start_time.weekday()  # Getting weekday (0=Mon, 6=Sun) for local start

    # Linear interpolation to 15-min resolution
    time_hourly = np.arange(0, 168, 1)  # Hourly time points (0 to 167)
    time_quarter = np.arange(0, 168, 0.25)  # Quarter-hourly time points (0 to 167.75)
    grid_price = np.interp(time_quarter, time_hourly, grid_price_hourly)  # Interpolating prices
    grid_buy_price = grid_price + 0.01  # Adding margin for buy price
    grid_sell_price = grid_price - 0.01  # Subtracting margin for sell price

    # Generating sample PV power profile: Sinusoidal daytime generation over 7 days with variations
    plant_size = 4000
    pv_power = np.zeros(n_steps)  # Initializing zero array for PV power (672 steps)
    multipliers = [1.0, 0.9, 0.5, 0.8, 1.0, 0.6, 1.0]  # Day-specific weather multipliers
    random.shuffle(multipliers)  # Randomly shuffling multipliers for realism
    for i, t in enumerate(time_steps):  # Looping over each time step
        local_t = t % 24  # Local time within day (0-23.75)
        day = int(t // 24)  # Day index (0-6)
        if 6 <= local_t <= 18:  # Daytime hours for PV generation
            amplitude = plant_size * multipliers[day]  # Scaling amplitude by multiplier
            # Sinusoidal generation with noise
            pv_power[i] = amplitude * np.sin(np.pi * (local_t - 6) / 12) + np.random.normal(0, 10)
        pv_power[i] = max(0, pv_power[i])  # Ensuring non-negative power

    # Generating consumer demand: Higher on weekdays (production), lower on weekends (standby)
    consumer_demand = np.zeros(n_steps)  # Initializing zero array for demand
    for i, t in enumerate(time_steps):  # Looping over each time step
        local_t = t % 24  # Local time within day
        day = int(t // 24)  # Day index
        actual_weekday = (start_weekday + day) % 7  # Calculating actual weekday based on shifted start
        if actual_weekday < 5:  # Weekdays: Production profile with ramps
            base = 200.0  # Base load
            if 6 <= local_t < 8:  # Morning ramp-up
                add = 1000.0 * (local_t - 6) / 2  # Linear increase
                consumer_demand[i] = base + add
            elif 8 <= local_t <= 16:  # Peak hours
                consumer_demand[i] = 1200.0  # Full production
            elif 16 < local_t <= 18:  # Evening ramp-down
                add = 1000.0 * (18 - local_t) / 2  # Linear decrease
                consumer_demand[i] = base + add
            else:  # Off-peak
                consumer_demand[i] = base
        else:  # Weekends: Flat standby load
            consumer_demand[i] = 70.0

    # Preparing plot information: Bidding zone and period strings
    bidding_zone_desc = f"Switzerland ({BIDDING_ZONE})"  # Description string
    period_start = local_start_time.strftime('%Y-%m-%d')  # Formatting start date
    period_end = (local_start_time + timedelta(days=7)).strftime('%Y-%m-%d')  # End date (7 days later)
    period_str = f"{period_start} to {period_end}"  # Period string

    # Returning all loaded data as a tuple, including plot info and start_weekday for dynamic labels
    return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
            lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
            eta_charge, eta_discharge, soc_initial, pi_consumer,
            bidding_zone_desc, period_str, start_weekday)

# Calling load_data to get all inputs
(pv_power, consumer_demand, grid_buy_price, grid_sell_price,
 lcoe_pv, lcoe_bess, bess_capacity, bess_power_limit,
 eta_charge, eta_discharge, soc_initial, pi_consumer,
 bidding_zone_desc, period_str, start_weekday) = load_data()

# Calculating Big-M constants for mutual exclusivity constraints (used in binary logic)
max_pv = np.max(pv_power)  # Maximum PV power for bounds
max_demand = np.max(consumer_demand)  # Maximum consumer demand for bounds
M_bess = bess_power_limit  # Big-M for BESS (power limit)
M_grid = max(max_pv + bess_power_limit, max_demand + bess_power_limit)  # Big-M for grid (larger bound)

# Defining optimization variables (all non-negative powers and SOC)
P_PV_consumer = cp.Variable(n_steps, nonneg=True)  # Power from PV to consumer
P_PV_BESS = cp.Variable(n_steps, nonneg=True)  # Power from PV to BESS
P_PV_grid = cp.Variable(n_steps, nonneg=True)  # Power from PV to grid
P_BESS_consumer = cp.Variable(n_steps, nonneg=True)  # Power from BESS to consumer
P_BESS_grid = cp.Variable(n_steps, nonneg=True)  # Power from BESS to grid
P_grid_consumer = cp.Variable(n_steps, nonneg=True)  # Power from grid to consumer
P_grid_BESS = cp.Variable(n_steps, nonneg=True)  # Power from grid to BESS
SOC = cp.Variable(n_steps + 1, nonneg=True)  # State of Charge (extra point for t=0 and t=end)
slack = cp.Variable(n_steps, nonneg=True)  # Slack variable for unmet demand

# Defining binary variables for mutual exclusivity (0 or 1)
delta_bess = cp.Variable(n_steps, boolean=True)  # Binary for BESS charge (1) or discharge (0)
delta_grid = cp.Variable(n_steps, boolean=True)  # Binary for grid buy (1) or sell (0)

# Initializing empty list for constraints
constraints = []

# Adding consumer balance constraints: Supply + slack = demand for each time step
constraints += [P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] + slack[t] == consumer_demand[t]
                for t in time_indices]  # List comprehension for all t

# Adding PV allocation constraints: Total PV usage <= available PV for each t
constraints += [P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= pv_power[t] for t in time_indices]

# Adding BESS power limit constraints for charge and discharge
for t in time_indices:  # Looping over time steps
    constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= bess_power_limit,  # Charge limit
                    P_BESS_consumer[t] + P_BESS_grid[t] <= bess_power_limit]  # Discharge limit

# Adding SOC dynamics constraints
constraints += [SOC[0] == soc_initial]  # Initial SOC
# SOC update equation: Next SOC = current + charge (efficient) - discharge (inefficient)
constraints += [SOC[t+1] == SOC[t] + eta_charge * (P_PV_BESS[t] + P_grid_BESS[t]) * delta_t -
                (P_BESS_consumer[t] + P_BESS_grid[t]) / eta_discharge * delta_t for t in range(n_steps)]
constraints += [SOC[t] <= bess_capacity for t in range(n_steps + 1)]  # Upper SOC bound
constraints += [SOC[t] >= 0.1 * bess_capacity for t in range(n_steps + 1)]  # Lower SOC bound (10%)

# Ensuring final SOC >= initial for sustainability
constraints += [SOC[n_steps] >= soc_initial]

# Adding mutual exclusivity for BESS: Can't charge and discharge simultaneously
for t in time_indices:  # Looping over time steps
    # Charge only if delta_bess=1 (Big-M method)
    constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= M_bess * delta_bess[t]]
    # Discharge only if delta_bess=0
    constraints += [P_BESS_consumer[t] + P_BESS_grid[t] <= M_bess * (1 - delta_bess[t])]

# Adding mutual exclusivity for Grid: Can't buy and sell simultaneously
for t in time_indices:  # Looping over time steps
    # Buy only if delta_grid=1
    constraints += [P_grid_consumer[t] + P_grid_BESS[t] <= M_grid * delta_grid[t]]
    # Sell only if delta_grid=0
    constraints += [P_PV_grid[t] + P_BESS_grid[t] <= M_grid * (1 - delta_grid[t])]

# Defining the objective: Maximize revenue (various terms for profits/costs)
revenue = (
    cp.sum(cp.multiply(P_PV_consumer, grid_buy_price - lcoe_pv) * delta_t) +  # PV to consumer: saved buy - PV cost
    cp.sum(cp.multiply(P_PV_grid, grid_sell_price - lcoe_pv) * delta_t) +  # PV to grid: sell revenue - PV cost
    cp.sum(cp.multiply(P_PV_BESS, - lcoe_pv) * delta_t) +  # PV to BESS: - PV cost (charging)
    cp.sum(cp.multiply(P_BESS_consumer, grid_buy_price - lcoe_bess) * delta_t) +  # BESS to consumer: saved buy - BESS cost
    cp.sum(cp.multiply(P_BESS_grid, grid_sell_price - lcoe_bess) * delta_t) -  # BESS to grid: sell revenue - BESS cost
    cp.sum(cp.multiply(P_grid_consumer + P_grid_BESS, grid_buy_price) * delta_t) -  # Grid buys: - buy cost
    1e5 * cp.sum(slack)  # Penalty
)
objective = cp.Maximize(revenue)  # Setting maximization objective

# Creating the optimization problem with objective and constraints
problem = cp.Problem(objective, constraints)
# Solving the problem using Gurobi solver with verbose output
problem.solve(solver=cp.GUROBI, verbose=True, MIPGap=0.0001) # 0.025% Error Tolerance for faster COMPUTATION TESTRUNS

# Checking the solver status
print("Status:", problem.status)
if problem.status == cp.OPTIMAL:  # Proceeding only if optimal solution found
    # Extracting variable values post-optimization
    P_PV_consumer_vals = P_PV_consumer.value  # Getting optimized values
    P_PV_BESS_vals = P_PV_BESS.value
    P_PV_grid_vals = P_PV_grid.value
    P_BESS_consumer_vals = P_BESS_consumer.value
    P_BESS_grid_vals = P_BESS_grid.value
    P_grid_consumer_vals = P_grid_consumer.value
    P_grid_BESS_vals = P_grid_BESS.value
    SOC_vals = SOC.value
    slack_vals = slack.value

    # Computing aggregate BESS charge and discharge
    P_BESS_charge = P_PV_BESS_vals + P_grid_BESS_vals  # Total charge power
    P_BESS_discharge = P_BESS_consumer_vals + P_BESS_grid_vals  # Total discharge power

    # Computing aggregate grid sold and bought
    P_grid_sold = P_PV_grid_vals + P_BESS_grid_vals  # Total sold to grid
    P_grid_bought = P_grid_consumer_vals + P_grid_BESS_vals  # Total bought from grid

    # Computing per-step revenues and costs for plotting, aligned to objective terms
    pv_to_consumer_rev = []  # PV to consumer: saved buy - PV cost
    pv_to_grid_rev = []  # PV to grid: sell - PV cost
    pv_to_bess_cost = []  # PV to BESS: - PV cost
    bess_to_consumer_rev = []  # BESS to consumer: saved buy - BESS cost
    bess_to_grid_rev = []  # BESS to grid: sell - BESS cost
    grid_buy_cost = []  # Grid buys: - buy cost (for consumer + BESS charge)
    penalty_per_step = []  # Penalty
    total_net_per_step = []  # Total net per step
    for t in time_indices:  # Looping over time steps
        pv_cons = P_PV_consumer_vals[t] * (grid_buy_price[t] - lcoe_pv) * delta_t
        pv_grid = P_PV_grid_vals[t] * (grid_sell_price[t] - lcoe_pv) * delta_t
        pv_bess = P_PV_BESS_vals[t] * (- lcoe_pv) * delta_t
        bess_cons = P_BESS_consumer_vals[t] * (grid_buy_price[t] - lcoe_bess) * delta_t
        bess_grid = P_BESS_grid_vals[t] * (grid_sell_price[t] - lcoe_bess) * delta_t
        grid_cost = - (P_grid_consumer_vals[t] + P_grid_BESS_vals[t]) * grid_buy_price[t] * delta_t
        penalty = -1e5 * slack_vals[t]
        net_rev = pv_cons + pv_grid + pv_bess + bess_cons + bess_grid + grid_cost + penalty
        pv_to_consumer_rev.append(pv_cons)
        pv_to_grid_rev.append(pv_grid)
        pv_to_bess_cost.append(pv_bess)
        bess_to_consumer_rev.append(bess_cons)
        bess_to_grid_rev.append(bess_grid)
        grid_buy_cost.append(grid_cost)
        penalty_per_step.append(penalty)
        total_net_per_step.append(net_rev)

    # Calculating total revenue
    total_revenue = sum(total_net_per_step)
    print(f"Total Revenue: Eur{total_revenue:.2f}")  # Printing formatted total

    # Checking and printing unmet demand times
    print("Time steps with unmet demand (kW):")
    for t in time_indices:
        if slack_vals[t] > 1e-6:  # Tolerance for numerical errors
            print(f"Time {time_steps[t]:.2f}h: Unmet demand = {slack_vals[t]:.2f} kW")

    # Setting global font size for plots
    plt.rcParams.update({'font.size': 8})

    # Defining dynamic day labels for x-ticks based on start_weekday
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']  # List of day abbreviations
    day_labels = [days[(start_weekday + d) % 7] for d in range(8)]  # Generating 8 labels starting from actual start day

    # Creating first figure: Energy Flows
    plt.figure(figsize=(12, 12))  # Setting figure size

    # Subplot 1: PV and Grid Flows
    plt.subplot(3, 1, 1)  # First subplot in 3x1 grid
    plt.plot(time_steps, pv_power, label='PV Gen (kW)', color='orange')  # Plotting PV generation
    plt.plot(time_steps, P_grid_sold, label='Grid Sold (kW)', color='blue')  # Plotting sold to grid
    plt.plot(time_steps, P_grid_bought, label='Grid Bought (kW)', color='magenta')  # Plotting bought from grid
    plt.xlabel('Time (h)')  # X-axis label
    plt.ylabel('Power (kW)')  # Y-axis label
    plt.title('PV and Grid Flows')  # Title
    plt.legend(loc='best')  # Legend
    plt.grid(True)  # Grid lines
    plt.xticks(np.arange(0, 169, 24), day_labels)  # X-ticks with dynamic day labels
    for d in range(1, 7):  # Adding vertical lines for day boundaries
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Subplot 2: BESS Flows and SOC
    plt.subplot(3, 1, 2)  # Second subplot
    ax1 = plt.gca()  # Getting current axis
    ax1.plot(time_steps, P_BESS_charge, label='BESS Charge (kW)', color='blue')  # Plotting charge
    ax1.plot(time_steps, P_BESS_discharge, label='BESS Discharge (kW)', color='red')  # Plotting discharge
    ax1.set_xlabel('Time (h)')  # X-label
    ax1.set_ylabel('Power (kW)')  # Y-label
    ax1.set_title('BESS Flows and SOC')  # Title
    ax1.legend(loc='best')  # Legend
    ax1.grid(True)  # Grid
    ax1.set_xticks(np.arange(0, 169, 24))  # X-ticks
    ax1.set_xticklabels(day_labels)  # Dynamic day labels
    for d in range(1, 7):  # Day boundaries
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()  # Creating twin axis for SOC
    ax2.plot(np.arange(0, 168.25, 0.25), SOC_vals, label='SOC (kWh)', color='green', linestyle='--')  # Plotting SOC
    ax2.set_ylabel('SOC (kWh)')  # Y-label for SOC
    ax2.legend(loc='upper right')  # Legend

    # Subplot 3: Consumer Flow Composition
    plt.subplot(3, 1, 3)  # Third subplot
    # Stackplot for sources to consumer
    plt.stackplot(time_steps, P_PV_consumer_vals, P_BESS_consumer_vals, P_grid_consumer_vals, slack_vals,
                  labels=['PV to Cons', 'BESS to Cons', 'Grid to Cons', 'Unmet'],
                  colors=['orange', 'green', 'blue', 'red'])
    plt.plot(time_steps, consumer_demand, label='Demand', color='black', linestyle='--')  # Demand line
    plt.xlabel('Time (h)')  # X-label
    plt.ylabel('Power (kW)')  # Y-label
    plt.title('Consumer Flow Composition')  # Title
    plt.legend(loc='best')  # Legend
    plt.grid(True)  # Grid
    plt.xticks(np.arange(0, 169, 24), day_labels)  # X-ticks with dynamic day labels
    for d in range(1, 7):  # Day boundaries
        plt.axvline(d * 24, color='gray', linestyle='--')

    plt.subplots_adjust(hspace=0.4)  # Adjusting vertical spacing
    plt.tight_layout(pad=1.5)  # Tight layout
    
    # Saving the first plot
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')  # Output directory path
    if not os.path.exists(output_dir):  # Creating directory if it doesn't exist
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Energy_Flows.png'))  # Saving PNG
    plt.show()  # Showing the plot

    # Creating second figure: Financials (adapted)
    plt.figure(figsize=(12, 12))  # Figure size

    # Subplot 1: Energy Market Price (kept as Prices and LCOEs, but focused on grid price)
    plt.subplot(3, 1, 1)  # First subplot
    plt.plot(time_steps, grid_buy_price, label='Grid Buy Price (Eur/kWh)', color='blue')  # Plotting buy price
    plt.plot(time_steps, grid_sell_price, label='Grid Sell Price (Eur/kWh)', color='cyan', linestyle='--')  # Adding sell price for completeness
    plt.plot(time_steps, np.full(n_steps, lcoe_pv), label='PV LCOE', color='orange', linestyle='--')  # PV LCOE line
    plt.plot(time_steps, np.full(n_steps, lcoe_bess), label='BESS LCOE', color='green', linestyle='--')  # BESS LCOE line
    plt.xlabel('Time (h)')  # X-label
    plt.ylabel('Price (Eur/kWh)')  # Y-label
    # Title with bidding zone and period info
    plt.title(f'Energy Market Prices\n{bidding_zone_desc}\n{period_str}')
    plt.legend(loc='best')  # Legend
    plt.grid(True)  # Grid
    plt.xticks(np.arange(0, 169, 24), day_labels)  # X-ticks with dynamic day labels
    for d in range(1, 7):  # Day boundaries
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Subplot 2: Revenue and Cost Streams (positive/negative, colored by source: PV orange, BESS green, GRID blue/magenta)
    plt.subplot(3, 1, 2)  # Second subplot
    # PV streams (orange variants)
    plt.plot(time_steps, pv_to_consumer_rev, label='PV to Consumer (Rev)', color='orange')
    plt.plot(time_steps, pv_to_grid_rev, label='PV to Grid (Rev)', color='darkorange')
    plt.plot(time_steps, pv_to_bess_cost, label='PV to BESS (Cost)', color='gold')
    # BESS streams (green variants)
    plt.plot(time_steps, bess_to_consumer_rev, label='BESS to Consumer (Rev)', color='green')
    plt.plot(time_steps, bess_to_grid_rev, label='BESS to Grid (Rev)', color='darkgreen')
    # Grid streams (blue/magenta)
    plt.plot(time_steps, grid_buy_cost, label='Grid Buy (Cost)', color='magenta')
    # Penalty (red, if any)
    plt.plot(time_steps, penalty_per_step, label='Penalty (Cost)', color='red', linestyle='--')
    plt.xlabel('Time (h)')  # X-label
    plt.ylabel('Eur/Step')  # Y-label
    plt.title('Revenue and Cost Streams')  # Title
    plt.legend(loc='best')  # Legend
    plt.grid(True)  # Grid
    plt.xticks(np.arange(0, 169, 24), day_labels)  # X-ticks with dynamic day labels
    for d in range(1, 7):  # Day boundaries
        plt.axvline(d * 24, color='gray', linestyle='--')

    # Subplot 3: Total revenue per time step and cumulative revenue
    plt.subplot(3, 1, 3)  # Third subplot
    ax1 = plt.gca()  # Current axis
    ax1.plot(time_steps, total_net_per_step, label='Total Rev per Step (Eur)', color='purple')  # Per-step total revenue
    ax1.set_xlabel('Time (h)')  # X-label
    ax1.set_ylabel('Rev per Step (Eur)')  # Y-label
    ax1.set_title('Total Revenue per Step and Cumulative')  # Title
    ax1.legend(loc='upper left')  # Legend
    ax1.grid(True)  # Grid
    ax1.set_xticks(np.arange(0, 169, 24))  # X-ticks
    ax1.set_xticklabels(day_labels)  # Dynamic day labels
    for d in range(1, 7):  # Day boundaries
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()  # Twin axis for cumulative
    cumulative_revenue = np.cumsum(total_net_per_step)  # Cumulative net revenue
    ax2.plot(time_steps, cumulative_revenue, label='Cum. Total Rev (Eur)', color='orange', linestyle='--')  # Plotting cumulative
    ax2.set_ylabel('Cum. Rev (Eur)')  # Y-label
    ax2.legend(loc='upper right')  # Legend

    plt.subplots_adjust(hspace=0.4)  # Vertical spacing
    plt.tight_layout(pad=1.5)  # Tight layout
    # Saving the second plot
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Financials.png'))
    plt.show()  # Showing the plot