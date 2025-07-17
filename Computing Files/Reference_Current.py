import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime, timedelta, timezone
import requests
from xml.etree import ElementTree
import warnings

print("--- Running the Simulation Script (No BESS) ---")

time_steps = np.arange(0, 168, 0.25)
n_steps = len(time_steps)
delta_t = 0.25
time_indices = range(n_steps)

ENTSOE_TOKEN = 'cd4a21d9-d58c-4b68-b233-ae5e0d8707f5'
BIDDING_ZONE = '10YCH-SWISSGRIDZ'
TIMEZONE_OFFSET = 2

def get_dayahead_prices(api_key: str, area_code: str, start: datetime = None, end: datetime = None):
    if not start:
        start = datetime.now(timezone.utc)
    elif start.tzinfo and start.tzinfo != timezone.utc:
        start = start.astimezone(timezone.utc)
    if not end:
        end = start + timedelta(days=1)
    elif end.tzinfo and end.tzinfo != timezone.utc:
        end = end.astimezone(timezone.utc)
    fmt = '%Y%m%d%H00'
    url = f'https://web-api.tp.entsoe.eu/api?securityToken={api_key}&documentType=A44&in_Domain={area_code}' \
          f'&out_Domain={area_code}&periodStart={start.strftime(fmt)}&periodEnd={end.strftime(fmt)}'
    response = requests.get(url)
    response.raise_for_status()
    xml_str = response.text
    result = {}
    for child in ElementTree.fromstring(xml_str):
        if child.tag.endswith("TimeSeries"):
            for ts_child in child:
                if ts_child.tag.endswith("Period"):
                    for pe_child in ts_child:
                        if pe_child.tag.endswith("timeInterval"):
                            for ti_child in pe_child:
                                if ti_child.tag.endswith("start"):
                                    start_time = datetime.strptime(ti_child.text, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc)
                        elif pe_child.tag.endswith("Point"):
                            for po_child in pe_child:
                                if po_child.tag.endswith("position"):
                                    delta = int(po_child.text) - 1
                                    time = start_time + timedelta(hours=delta)
                                elif po_child.tag.endswith("price.amount"):
                                    price = float(po_child.text)
                                    result[time] = price
    return result

def load_data():
    pv_lcoe_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/PV_LCOE.csv', comment='#')
    lcoe_pv = pv_lcoe_data['LCOE_PV'].iloc[0]

    constants_data = pd.read_csv('C:/Users/dell/V1_First_Model/Input Data Files/Constants_Plant.csv', comment='#')
    pi_consumer = float(constants_data[constants_data['Parameter'] == 'Consumer_Price']['Value'].iloc[0])

    utc_now = datetime.now(timezone.utc)
    start_date = (utc_now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        prices_dict = get_dayahead_prices(ENTSOE_TOKEN, BIDDING_ZONE, start_date, end_date)
        sorted_times = sorted(prices_dict.keys())
        print(f"API debug: Fetched {len(sorted_times)} prices for zone {BIDDING_ZONE}")
        if sorted_times:
            print(f"API debug: Date range: {sorted_times[0]} to {sorted_times[-1]}")
        
        filtered_times = [t for t in sorted_times if start_date <= t < end_date]
        print(f"API debug: Filtered to {len(filtered_times)} prices within {start_date} to {end_date}")
        
        current_times = filtered_times
        if len(current_times) != 168:
            warnings.warn(f"Warning: Non-standard data ({len(current_times)} prices). Adjusting to 168.")
            mean_price = np.mean([prices_dict[t] for t in current_times]) if current_times else 0.103 * 1000
            
            while len(current_times) < 168:
                last_t = current_times[-1] if current_times else start_date
                next_t = last_t + timedelta(hours=1)
                prices_dict[next_t] = mean_price
                current_times.append(next_t)
            
            if len(current_times) > 168:
                current_times = current_times[:168]
        
        sorted_times = sorted(current_times)
    except Exception as e:
        print(f"API error: {e}. Falling back to synthetic data.")
        grid_price_hourly = np.full(168, 0.103)
        sorted_times = [start_date + timedelta(hours=i) for i in range(168)]
    else:
        grid_price_hourly = np.array([prices_dict[t] for t in sorted_times]) / 1000

    local_start_time = sorted_times[0] + timedelta(hours=TIMEZONE_OFFSET)
    start_weekday = local_start_time.weekday()

    time_hourly = np.arange(0, 168, 1)
    time_quarter = np.arange(0, 168, 0.25)
    grid_price = np.interp(time_quarter, time_hourly, grid_price_hourly)
    grid_buy_price = grid_price + 0.01
    grid_sell_price = grid_price - 0.01

    pv_power = np.zeros(n_steps)
    multipliers = [1.0, 0.9, 0.5, 0.8, 1.0, 0.6, 1.0]
    random.shuffle(multipliers)
    for i, t in enumerate(time_steps):
        local_t = t % 24
        day = int(t // 24)
        if 6 <= local_t <= 18:
            amplitude = 2327 * multipliers[day]
            pv_power[i] = amplitude * np.sin(np.pi * (local_t - 6) / 12) + np.random.normal(0, 10)
        pv_power[i] = max(0, pv_power[i])

    consumer_demand = np.zeros(n_steps)
    for i, t in enumerate(time_steps):
        local_t = t % 24
        day = int(t // 24)
        actual_weekday = (start_weekday + day) % 7
        if actual_weekday < 5:
            base = 200.0
            if 6 <= local_t < 8:
                add = 1000.0 * (local_t - 6) / 2
                consumer_demand[i] = base + add
            elif 8 <= local_t <= 16:
                consumer_demand[i] = 1200.0
            elif 16 < local_t <= 18:
                add = 1000.0 * (18 - local_t) / 2
                consumer_demand[i] = base + add
            else:
                consumer_demand[i] = base
        else:
            consumer_demand[i] = 70.0

    bidding_zone_desc = f"Switzerland ({BIDDING_ZONE})"
    period_start = local_start_time.strftime('%Y-%m-%d')
    period_end = (local_start_time + timedelta(days=7)).strftime('%Y-%m-%d')
    period_str = f"{period_start} to {period_end}"

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_labels = [days[(start_weekday + d) % 7] for d in range(8)]

    return (pv_power, consumer_demand, grid_buy_price, grid_sell_price,
            lcoe_pv, pi_consumer,
            bidding_zone_desc, period_str, day_labels)

(pv_power, consumer_demand, grid_buy_price, grid_sell_price,
 lcoe_pv, pi_consumer,
 bidding_zone_desc, period_str, day_labels) = load_data()

P_PV_consumer_vals = np.minimum(pv_power, consumer_demand)
P_PV_grid_vals = np.maximum(pv_power - consumer_demand, 0)
P_grid_consumer_vals = np.maximum(consumer_demand - pv_power, 0)
slack_vals = np.zeros(n_steps)

P_grid_sold = P_PV_grid_vals
P_grid_bought = P_grid_consumer_vals

rev_pv_per_step = P_PV_consumer_vals * (grid_buy_price - lcoe_pv) * delta_t
rev_sell_per_step = P_PV_grid_vals * grid_sell_price * delta_t
cost_grid_per_step = - P_grid_consumer_vals * grid_buy_price * delta_t
penalty_per_step = -1e5 * slack_vals
total_net_per_step = rev_pv_per_step + rev_sell_per_step + cost_grid_per_step + penalty_per_step

total_revenue = sum(total_net_per_step)
print(f"Total Revenue: Eur{total_revenue:.2f}")

print("Time steps with unmet demand (kW):")
for t in time_indices:
    if slack_vals[t] > 1e-6:
        print(f"Time {time_steps[t]:.2f}h: Unmet demand = {slack_vals[t]:.2f} kW")

plt.rcParams.update({'font.size': 8})

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_steps, pv_power, label='PV Gen (kW)', color='orange')
plt.plot(time_steps, P_grid_sold, label='Grid Sold (kW)', color='blue')
plt.plot(time_steps, P_grid_bought, label='Grid Bought (kW)', color='magenta')
plt.xlabel('Time (h)')
plt.ylabel('Power (kW)')
plt.title('PV and Grid Flows')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(np.arange(0, 169, 24), day_labels)
for d in range(1, 7):
    plt.axvline(d * 24, color='gray', linestyle='--')

plt.subplot(2, 1, 2)
plt.stackplot(time_steps, P_PV_consumer_vals, P_grid_consumer_vals, slack_vals,
              labels=['PV to Cons', 'Grid to Cons', 'Unmet'],
              colors=['orange', 'blue', 'red'])
plt.plot(time_steps, consumer_demand, label='Demand', color='black', linestyle='--')
plt.xlabel('Time (h)')
plt.ylabel('Power (kW)')
plt.title('Consumer Flow Composition')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(np.arange(0, 169, 24), day_labels)
for d in range(1, 7):
    plt.axvline(d * 24, color='gray', linestyle='--')

plt.subplots_adjust(hspace=0.4)
plt.tight_layout(pad=1.5)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'Energy_Flows_Reference.png'))
plt.show()

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(time_steps, grid_buy_price, label='Grid Price (Eur/kWh)', color='blue')
plt.plot(time_steps, np.full(n_steps, lcoe_pv), label='PV LCOE', color='orange', linestyle='--')
plt.xlabel('Time (h)')
plt.ylabel('Price (Eur/kWh)')
plt.title('Prices and LCOEs\n' + bidding_zone_desc + '\n' + period_str)
plt.legend(loc='best')
plt.grid(True)
plt.xticks(np.arange(0, 169, 24), day_labels)
for d in range(1, 7):
    plt.axvline(d * 24, color='gray', linestyle='--')

plt.subplot(3, 1, 2)
plt.plot(time_steps, rev_sell_per_step, label='Grid Sell Rev (Eur)', color='cyan')
plt.plot(time_steps, cost_grid_per_step, label='Grid Buy Cost (Eur)', color='red')
plt.plot(time_steps, rev_pv_per_step, label='PV Avoided Cost (Eur)', color='green')
plt.xlabel('Time (h)')
plt.ylabel('Eur/Step')
plt.title('Revenues and Costs')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(np.arange(0, 169, 24), day_labels)
for d in range(1, 7):
    plt.axvline(d * 24, color='gray', linestyle='--')

plt.subplot(3, 1, 3)
ax1 = plt.gca()
ax1.plot(time_steps, total_net_per_step, label='Rev per Step (Eur)', color='purple')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Rev per Step (Eur)')
ax1.set_title('Timestep and Cum. Revenue')
ax1.legend(loc='upper left')
ax1.grid(True)
ax1.set_xticks(np.arange(0, 169, 24))
ax1.set_xticklabels(day_labels)
for d in range(1, 7):
    ax1.axvline(d * 24, color='gray', linestyle='--')

ax2 = ax1.twinx()
cumulative_revenue = np.cumsum(total_net_per_step)
ax2.plot(time_steps, cumulative_revenue, label='Cum. Rev (Eur)', color='orange', linestyle='--')
ax2.set_ylabel('Cum. Rev (Eur)')
ax2.legend(loc='upper right')

plt.subplots_adjust(hspace=0.4)
plt.tight_layout(pad=1.5)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'Financials_Reference.png'))
plt.show()
