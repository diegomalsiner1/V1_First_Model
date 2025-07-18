import numpy as np
import matplotlib.pyplot as plt
import os
import load_data  # Import the modular data loader

print("--- Running the Simulation Script (No BESS) ---")

# Load all input data from load_data.py (ignores BESS-related params)
data = load_data.load()

# Extract relevant no-BESS data
pv_power = data['pv_power']
consumer_demand = data['consumer_demand']
grid_buy_price = data['grid_buy_price']
grid_sell_price = data['grid_sell_price']
lcoe_pv = data['lcoe_pv']
bidding_zone_desc = data['bidding_zone_desc']
period_str = data['period_str']
start_weekday = data['start_weekday']
time_steps = data['time_steps']
time_indices = data['time_indices']
delta_t = data['delta_t']
n_steps = data['n_steps']

# Compute flows (no optimization: prioritize self-consumption, sell excess, buy deficit)
P_PV_consumer_vals = np.minimum(pv_power, consumer_demand)
P_PV_grid_vals = np.maximum(pv_power - consumer_demand, 0)
P_grid_consumer_vals = np.maximum(consumer_demand - pv_power, 0)
slack_vals = np.zeros(n_steps)

P_grid_sold = P_PV_grid_vals
P_grid_bought = P_grid_consumer_vals

# Compute revenues/costs
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

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_labels = [days[(start_weekday + d) % 7] for d in range(8)]

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