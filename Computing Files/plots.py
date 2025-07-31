import matplotlib.pyplot as plt
import numpy as np
import os

def plot_energy_flows(results, data, revenues, save_dir=None):
    """
    Plot PV, grid, BESS flows, and consumer demand composition.
    """
    plt.figure(figsize=(14, 12))
    
    # Common x-axis setup for 7 days (15-min intervals)
    n_steps = len(data['time_steps'])
    max_time = np.max(data['time_steps']) + data['delta_t']
    x_ticks = np.arange(0, max_time + 0.01, 24)  # Ticks every 24 hours in hour units
    if 'day_labels' in data and len(data['day_labels']) == 7:
        x_labels = data['day_labels'] + [data['day_labels'][-1] + ' End']  # 8 labels for 8 ticks
    else:
        x_labels = [str(d) for d in range(len(x_ticks))]  # Default to step numbers if no labels
    
    # Subplot 1: PV and Grid Flows
    plt.subplot(3, 1, 1)
    plt.plot(data['time_steps'], data['pv_power'], label='PV Generation (kW)', color='orange', linewidth=1)
    plt.plot(data['time_steps'], results.get('P_grid_sold', np.zeros_like(data['pv_power'])), label='Grid Sold (kW)', color='purple', linewidth=1)
    plt.plot(data['time_steps'], results.get('P_grid_bought', np.zeros_like(data['pv_power'])), label='Grid Bought (kW)', color='magenta', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title('PV and Grid Flows')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(x_ticks, x_labels)
    for d in range(1, len(x_ticks)):
        plt.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)

    # Subplot 2: BESS Flows and SOC (absolute units)
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax1.plot(data['time_steps'], results.get('P_BESS_charge', np.zeros_like(data['pv_power'])), label='BESS Charge (kW)', color='blue', linewidth=1)
    ax1.plot(data['time_steps'], results.get('P_BESS_discharge', np.zeros_like(data['pv_power'])), label='BESS Discharge (kW)', color='red', linewidth=1)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS Flows and SOC')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle=':', linewidth=0.7)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    for d in range(1, len(x_ticks)):
        ax1.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)
    ax2 = ax1.twinx()
    ax2.plot(data['time_steps'], results.get('SOC_vals', np.zeros(n_steps + 1))[:n_steps], label='SOC (kWh)', color='green', linestyle='--', linewidth=1)
    ax2.set_ylabel('SOC (kWh)')
    ax2.legend(loc='upper right', fontsize=9)

    # Subplot 3: Consumer and EV Flow Composition (show BESS flows if available)
    plt.subplot(3, 1, 3)
    plt.stackplot(
        data['time_steps'],
        results.get('P_PV_consumer_vals', np.zeros_like(data['pv_power'])),
        results.get('P_BESS_consumer_vals', np.zeros_like(data['pv_power'])),
        results.get('P_grid_consumer_vals', np.zeros_like(data['pv_power'])),
        results.get('P_PV_ev_vals', np.zeros_like(data['pv_power'])),
        results.get('P_BESS_ev_vals', np.zeros_like(data['pv_power'])),
        results.get('P_grid_ev_vals', np.zeros_like(data['pv_power'])),
        labels=['PV to Consumer', 'BESS to Consumer', 'Grid to Consumer',
                'PV to EV', 'BESS to EV', 'Grid to EV'],
        colors=['orange', 'green', 'blue', 'yellow', 'lightgreen', 'lightblue']
    )
    plt.plot(data['time_steps'], data['consumer_demand'] + data['ev_demand'],
             label='Total Demand', color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title(f'Consumer & EV Flow Composition (Self-Sufficiency: {revenues.get("self_sufficiency", 0):.2f}%)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(x_ticks, x_labels)
    for d in range(1, len(x_ticks)):
        plt.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)

    plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.06, left=0.07, right=0.97)
    plt.tight_layout(pad=1.5)
    
    # Save plot
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = data.get('plot_suffix', '')
    plt.savefig(os.path.join(save_dir, f'Energy_Flows{suffix}.png'), dpi=200, bbox_inches='tight')
    plt.close()

def plot_financials(revenues, data, save_dir=None):
    """
    Plot financial results: prices, revenue streams, and cumulative revenue.
    """
    plt.figure(figsize=(14, 12))
    n_steps = len(data['time_steps'])
    max_time = np.max(data['time_steps']) + data['delta_t']
    x_ticks = np.arange(0, max_time + 0.01, 24)  # Ticks every 24 hours in hour units
    if 'day_labels' in data and len(data['day_labels']) == 7:
        x_labels = data['day_labels'] + [data['day_labels'][-1] + ' End']  # 8 labels for 8 ticks
    else:
        x_labels = [str(d) for d in range(len(x_ticks))]
    # Subplot 1: Market Prices
    plt.subplot(3, 1, 1)
    plt.plot(data['time_steps'], data['grid_buy_price'], label='Grid Buy Price (Euro/kWh)', color='blue', linewidth=1)
    plt.plot(data['time_steps'], data['grid_sell_price'], label='Grid Sell Price (Euro/kWh)', color='lightblue', linestyle='--', linewidth=1)
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_pv']), label='PV LCOE (Euro/kWh)', color='orange', linestyle='--', linewidth=1)
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_bess']), label='BESS LCOE (Euro/kWh)', color='green', linestyle='--', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Price (Euro/kWh)')
    plt.title(f'Energy Market Price ITA (Matrix)\n{data["period_str"]}')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(x_ticks, x_labels)
    for d in range(1, len(x_ticks)):
        plt.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)
    # Subplot 2: Revenue and Cost Streams (use new separated terms)
    plt.subplot(3, 1, 2)
    plt.plot(data['time_steps'], revenues.get('pv_to_grid_rev', np.zeros_like(data['time_steps'])), label='PV to Grid Revenue (Euro/step)', color='darkorange', linewidth=1)
    plt.plot(data['time_steps'], -revenues.get('grid_buy_cost', np.zeros_like(data['time_steps'])), label='Grid Buy Cost (Euro/step)', color='blue', linewidth=1, linestyle='--')
    plt.plot(data['time_steps'], revenues.get('bess_to_grid_rev', np.zeros_like(data['time_steps'])), label='BESS to Grid Revenue (Euro/step)', color='darkgreen', linewidth=1)
    plt.plot(data['time_steps'], revenues.get('pv_to_ev_rev', np.zeros_like(data['time_steps'])), label='PV to EV Revenue (Euro/step)', color='orange', linewidth=1)
    plt.plot(data['time_steps'], revenues.get('bess_to_ev_rev', np.zeros_like(data['time_steps'])), label='BESS to EV Revenue (Euro/step)', color='green', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Euro/step')
    plt.title('Revenue and Cost Streams')
    plt.legend(loc='upper right', fontsize=9, ncol=2)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(x_ticks, x_labels)
    for d in range(1, len(x_ticks)):
        plt.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)
    # Subplot 3: Total and Cumulative Revenue
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax1.plot(data['time_steps'], revenues.get('total_net_per_step', np.zeros_like(data['time_steps'])), label='Total Revenue per Step (Euro)', color='purple', linewidth=1)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Revenue per Step (Euro)')
    ax1.set_title('Total Revenue per Step and Cumulative')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle=':', linewidth=0.7)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    for d in range(1, len(x_ticks)):
        ax1.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)
    ax2 = ax1.twinx()
    cumulative_revenue = np.cumsum(revenues.get('total_net_per_step', np.zeros_like(data['time_steps'])))
    ax2.plot(data['time_steps'], cumulative_revenue, label='Cumulative Total Revenue (Euro)', color='orange', linestyle='--', linewidth=1)
    ax2.set_ylabel('Cumulative Revenue (Euro)')
    ax2.legend(loc='upper right', fontsize=9)
    plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.06, left=0.07, right=0.97)
    plt.tight_layout(pad=1.5)
    # Always save to Output Files in the V1_First_Model folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # V1_First_Model
    output_dir = os.path.join(base_dir, 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    suffix = data.get('plot_suffix', '')
    save_path = os.path.join(output_dir, f'Financials{suffix}.png')
    print(f"Saving financials plot to: {save_path}")  # Debug print
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
