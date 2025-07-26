import matplotlib.pyplot as plt
import numpy as np
import os

def plot_energy_flows(results, data, revenues, save_dir=None):
    """
    Plot PV, grid, BESS flows, consumer demand composition, and EV demand composition.
    """
    plt.figure(figsize=(14, 16))
    # Subplot 1: PV and Grid Flows
    plt.subplot(4, 1, 1)
    plt.plot(data['time_steps'], data['pv_power'], label='PV Generation (kW)', color='orange', linewidth=1)
    plt.plot(data['time_steps'], results['P_grid_sold'], label='Grid Sold (kW)', color='blue', linewidth=1)
    plt.plot(data['time_steps'], results['P_grid_bought'], label='Grid Bought (kW)', color='magenta', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title('PV and Grid Flows')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0, 168, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    # Subplot 2: BESS Flows and SOC
    plt.subplot(4, 1, 2)
    ax1 = plt.gca()
    ax1.plot(data['time_steps'], results['P_BESS_charge'], label='BESS Charge (kW)', color='blue', linewidth=1)
    ax1.plot(data['time_steps'], results['P_BESS_discharge'], label='BESS Discharge (kW)', color='red', linewidth=1)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS Flows and SOC')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle=':', linewidth=0.7)
    ax1.set_xticks(np.arange(0, 168, 24))
    ax1.set_xticklabels(data['day_labels'])
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0, 168, len(results['SOC_vals'])), results['SOC_vals'], label='SOC (kWh)', color='green', linestyle='--', linewidth=1)
    ax2.set_ylabel('SOC (kWh)')
    ax2.legend(loc='upper right', fontsize=9)
    # Subplot 3: Consumer Flow Composition
    plt.subplot(4, 1, 3)
    plt.stackplot(data['time_steps'], results['P_PV_consumer_vals'], results['P_BESS_consumer_vals'], results['P_grid_consumer_vals'], results['slack_vals'],
                  labels=['PV to Consumer (kW)', 'BESS to Consumer (kW)', 'Grid to Consumer (kW)', 'Unmet Demand (kW)'],
                  colors=['orange', 'green', 'blue', 'red'], alpha=0.7)
    plt.plot(data['time_steps'], data['consumer_demand'], label='Demand (kW)', color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title(f'Consumer Flow Composition (Self-Sufficiency: {revenues["self_sufficiency"]:.2f}%)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0, 168, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    # Subplot 4: EV Flow Composition
    plt.subplot(4, 1, 4)
    P_PV_ev = results.get('P_PV_ev_vals', np.zeros(data['n_steps']))
    P_BESS_ev = results.get('P_BESS_ev_vals', np.zeros(data['n_steps']))
    P_grid_ev = results.get('P_grid_ev_vals', np.zeros(data['n_steps']))
    plt.stackplot(data['time_steps'], P_PV_ev, P_BESS_ev, P_grid_ev,
                  labels=['PV to EV (kW)', 'BESS to EV (kW)', 'Grid to EV (kW)'],
                  colors=['orange', 'green', 'blue'], alpha=0.7)
    plt.plot(data['time_steps'], data.get('ev_demand', np.zeros(data['n_steps'])), label='EV Demand (kW)', color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title(f'EV Flow Composition (Renewable Share: {revenues.get("ev_renewable_share", 0):.2f}%)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0, 168, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.06, left=0.07, right=0.97)
    plt.tight_layout(pad=1.5)
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
    # Subplot 1: Market Prices
    plt.subplot(3, 1, 1)
    plt.plot(data['time_steps'], data['grid_buy_price'], label='Grid Buy Price (Euro/kWh)', color='blue', linewidth=1)
    plt.plot(data['time_steps'], data['grid_sell_price'], label='Grid Sell Price (Euro/kWh)', color='lightblue', linestyle='--', linewidth=1)
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_pv']), label='PV LCOE (Euro/kWh)', color='orange', linestyle='--', linewidth=1)
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_bess']), label='BESS LCOE (Euro/kWh)', color='green', linestyle='--', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Price (Euro/kWh)')
    plt.title(f'Energy Market Prices\n{data["bidding_zone_desc"]}\n{data["period_str"]}')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0, 168, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    # Subplot 2: Revenue and Cost Streams
    plt.subplot(3, 1, 2)
    plt.plot(data['time_steps'], revenues['pv_to_consumer_rev'], label='PV to Consumer Revenue (Euro/step)', color='orange', linewidth=1)
    plt.plot(data['time_steps'], revenues['pv_to_grid_rev'], label='PV to Grid Revenue (Euro/step)', color='darkorange', linewidth=1)
    plt.plot(data['time_steps'], revenues['pv_to_bess_cost'], label='PV to BESS Cost (Euro/step)', color='gold', linewidth=1)
    plt.plot(data['time_steps'], revenues['bess_to_consumer_rev'], label='BESS to Consumer Revenue (Euro/step)', color='green', linewidth=1)
    plt.plot(data['time_steps'], revenues['bess_to_grid_rev'], label='BESS to Grid Revenue (Euro/step)', color='darkgreen', linewidth=1)
    plt.plot(data['time_steps'], revenues['grid_buy_cost'], label='Grid Buy Cost (Euro/step)', color='magenta', linewidth=1)
    plt.plot(data['time_steps'], revenues['penalty_per_step'], label='Penalty Cost (Euro/step)', color='red', linestyle='--', linewidth=1)
    plt.xlabel('Time (h)')
    plt.ylabel('Euro/step')
    plt.title('Revenue and Cost Streams')
    plt.legend(loc='upper right', fontsize=9, ncol=2)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0, 168, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    # Subplot 3: Total and Cumulative Revenue
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax1.plot(data['time_steps'], revenues['total_net_per_step'], label='Total Revenue per Step (Euro)', color='purple', linewidth=1)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Revenue per Step (Euro)')
    ax1.set_title('Total Revenue per Step and Cumulative')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle=':', linewidth=0.7)
    ax1.set_xticks(np.arange(0, 168, 24))
    ax1.set_xticklabels(data['day_labels'])
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--', linewidth=0.7)
    ax2 = ax1.twinx()
    cumulative_revenue = np.cumsum(revenues['total_net_per_step'])
    ax2.plot(data['time_steps'], cumulative_revenue, label='Cumulative Total Revenue (Euro)', color='orange', linestyle='--', linewidth=1)
    ax2.set_ylabel('Cumulative Revenue (Euro)')
    ax2.legend(loc='upper right', fontsize=9)
    plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.06, left=0.07, right=0.97)
    plt.tight_layout(pad=1.5)
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = data.get('plot_suffix', '')
    plt.savefig(os.path.join(save_dir, f'Financials{suffix}.png'), dpi=200, bbox_inches='tight')
    plt.close()