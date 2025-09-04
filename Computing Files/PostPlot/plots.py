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
    # Subplot 1: Market Prices with per-timestep coloring based ONLY on BESS↔Grid exchange
    plt.subplot(3, 1, 1)
    # Get per-step BESS-specific import/export (Grid→BESS and BESS→Grid)
    bess_grid_import = np.asarray(data.get('bess_grid_import_vals', np.zeros_like(data['time_steps'])))
    bess_grid_export = np.asarray(data.get('bess_grid_export_vals', np.zeros_like(data['time_steps'])))

    # Build robust buy/sell masks (avoid numerical noise and enforce exclusivity)
    thr = max(1e-3, 1e-4 * (np.max(np.abs(bess_grid_import) + np.abs(bess_grid_export)) + 1.0))
    # Green when BESS is buying from grid (Grid→BESS), Red when BESS is selling to grid (BESS→Grid)
    buy_mask = bess_grid_import > thr
    sell_mask = bess_grid_export > thr

    # Color each timestep: green when buying, red when selling
    ts = np.asarray(data['time_steps'])
    dt = float(data.get('delta_t', 0.25))
    for i in range(len(ts)):
        x0 = ts[i]
        x1 = x0 + dt
        if sell_mask[i]:
            plt.axvspan(x0, x1, color='red', alpha=0.18, linewidth=0, zorder=0)
        elif buy_mask[i]:
            plt.axvspan(x0, x1, color='green', alpha=0.18, linewidth=0, zorder=0)

    # Plot price lines on top
    plt.plot(data['time_steps'], data['grid_buy_price'], label='Grid Buy Price (Euro/kWh)', color='blue', linewidth=2, zorder=2)
    plt.plot(data['time_steps'], data['grid_sell_price'], label='Grid Sell Price (Euro/kWh)', color='lightblue', linestyle='--', linewidth=2, zorder=2)
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_pv']), label='PV LCOE (Euro/kWh)', color='orange', linestyle='--', linewidth=1)
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_bess']), label='BESS LCOE (Euro/kWh)', color='green', linestyle='--', linewidth=1)
    
    # Calculate and display arbitrage statistics
    buying_periods = grid_import > 1e-6
    selling_periods = grid_export > 1e-6
    total_buying_time = np.sum(buying_periods) * data['delta_t']  # hours
    total_selling_time = np.sum(selling_periods) * data['delta_t']  # hours
    total_energy_bought = np.sum(grid_import) * data['delta_t']  # kWh
    total_energy_sold = np.sum(grid_export) * data['delta_t']  # kWh
    avg_buy_price = float(np.mean(np.array(data['grid_buy_price'])[buying_periods])) if np.any(buying_periods) else 0
    avg_sell_price = float(np.mean(np.array(data['grid_sell_price'])[selling_periods])) if np.any(selling_periods) else 0
    
    plt.xlabel('Time (h)')
    plt.ylabel('Price (Euro/kWh)')
    plt.title(f'Energy Market Price {data.get("price_source")} - Arbitrage Visualization\n'
              f'{data["period_str"]} | Buy: {total_buying_time:.1f}h ({total_energy_bought:.0f}kWh) @ {avg_buy_price:.3f}€/kWh | '
              f'Sell: {total_selling_time:.1f}h ({total_energy_sold:.0f}kWh) @ {avg_sell_price:.3f}€/kWh')
    # Add legend entries for coloring (BESS↔Grid only)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='green', alpha=0.1, label='BESS buying from Grid'),
        Patch(facecolor='red', alpha=0.1, label='BESS selling to Grid')
    ]
    price_lines = plt.legend(loc='upper right', fontsize=9)
    plt.gca().add_artist(price_lines)
    plt.legend(handles=legend_handles, loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(x_ticks, x_labels)
    for d in range(1, len(x_ticks)):
        plt.axvline(x_ticks[d], color='gray', linestyle='--', linewidth=0.7)
    # Subplot 2: Revenue and Cost Streams (use new separated terms)
    plt.subplot(3, 1, 2)
    plt.plot(data['time_steps'], revenues.get('pv_to_grid_rev', np.zeros_like(data['time_steps'])), label='PV to Grid Revenue (Euro/step)', color='darkorange', linewidth=1)
    plt.plot(data['time_steps'], -revenues.get('grid_buy_cost', np.zeros_like(data['time_steps'])), label='Grid Buy Cost (Euro/step)', color='blue', linewidth=1, linestyle='--')
    plt.plot(data['time_steps'], revenues.get('bess_to_grid_rev', np.zeros_like(data['time_steps'])), label='BESS to Grid Revenue (Euro/step)', color='black', linewidth=1)
    plt.plot(data['time_steps'], revenues.get('pv_to_ev_rev', np.zeros_like(data['time_steps'])), label='PV to EV Revenue (Euro/step)', color='red', linewidth=1)
    plt.plot(data['time_steps'], revenues.get('bess_to_ev_rev', np.zeros_like(data['time_steps'])), label='BESS to EV Revenue (Euro/step)', color='green', linewidth=1)
    plt.plot(data['time_steps'], -revenues.get('bess_grid_charging_cost', np.zeros_like(data['time_steps'])), label='BESS Grid Charging Cost (Euro/step)', color='purple', linewidth=1, linestyle='--')
    plt.plot(data['time_steps'], -revenues.get('bess_efficiency_loss_cost', np.zeros_like(data['time_steps'])), label='BESS Efficiency Loss Cost (Euro/step)', color='brown', linewidth=1, linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Euro/step')
    plt.title('Revenue and Cost Streams')
    plt.legend(loc='upper right', fontsize=8, ncol=2, bbox_to_anchor=(1.0, 1.0))
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
    # Save plot - honor provided save_dir if given, otherwise default to Output Files
    if save_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(base_dir, 'Output Files')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = data.get('plot_suffix', '')
    save_path = os.path.join(save_dir, f'Financials{suffix}.png')
    print(f"Saving financials plot to: {save_path}")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
