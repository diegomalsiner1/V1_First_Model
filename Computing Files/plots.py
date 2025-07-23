import matplotlib.pyplot as plt
import numpy as np
import os

def plot_energy_flows(results, data, revenues):
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(data['time_steps'], data['pv_power'], label='PV Generation (kW)', color='orange')
    plt.plot(data['time_steps'], results['P_grid_sold'], label='Grid Sold (kW)', color='blue')
    plt.plot(data['time_steps'], results['P_grid_bought'], label='Grid Bought (kW)', color='magenta')
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title('PV and Grid Flows')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax1.plot(data['time_steps'], results['P_BESS_charge'], label='BESS Charge (kW)', color='blue')
    ax1.plot(data['time_steps'], results['P_BESS_discharge'], label='BESS Discharge (kW)', color='red')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('BESS Flows and SOC')
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, 169, 24))
    ax1.set_xticklabels(data['day_labels'])
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0, 168, len(results['SOC_vals'])), results['SOC_vals'], label='SOC (kWh)', color='green', linestyle='--')
    ax2.set_ylabel('SOC (kWh)')
    ax2.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    plt.stackplot(data['time_steps'], results['P_PV_consumer_vals'], results['P_BESS_consumer_vals'], results['P_grid_consumer_vals'], results['slack_vals'],
                  labels=['PV to Consumer (kW)', 'BESS to Consumer (kW)', 'Grid to Consumer (kW)', 'Unmet Demand (kW)'],
                  colors=['orange', 'green', 'blue', 'red'])
    plt.plot(data['time_steps'], data['consumer_demand'], label='Demand (kW)', color='black', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Power (kW)')
    plt.title(f'Consumer Flow Composition (Self-Sufficiency: {revenues["self_sufficiency"]:.2f}%)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(pad=1.5)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Energy_Flows.png'))
    plt.show()

def plot_financials(revenues, data):
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(data['time_steps'], data['grid_buy_price'], label='Grid Buy Price (Euro/kWh)', color='blue')
    plt.plot(data['time_steps'], data['grid_sell_price'], label='Grid Sell Price (Euro/kWh)', color='cyan', linestyle='--')
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_pv']), label='PV LCOE (Euro/kWh)', color='orange', linestyle='--')
    plt.plot(data['time_steps'], np.full(data['n_steps'], data['lcoe_bess']), label='BESS LCOE (Euro/kWh)', color='green', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Price (Euro/kWh)')
    plt.title(f'Energy Market Prices\n{data["bidding_zone_desc"]}\n{data["period_str"]}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    plt.subplot(3, 1, 2)
    plt.plot(data['time_steps'], revenues['pv_to_consumer_rev'], label='PV to Consumer Revenue (Euro/step)', color='orange')
    plt.plot(data['time_steps'], revenues['pv_to_grid_rev'], label='PV to Grid Revenue (Euro/step)', color='darkorange')
    plt.plot(data['time_steps'], revenues['pv_to_bess_cost'], label='PV to BESS Cost (Euro/step)', color='gold')
    plt.plot(data['time_steps'], revenues['bess_to_consumer_rev'], label='BESS to Consumer Revenue (Euro/step)', color='green')
    plt.plot(data['time_steps'], revenues['bess_to_grid_rev'], label='BESS to Grid Revenue (Euro/step)', color='darkgreen')
    plt.plot(data['time_steps'], revenues['grid_buy_cost'], label='Grid Buy Cost (Euro/step)', color='magenta')
    plt.plot(data['time_steps'], revenues['penalty_per_step'], label='Penalty Cost (Euro/step)', color='red', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Euro/step')
    plt.title('Revenue and Cost Streams')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(np.arange(0, 169, 24), data['day_labels'])
    for d in range(1, 7):
        plt.axvline(d * 24, color='gray', linestyle='--')

    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax1.plot(data['time_steps'], revenues['total_net_per_step'], label='Total Revenue per Step (Euro)', color='purple')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Revenue per Step (Euro)')
    ax1.set_title('Total Revenue per Step and Cumulative')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, 169, 24))
    ax1.set_xticklabels(data['day_labels'])
    for d in range(1, 7):
        ax1.axvline(d * 24, color='gray', linestyle='--')

    ax2 = ax1.twinx()
    cumulative_revenue = np.cumsum(revenues['total_net_per_step'])
    ax2.plot(data['time_steps'], cumulative_revenue, label='Cumulative Total Revenue (Euro)', color='orange', linestyle='--')
    ax2.set_ylabel('Cumulative Revenue (Euro)')
    ax2.legend(loc='upper right')

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(pad=1.5)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'Financials.png'))
    plt.show()