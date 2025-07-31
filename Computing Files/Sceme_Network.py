import pypsa
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx

# Import the MPC class from mpc.py (assuming it's in the same directory)
from mpc import MPC

# Function to build and plot the network with enhanced labels
def plot_network_scheme():
    # Instantiate MPC with a sample delta_t (e.g., 0.25 hours = 15 minutes)
    mpc = MPC(delta_t=0.25)

    # Provide dummy forecasts and parameters for a short horizon to build the network
    # These are placeholders; actual values aren't needed for topology plot
    horizon = 1  # Minimal horizon to build snapshots
    soc = mpc.soc_initial  # Use initial SOC
    pv_forecast = [10.0] * horizon  # Dummy PV forecast (kW)
    demand_forecast = [5.0] * horizon  # Dummy consumer demand (kW)
    ev_forecast = [2.0] * horizon  # Dummy EV demand (kW)
    buy_forecast = [0.1] * horizon  # Dummy buy prices ($/kWh)
    sell_forecast = [0.05] * horizon  # Dummy sell prices ($/kWh)
    lcoe_pv = 0.05  # Unused, but required
    pi_ev = 0.2  # Dummy EV incentive
    pi_consumer = 0.15  # Dummy consumer incentive

    # Simulate the network building here (copied from mpc.py's predict method)
    snapshots = pd.date_range("2024-01-01", periods=horizon, freq=f'{int(mpc.delta_t*60)}min')
    n = pypsa.Network()
    n.set_snapshots(snapshots)

    n.snapshot_weightings.objective = mpc.delta_t
    n.snapshot_weightings.generators = mpc.delta_t
    n.snapshot_weightings.stores = mpc.delta_t

    # Add buses
    n.add("Bus", "AC", carrier='AC')
    n.add("Bus", "PV", carrier='DC')
    n.add("Bus", "Grid", carrier='AC')
    n.add("Bus", "BESS", carrier='DC')

    # PV generator (PV bus)
    pv_nom = max(pv_forecast)
    pv_max = np.max(pv_forecast)
    if pv_max == 0:
        pv_max = 1
    n.add("Generator", "PV", bus="PV", p_nom=pv_nom, p_max_pu=pv_forecast/pv_max, marginal_cost=0)

    # BESS StorageUnit (BESS bus)
    if mpc.bess_power_limit > 0 and mpc.bess_capacity > 0:
        n.add("StorageUnit", "BESS", bus="BESS",
              p_nom=mpc.bess_power_limit,
              max_hours=mpc.bess_capacity/mpc.bess_power_limit,
              efficiency_store=1.0,
              efficiency_dispatch=1.0,
              marginal_cost=0,
              state_of_charge_initial=soc)
    
    # Link for grid charging BESS
    n.add("Link", "Grid_to_BESS", bus0="Grid", bus1="BESS", p_nom=mpc.bess_power_limit, efficiency=mpc.eta_charge, marginal_cost=0)
    
    # Link for PV charging BESS
    n.add("Link", "PV_to_BESS", bus0="PV", bus1="BESS", p_nom=mpc.bess_power_limit, efficiency=mpc.eta_charge, marginal_cost=0)
    
    # Link for BESS discharge to AC bus
    n.add("Link", "BESS_to_AC", bus0="BESS", bus1="AC", p_nom=mpc.bess_power_limit, efficiency=mpc.eta_discharge, marginal_cost=0)

    # DC/AC converter for PV direct supply to AC bus
    n.add("Link", "PV_to_AC", bus0="PV", bus1="AC", p_nom=pv_nom, efficiency=0.98, marginal_cost=0)

    # Grid import/export as Links
    max_grid_import = np.max(demand_forecast) + np.max(ev_forecast) + mpc.bess_power_limit
    max_grid_export = np.max(pv_forecast) + mpc.bess_power_limit
    n.add("Link", "Grid_Import", bus0="Grid", bus1="AC", p_nom=max_grid_import, efficiency=1.0, marginal_cost=0, carrier='AC')
    n.add("Link", "Grid_Export", bus0="AC", bus1="Grid", p_nom=max_grid_export, efficiency=1.0, marginal_cost=0, carrier='AC')
    
    # Grid source generator
    n.add("Generator", "Grid_Source", bus="Grid", p_nom=1e9, marginal_cost=0)

    # Loads
    n.add("Load", "Consumer", bus="AC", p_set=0, marginal_cost=-pi_consumer)
    n.add("Load", "EV", bus="AC", p_set=0, marginal_cost=-pi_ev)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (buses)
    for bus in n.buses.index:
        G.add_node(bus)

    # Add directed edges for links
    for idx, row in n.links.iterrows():
        G.add_edge(row.bus0, row.bus1, name=idx, efficiency=row.efficiency, marginal_cost=row.marginal_cost)

    # Define custom positions for better layout
    pos = {
        'Grid': (-2, 0),
        'AC': (2, 0),
        'PV': (0, 2),
        'BESS': (0, -2)
    }

    # Bus colors
    bus_colors_dict = {'AC': 'blue', 'PV': 'green', 'Grid': 'red', 'BESS': 'orange'}
    node_colors = [bus_colors_dict.get(node, 'gray') for node in G.nodes()]

    # Prepare edge labels
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        label = f"{d['name']}\neff: {d['efficiency']:.2f}\ncost: {d['marginal_cost']:.2f}"
        edge_labels[(u, v)] = label

    # Plot using networkx
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)  # Bus labels

    # Separate edges: straight for most, curved for Grid_Export to avoid overlap
    straight_edges = [(u, v) for u, v in G.edges if not (u == 'AC' and v == 'Grid')]
    curved_edges = [(u, v) for u, v in G.edges if u == 'AC' and v == 'Grid']

    # Draw straight edges with arrows
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, arrows=True, arrowstyle='->', arrowsize=20, width=2.0, ax=ax)

    # Draw curved edge for export with arrow
    nx.draw_networkx_edges(G, pos, edgelist=curved_edges, arrows=True, arrowstyle='->', arrowsize=20, width=2.0, connectionstyle='arc3,rad=0.3', ax=ax)

    # Draw edge labels with horizontal orientation
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8, ax=ax, rotate=False)

    # Add title
    plt.title("Energy Network Topology with Labels")

    # Add annotations for attached components
    annotation_offset = 0.3  # Increased slightly for better spacing
    for bus in n.buses.index:
        bus_pos = pos[bus]
        # Generators at bus
        gens = n.generators[n.generators.bus == bus].index
        if not gens.empty:
            ax.text(bus_pos[0], bus_pos[1] + annotation_offset, f"Gens: {', '.join(gens)}", ha='center', va='bottom', fontsize=8)
        # Loads
        loads = n.loads[n.loads.bus == bus].index
        if not loads.empty:
            ax.text(bus_pos[0], bus_pos[1] - annotation_offset, f"Loads: {', '.join(loads)}", ha='center', va='top', fontsize=8)
        # Storage
        stores = n.storage_units[n.storage_units.bus == bus].index
        if not stores.empty:
            ax.text(bus_pos[0], bus_pos[1] - 2*annotation_offset, f"Storage: {', '.join(stores)}", ha='center', va='top', fontsize=8)

    # Adjust limits and show
    plt.tight_layout()
    plt.show()

    # Optionally save it
    plt.savefig('energy_network_scheme_enhanced.png', dpi=300, bbox_inches='tight')

# Call the function to plot
if __name__ == "__main__":
    plot_network_scheme()