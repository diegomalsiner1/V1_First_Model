import pypsa
import numpy as np
import logging
import pandas as pd
import os
import datetime  # For pd.date_range

logging.basicConfig(level=logging.INFO)

def load_bess_percent_limit(constants_path=None):
    """Load BESS percent limit from constants CSV."""
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#')
    return float(constants_data[constants_data['Parameter'] == 'BESS_limit']['Value'].iloc[0])

def load_pi_ev(constants_path=None):
    """Load EV charging price from constants CSV."""
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#')
    try:
        return float(constants_data[constants_data['Parameter'] == 'EV_PRICE']['Value'].iloc[0])
    except IndexError:
        logging.warning("pi_ev not found in constants CSV; using default value 0.3")
        return 0.3  # Default value, e.g., 0.3 $/kWh; adjust as needed

def load_converter_efficiency(constants_path=None):
    """Load converter efficiency from constants CSV."""
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#')
    try:
        return float(constants_data[constants_data['Parameter'] == 'CONVERTER_EFFICIENCY']['Value'].iloc[0])
    except IndexError:
        logging.warning("CONVERTER_EFFICIENCY not found in constants CSV; using default value 0.95")
        return 0.95

class MPC:
    def __init__(self, bess_capacity, bess_power_limit, eta_charge, eta_discharge, lcoe_bess, delta_t, constants_path=None):
        self.bess_capacity = bess_capacity
        self.bess_power_limit = bess_power_limit
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.lcoe_bess = lcoe_bess
        self.delta_t = delta_t
        self.bess_percent_limit = load_bess_percent_limit(constants_path)
        self.pi_ev = load_pi_ev(constants_path)
        self.converter_efficiency = load_converter_efficiency(constants_path)
        self.alpha = 0.00001  # Small incentive for BESS usage

    def predict(self, soc_current, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, horizon):
        # Create PyPSA network
        net = pypsa.Network()

        # Add buses
        net.add("Bus", "DC_bus")
        net.add("Bus", "AC_bus")

        # Set snapshots with date_range for compatibility (assuming delta_t=1 hour)
        net.set_snapshots(pd.date_range("2024-01-01", periods=horizon, freq="h"))

        # Add PV generator on DC_bus
        net.add("Generator", "PV", bus="DC_bus", p_nom=1e6, marginal_cost=lcoe_pv, 
                p_set=pv_forecast)

        # Add BESS as Store on DC_bus
        net.add("Store", "BESS", bus="DC_bus", e_nom=self.bess_capacity, e_initial=soc_current * self.bess_capacity,
                e_min_pu=self.bess_percent_limit, e_cyclic=True, 
                marginal_cost=self.lcoe_bess, standing_loss=0.001)

        # Add converter (bidirectional Link)
        net.add("Link", "Converter DC to AC", bus0="DC_bus", bus1="AC_bus", efficiency=self.converter_efficiency, p_nom=1e6, marginal_cost=0.001)
        net.add("Link", "Converter AC to DC", bus0="AC_bus", bus1="DC_bus", efficiency=self.converter_efficiency, p_nom=1e6, marginal_cost=0.001)

        # Add loads on AC_bus
        net.add("Load", "Consumer", bus="AC_bus", p_set=demand_forecast)
        net.add("Load", "EV", bus="AC_bus", p_set=ev_forecast)

        # Add grid as bidirectional link with time-varying costs
        net.add("Bus", "Grid_bus")
        net.add("Generator", "Grid Source", bus="Grid_bus", p_nom=1e6, marginal_cost=0)
        net.add("Link", "Grid Import Link", bus0="Grid_bus", bus1="AC_bus", p_nom=1e6, efficiency=1.0)
        net.add("Link", "Grid Export Link", bus0="AC_bus", bus1="Grid_bus", p_nom=1e6, efficiency=1.0)

        # Set time-varying marginal costs
        net.links_t.marginal_cost["Grid Import Link"] = buy_forecast
        net.links_t.marginal_cost["Grid Export Link"] = -sell_forecast  # Negative for revenue

        # Optimize the network (pyomo=False to avoid Pyomo issues, use linopy directly)
        net.optimize(solver_name="gurobi", pyomo=False, solver_options={'MIPGap': 0.025, 'TimeLimit': 30})

        # Extract results for first step (k=0)
        k = 0
        pv_output = net.generators_t.p.iloc[k]["PV"]
        bess_p = net.stores_t.p.iloc[k]["BESS"]  # Positive = charge, negative = discharge
        bess_charge = max(bess_p, 0)
        bess_discharge = -min(bess_p, 0)
        soc_next = net.stores_t.e.iloc[min(k+1, horizon-1)]["BESS"] / self.bess_capacity if self.bess_capacity > 0 else 0
        converter_dc_ac = net.links_t.p.iloc[k]["Converter DC to AC"]
        converter_ac_dc = net.links_t.p.iloc[k]["Converter AC to DC"]
        grid_import_val = net.links_t.p.iloc[k]["Grid Import Link"]
        grid_export_val = net.links_t.p.iloc[k]["Grid Export Link"]

        # Post-process to derive explicit flows
        local_ac = converter_dc_ac * self.converter_efficiency - converter_ac_dc
        total_load = demand_forecast[k] + ev_forecast[k]

        local_to_consumer = min(local_ac, demand_forecast[k])
        local_to_ev = min(max(local_ac - demand_forecast[k], 0), ev_forecast[k])
        local_to_grid = max(local_ac - demand_forecast[k] - ev_forecast[k], 0)

        net_dc_in = pv_output + bess_discharge
        pv_share = pv_output / max(net_dc_in, 1e-6)
        bess_share = bess_discharge / max(net_dc_in, 1e-6)

        pv_to_consumer = local_to_consumer * pv_share
        bess_to_consumer = local_to_consumer * bess_share
        pv_to_ev = local_to_ev * pv_share
        bess_to_ev = local_to_ev * bess_share
        pv_to_grid = local_to_grid * pv_share
        bess_to_grid = local_to_grid * bess_share

        pv_to_bess = min(pv_output, bess_charge)
        grid_to_bess = max(0, bess_charge - pv_to_bess)

        grid_to_consumer = max(0, demand_forecast[k] - local_to_consumer)
        grid_to_ev = max(0, ev_forecast[k] - local_to_ev)

        p_bess = bess_discharge - bess_charge

        return {
            'P_BESS': p_bess,
            'SOC_next': soc_next,
            'pv_to_consumer': pv_to_consumer,
            'pv_to_ev': pv_to_ev,
            'bess_to_consumer': bess_to_consumer,
            'bess_to_ev': bess_to_ev,
            'bess_to_grid': bess_to_grid,
            'pv_to_bess': pv_to_bess,
            'pv_to_grid': pv_to_grid,
            'grid_to_consumer': grid_to_consumer,
            'grid_to_ev': grid_to_ev,
            'grid_to_bess': grid_to_bess,
            'slack': 0.0
        }