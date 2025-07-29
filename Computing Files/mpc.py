import pypsa
import pandas as pd
import numpy as np
import logging
import os

print("PyPSA version:", pypsa.__version__)
print("PyPSA file:", pypsa.__file__)
print("Has lopf:", hasattr(pypsa.Network(), 'lopf'))
print("Network dir:", dir(pypsa.Network()))

logging.basicConfig(level=logging.INFO)

def load_bess_params(constants_path=None):
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#')
    params = {}
    for key in ['BESS_Capacity', 'BESS_Power_Limit', 'BESS_Efficiency_Charge', 'BESS_Efficiency_Discharge', 'SOC_Initial', 'BESS_limit']:
        params[key] = float(constants_data[constants_data['Parameter'] == key]['Value'].iloc[0])
    return params

class MPC:
    def __init__(self, constants_path=None):
        params = load_bess_params(constants_path)
        self.bess_capacity = params['BESS_Capacity']
        self.bess_power_limit = params['BESS_Power_Limit']
        self.eta_charge = params['BESS_Efficiency_Charge']
        self.eta_discharge = params['BESS_Efficiency_Discharge']
        self.lcoe_bess = 0
        self.soc_initial = params['SOC_Initial']
        self.delta_t = 0.25  # 15 min
        self.bess_percent_limit = params['BESS_limit']

    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, pi_ev, horizon):
        # Prepare time index for PyPSA
        snapshots = pd.date_range("2024-01-01", periods=horizon, freq=f'{int(self.delta_t*60)}min')
        n = pypsa.Network()
        n.set_snapshots(snapshots)

        # Add buses
        n.add("Bus", "AC")
        n.add("Bus", "DC")
        n.add("Bus", "Grid")

        # Add DC/AC converter (trafo) with zero marginal cost and very large power limit
        n.add("Link", "DC_AC_Converter", bus0="DC", bus1="AC", p_nom=1e6, efficiency=0.98, marginal_cost=0)

        # Add PV generator (DC bus) with zero marginal cost
        n.add("Generator", "PV", bus="DC", p_nom=max(pv_forecast), p_max_pu=pv_forecast/np.max(pv_forecast), marginal_cost=0)

        # Add BESS (DC bus) with zero marginal cost
        n.add("StorageUnit", "BESS", bus="DC",
              p_nom=self.bess_power_limit,
              max_hours=self.bess_capacity/self.bess_power_limit,
              efficiency_store=self.eta_charge,
              efficiency_dispatch=self.eta_discharge,
              marginal_cost=0,
              state_of_charge_initial=soc/self.bess_capacity,
              state_of_charge_min=self.bess_percent_limit,
              state_of_charge_max=1.0)

        # Add grid import/export as Links to Grid bus
        n.add("Link", "Grid_Import", bus0="Grid", bus1="AC", p_nom=1e6, efficiency=1.0, marginal_cost=buy_forecast)
        n.add("Link", "Grid_Export", bus0="AC", bus1="Grid", p_nom=1e6, efficiency=1.0, marginal_cost=-sell_forecast)

        # Add infinite generator at Grid bus (reference)
        n.add("Generator", "Grid_Source", bus="Grid", p_nom=1e6, marginal_cost=0)

        # Add consumer and EV loads (AC bus) with scalar p_set
        n.add("Load", "Consumer", bus="AC", p_set=0)
        n.add("Load", "EV", bus="AC", p_set=0)

        # Assign time series to loads
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Run linear optimal power flow (LOPF)
        n.optimize.create_model()
        n.optimize.solve_model(solver_name="gurobi")

        # Extract flows for the first step (MPC receding horizon)
        pv_gen = n.generators_t.p["PV"].values[0]
        bess_dispatch = n.storage_units_t.p["BESS"].values[0]  # +ve: discharge, -ve: charge
        soc_next = n.storage_units_t.state_of_charge["BESS"].values[1] * self.bess_capacity if len(n.storage_units_t.state_of_charge["BESS"]) > 1 else soc
        link_dc_to_ac = n.links_t.p0["DC_AC_Converter"].values[0]  # Power from DC to AC (positive: DC->AC, negative: AC->DC)
        grid_import = n.links_t.p0["Grid_Import"].values[0]  # Power from Grid to AC (positive: import)
        grid_export = n.links_t.p0["Grid_Export"].values[0]  # Power from AC to Grid (positive: export)
        consumer_load = n.loads_t.p["Consumer"].values[0]
        ev_load = n.loads_t.p["EV"].values[0]

        # Map physical flows at the inverter (AC) bus
        total_ac_demand = consumer_load + ev_load
        dc_to_ac_available = max(link_dc_to_ac, 0)  # Only positive flow (DC->AC)
        pv_bess_to_consumer = min(dc_to_ac_available, consumer_load)
        pv_bess_to_ev = min(max(0, dc_to_ac_available - pv_bess_to_consumer), ev_load)
        pv_bess_to_grid = max(0, dc_to_ac_available - pv_bess_to_consumer - pv_bess_to_ev)
        grid_to_consumer = max(0, consumer_load - pv_bess_to_consumer)
        grid_to_ev = max(0, ev_load - pv_bess_to_ev)

        return {
            'P_BESS': bess_dispatch,
            'SOC_next': soc_next,
            'P_BESS_discharge': np.maximum(bess_dispatch, 0),
            'P_BESS_charge': np.abs(np.minimum(bess_dispatch, 0)),
            'P_PV_gen': pv_gen,
            'P_link_dc_to_ac': link_dc_to_ac,
            'P_grid_import': grid_import,
            'P_grid_export': grid_export,
            'pv_bess_to_consumer': pv_bess_to_consumer,
            'pv_bess_to_ev': pv_bess_to_ev,
            'pv_bess_to_grid': pv_bess_to_grid,
            'grid_to_consumer': grid_to_consumer,
            'grid_to_ev': grid_to_ev,
            'bess_capacity': self.bess_capacity,
            'bess_power_limit': self.bess_power_limit,
            'bess_efficiency_charge': self.eta_charge,
            'bess_efficiency_discharge': self.eta_discharge,
            'soc_initial': self.soc_initial,
            'bess_percent_limit': self.bess_percent_limit,
            'slack': 0.0
        }