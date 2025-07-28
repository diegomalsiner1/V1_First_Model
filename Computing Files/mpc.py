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

def load_bess_percent_limit(constants_path=None):
    """Load BESS percent limit from constants CSV."""
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#')
    return float(constants_data[constants_data['Parameter'] == 'BESS_limit']['Value'].iloc[0])

class MPC:
    def __init__(self, bess_capacity, bess_power_limit, eta_charge, eta_discharge, lcoe_bess, soc_initial, delta_t, constants_path=None):
        self.bess_capacity = bess_capacity
        self.bess_power_limit = bess_power_limit
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.lcoe_bess = lcoe_bess
        self.soc_initial = soc_initial
        self.delta_t = delta_t
        self.bess_percent_limit = load_bess_percent_limit(constants_path)

    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, pi_ev, horizon):
        # Prepare time index for PyPSA
        snapshots = pd.date_range("2024-01-01", periods=horizon, freq=f'{int(self.delta_t*60)}min')
        n = pypsa.Network()
        n.set_snapshots(snapshots)

        # Add buses
        n.add("Bus", "AC")
        n.add("Bus", "DC")

        # Add DC/AC converter (trafo)
        n.add("Link", "DC_AC_Converter", bus0="DC", bus1="AC", p_nom=self.bess_power_limit*2, efficiency=0.98)

        # Add PV generator (DC bus)
        n.add("Generator", "PV", bus="DC", p_nom=max(pv_forecast), p_max_pu=pv_forecast/np.max(pv_forecast), marginal_cost=lcoe_pv)

        # Add BESS (DC bus) with zero marginal cost to enable arbitrage
        n.add("StorageUnit", "BESS", bus="DC",
              p_nom=self.bess_power_limit,
              max_hours=self.bess_capacity/self.bess_power_limit,
              efficiency_store=self.eta_charge,
              efficiency_dispatch=self.eta_discharge,
              marginal_cost=0,  # Set to zero to allow arbitrage
              state_of_charge_initial=soc/self.bess_capacity)

        # Add grid import/export (AC bus) with scalar marginal_cost
        n.add("Generator", "Grid_Import", bus="AC", p_nom=1e6, marginal_cost=0)
        n.add("Generator", "Grid_Export", bus="AC", p_nom=1e6, marginal_cost=0)

        # Add consumer and EV loads (AC bus) with scalar p_set
        n.add("Load", "Consumer", bus="AC", p_set=0)
        n.add("Load", "EV", bus="AC", p_set=0)

        # Assign time series to generators and loads
        n.generators_t.marginal_cost.loc[:, "Grid_Import"] = buy_forecast
        n.generators_t.marginal_cost.loc[:, "Grid_Export"] = -sell_forecast
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Run linear optimal power flow (LOPF)
        n.optimize.create_model()
        n.optimize.solve_model(solver_name="gurobi", verborse=True, MIPGap=0.025)

        # Extract results for the first step (MPC receding horizon)
        pv_to_dc = n.generators_t.p["PV"].values[0]
        bess_to_dc = n.storage_units_t.p["BESS"].values[0]
        grid_import = n.generators_t.p["Grid_Import"].values[0]
        grid_export = n.generators_t.p["Grid_Export"].values[0]
        consumer_load = n.loads_t.p["Consumer"].values[0]
        ev_load = n.loads_t.p["EV"].values[0]
        soc_next = n.storage_units_t.state_of_charge["BESS"].values[1] * self.bess_capacity if len(n.storage_units_t.state_of_charge["BESS"]) > 1 else soc

        # For compatibility with the rest of the code, map flows to expected outputs
        return {
            'P_BESS': bess_to_dc,
            'SOC_next': soc_next,
            'pv_to_consumer': min(pv_to_dc, consumer_load),
            'pv_to_ev': min(pv_to_dc - min(pv_to_dc, consumer_load), ev_load),
            'bess_to_consumer': 0,  # Not directly tracked in PyPSA, can be inferred from storage dispatch if needed
            'bess_to_ev': 0,        # Not directly tracked in PyPSA, can be inferred from storage dispatch if needed
            'bess_to_grid': 0,      # Not directly tracked in PyPSA, can be inferred from storage dispatch if needed
            'pv_to_bess': 0,        # Not directly tracked in PyPSA, can be inferred from storage charging if needed
            'pv_to_grid': max(0, pv_to_dc - consumer_load - ev_load),
            'grid_to_consumer': max(0, consumer_load - pv_to_dc),
            'grid_to_ev': max(0, ev_load - max(0, pv_to_dc - consumer_load)),
            'grid_to_bess': 0,      # Not directly tracked in PyPSA, can be inferred from storage charging if needed
            'slack': 0.0
        }