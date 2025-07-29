import pypsa
import pandas as pd
import numpy as np
import os

def load_bess_params(constants_path=None):
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#', header=None, names=['Parameter', 'Value'])
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
        self.soc_initial = params['SOC_Initial']
        self.delta_t = 0.25  # 15 min
        self.bess_percent_limit = params['BESS_limit']

    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, pi_ev, horizon):
        # Debug: print input parameters
        print("PV forecast (first 5):", pv_forecast[:5])
        print("BESS power limit:", self.bess_power_limit)
        print("BESS capacity:", self.bess_capacity)
        print("Grid buy price (first 5):", buy_forecast[:5])
        print("Consumer demand (first 5):", demand_forecast[:5])
        print("EV demand (first 5):", ev_forecast[:5])

        snapshots = pd.date_range("2024-01-01", periods=horizon, freq=f'{int(self.delta_t*60)}min')
        n = pypsa.Network()
        n.set_snapshots(snapshots)

        # Add buses
        n.add("Bus", "AC")
        n.add("Bus", "DC")
        n.add("Bus", "Grid")

        # DC/AC converter (trafo)
        n.add("Link", "DC_AC_Converter", bus0="DC", bus1="AC", p_nom=1e6, efficiency=0.98, marginal_cost=0)

        # PV generator (DC bus)
        pv_nom = max(pv_forecast)
        pv_max = np.max(pv_forecast)
        if pv_max == 0:
            pv_max = 1  # Prevent division by zero
        n.add("Generator", "PV", bus="DC", p_nom=pv_nom, p_max_pu=pv_forecast/pv_max, marginal_cost=0)

        # BESS (DC bus)
        print(f"BESS params: p_nom={self.bess_power_limit}, capacity={self.bess_capacity}")
        if self.bess_power_limit > 0 and self.bess_capacity > 0:
            n.add("StorageUnit", "BESS", bus="DC",
                  p_nom=self.bess_power_limit,
                  max_hours=self.bess_capacity/self.bess_power_limit,
                  efficiency_store=self.eta_charge,
                  efficiency_dispatch=self.eta_discharge,
                  marginal_cost=0,
                  state_of_charge_initial=soc,
                  state_of_charge_min=self.bess_percent_limit * self.bess_capacity,
                  state_of_charge_max=self.bess_capacity)

        # Grid import/export as Links (no artificial limits, p_nom=1e6)
        n.add("Link", "Grid_Import", bus0="Grid", bus1="AC", p_nom=1e6, efficiency=1.0, marginal_cost=pd.Series(buy_forecast, index=snapshots))
        n.add("Link", "Grid_Export", bus0="AC", bus1="Grid", p_nom=1e6, efficiency=1.0, marginal_cost=-pd.Series(sell_forecast, index=snapshots))

        # Infinite generator at Grid bus (reference, marginal_cost=buy_forecast)
        # This ensures grid can always supply at market price
        n.add("Generator", "Grid_Source", bus="Grid", p_nom=1e6, marginal_cost=pd.Series(buy_forecast, index=snapshots))

        # Loads (AC bus)
        n.add("Load", "Consumer", bus="AC", p_set=0)
        n.add("Load", "EV", bus="AC", p_set=0)
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Run optimization
        n.optimize.create_model()
        n.optimize.solve_model(solver_name="gurobi")

        # Fail-safe: check if solved
        if not n.is_solved:
            print("Optimization failed: infeasible or unbounded.")
            return {
                'P_BESS': 0,
                'SOC_next': soc,
                'P_BESS_discharge': 0,
                'P_BESS_charge': 0,
                'P_PV_gen': 0,
                'P_link_dc_to_ac': 0,
                'P_grid_import': 0,
                'P_grid_export': 0,
                'pv_bess_to_consumer': 0,
                'pv_bess_to_ev': 0,
                'pv_bess_to_grid': 0,
                'grid_to_consumer': 0,
                'grid_to_ev': 0,
                'ev_renewable_share': 0,
                'ev_revenue': 0,
                'bess_capacity': self.bess_capacity,
                'bess_power_limit': self.bess_power_limit,
                'bess_efficiency_charge': self.eta_charge,
                'bess_efficiency_discharge': self.eta_discharge,
                'soc_initial': self.soc_initial,
                'bess_percent_limit': self.bess_percent_limit,
                'slack': 0.0
            }

        # Defensive extraction for PV
        if "PV" in n.generators_t.p.columns:
            pv_gen = n.generators_t.p["PV"].values[0]
        else:
            pv_gen = 0

        # Defensive extraction for BESS
        if "BESS" in n.storage_units_t.p.columns:
            bess_dispatch = n.storage_units_t.p["BESS"].values[0]
            soc_next = n.storage_units_t.state_of_charge["BESS"].values[1] * self.bess_capacity if len(n.storage_units_t.state_of_charge["BESS"]) > 1 else soc
        else:
            bess_dispatch = 0
            soc_next = soc

        # Defensive extraction for links
        if "DC_AC_Converter" in n.links_t.p0.columns:
            link_dc_to_ac = n.links_t.p0["DC_AC_Converter"].values[0]
        else:
            link_dc_to_ac = 0
        if "Grid_Import" in n.links_t.p0.columns:
            grid_import = n.links_t.p0["Grid_Import"].values[0]
        else:
            grid_import = 0
        if "Grid_Export" in n.links_t.p0.columns:
            grid_export = n.links_t.p0["Grid_Export"].values[0]
        else:
            grid_export = 0

        # Defensive extraction for loads
        if "Consumer" in n.loads_t.p.columns:
            consumer_load = n.loads_t.p["Consumer"].values[0]
        else:
            consumer_load = 0
        if "EV" in n.loads_t.p.columns:
            ev_load = n.loads_t.p["EV"].values[0]
        else:
            ev_load = 0

        # Map DC to AC flow to demand and grid export
        dc_to_ac_available = max(link_dc_to_ac, 0)
        pv_bess_to_consumer = min(dc_to_ac_available, consumer_load)
        pv_bess_to_ev = min(max(0, dc_to_ac_available - pv_bess_to_consumer), ev_load)
        pv_bess_to_grid = max(0, dc_to_ac_available - pv_bess_to_consumer - pv_bess_to_ev)
        grid_to_consumer = max(0, consumer_load - pv_bess_to_consumer)
        grid_to_ev = max(0, ev_load - pv_bess_to_ev)

        # Calculate renewable share for EV charging and its revenue
        ev_renewable_share = pv_bess_to_ev / ev_load if ev_load > 0 else 0
        ev_revenue = pv_bess_to_ev * pi_ev  # Only renewable share, grid share is cost

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
            'ev_renewable_share': ev_renewable_share,
            'ev_revenue': ev_revenue,
            'bess_capacity': self.bess_capacity,
            'bess_power_limit': self.bess_power_limit,
            'bess_efficiency_charge': self.eta_charge,
            'bess_efficiency_discharge': self.eta_discharge,
            'soc_initial': self.soc_initial,
            'bess_percent_limit': self.bess_percent_limit,
            'slack': 0.0
        }