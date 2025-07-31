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
    for key in ['BESS_Capacity', 'BESS_Power_Limit', 'BESS_Efficiency_Charge', 'BESS_Efficiency_Discharge', 'SOC_Initial', 'BESS_limit', 'CONVERTER_EFFICIENCY']:
        if key in constants_data['Parameter'].values:
            params[key] = float(constants_data[constants_data['Parameter'] == key]['Value'].iloc[0])
    return params

class MPC:
    def __init__(self, delta_t, constants_path=None):
        params = load_bess_params(constants_path)
        self.bess_capacity = params['BESS_Capacity']
        self.bess_power_limit = params['BESS_Power_Limit']
        self.eta_charge = params['BESS_Efficiency_Charge']
        self.eta_discharge = params['BESS_Efficiency_Discharge']
        self.soc_initial = params['SOC_Initial']
        self.delta_t = delta_t
        self.bess_percent_limit = params['BESS_limit']
        self.converter_efficiency = params.get('CONVERTER_EFFICIENCY', 0.98)

    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, pi_ev, pi_consumer, horizon, start_dt):
        # Debug: print input parameters
        print("PV forecast (first 5):", pv_forecast[:5])
        print("BESS power limit:", self.bess_power_limit)
        print("BESS capacity:", self.bess_capacity)
        print("Grid buy price (first 5):", buy_forecast[:5])
        print("Consumer demand (first 5):", demand_forecast[:5])
        print("EV demand (first 5):", ev_forecast[:5])

        # Use actual simulation start date for snapshots
        snapshots = pd.date_range(start_dt, periods=horizon, freq=f'{int(self.delta_t*60)}min')
        n = pypsa.Network()
        n.set_snapshots(snapshots)

        n.snapshot_weightings.objective = self.delta_t
        n.snapshot_weightings.generators = self.delta_t
        n.snapshot_weightings.stores = self.delta_t

        # Add buses
        n.add("Bus", "AC", carrier='AC')
        n.add("Bus", "DC", carrier='DC')
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
        if self.bess_power_limit > 0 and self.bess_capacity > 0:
            n.add("StorageUnit", "BESS", bus="BESS",
                  p_nom=self.bess_power_limit,
                  max_hours=self.bess_capacity/self.bess_power_limit,
                  efficiency_store=1.0,
                  efficiency_dispatch=1.0,
                  marginal_cost=0,
                  state_of_charge_initial=soc)
        
        # Link for grid charging BESS
        n.add("Link", "Grid_to_BESS", bus0="Grid", bus1="BESS", p_nom=self.bess_power_limit, efficiency=self.eta_charge, marginal_cost=0)
        #Cost when charging from Grid
        n.links_t.marginal_cost["Grid_to_BESS"] = pd.Series(buy_forecast, index=n.snapshots)
        
        # Link for PV charging BESS
        n.add("Link", "PV_to_BESS", bus0="PV", bus1="BESS", p_nom=self.bess_power_limit, efficiency=self.eta_charge, marginal_cost=0)
        
        # Link for BESS discharge to AC bus
        n.add("Link", "BESS_to_AC", bus0="BESS", bus1="AC", p_nom=self.bess_power_limit, efficiency=self.eta_discharge, marginal_cost=0)

        # DC/AC converter for PV direct supply to AC bus
        n.add("Link", "PV_to_AC", bus0="PV", bus1="AC", p_nom=pv_nom, efficiency=0.98, marginal_cost=0)

        # Grid import/export as Links
        max_grid_import = np.max(demand_forecast) + np.max(ev_forecast) + self.bess_power_limit
        max_grid_export = np.max(pv_forecast) + self.bess_power_limit
        n.add("Link", "Grid_Import", bus0="Grid", bus1="AC", p_nom=max_grid_import, efficiency=1.0, marginal_cost=0, carrier='AC')
        n.add("Link", "Grid_Export", bus0="AC", bus1="Grid", p_nom=max_grid_export, efficiency=1.0, marginal_cost=0, carrier='AC')
        
        # BIG generator at Grid bus (reference, cost = buy_forecast)
        n.add("Generator", "Grid_Source", bus="Grid", p_nom=max_grid_import, marginal_cost=0)
        max_grid_import = 1e9
        
        #Revenue of selling Energy
        n.links_t.marginal_cost["Grid_Export"] = -pd.Series(sell_forecast, index=n.snapshots)
        
        #Cost of Energy Bought from Grid
        n.generators_t.marginal_cost["Grid_Source"] = pd.Series(buy_forecast, index=n.snapshots)

        # Loads with incentive EV at pi_ev, cons at pi_consumer
        n.add("Load", "Consumer", bus="AC", p_set=0, marginal_cost=0)
        n.add("Load", "EV", bus="AC", p_set=0, marginal_cost=0)
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Add dummy curtailment load on DC to penalize curtailment (lower penalty)
        #curtailment_penalty = 0.01  # Much lower penalty, allows curtailment if needed
        #n.add("Load", "Curtail", bus="DC", p_set=0, marginal_cost=curtailment_penalty)

        # Create model
        n.optimize.create_model()

        #PyPSA Handels SOC bounds of Storage systems internally (later set lower bound)
        # Add SOC constraints (if BESS exists)
        #if "BESS" in n.storage_units.index:
        #    soc_var = n.model["StorageUnit-state_of_charge"]
        #    n.model.add_constraints(soc_var >= min_soc, name="SOC_min")
        #    n.model.add_constraints(soc_var <= max_soc, name="SOC_max")

        try:
            n.optimize.solve_model(solver_name="highs", solver_options={"parallel": "on"})
        except Exception as e:
            print(f"HiGHS failed: {e}. Falling back to Gurobi.")
            n.optimize.solve_model(solver_name="gurobi", solver_options={"DualReductions": 0})
        
        # Fail-safe: check if solved
        if not n.is_solved:
            print("Optimization failed: infeasible or unbounded.")
            # Compute IIS for debug
            n.model.solver_model.computeIIS()
            n.model.solver_model.write("iis.ilp")  # Inspect this file for conflicting constraints
            print("IIS written to iis.ilp - open it to see conflicting constraints (e.g., SOC bounds vs balance).")
            
            # Define variables for failed case
            bess_dispatch = 0
            soc_next = soc
            pv_gen = 0
            link_dc_to_ac = 0
            grid_import = 0
            grid_export = 0
            consumer_load = demand_forecast[0]
            ev_load = ev_forecast[0]
        else:
            # Defensive extraction for PV
            if "PV" in n.generators_t.p.columns:
                pv_gen = n.generators_t.p["PV"].values[0]
            else:
                pv_gen = 0

            # Defensive extraction for BESS
            if "BESS" in n.storage_units_t.p.columns:
                bess_dispatch = n.storage_units_t.p["BESS"].values[0]
                soc_next = n.storage_units_t.state_of_charge["BESS"].values[0]
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
            
            grid_to_bess = 0
            if "Grid_to_BESS" in n.links_t.p0.columns:
                grid_to_bess = n.links_t.p0["Grid_to_BESS"].values[0]
            pv_to_bess = 0
            if "PV_to_BESS" in n.links_t.p0.columns:
                pv_to_bess = n.links_t.p0["PV_to_BESS"].values[0]

            # Defensive extraction for loads
            if "Consumer" in n.loads_t.p.columns:
                consumer_load = n.loads_t.p["Consumer"].values[0]
            else:
                consumer_load = 0
            if "EV" in n.loads_t.p.columns:
                ev_load = n.loads_t.p["EV"].values[0]
            else:
                ev_load = 0

        # Add balance debug
        dc_balance = pv_gen + grid_to_bess + pv_to_bess - bess_dispatch - (link_dc_to_ac / self.converter_efficiency)
        print(f"DC balance check: {dc_balance:.2f} (should ~0)")

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
            'P_BESS': bess_dispatch,           # kW
            'SOC_next': soc_next,              # kWh
            'P_BESS_discharge': np.maximum(bess_dispatch, 0),  # kW
            'P_grid_to_bess': grid_to_bess,  # kW
            'P_BESS_charge': np.abs(np.minimum(bess_dispatch, 0)),  # kW
            'P_PV_gen': pv_gen,                # kW
            'P_link_dc_to_ac': link_dc_to_ac,  # kW
            'P_grid_import': grid_import,      # kW
            'P_grid_export': grid_export,      # kW
            'pv_bess_to_consumer': pv_bess_to_consumer,  # kW
            'pv_bess_to_ev': pv_bess_to_ev,    # kW
            'pv_bess_to_grid': pv_bess_to_grid, # kW
            'grid_to_consumer': grid_to_consumer, # kW
            'grid_to_ev': grid_to_ev,          # kW
            'ev_renewable_share': ev_renewable_share,
            'ev_revenue': ev_revenue,
            'bess_capacity': self.bess_capacity, # kWh
            'bess_power_limit': self.bess_power_limit, # kW
            'bess_efficiency_charge': self.eta_charge,
            'bess_efficiency_discharge': self.eta_discharge,
            'soc_initial': self.soc_initial,    # kWh
            'bess_percent_limit': self.bess_percent_limit,
            'slack': 0.0
        }

