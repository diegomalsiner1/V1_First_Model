import pypsa
import pandas as pd
import numpy as np
import os


def load_bess_params(constants_path=None):
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#', header=None, names=['Parameter', 'Value'])
    params = {}
    for key in ['BESS_Capacity', 'BESS_Power_Limit', 'BESS_Efficiency_Charge', 'BESS_Efficiency_Discharge', 'SOC_Initial', 'BESS_limit', 'CONVERTER_EFFICIENCY', 'DAMPER_COEFFICIENT']:
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
        self.converter_efficiency = params.get('CONVERTER_EFFICIENCY')
        # Damper coefficient (€/kW-change) for smoothing power transitions (from constants; default 0.001 if missing)
        self.damper_coefficient = params.get('DAMPER_COEFFICIENT')

        # Reusable network handle (lazy-initialized on first predict)
        self._n = None
        # Cache current nominal limits for quick updates
        self._pv_nom = None
        self._grid_imp_nom = None
        self._grid_exp_nom = None

    def _ensure_network(self, pv_nom, max_grid_import, max_grid_export):
        # Create or update a reusable PyPSA network with fixed topology
        if self._n is None:
            n = pypsa.Network()
            # Minimal snapshot to initialize; will be reset per predict()
            n.set_snapshots(pd.date_range("2000-01-01", periods=1, freq=f"{int(self.delta_t*60)}min"))

            n.snapshot_weightings.objective = self.delta_t
            n.snapshot_weightings.generators = self.delta_t
            n.snapshot_weightings.stores = self.delta_t

            # Carriers
            if "AC" not in getattr(n, "carriers", pd.DataFrame(index=[])).index:
                n.add("Carrier", "AC")
            if "DC" not in getattr(n, "carriers", pd.DataFrame(index=[])).index:
                n.add("Carrier", "DC")

            # Buses
            n.add("Bus", "AC", carrier='AC')
            n.add("Bus", "PV", carrier='DC')
            n.add("Bus", "Grid", carrier='AC')
            n.add("Bus", "BESS", carrier='DC')

            # PV
            n.add("Generator", "PV", bus="PV", p_nom=max(1.0, pv_nom), marginal_cost=0)

            # BESS
            if self.bess_power_limit > 0 and self.bess_capacity > 0:
                n.add("StorageUnit", "BESS", bus="BESS",
                      p_nom=self.bess_power_limit,
                      max_hours=self.bess_capacity/self.bess_power_limit,
                      efficiency_store=self.eta_charge,
                      efficiency_dispatch=self.eta_discharge,
                      marginal_cost=0,
                      state_of_charge_initial=self.soc_initial)

            # Links
            n.add("Link", "AC_to_BESS", bus0="AC", bus1="BESS", p_nom=self.bess_power_limit, efficiency=self.eta_charge, marginal_cost=0, carrier='AC')
            n.add("Link", "BESS_to_AC", bus0="BESS", bus1="AC", p_nom=self.bess_power_limit, efficiency=self.eta_discharge, marginal_cost=0, carrier='AC')
            n.add("Link", "PV_to_AC", bus0="PV", bus1="AC", p_nom=max(1.0, pv_nom), efficiency=self.converter_efficiency, marginal_cost=0, carrier='AC')

            # Grid IO
            n.add("Link", "Grid_Import", bus0="Grid", bus1="AC", p_nom=max_grid_import, efficiency=1.0, marginal_cost=0, carrier='AC')
            n.add("Link", "Grid_Export", bus0="AC", bus1="Grid", p_nom=max_grid_export, efficiency=1.0, marginal_cost=0, carrier='AC')

            # Grid source and loads
            n.add("Generator", "Grid_Source", bus="Grid", p_nom=1e9, marginal_cost=0)
            n.add("Load", "Consumer", bus="AC", p_set=0, marginal_cost=0)
            n.add("Load", "EV", bus="AC", p_set=0, marginal_cost=0)

            self._n = n
        else:
            n = self._n
            # Update p_nom where needed to avoid recreating components
            if pv_nom != self._pv_nom:
                n.generators.at["PV", "p_nom"] = max(1.0, pv_nom)
                n.links.at["PV_to_AC", "p_nom"] = max(1.0, pv_nom)
            if max_grid_import != self._grid_imp_nom:
                n.links.at["Grid_Import", "p_nom"] = max_grid_import
            if max_grid_export != self._grid_exp_nom:
                n.links.at["Grid_Export", "p_nom"] = max_grid_export

        # Update cached values
        self._pv_nom = pv_nom
        self._grid_imp_nom = max_grid_import
        self._grid_exp_nom = max_grid_export
        return self._n


    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, pi_ev, pi_consumer, horizon, start_dt):
        # Use actual simulation start date for snapshots
        snapshots = pd.date_range(start_dt, periods=horizon, freq=f'{int(self.delta_t*60)}min')

        # Compute nominal values for this horizon
        pv_nom = float(np.max(pv_forecast)) if np.max(pv_forecast) > 0 else 1.0
        max_grid_import = float(np.max(demand_forecast) + np.max(ev_forecast) + self.bess_power_limit)
        max_grid_export = float(np.max(pv_forecast) + self.bess_power_limit)
        if max_grid_import < float(np.max(demand_forecast) + np.max(ev_forecast)):
            max_grid_import = float(np.max(demand_forecast) + np.max(ev_forecast) + 100)

        # Ensure/reuse network and update p_nom as needed
        n = self._ensure_network(pv_nom, max_grid_import, max_grid_export)

        # Reset snapshots and weights for this horizon
        n.set_snapshots(snapshots)
        n.snapshot_weightings.objective = self.delta_t
        n.snapshot_weightings.generators = self.delta_t
        n.snapshot_weightings.stores = self.delta_t

        # Time-series updates
        n.generators_t.p_max_pu.loc[:, "PV"] = pd.Series(np.array(pv_forecast) / max(1.0, pv_nom), index=n.snapshots)
        n.links_t.marginal_cost["Grid_Export"] = -pd.Series(sell_forecast, index=n.snapshots)
        n.generators_t.marginal_cost["Grid_Source"] = pd.Series(buy_forecast, index=n.snapshots)

        # --- DEBUG: Print key marginal costs and link parameters ---
        try:
            print("PV marginal cost:", n.generators.loc["PV", "marginal_cost"])
            print("Grid import marginal cost (first 5):", n.generators_t.marginal_cost["Grid_Source"].head())
            print("Grid export marginal cost (first 5):", n.links_t.marginal_cost["Grid_Export"].head())
            print("PV_to_AC link p_nom:", n.links.loc["PV_to_AC", "p_nom"], "efficiency:", n.links.loc["PV_to_AC", "efficiency"])
            print("AC_to_BESS link p_nom:", n.links.loc["AC_to_BESS", "p_nom"], "efficiency:", n.links.loc["AC_to_BESS", "efficiency"])
            print("BESS_to_AC link p_nom:", n.links.loc["BESS_to_AC", "p_nom"], "efficiency:", n.links.loc["BESS_to_AC", "efficiency"])
            print("Initial SOC:", soc, "BESS capacity:", self.bess_capacity)
            print("PV forecast sample:", pv_forecast[:10])
            print("Consumer demand sample:", demand_forecast[:10])
        except Exception as e:
            print(f"DEBUG print error: {e}")

        # Loads: update time series
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Update initial SOC for this window
        if "BESS" in n.storage_units.index:
            n.storage_units.at["BESS", "state_of_charge_initial"] = soc

        # No manual constraints/damping: rely on PyPSA standard LP and costs

        # Build model (or rebuild with updated time series) and add MILP mutual exclusivity for grid import/export
        n.optimize.create_model()
        try:
            m = n.model
            # Access link power variables for import/export (vector over snapshots)
            link_p = m["Link-p"]
            if "Grid_Import" in n.links.index and "Grid_Export" in n.links.index:
                # Reduce the 'Link' dimension using .sel to keep only the 'snapshots' dimension
                grid_imp_p = link_p.sel(Link="Grid_Import")
                grid_exp_p = link_p.sel(Link="Grid_Export")

                z_imp = m.add_variables(binary=True, name="z_grid_import", coords=[n.snapshots])
                z_exp = m.add_variables(binary=True, name="z_grid_export", coords=[n.snapshots])

                bigM_imp = float(n.links.loc["Grid_Import", "p_nom"]) if "Grid_Import" in n.links.index else 0.0
                bigM_exp = float(n.links.loc["Grid_Export", "p_nom"]) if "Grid_Export" in n.links.index else 0.0

                # Vectorized constraints over all snapshots (aligns on 'snapshots' coordinate)
                # Enforce non-negativity and big-M upper bounds
                m.add_constraints(grid_imp_p >= 0)
                m.add_constraints(grid_exp_p >= 0)
                m.add_constraints(grid_imp_p <= bigM_imp * z_imp)
                m.add_constraints(grid_exp_p <= bigM_exp * z_exp)
                m.add_constraints(z_imp + z_exp <= 1)
        except Exception as e:
            print(f"Warning: could not add MILP mutual exclusivity: {e}")

        # Solve (default: gurobi MILP; fallback: highs)
        try:
            n.optimize.solve_model(solver_name="gurobi", solver_options={"MIPGap": 0.001})
        except Exception as e:
            print(f"Gurobi failed: {e}. Falling back to HiGHS.")
            n.optimize.solve_model(solver_name="highs", solver_options={"parallel": "on"})
        except Exception as e:
            print(f"HiGHS failed: {e}. Falling back to Gurobi.")
            n.optimize.solve_model(solver_name="gurobi", solver_options={"DualReductions": 0})

        # If failed
        if not n.is_solved:
            print("Optimization failed: infeasible or unbounded.")
            bess_dispatch = 0
            soc_next = soc
            pv_gen = 0
            pv_to_ac = 0
            bess_to_ac = 0
            grid_import = demand_forecast[0] + ev_forecast[0]  # Import what's needed
            grid_export = 0
            consumer_load = demand_forecast[0]
            ev_load = ev_forecast[0]
            ac_to_bess = 0
            
            # Set default values for separated flows
            pv_bess_to_consumer = 0
            pv_bess_to_ev = 0
            pv_bess_to_grid = 0
            pv_to_consumer = 0
            pv_to_ev = 0
            pv_to_grid = 0
            bess_to_consumer = 0
            bess_to_ev = 0
            bess_to_grid = 0
            grid_to_consumer = consumer_load
            grid_to_ev = ev_load
            ev_renewable_share = 0
            ev_revenue = 0
        else:
            pv_gen = n.generators_t.p["PV"].values[0] if "PV" in n.generators_t.p.columns else 0
            bess_dispatch = n.storage_units_t.p["BESS"].values[0] if "BESS" in n.storage_units_t.p.columns else 0
            soc_next = n.storage_units_t.state_of_charge["BESS"].values[0] if "BESS" in n.storage_units_t.state_of_charge.columns else soc
            pv_to_ac = n.links_t.p0["PV_to_AC"].values[0] if "PV_to_AC" in n.links_t.p0.columns else 0
            bess_to_ac = n.links_t.p0["BESS_to_AC"].values[0] if "BESS_to_AC" in n.links_t.p0.columns else 0
            grid_import = n.links_t.p0["Grid_Import"].values[0] if "Grid_Import" in n.links_t.p0.columns else 0
            grid_export = n.links_t.p0["Grid_Export"].values[0] if "Grid_Export" in n.links_t.p0.columns else 0
            ac_to_bess = n.links_t.p0["AC_to_BESS"].values[0] if "AC_to_BESS" in n.links_t.p0.columns else 0
            consumer_load = n.loads_t.p["Consumer"].values[0] if "Consumer" in n.loads_t.p.columns else 0
            ev_load = n.loads_t.p["EV"].values[0] if "EV" in n.loads_t.p.columns else 0
            
            # DEBUG: Print actual PyPSA results (only for first few iterations)
            if soc < 100:  # Only print for first few iterations
                print(f"DEBUG - PyPSA Results: PV_gen={pv_gen:.2f}, PV_to_AC={pv_to_ac:.2f}, BESS_to_AC={bess_to_ac:.2f}, Grid_import={grid_import:.2f}, Grid_export={grid_export:.2f}")
                print(f"DEBUG - Loads: Consumer={consumer_load:.2f}, EV={ev_load:.2f}")

            # FIXED: Proper energy flow separation and prioritization
            # Priority 1: PV energy to loads (highest value)
            pv_bess_to_consumer = min(pv_to_ac + bess_to_ac, consumer_load)
            pv_bess_to_ev = min(max(0, pv_to_ac + bess_to_ac - pv_bess_to_consumer), ev_load)
            pv_bess_to_grid = max(0, pv_to_ac + bess_to_ac - pv_bess_to_consumer - pv_bess_to_ev)
            
            # Priority 2: Separate PV and BESS contributions based on availability
            # PV gets priority for local use (higher revenue)
            pv_to_consumer = min(pv_to_ac, consumer_load)
            remaining_pv = pv_to_ac - pv_to_consumer
            pv_to_ev = min(remaining_pv, ev_load)
            remaining_pv = remaining_pv - pv_to_ev
            pv_to_grid = remaining_pv  # Remaining PV goes to grid
            
            # BESS fills remaining demand
            remaining_consumer = max(0, consumer_load - pv_to_consumer)
            remaining_ev = max(0, ev_load - pv_to_ev)
            bess_to_consumer = min(bess_to_ac, remaining_consumer)
            remaining_bess = bess_to_ac - bess_to_consumer
            bess_to_ev = min(remaining_bess, remaining_ev)
            bess_to_grid = remaining_bess - bess_to_ev
            
            # Grid fills any remaining gaps
            grid_to_consumer = max(0, consumer_load - pv_to_consumer - bess_to_consumer)
            grid_to_ev = max(0, ev_load - pv_to_ev - bess_to_ev)
            
            ev_renewable_share = (pv_to_ev + bess_to_ev) / ev_load if ev_load > 0 else 0
            ev_revenue = (pv_to_ev + bess_to_ev) * pi_ev

        return {
            'P_BESS': bess_dispatch,
            'SOC_next': soc_next,
            'P_BESS_discharge': np.maximum(bess_dispatch, 0),
            'P_grid_to_bess': ac_to_bess,
            'P_BESS_charge': np.abs(np.minimum(bess_dispatch, 0)),
            'P_PV_gen': pv_gen,
            'P_link_dc_to_ac': pv_to_ac + bess_to_ac,
            'P_grid_import': grid_import,
            'P_grid_export': grid_export,
            # Separated PV and BESS flows
            'pv_bess_to_consumer': pv_bess_to_consumer,
            'pv_bess_to_ev': pv_bess_to_ev,
            'pv_bess_to_grid': pv_bess_to_grid,
            'pv_to_consumer': pv_to_consumer,
            'pv_to_ev': pv_to_ev,
            'pv_to_grid': pv_to_grid,
            'bess_to_consumer': bess_to_consumer,
            'bess_to_ev': bess_to_ev,
            'bess_to_grid': bess_to_grid,
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

