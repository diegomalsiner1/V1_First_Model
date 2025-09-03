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


    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, pi_ev, pi_consumer, horizon, start_dt):
        # Use actual simulation start date for snapshots
        snapshots = pd.date_range(start_dt, periods=horizon, freq=f'{int(self.delta_t*60)}min')
        n = pypsa.Network()
        n.set_snapshots(snapshots)

        n.snapshot_weightings.objective = self.delta_t
        n.snapshot_weightings.generators = self.delta_t
        n.snapshot_weightings.stores = self.delta_t

        # Define carriers explicitly to avoid warnings and for clarity
        if "AC" not in getattr(n, "carriers", pd.DataFrame(index=[])).index:
            n.add("Carrier", "AC")
        if "DC" not in getattr(n, "carriers", pd.DataFrame(index=[])).index:
            n.add("Carrier", "DC")

        # Add buses (AC grid side; DC side split by PV and BESS buses)
        n.add("Bus", "AC", carrier='AC')
        n.add("Bus", "PV", carrier='DC')
        n.add("Bus", "Grid", carrier='AC')
        n.add("Bus", "BESS", carrier='DC')

        # PV generator (PV bus)
        pv_nom = np.max(pv_forecast)
        if pv_nom == 0:
            pv_nom = 1
        p_max_pu = np.array(pv_forecast) / pv_nom
        n.add("Generator", "PV", bus="PV", p_nom=pv_nom, p_max_pu=p_max_pu, marginal_cost=0)

        # BESS StorageUnit (BESS bus)
        if self.bess_power_limit > 0 and self.bess_capacity > 0:
            n.add("StorageUnit", "BESS", bus="BESS",
                  p_nom=self.bess_power_limit,
                  max_hours=self.bess_capacity/self.bess_power_limit,
                  efficiency_store=self.eta_charge,
                  efficiency_dispatch=self.eta_discharge,
                  marginal_cost=0,
                  state_of_charge_initial=soc)

        # Links between buses
        n.add("Link", "AC_to_BESS", bus0="AC", bus1="BESS", p_nom=self.bess_power_limit, efficiency=self.eta_charge, marginal_cost=0, carrier='AC')
        n.add("Link", "BESS_to_AC", bus0="BESS", bus1="AC", p_nom=self.bess_power_limit, efficiency=self.eta_discharge, marginal_cost=0, carrier='AC')
        n.add("Link", "PV_to_AC", bus0="PV", bus1="AC", p_nom=pv_nom, efficiency=self.converter_efficiency, marginal_cost=0, carrier='AC')

        # Grid import/export as Links
        max_grid_import = np.max(demand_forecast) + np.max(ev_forecast) + self.bess_power_limit
        max_grid_export = np.max(pv_forecast) + self.bess_power_limit
        n.add("Link", "Grid_Import", bus0="Grid", bus1="AC", p_nom=max_grid_import, efficiency=1.0, marginal_cost=0, carrier='AC')
        n.add("Link", "Grid_Export", bus0="AC", bus1="Grid", p_nom=max_grid_export, efficiency=1.0, marginal_cost=0, carrier='AC')

        # Grid source generator (reference, cost = buy_forecast)
        n.add("Generator", "Grid_Source", bus="Grid", p_nom=1e9, marginal_cost=0)
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
        except Exception as e:
            print(f"DEBUG print error: {e}")
        # --- Mutual exclusivity for grid import/export ---
        def add_mutual_exclusivity_constraint(network):
            m = network.model
            import pyomo.environ as po
            m.grid_import_bin = po.Var(network.snapshots, within=po.Binary)
            big_M_import = max_grid_import
            big_M_export = max_grid_export
            for t in network.snapshots:
                m.add_constraint(m["Link-p"]["Grid_Import"][t] <= big_M_import * m.grid_import_bin[t],
                                 name=f"GridImportBin_{t}")
                m.add_constraint(m["Link-p"]["Grid_Export"][t] <= big_M_export * (1 - m.grid_import_bin[t]),
                                 name=f"GridExportBin_{t}")

        n.optimize.create_model()
        try:
            add_mutual_exclusivity_constraint(n)
        except Exception as e:
            print(f"Warning: Could not add grid import/export mutual exclusivity constraint: {e}")

        # Loads: keep Consumer and EV on AC side (EV chargers are typically AC-coupled externally)
        n.add("Load", "Consumer", bus="AC", p_set=0, marginal_cost=0)
        n.add("Load", "EV", bus="AC", p_set=0, marginal_cost=0)
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Explicit balances for clarity (PyPSA already ensures nodal balances internally)
        try:
            if "PV_to_AC" in n.links.index and "BESS_to_AC" in n.links.index and "AC_to_BESS" in n.links.index:
                pv_to_ac_var = n.model["Link-p"]["PV_to_AC"]
                bess_to_ac_var = n.model["Link-p"]["BESS_to_AC"]
                grid_import_var = n.model["Link-p"]["Grid_Import"]
                grid_export_var = n.model["Link-p"]["Grid_Export"]
                consumer_load_var = n.model["Load-p"]["Consumer"]
                ev_load_var = n.model["Load-p"]["EV"]
                ac_to_bess_var = n.model["Link-p"]["AC_to_BESS"]
                n.model.add_constraints(pv_to_ac_var + bess_to_ac_var + grid_import_var == consumer_load_var + ev_load_var + grid_export_var + ac_to_bess_var, name="AC_Balance")
        except Exception as e:
            print(f"Warning: Could not add explicit balance constraints: {e}")

        # Add artificial damper (power-change penalty) to smooth BESS and Grid imports
        try:
            if "Link-p" in n.model:
                if "BESS_to_AC" in n.links.index:
                    bess_to_ac_p = n.model["Link-p"]["BESS_to_AC"]
                    b2a_pos = n.model.add_variables(lower=0, name="B2A_delta_pos", coords=[n.snapshots])
                    b2a_neg = n.model.add_variables(lower=0, name="B2A_delta_neg", coords=[n.snapshots])
                    for idx in range(1, len(n.snapshots)):
                        prev_ts = n.snapshots[idx - 1]
                        curr_ts = n.snapshots[idx]
                        n.model.add_constraints(
                            bess_to_ac_p[curr_ts] - bess_to_ac_p[prev_ts] == b2a_pos[curr_ts] - b2a_neg[curr_ts],
                            name=f"B2A_change_{idx}"
                        )
                    n.model.objective += self.damper_coefficient * (b2a_pos.sum() + b2a_neg.sum())

                if "AC_to_BESS" in n.links.index:
                    ac_to_bess_p = n.model["Link-p"]["AC_to_BESS"]
                    a2b_pos = n.model.add_variables(lower=0, name="A2B_delta_pos", coords=[n.snapshots])
                    a2b_neg = n.model.add_variables(lower=0, name="A2B_delta_neg", coords=[n.snapshots])
                    for idx in range(1, len(n.snapshots)):
                        prev_ts = n.snapshots[idx - 1]
                        curr_ts = n.snapshots[idx]
                        n.model.add_constraints(
                            ac_to_bess_p[curr_ts] - ac_to_bess_p[prev_ts] == a2b_pos[curr_ts] - a2b_neg[curr_ts],
                            name=f"A2B_change_{idx}"
                        )
                    n.model.objective += self.damper_coefficient * (a2b_pos.sum() + a2b_neg.sum())

            if "Grid_Import" in n.links.index and "Link-p" in n.model:
                grid_imp_p = n.model["Link-p"]["Grid_Import"]
                grid_delta_pos = n.model.add_variables(lower=0, name="Grid_import_delta_pos", coords=[n.snapshots])
                grid_delta_neg = n.model.add_variables(lower=0, name="Grid_import_delta_neg", coords=[n.snapshots])
                for idx in range(1, len(n.snapshots)):
                    prev_ts = n.snapshots[idx - 1]
                    curr_ts = n.snapshots[idx]
                    n.model.add_constraints(
                        grid_imp_p[curr_ts] - grid_imp_p[prev_ts] == grid_delta_pos[curr_ts] - grid_delta_neg[curr_ts],
                        name=f"Grid_import_change_{idx}"
                    )
                n.model.objective += self.damper_coefficient * (grid_delta_pos.sum() + grid_delta_neg.sum())
        except Exception as e:
            print(f"Warning adding damper terms: {e}")

        # Solve
        try:
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
            grid_import = 0
            grid_export = 0
            consumer_load = demand_forecast[0]
            ev_load = ev_forecast[0]
            ac_to_bess = 0
            dc_to_ac_flow = 0
            pv_bess_to_consumer = 0
            pv_bess_to_ev = 0
            pv_bess_to_grid = 0
            grid_to_consumer = 0
            grid_to_ev = 0
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

            dc_to_ac_flow = pv_to_ac + bess_to_ac
            pv_bess_to_consumer = min(dc_to_ac_flow, consumer_load)
            pv_bess_to_ev = min(max(0, dc_to_ac_flow - pv_bess_to_consumer), ev_load)
            pv_bess_to_grid = max(0, dc_to_ac_flow - pv_bess_to_consumer - pv_bess_to_ev)
            grid_to_consumer = max(0, consumer_load - pv_bess_to_consumer)
            grid_to_ev = max(0, ev_load - pv_bess_to_ev)
            ev_renewable_share = pv_bess_to_ev / ev_load if ev_load > 0 else 0
            ev_revenue = pv_bess_to_ev * pi_ev

        return {
            'P_BESS': bess_dispatch,
            'SOC_next': soc_next,
            'P_BESS_discharge': np.maximum(bess_dispatch, 0),
            'P_grid_to_bess': ac_to_bess,
            'P_BESS_charge': np.abs(np.minimum(bess_dispatch, 0)),
            'P_PV_gen': pv_gen,
            'P_link_dc_to_ac': dc_to_ac_flow,
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

