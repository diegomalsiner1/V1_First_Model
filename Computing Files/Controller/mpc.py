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
        self.converter_efficiency = params.get('CONVERTER_EFFICIENCY', 0.98)
        # Damper coefficient (€/kW-change) for smoothing power transitions (from constants; default 0.01 if missing)
        self.damper_coefficient = params.get('DAMPER_COEFFICIENT', 0.01)

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
                  efficiency_store=self.eta_charge,
                  efficiency_dispatch=self.eta_discharge,
                  marginal_cost=0,
                  state_of_charge_initial=soc)

        # Links between buses
        # Charge BESS from AC side (so grid-to-BESS passes via AC)
        n.add("Link", "AC_to_BESS", bus0="AC", bus1="BESS", p_nom=self.bess_power_limit, efficiency=self.eta_charge, marginal_cost=0, carrier='AC')

        n.add("Link", "PV_to_BESS", bus0="PV", bus1="BESS", p_nom=self.bess_power_limit, efficiency=self.eta_charge, marginal_cost=0, carrier='DC')
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

        # Loads: keep Consumer and EV on AC side (EV chargers are typically AC-coupled externally)
        n.add("Load", "Consumer", bus="AC", p_set=0, marginal_cost=0)
        n.add("Load", "EV", bus="AC", p_set=0, marginal_cost=0)
        n.loads_t.p_set.loc[:, "Consumer"] = demand_forecast
        n.loads_t.p_set.loc[:, "EV"] = ev_forecast

        # Build model
        n.optimize.create_model()

        # Explicit balances for clarity (PyPSA already ensures nodal balances internally)
        try:
            if "PV" in n.generators.index and "PV_to_BESS" in n.links.index and "PV_to_AC" in n.links.index:
                pv_gen_var = n.model["Generator-p"]["PV"]
                pv_to_bess_var = n.model["Link-p"]["PV_to_BESS"]
                pv_to_ac_var = n.model["Link-p"]["PV_to_AC"]
                n.model.add_constraints(pv_gen_var == pv_to_bess_var + pv_to_ac_var, name="DC_PV_Balance")

            if "PV_to_AC" in n.links.index and "BESS_to_AC" in n.links.index and "AC_to_BESS" in n.links.index:
                pv_to_ac_var = n.model["Link-p"]["PV_to_AC"]
                bess_to_ac_var = n.model["Link-p"]["BESS_to_AC"]
                grid_import_var = n.model["Link-p"]["Grid_Import"]
                grid_export_var = n.model["Link-p"]["Grid_Export"]
                consumer_load_var = n.model["Load-p"]["Consumer"]
                ev_load_var = n.model["Load-p"]["EV"]
                ac_to_bess_var = n.model["Link-p"]["AC_to_BESS"]
                # AC balance: supply (PV_to_AC + BESS_to_AC + Grid_Import) = demand (Consumer + EV + Grid_Export + AC_to_BESS)
                n.model.add_constraints(pv_to_ac_var + bess_to_ac_var + grid_import_var == consumer_load_var + ev_load_var + grid_export_var + ac_to_bess_var, name="AC_Balance")
        except Exception as e:
            print(f"Warning: Could not add explicit balance constraints: {e}")

        # Add artificial damper (power-change penalty) to smooth BESS and Grid imports
        try:
            # Penalize changes on link flows that represent BESS charge/discharge on AC side
            if "Link-p" in n.model:
                # BESS discharge to AC
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

                # BESS charge from AC
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

            # Grid import change penalty (Link-p for Grid_Import)
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
            grid_to_bess = 0
            pv_to_bess = 0
        else:
            # Extract generator/storage
            if "PV" in getattr(n.generators_t, "p", pd.DataFrame()).columns:
                pv_gen = n.generators_t.p["PV"].values[0]
            else:
                pv_gen = 0
            if "BESS" in getattr(n.storage_units_t, "p", pd.DataFrame()).columns:
                bess_dispatch = n.storage_units_t.p["BESS"].values[0]
                soc_next = n.storage_units_t.state_of_charge["BESS"].values[0]
            else:
                bess_dispatch = 0
                soc_next = soc

            # Extract links (guard missing columns)
            cols_p0 = getattr(n.links_t, "p0", pd.DataFrame()).columns
            pv_to_ac = n.links_t.p0["PV_to_AC"].values[0] if "PV_to_AC" in cols_p0 else 0
            bess_to_ac = n.links_t.p0["BESS_to_AC"].values[0] if "BESS_to_AC" in cols_p0 else 0
            grid_import = n.links_t.p0["Grid_Import"].values[0] if "Grid_Import" in cols_p0 else 0
            grid_export = n.links_t.p0["Grid_Export"].values[0] if "Grid_Export" in cols_p0 else 0
            ac_to_bess = n.links_t.p0["AC_to_BESS"].values[0] if "AC_to_BESS" in cols_p0 else 0
            pv_to_bess = n.links_t.p0["PV_to_BESS"].values[0] if "PV_to_BESS" in cols_p0 else 0

            # Extract loads
            cols_load = getattr(n.loads_t, "p", pd.DataFrame()).columns
            consumer_load = n.loads_t.p["Consumer"].values[0] if "Consumer" in cols_load else 0
            ev_load = n.loads_t.p["EV"].values[0] if "EV" in cols_load else 0

        # Balance diagnostics (AC side only; DC is enforced by DC_PV_Balance)
        dc_to_ac_flow = max(pv_to_ac + bess_to_ac, 0)
        ac_balance = pv_to_ac + bess_to_ac + grid_import - consumer_load - ev_load - grid_export - ac_to_bess
        balance_tolerance = 0.1
        if abs(ac_balance) > balance_tolerance:
            print(f"⚠ AC balance mismatch: {ac_balance:.3f} kW")

        # Map DC→AC renewable flow to uses on AC bus
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
            # Maintain key name for compatibility; now represents AC_to_BESS
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

