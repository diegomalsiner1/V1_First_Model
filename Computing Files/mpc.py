import cvxpy as cp
import numpy as np
import logging
import pandas as pd
import os

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

    def predict(self, soc_current, pv_forecast, demand_forecast, buy_forecast, sell_forecast, lcoe_pv, horizon):
        # Ensure initial SOC is always set to soc_initial for the first time step
        if hasattr(self, '_first_call'):
            pass
        else:
            self._first_call = True
        if self._first_call:
            soc_current = self.soc_initial
            self._first_call = False
        # Define variables for horizon
        P_PV_cons = cp.Variable(horizon, nonneg=True)
        P_PV_BESS = cp.Variable(horizon, nonneg=True)
        P_PV_grid = cp.Variable(horizon, nonneg=True)
        P_BESS_cons = cp.Variable(horizon, nonneg=True)
        P_BESS_grid = cp.Variable(horizon, nonneg=True)
        P_grid_cons = cp.Variable(horizon, nonneg=True)
        P_grid_BESS = cp.Variable(horizon, nonneg=True)
        SOC = cp.Variable(horizon + 1, nonneg=True)
        slack = cp.Variable(horizon, nonneg=True)
        delta_bess = cp.Variable(horizon, boolean=True)
        delta_grid = cp.Variable(horizon, boolean=True)

        constraints = []
        # Consumer balance
        constraints += [P_PV_cons[k] + P_BESS_cons[k] + P_grid_cons[k] + slack[k] == demand_forecast[k] for k in range(horizon)]
        # PV allocation
        constraints += [P_PV_cons[k] + P_PV_BESS[k] + P_PV_grid[k] + slack[k] == pv_forecast[k] for k in range(horizon)]
        # BESS limits
        for k in range(horizon):
            constraints += [P_PV_BESS[k] + P_grid_BESS[k] <= self.bess_power_limit]
            constraints += [P_BESS_cons[k] + P_BESS_grid[k] <= self.bess_power_limit]
        # SOC dynamics
        constraints += [SOC[0] == soc_current]
        constraints += [SOC[k+1] == SOC[k] + self.eta_charge * (P_PV_BESS[k] + P_grid_BESS[k]) * self.delta_t -
                        (P_BESS_cons[k] + P_BESS_grid[k]) / self.eta_discharge * self.delta_t for k in range(horizon)]
        constraints += [SOC[k] <= self.bess_capacity for k in range(horizon + 1)]
        constraints += [SOC[k] >= self.bess_percent_limit * self.bess_capacity for k in range(horizon + 1)]
        constraints += [SOC[horizon] >= soc_current]  # Terminal constraint
        # Mutual exclusivity (Big-M)
        M_bess = self.bess_power_limit
        M_grid = max(np.max(pv_forecast), np.max(demand_forecast)) + self.bess_power_limit
        for k in range(horizon):
            constraints += [P_PV_BESS[k] + P_grid_BESS[k] <= M_bess * delta_bess[k]]
            constraints += [P_BESS_cons[k] + P_BESS_grid[k] <= M_bess * (1 - delta_bess[k])]
            constraints += [P_grid_cons[k] + P_grid_BESS[k] <= M_grid * delta_grid[k]]
            constraints += [P_PV_grid[k] + P_BESS_grid[k] <= M_grid * (1 - delta_grid[k])]
        # Objective (revenue over horizon)
        revenue = (
            cp.sum(cp.multiply(P_PV_cons, buy_forecast - lcoe_pv) * self.delta_t) +
            cp.sum(cp.multiply(P_PV_grid, sell_forecast - lcoe_pv) * self.delta_t) +
            cp.sum(cp.multiply(P_PV_BESS, - lcoe_pv) * self.delta_t) +
            cp.sum(cp.multiply(P_BESS_cons, buy_forecast - self.lcoe_bess) * self.delta_t) +
            cp.sum(cp.multiply(P_BESS_grid, sell_forecast - self.lcoe_bess) * self.delta_t) -
            cp.sum(cp.multiply(P_grid_cons + P_grid_BESS, buy_forecast) * self.delta_t) -
            1e5 * cp.sum(slack)
        )
        objective = cp.Maximize(revenue)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI, verbose=True, MIPGap=0.005)
        if problem.status == 'optimal':
            return {
                'P_PV_cons': P_PV_cons.value[0],
                'P_PV_BESS': P_PV_BESS.value[0],
                'P_PV_grid': P_PV_grid.value[0],
                'P_BESS_cons': P_BESS_cons.value[0],
                'P_BESS_grid': P_BESS_grid.value[0],
                'P_grid_cons': P_grid_cons.value[0],
                'P_grid_BESS': P_grid_BESS.value[0],
                'slack': slack.value[0],
                'SOC_next': SOC.value[1]
            }
        else:
            logging.info("MPC infeasible at current step.")
            return None