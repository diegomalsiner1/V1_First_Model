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
        # Only BESS power is controllable: positive = discharge, negative = charge
        P_BESS = cp.Variable(horizon)
        SOC = cp.Variable(horizon + 1)
        slack = cp.Variable(horizon, nonneg=True)
        # Auxiliary variables for all flows (all non-negative)
        pv_to_consumer = cp.Variable(horizon, nonneg=True)
        bess_to_consumer = cp.Variable(horizon, nonneg=True)
        pv_to_bess = cp.Variable(horizon, nonneg=True)
        pv_to_grid = cp.Variable(horizon, nonneg=True)
        grid_to_consumer = cp.Variable(horizon, nonneg=True)
        grid_to_bess = cp.Variable(horizon, nonneg=True)
        pv_surplus = cp.Variable(horizon, nonneg=True)
        pv_after_bess = cp.Variable(horizon, nonneg=True)
        charge = cp.Variable(horizon, nonneg=True)
        discharge = cp.Variable(horizon, nonneg=True)
        constraints = []
        constraints += [SOC[0] == soc_current]
        for k in range(horizon):
            pv = pv_forecast[k]
            demand = demand_forecast[k]
            # BESS power limits
            constraints += [P_BESS[k] <= self.bess_power_limit]
            constraints += [P_BESS[k] >= -self.bess_power_limit]
            # Charge/discharge split
            constraints += [charge[k] >= -P_BESS[k]]
            constraints += [charge[k] >= 0]
            constraints += [discharge[k] >= P_BESS[k]]
            constraints += [discharge[k] >= 0]
            # SOC update
            constraints += [SOC[k+1] == SOC[k] + self.eta_charge * charge[k] * self.delta_t - (discharge[k] / self.eta_discharge) * self.delta_t]
            constraints += [SOC[k+1] <= self.bess_capacity]
            constraints += [SOC[k+1] >= self.bess_percent_limit * self.bess_capacity]
            # PV to consumer: up to demand and PV available
            constraints += [pv_to_consumer[k] <= pv]
            constraints += [pv_to_consumer[k] <= demand]
            # PV surplus after consumer
            constraints += [pv_surplus[k] >= pv - pv_to_consumer[k]]
            constraints += [pv_surplus[k] >= 0]
            # PV to BESS: up to charge and PV surplus
            constraints += [pv_to_bess[k] <= charge[k]]
            constraints += [pv_to_bess[k] <= pv_surplus[k]]
            # PV after BESS (for grid export)
            constraints += [pv_after_bess[k] >= pv_surplus[k] - pv_to_bess[k]]
            constraints += [pv_after_bess[k] >= 0]
            # PV to grid: remaining PV after consumer and BESS
            constraints += [pv_to_grid[k] == pv_after_bess[k]]
            # BESS to consumer: up to discharge and remaining demand
            constraints += [bess_to_consumer[k] <= discharge[k]]
            constraints += [bess_to_consumer[k] <= demand - pv_to_consumer[k]]
            # Grid to consumer: remaining demand
            constraints += [grid_to_consumer[k] == demand - pv_to_consumer[k] - bess_to_consumer[k] + slack[k]]
            # Grid to BESS: up to charge not covered by PV
            constraints += [grid_to_bess[k] == charge[k] - pv_to_bess[k]]
            constraints += [grid_to_bess[k] >= 0]
        # Objective: revenue + battery usage reward
        alpha = 0.01  # Usage reward coefficient
        revenue = cp.sum(
            pv_to_consumer * (buy_forecast[:horizon] - lcoe_pv) * self.delta_t +
            pv_to_grid * (sell_forecast[:horizon] - lcoe_pv) * self.delta_t +
            pv_to_bess * (-lcoe_pv) * self.delta_t +
            bess_to_consumer * (buy_forecast[:horizon] - self.lcoe_bess) * self.delta_t +
            grid_to_consumer * (-buy_forecast[:horizon]) * self.delta_t +
            grid_to_bess * (-self.lcoe_bess) * self.delta_t -
            1e5 * slack * self.delta_t
        )
        bess_usage = cp.sum(charge + discharge) * self.delta_t
        objective = cp.Maximize(revenue + alpha * bess_usage)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI, verbose=True, MIPGap=0.005)
        if problem.status == 'optimal':
            return {
                'P_BESS': P_BESS.value[0],
                'SOC_next': SOC.value[1],
                'pv_to_consumer': pv_to_consumer.value[0],
                'bess_to_consumer': bess_to_consumer.value[0],
                'pv_to_bess': pv_to_bess.value[0],
                'pv_to_grid': pv_to_grid.value[0],
                'grid_to_consumer': grid_to_consumer.value[0],
                'grid_to_bess': grid_to_bess.value[0],
                'slack': slack.value[0]
            }
        else:
            logging.info("MPC infeasible at current step.")
            return None