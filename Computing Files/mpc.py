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
        bess_to_grid = cp.Variable(horizon, nonneg=True)
        pv_to_bess = cp.Variable(horizon, nonneg=True)
        pv_to_grid = cp.Variable(horizon, nonneg=True)
        grid_to_consumer = cp.Variable(horizon, nonneg=True)
        grid_to_bess = cp.Variable(horizon, nonneg=True)
        pv_surplus = cp.Variable(horizon, nonneg=True)
        pv_after_bess = cp.Variable(horizon, nonneg=True)
        charge = cp.Variable(horizon, nonneg=True)
        discharge = cp.Variable(horizon, nonneg=True)
        is_charging = cp.Variable(horizon, boolean=True)
        constraints = []
        constraints += [SOC[0] == soc_current]
        M = self.bess_power_limit
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
            # Big-M mutual exclusivity
            constraints += [charge[k] <= M * is_charging[k]]
            constraints += [discharge[k] <= M * (1 - is_charging[k])]
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
            # BESS to consumer: up to discharge
            constraints += [bess_to_consumer[k] <= discharge[k]]
            # BESS to grid: up to discharge
            constraints += [bess_to_grid[k] <= discharge[k]]
            # Discharge split
            constraints += [bess_to_consumer[k] + bess_to_grid[k] == discharge[k]]
            # Energy balance constraints
            # PV balance
            constraints += [pv_to_consumer[k] + pv_to_bess[k] + pv_to_grid[k] == pv]
            
            # Consumer balance
            constraints += [pv_to_consumer[k] + bess_to_consumer[k] + grid_to_consumer[k] == demand]
            
            # BESS balance
            constraints += [charge[k] == pv_to_bess[k] + grid_to_bess[k]]
            constraints += [discharge[k] == bess_to_consumer[k] + bess_to_grid[k]]
            
            # Grid balance (grid acts as slack)
            constraints += [grid_to_consumer[k] == demand - pv_to_consumer[k] - bess_to_consumer[k]]
            constraints += [grid_to_bess[k] == charge[k] - pv_to_bess[k]]
            constraints += [grid_to_bess[k] >= 0]
            # Remove slack since we don't need it
            constraints += [slack[k] == 0]
        # Grid balance at each timestep
        grid_import = []
        grid_export = []
        for k in range(horizon):
            grid_imp_k = grid_to_consumer[k] + grid_to_bess[k]
            grid_exp_k = pv_to_grid[k] + bess_to_grid[k]
            grid_import.append(grid_imp_k)
            grid_export.append(grid_exp_k)
        
        # Objective: true financial flows only (no virtual rewards)
        alpha = 0.0001  # Usage reward coefficient (reduced to avoid excessive cycling)
        
        savings = 0
        for k in range(horizon):
            # Grid exports (revenue)
            savings += cp.multiply(grid_export[k], sell_forecast[k]) * self.delta_t
            # Grid imports (cost)
            savings -= cp.multiply(grid_import[k], buy_forecast[k]) * self.delta_t
            # Internal costs (LCOE)
            savings -= cp.multiply(pv_to_consumer[k], lcoe_pv) * self.delta_t
            savings -= cp.multiply(bess_to_consumer[k], self.lcoe_bess) * self.delta_t
        
        bess_usage = cp.sum(charge + discharge) * self.delta_t
        objective = cp.Maximize(savings + alpha * bess_usage)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI, verbose=True, MIPGap=0.05, TimeLimit=30)
        if problem.status == 'optimal':
            try:
                return {
                    'P_BESS': P_BESS.value[0],
                    'SOC_next': SOC.value[1],
                    'pv_to_consumer': pv_to_consumer.value[0],
                    'bess_to_consumer': bess_to_consumer.value[0],
                    'bess_to_grid': bess_to_grid.value[0],
                    'pv_to_bess': pv_to_bess.value[0],
                    'pv_to_grid': pv_to_grid.value[0],
                    'grid_to_consumer': grid_to_consumer.value[0],
                    'grid_to_bess': grid_to_bess.value[0],
                    'slack': 0.0  # Since we constrain slack to be 0
                }
            except AttributeError as e:
                logging.error(f"Error accessing solution values: {e}")
                return None
        else:
            logging.info("MPC infeasible at current step.")
            return None