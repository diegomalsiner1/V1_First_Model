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

def load_pi_ev(constants_path=None):
    """Load EV charging price from constants CSV."""
    if constants_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_path = os.path.join(script_dir, '..', 'Input Data Files', 'Constants_Plant.csv')
    constants_data = pd.read_csv(constants_path, comment='#')
    try:
        return float(constants_data[constants_data['Parameter'] == 'EV_PRICE']['Value'].iloc[0])
    except IndexError:
        logging.warning("pi_ev not found in constants CSV; using default value 0.3")
        return 0.3  # Default value, e.g., 0.3 $/kWh; adjust as needed

class MPC:
    def __init__(self, bess_capacity, bess_power_limit, eta_charge, eta_discharge, lcoe_bess, delta_t, constants_path=None):
        self.bess_capacity = bess_capacity
        self.bess_power_limit = bess_power_limit
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.lcoe_bess = lcoe_bess
        self.delta_t = delta_t
        self.bess_percent_limit = load_bess_percent_limit(constants_path)
        self.pi_ev = load_pi_ev(constants_path)
        self.big_M = 1e6  # Large constant for Big-M constraints

    def predict(self, soc_current, pv_forecast, demand_forecast, ev_forecast, buy_forecast, sell_forecast, lcoe_pv, horizon):
        # Variables
        P_BESS = cp.Variable(horizon)
        SOC = cp.Variable(horizon + 1)
        
        # Flow variables including EV
        pv_to_consumer = cp.Variable(horizon, nonneg=True)
        pv_to_ev = cp.Variable(horizon, nonneg=True)
        pv_to_bess = cp.Variable(horizon, nonneg=True)
        pv_to_grid = cp.Variable(horizon, nonneg=True)
        pv_surplus_after_consumer = cp.Variable(horizon, nonneg=True)
        pv_surplus_after_ev = cp.Variable(horizon, nonneg=True)
        pv_after_bess = cp.Variable(horizon, nonneg=True)

        bess_to_consumer = cp.Variable(horizon, nonneg=True)
        bess_to_ev = cp.Variable(horizon, nonneg=True)
        bess_to_grid = cp.Variable(horizon, nonneg=True)
        bess_surplus_after_consumer = cp.Variable(horizon, nonneg=True)
        bess_surplus_after_ev = cp.Variable(horizon, nonneg=True)
        
        grid_to_consumer = cp.Variable(horizon, nonneg=True)
        grid_to_ev = cp.Variable(horizon, nonneg=True)
        grid_to_bess = cp.Variable(horizon, nonneg=True)
        
        charge = cp.Variable(horizon, nonneg=True)
        discharge = cp.Variable(horizon, nonneg=True)
        is_charging = cp.Variable(horizon, boolean=True)
        
        # Binary variable to prevent simultaneous import and export
        u = cp.Variable(horizon, boolean=True)  # 1 if importing, 0 if exporting

        constraints = []
        constraints += [SOC[0] == soc_current]
        
        M = self.bess_power_limit
        for k in range(horizon):
            pv = pv_forecast[k]
            consumer_demand = demand_forecast[k]
            ev_demand = ev_forecast[k]
            
            # BESS power limits
            constraints += [P_BESS[k] <= self.bess_power_limit]
            constraints += [P_BESS[k] >= -self.bess_power_limit]
            # Charge/discharge split
            constraints += [charge[k] >= -P_BESS[k]]
            constraints += [charge[k] >= 0]
            constraints += [discharge[k] >= P_BESS[k]]
            constraints += [discharge[k] >= 0]
            # Big-M mutual exclusivity for charge/discharge
            constraints += [charge[k] <= M * is_charging[k]]
            constraints += [discharge[k] <= M * (1 - is_charging[k])]
            # SOC update
            constraints += [SOC[k+1] == SOC[k] + self.eta_charge * charge[k] * self.delta_t - (discharge[k] / self.eta_discharge) * self.delta_t]
            constraints += [SOC[k+1] <= self.bess_capacity]
            constraints += [SOC[k+1] >= self.bess_percent_limit * self.bess_capacity]
            # PV to consumer: up to demand and PV available
            constraints += [pv_to_consumer[k] <= pv]
            constraints += [pv_to_consumer[k] <= consumer_demand]
            # PV surplus after consumer
            constraints += [pv_surplus_after_consumer[k] >= pv - pv_to_consumer[k]]
            constraints += [pv_surplus_after_consumer[k] >= 0]
            # PV to EV: up to EV demand and surplus after consumer
            constraints += [pv_to_ev[k] <= pv_surplus_after_consumer[k]]
            constraints += [pv_to_ev[k] <= ev_demand]
            # PV surplus after EV
            constraints += [pv_surplus_after_ev[k] >= pv_surplus_after_consumer[k] - pv_to_ev[k]]
            constraints += [pv_surplus_after_ev[k] >= 0]
            # PV to BESS: up to charge and surplus after EV
            constraints += [pv_to_bess[k] <= charge[k]]
            constraints += [pv_to_bess[k] <= pv_surplus_after_ev[k]]
            # PV after BESS (for grid export)
            constraints += [pv_after_bess[k] >= pv_surplus_after_ev[k] - pv_to_bess[k]]
            constraints += [pv_after_bess[k] >= 0]
            # PV to grid: remaining PV after loads and BESS
            constraints += [pv_to_grid[k] == pv_after_bess[k]]
            # BESS to consumer: up to discharge
            constraints += [bess_to_consumer[k] <= discharge[k]]
            # BESS surplus after consumer
            constraints += [bess_surplus_after_consumer[k] >= discharge[k] - bess_to_consumer[k]]
            constraints += [bess_surplus_after_consumer[k] >= 0]
            # BESS to EV: up to surplus after consumer
            constraints += [bess_to_ev[k] <= bess_surplus_after_consumer[k]]
            # BESS surplus after EV
            constraints += [bess_surplus_after_ev[k] >= bess_surplus_after_consumer[k] - bess_to_ev[k]]
            constraints += [bess_surplus_after_ev[k] >= 0]
            # BESS to grid: remaining after loads
            constraints += [bess_to_grid[k] == bess_surplus_after_ev[k]]
            # Discharge split
            constraints += [discharge[k] == bess_to_consumer[k] + bess_to_ev[k] + bess_to_grid[k]]
            # Energy balance constraints
            # PV balance
            constraints += [pv_to_consumer[k] + pv_to_ev[k] + pv_to_bess[k] + pv_to_grid[k] == pv]
            
            # Consumer balance
            constraints += [pv_to_consumer[k] + bess_to_consumer[k] + grid_to_consumer[k] == consumer_demand]
           
            # EV balance
            constraints += [pv_to_ev[k] + bess_to_ev[k] + grid_to_ev[k] == ev_demand]
            
            # BESS balance
            constraints += [charge[k] == pv_to_bess[k] + grid_to_bess[k]]
            
            # Grid balance with strict matching of imports to needs
            constraints += [grid_to_consumer[k] == consumer_demand - pv_to_consumer[k] - bess_to_consumer[k]]
            constraints += [grid_to_ev[k] == ev_demand - pv_to_ev[k] - bess_to_ev[k]]
            constraints += [grid_to_bess[k] == charge[k] - pv_to_bess[k]]
            # Non-negativity
            constraints += [grid_to_bess[k] >= 0]
        # Grid balance at each timestep
        grid_import = []
        grid_export = []
        for k in range(horizon):
            grid_imp_k = grid_to_consumer[k] + grid_to_ev[k] + grid_to_bess[k]
            grid_exp_k = pv_to_grid[k] + bess_to_grid[k]
            grid_import.append(grid_imp_k)
            grid_export.append(grid_exp_k)
            
            # Prevent simultaneous import and export
            constraints += [grid_imp_k <= self.big_M * u[k]]
            constraints += [grid_exp_k <= self.big_M * (1 - u[k])]
        
        # Objective: true financial flows only (no virtual rewards)
        alpha = 0.00001  # Usage reward coefficient (reduced to avoid excessive cycling)
        
        savings = 0
        for k in range(horizon):
            # Revenue from EV charging (real cash inflow)
            savings += (pv_to_ev[k] + bess_to_ev[k] + grid_to_ev[k]) * self.pi_ev * self.delta_t
            
            # Grid exports (revenue)
            savings += grid_export[k] * sell_forecast[k] * self.delta_t
            # Grid imports (cost)
            savings -= grid_import[k] * buy_forecast[k] * self.delta_t
            # Internal costs (LCOE) for all PV and BESS flows
            savings -= (pv_to_consumer[k] + pv_to_ev[k] + pv_to_bess[k] + pv_to_grid[k]) * lcoe_pv * self.delta_t
            savings -= (bess_to_consumer[k] + bess_to_ev[k] + bess_to_grid[k]) * self.lcoe_bess * self.delta_t
        
        bess_usage = cp.sum(charge + discharge) * self.delta_t
        objective = cp.Maximize(savings + alpha * bess_usage)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI, verbose=True, MIPGap=0.025)
        if problem.status == 'optimal':
            try:
                return {
                    'P_BESS': P_BESS.value[0],
                    'SOC_next': SOC.value[1],
                    'pv_to_consumer': pv_to_consumer.value[0],
                    'pv_to_ev': pv_to_ev.value[0],  # Add this
                    'bess_to_consumer': bess_to_consumer.value[0],
                    'bess_to_ev': bess_to_ev.value[0],  # Add this
                    'bess_to_grid': bess_to_grid.value[0],
                    'pv_to_bess': pv_to_bess.value[0],
                    'pv_to_grid': pv_to_grid.value[0],
                    'grid_to_consumer': grid_to_consumer.value[0],
                    'grid_to_ev': grid_to_ev.value[0],  # Add this
                    'grid_to_bess': grid_to_bess.value[0],
                    'slack': 0.0
                }
            except AttributeError as e:
                logging.error(f"Error accessing solution values: {e}")
                return None