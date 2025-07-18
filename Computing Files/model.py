import cvxpy as cp
import numpy as np

def define_variables(n_steps):
    P_PV_consumer = cp.Variable(n_steps, nonneg=True)
    P_PV_BESS = cp.Variable(n_steps, nonneg=True)
    P_PV_grid = cp.Variable(n_steps, nonneg=True)
    P_BESS_consumer = cp.Variable(n_steps, nonneg=True)
    P_BESS_grid = cp.Variable(n_steps, nonneg=True)
    P_grid_consumer = cp.Variable(n_steps, nonneg=True)
    P_grid_BESS = cp.Variable(n_steps, nonneg=True)
    SOC = cp.Variable(n_steps + 1, nonneg=True)
    slack = cp.Variable(n_steps, nonneg=True)
    delta_bess = cp.Variable(n_steps, boolean=True)
    delta_grid = cp.Variable(n_steps, boolean=True)
    return {
        'P_PV_consumer': P_PV_consumer,
        'P_PV_BESS': P_PV_BESS,
        'P_PV_grid': P_PV_grid,
        'P_BESS_consumer': P_BESS_consumer,
        'P_BESS_grid': P_BESS_grid,
        'P_grid_consumer': P_grid_consumer,
        'P_grid_BESS': P_grid_BESS,
        'SOC': SOC,
        'slack': slack,
        'delta_bess': delta_bess,
        'delta_grid': delta_grid
    }

def build_constraints(variables, data):
    constraints = []
    P_PV_consumer = variables['P_PV_consumer']
    P_PV_BESS = variables['P_PV_BESS']
    P_PV_grid = variables['P_PV_grid']
    P_BESS_consumer = variables['P_BESS_consumer']
    P_BESS_grid = variables['P_BESS_grid']
    P_grid_consumer = variables['P_grid_consumer']
    P_grid_BESS = variables['P_grid_BESS']
    SOC = variables['SOC']
    slack = variables['slack']
    delta_bess = variables['delta_bess']
    delta_grid = variables['delta_grid']

    constraints += [P_PV_consumer[t] + P_BESS_consumer[t] + P_grid_consumer[t] + slack[t] == data['consumer_demand'][t]
                    for t in data['time_indices']]

    constraints += [P_PV_consumer[t] + P_PV_BESS[t] + P_PV_grid[t] <= data['pv_power'][t] for t in data['time_indices']]

    for t in data['time_indices']:
        constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= data['bess_power_limit'],
                        P_BESS_consumer[t] + P_BESS_grid[t] <= data['bess_power_limit']]

    constraints += [SOC[0] == data['soc_initial']]
    constraints += [SOC[t+1] == SOC[t] + data['eta_charge'] * (P_PV_BESS[t] + P_grid_BESS[t]) * data['delta_t'] -
                    (P_BESS_consumer[t] + P_BESS_grid[t]) / data['eta_discharge'] * data['delta_t'] for t in range(data['n_steps'])]
    constraints += [SOC[t] <= data['bess_capacity'] for t in range(data['n_steps'] + 1)]
    constraints += [SOC[t] >= 0.05 * data['bess_capacity'] for t in range(data['n_steps'] + 1)]

    constraints += [SOC[data['n_steps']] >= data['soc_initial']]

    max_pv = np.max(data['pv_power'])
    max_demand = np.max(data['consumer_demand'])
    M_bess = data['bess_power_limit']
    M_grid = max(max_pv + data['bess_power_limit'], max_demand + data['bess_power_limit'])

    for t in data['time_indices']:
        constraints += [P_PV_BESS[t] + P_grid_BESS[t] <= M_bess * delta_bess[t]]
        constraints += [P_BESS_consumer[t] + P_BESS_grid[t] <= M_bess * (1 - delta_bess[t])]

    for t in data['time_indices']:
        constraints += [P_grid_consumer[t] + P_grid_BESS[t] <= M_grid * delta_grid[t]]
        constraints += [P_PV_grid[t] + P_BESS_grid[t] <= M_grid * (1 - delta_grid[t])]

    return constraints

def build_objective(variables, data):
    revenue = (
        cp.sum(cp.multiply(variables['P_PV_consumer'], data['grid_buy_price'] - data['lcoe_pv']) * data['delta_t']) +
        cp.sum(cp.multiply(variables['P_PV_grid'], data['grid_sell_price'] - data['lcoe_pv']) * data['delta_t']) +
        cp.sum(cp.multiply(variables['P_PV_BESS'], - data['lcoe_pv']) * data['delta_t']) +
        cp.sum(cp.multiply(variables['P_BESS_consumer'], data['grid_buy_price'] - data['lcoe_bess']) * data['delta_t']) +
        cp.sum(cp.multiply(variables['P_BESS_grid'], data['grid_sell_price'] - data['lcoe_bess']) * data['delta_t']) -
        cp.sum(cp.multiply(variables['P_grid_consumer'] + variables['P_grid_BESS'], data['grid_buy_price']) * data['delta_t']) -
        1e5 * cp.sum(variables['slack'])
    )
    return cp.Maximize(revenue)