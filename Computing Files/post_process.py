import numpy as np

def extract_results(variables, status):
    if status != cp.OPTIMAL:
        return None
    results = {}
    results['P_PV_consumer_vals'] = variables['P_PV_consumer'].value
    results['P_PV_BESS_vals'] = variables['P_PV_BESS'].value
    results['P_PV_grid_vals'] = variables['P_PV_grid'].value
    results['P_BESS_consumer_vals'] = variables['P_BESS_consumer'].value
    results['P_BESS_grid_vals'] = variables['P_BESS_grid'].value
    results['P_grid_consumer_vals'] = variables['P_grid_consumer'].value
    results['P_grid_BESS_vals'] = variables['P_grid_BESS'].value
    results['SOC_vals'] = variables['SOC'].value
    results['slack_vals'] = variables['slack'].value

    results['P_BESS_charge'] = results['P_PV_BESS_vals'] + results['P_grid_BESS_vals']
    results['P_BESS_discharge'] = results['P_BESS_consumer_vals'] + results['P_BESS_grid_vals']

    results['P_grid_sold'] = results['P_PV_grid_vals'] + results['P_BESS_grid_vals']
    results['P_grid_bought'] = results['P_grid_consumer_vals'] + results['P_grid_BESS_vals']
    return results

def compute_revenues(results, data):
    pv_to_consumer_rev = []
    pv_to_grid_rev = []
    pv_to_bess_cost = []
    bess_to_consumer_rev = []
    bess_to_grid_rev = []
    grid_buy_cost = []
    penalty_per_step = []
    total_net_per_step = []
    for t in data['time_indices']:
        pv_cons = results['P_PV_consumer_vals'][t] * (data['grid_buy_price'][t] - data['lcoe_pv']) * data['delta_t']
        pv_grid = results['P_PV_grid_vals'][t] * (data['grid_sell_price'][t] - data['lcoe_pv']) * data['delta_t']
        pv_bess = results['P_PV_BESS_vals'][t] * (- data['lcoe_pv']) * data['delta_t']
        bess_cons = results['P_BESS_consumer_vals'][t] * (data['grid_buy_price'][t] - data['lcoe_bess']) * data['delta_t']
        bess_grid = results['P_BESS_grid_vals'][t] * (data['grid_sell_price'][t] - data['lcoe_bess']) * data['delta_t']
        grid_cost = - (results['P_grid_consumer_vals'][t] + results['P_grid_BESS_vals'][t]) * data['grid_buy_price'][t] * data['delta_t']
        penalty = -1e5 * results['slack_vals'][t]
        net_rev = pv_cons + pv_grid + pv_bess + bess_cons + bess_grid + grid_cost + penalty
        pv_to_consumer_rev.append(pv_cons)
        pv_to_grid_rev.append(pv_grid)
        pv_to_bess_cost.append(pv_bess)
        bess_to_consumer_rev.append(bess_cons)
        bess_to_grid_rev.append(bess_grid)
        grid_buy_cost.append(grid_cost)
        penalty_per_step.append(penalty)
        total_net_per_step.append(net_rev)
    revenues = {
        'pv_to_consumer_rev': pv_to_consumer_rev,
        'pv_to_grid_rev': pv_to_grid_rev,
        'pv_to_bess_cost': pv_to_bess_cost,
        'bess_to_consumer_rev': bess_to_consumer_rev,
        'bess_to_grid_rev': bess_to_grid_rev,
        'grid_buy_cost': grid_buy_cost,
        'penalty_per_step': penalty_per_step,
        'total_net_per_step': total_net_per_step,
        'total_revenue': sum(total_net_per_step)
    }
    return revenues

def print_results(revenues, results, data):
    print(f"Total Revenue: Eur{revenues['total_revenue']:.2f}")
    print("Time steps with unmet demand (kW):")
    for t in data['time_indices']:
        if results['slack_vals'][t] > 1e-6:
            print(f"Time {data['time_steps'][t]:.2f}h: Unmet demand = {results['slack_vals'][t]:.2f} kW")