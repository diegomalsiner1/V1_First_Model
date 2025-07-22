import numpy as np

def compute_revenues(results, data):
    # Vectorized revenue calculations using NumPy for efficiency and consistency
    pv_to_consumer_rev = results['P_PV_consumer_vals'] * (data['grid_buy_price'] - data['lcoe_pv']) * data['delta_t']
    pv_to_grid_rev = results['P_PV_grid_vals'] * (data['grid_sell_price'] - data['lcoe_pv']) * data['delta_t']
    pv_to_bess_cost = results['P_PV_BESS_vals'] * (-data['lcoe_pv']) * data['delta_t']
    bess_to_consumer_rev = results['P_BESS_consumer_vals'] * (data['grid_buy_price'] - data['lcoe_bess']) * data['delta_t']
    bess_to_grid_rev = results['P_BESS_grid_vals'] * (data['grid_sell_price'] - data['lcoe_bess']) * data['delta_t']
    grid_buy_cost = - (results['P_grid_consumer_vals'] + results['P_grid_BESS_vals']) * data['grid_buy_price'] * data['delta_t']
    penalty_per_step = -1e5 * results['slack_vals'] * data['delta_t']  # Penalty adjusted for energy units (kWh)
    total_net_per_step = pv_to_consumer_rev + pv_to_grid_rev + pv_to_bess_cost + bess_to_consumer_rev + bess_to_grid_rev + grid_buy_cost + penalty_per_step
    
    revenues = {
        'pv_to_consumer_rev': pv_to_consumer_rev,
        'pv_to_grid_rev': pv_to_grid_rev,
        'pv_to_bess_cost': pv_to_bess_cost,
        'bess_to_consumer_rev': bess_to_consumer_rev,
        'bess_to_grid_rev': bess_to_grid_rev,
        'grid_buy_cost': grid_buy_cost,
        'penalty_per_step': penalty_per_step,
        'total_net_per_step': total_net_per_step,
        'total_revenue': np.sum(total_net_per_step)
    }
    
    # Calculate self-sufficiency ratio (unitless percentage)
    total_demand = np.sum(data['consumer_demand']) * data['delta_t']  # Total demand energy (kWh)
    total_renewable_to_cons = np.sum(results['P_PV_consumer_vals'] + results['P_BESS_consumer_vals']) * data['delta_t']  # Renewable supplied to consumer (kWh)
    self_sufficiency = (total_renewable_to_cons / total_demand * 100) if total_demand > 0 else 0
    
    revenues['self_sufficiency'] = self_sufficiency  # Add to revenues dict
    
    return revenues

def print_results(revenues, results, data):
    print(f"Total Revenue: Eur{revenues['total_revenue']:.2f}")
    print("Time steps with unmet demand (kW):")
    for t in data['time_indices']:
        if results['slack_vals'][t] > 1e-6:
            print(f"Time {data['time_steps'][t]:.2f}h: Unmet demand = {results['slack_vals'][t]:.2f} kW")