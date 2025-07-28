import numpy as np

def compute_revenues(results, data):
    """
    Compute detailed revenue streams and self-sufficiency for the simulation results.

    Args:
        results (dict): Power flows and slack values.
        data (dict): Simulation data and parameters.

    Returns:
        dict: Revenue streams and self-sufficiency ratio.
    """
    # Map BESS discharge to consumer, EV, and grid
    bess_discharge = results.get('P_BESS_discharge', np.zeros_like(results['P_BESS_consumer_vals']))
    consumer_demand = data['consumer_demand']
    ev_demand = data['ev_demand']
    pv_bess_to_consumer = results['P_PV_consumer_vals']
    pv_bess_to_ev = results['P_PV_ev_vals']
    pv_bess_to_grid = results['P_PV_grid_vals']
    # Assign BESS discharge to consumer, then EV, then grid
    bess_to_consumer = np.minimum(bess_discharge, np.maximum(consumer_demand - pv_bess_to_consumer, 0))
    bess_to_ev = np.minimum(np.maximum(bess_discharge - bess_to_consumer, 0), np.maximum(ev_demand - pv_bess_to_ev, 0))
    bess_to_grid = np.maximum(bess_discharge - bess_to_consumer - bess_to_ev, 0)
    results['P_BESS_consumer_vals'] = bess_to_consumer
    results['P_BESS_ev_vals'] = bess_to_ev
    results['P_BESS_grid_vals'] = bess_to_grid
    # Grid sold is PV+BESS to grid
    results['P_grid_sold'] = pv_bess_to_grid + bess_to_grid

    # Vectorized revenue calculations using NumPy for efficiency and consistency
    # Regular consumer revenue
    pv_to_consumer_rev = results['P_PV_consumer_vals'] * (data['grid_buy_price'] - data['lcoe_pv']) * data['delta_t']
    # EV charging revenue (at EV_PRICE)
    pv_to_ev_rev = results['P_PV_ev_vals'] * (data['pi_ev'] - data['lcoe_pv']) * data['delta_t']
    # Grid export revenue
    pv_to_grid_rev = results['P_PV_grid_vals'] * (data['grid_sell_price'] - data['lcoe_pv']) * data['delta_t']
    pv_to_bess_cost = results['P_PV_BESS_vals'] * (-data['lcoe_pv']) * data['delta_t']
    bess_to_consumer_rev = results['P_BESS_consumer_vals'] * (data['grid_buy_price'] - data['lcoe_bess']) * data['delta_t']
    bess_to_grid_rev = results['P_BESS_grid_vals'] * (data['grid_sell_price'] - data['lcoe_bess']) * data['delta_t']
    grid_buy_cost = - (results['P_grid_consumer_vals'] + results['P_grid_BESS_vals']) * data['grid_buy_price'] * data['delta_t']
    penalty_per_step = -1e5 * results['slack_vals'] * data['delta_t']  # Penalty adjusted for energy units (kWh)
    total_net_per_step = pv_to_consumer_rev + pv_to_grid_rev + pv_to_bess_cost + bess_to_consumer_rev + bess_to_grid_rev + grid_buy_cost + penalty_per_step
    
    # Add EV charging revenues (real cash inflow)
    ev_charging_revenue = (results['P_PV_ev_vals'] + results['P_BESS_ev_vals'] + results['P_grid_ev_vals']) * data['pi_ev'] * data['delta_t']
    
    revenues = {
        'pv_to_consumer_rev': pv_to_consumer_rev,
        'pv_to_grid_rev': pv_to_grid_rev,
        'pv_to_bess_cost': pv_to_bess_cost,
        'bess_to_consumer_rev': bess_to_consumer_rev,
        'bess_to_grid_rev': bess_to_grid_rev,
        'grid_buy_cost': grid_buy_cost,
        'ev_charging_revenue': ev_charging_revenue,  # Real cash inflow from EV charging
        'penalty_per_step': penalty_per_step,
        'total_net_per_step': total_net_per_step + ev_charging_revenue,
        'total_revenue': np.sum(total_net_per_step + ev_charging_revenue)
    }
    
    # Calculate self-sufficiency ratio (unitless percentage) - excluding EV demand
    total_consumer_demand = np.sum(data['consumer_demand']) * data['delta_t']  # Consumer demand only
    total_renewable_to_cons = np.sum(results['P_PV_consumer_vals'] + results['P_BESS_consumer_vals']) * data['delta_t']
    self_sufficiency = (total_renewable_to_cons / total_consumer_demand * 100) if total_consumer_demand > 0 else 0
    
    # Calculate EV renewable service level
    total_ev_demand = np.sum(data['ev_demand']) * data['delta_t']
    renewable_to_ev = np.sum(results['P_PV_ev_vals'] + results['P_BESS_ev_vals']) * data['delta_t']
    ev_renewable_share = (renewable_to_ev / total_ev_demand * 100) if total_ev_demand > 0 else 0
    revenues['ev_renewable_share'] = ev_renewable_share  # Add to revenues dict
    
    revenues['self_sufficiency'] = self_sufficiency  # Add to revenues dict
    
    return revenues

def print_results(revenues, results, data):
    """
    Print summary results for the simulation.

    Args:
        revenues (dict): Revenue streams and self-sufficiency.
        results (dict): Power flows and slack values.
        data (dict): Simulation data and parameters.
    """
    print(f"Total Revenue: Eur{revenues['total_revenue']:.2f}")
    print("Time steps with unmet demand (kW):")
    for t in data['time_indices']:
        if results['slack_vals'][t] > 1e-6:
            print(f"Time {data['time_steps'][t]:.2f}h: Unmet demand = {results['slack_vals'][t]:.2f} kW")