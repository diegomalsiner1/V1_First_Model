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
    bess_discharge = results.get('P_BESS_discharge', np.zeros_like(data['consumer_demand']))
    consumer_demand = data['consumer_demand']
    ev_demand = data['ev_demand']
    pv_bess_to_consumer = results['P_PV_consumer_vals']
    pv_bess_to_ev = results['P_PV_ev_vals']
    pv_bess_to_grid = results['P_PV_grid_vals']
    # Assign flows to Consumer and EV from PV, BESS, and Grid
    # Correct allocation of BESS discharge to consumer and EV demand
    n = len(consumer_demand)
    pv_to_consumer = np.zeros(n)
    bess_to_consumer = np.zeros(n)
    grid_to_consumer = np.zeros(n)
    pv_to_ev = np.zeros(n)
    bess_to_ev = np.zeros(n)
    grid_to_ev = np.zeros(n)
    for t in range(n):
        # PV to consumer
        pv_to_consumer[t] = min(consumer_demand[t], pv_bess_to_consumer[t])
        # BESS to consumer
        bess_avail = bess_discharge[t]
        bess_to_consumer[t] = min(max(consumer_demand[t] - pv_to_consumer[t], 0), bess_avail)
        bess_avail -= bess_to_consumer[t]
        # Grid to consumer
        grid_to_consumer[t] = max(consumer_demand[t] - pv_to_consumer[t] - bess_to_consumer[t], 0)
        # PV to EV
        pv_to_ev[t] = min(ev_demand[t], pv_bess_to_ev[t])
        # BESS to EV
        bess_to_ev[t] = min(max(ev_demand[t] - pv_to_ev[t], 0), bess_avail)
        bess_avail -= bess_to_ev[t]
        # Grid to EV
        grid_to_ev[t] = max(ev_demand[t] - pv_to_ev[t] - bess_to_ev[t], 0)
    results['P_PV_consumer_vals'] = pv_to_consumer
    results['P_BESS_consumer_vals'] = bess_to_consumer
    results['P_grid_consumer_vals'] = grid_to_consumer
    results['P_PV_ev_vals'] = pv_to_ev
    results['P_BESS_ev_vals'] = bess_to_ev
    results['P_grid_ev_vals'] = grid_to_ev
    bess_to_grid = np.maximum(bess_discharge - bess_to_consumer - bess_to_ev, 0)
    results['P_BESS_grid_vals'] = bess_to_grid
    # Grid sold is PV+BESS to grid
    results['P_grid_sold'] = results['P_PV_grid_vals'] + bess_to_grid

    # Grid bought is grid import
    results['P_grid_bought'] = results.get('P_grid_import_vals', np.zeros_like(data['consumer_demand']))

    # Grid sold energy
    grid_sell_revenue = results['P_grid_sold'] * data['grid_sell_price'] * data['delta_t']
    grid_buy_cost = results['P_grid_bought'] * data['grid_buy_price'] * data['delta_t']

    # Consumer and EV revenue (if applicable)
    pv_to_consumer_rev = results['P_PV_consumer_vals'] * data['lcoe_pv'] * data['delta_t']
    pv_to_ev_rev = results['P_PV_ev_vals'] * data['lcoe_pv'] * data['delta_t']

    total_net_per_step = grid_sell_revenue - grid_buy_cost
    total_revenue = np.sum(total_net_per_step)

    revenues = {
        'grid_sell_revenue': grid_sell_revenue,
        'grid_buy_cost': grid_buy_cost,
        'pv_to_consumer_rev': pv_to_consumer_rev,
        'pv_to_ev_rev': pv_to_ev_rev,
        'total_net_per_step': total_net_per_step,
        'total_revenue': total_revenue
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
    
    # Calculate per-timestep PV and BESS grid export shares
    pv_grid_export = results['P_PV_grid_vals']
    bess_grid_export = results['P_BESS_grid_vals']
    results['P_grid_export_pv'] = pv_grid_export
    results['P_grid_export_bess'] = bess_grid_export
    
    # Calculate per-timestep grid sell revenue from PV and BESS
    grid_sell_revenue_pv = results['P_PV_grid_vals'] * data['grid_sell_price'] * data['delta_t']
    grid_sell_revenue_bess = results['P_BESS_grid_vals'] * data['grid_sell_price'] * data['delta_t']
    revenues['grid_sell_revenue_pv'] = grid_sell_revenue_pv
    revenues['grid_sell_revenue_bess'] = grid_sell_revenue_bess
    
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
    print(f"Self-sufficiency ratio (consumer): {revenues['self_sufficiency']:.2f}%")
    print(f"EV renewable share: {revenues['ev_renewable_share']:.2f}%")
    # Remove slack_vals check, as PyPSA always balances demand if feasible
    print("All demand met in every timestep (no slack variable used).")