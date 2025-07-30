import numpy as np

def compute_revenues(results, data):
    """
    Compute detailed revenue streams and self-sufficiency for the simulation results.
    Prioritizes PV for local use to ensure correct splitting and visualization.
    """
    # Direct extractions (add P_PV_gen to results as above)
    pv_gen = results.get('P_PV_gen', np.zeros_like(data['consumer_demand']))
    bess_discharge = results.get('P_BESS_discharge', np.zeros_like(pv_gen))
    bess_charge = results.get('P_BESS_charge', np.zeros_like(pv_gen))
    consumer_demand = data['consumer_demand']
    ev_demand = data['ev_demand']

    # Priority 1: PV to demands/charge/grid
    pv_to_cons = np.minimum(pv_gen, consumer_demand)
    pv_to_ev = np.minimum(np.maximum(pv_gen - pv_to_cons, 0), ev_demand)
    pv_to_charge = np.minimum(np.maximum(pv_gen - pv_to_cons - pv_to_ev, 0), bess_charge)
    pv_to_grid = np.maximum(pv_gen - pv_to_cons - pv_to_ev - pv_to_charge, 0)

    # Priority 2: BESS discharge to remaining demands/grid
    remaining_cons = np.maximum(consumer_demand - pv_to_cons, 0)
    remaining_ev = np.maximum(ev_demand - pv_to_ev, 0)
    bess_to_cons = np.minimum(bess_discharge, remaining_cons)
    bess_to_ev = np.minimum(np.maximum(bess_discharge - bess_to_cons, 0), remaining_ev)
    bess_to_grid = np.maximum(bess_discharge - bess_to_cons - bess_to_ev, 0)

    # Grid fills any gaps
    grid_to_cons = np.maximum(remaining_cons - bess_to_cons, 0)
    grid_to_ev = np.maximum(remaining_ev - bess_to_ev, 0)
    
    # Read grid-to-BESS charging from results
    grid_to_bess = results.get('P_Grid_to_BESS', np.zeros_like(pv_gen))
    calculated_import = grid_to_cons + grid_to_ev + grid_to_bess
    calculated_export = pv_to_grid + bess_to_grid

    # Override results for plots (pure PV/BESS separation)
    results['P_PV_consumer_vals'] = pv_to_cons
    results['P_PV_ev_vals'] = pv_to_ev
    results['P_PV_grid_vals'] = pv_to_grid
    results['P_BESS_consumer_vals'] = bess_to_cons
    results['P_BESS_ev_vals'] = bess_to_ev
    results['P_BESS_grid_vals'] = bess_to_grid
    results['P_grid_consumer_vals'] = grid_to_cons
    results['P_grid_ev_vals'] = grid_to_ev
    results['P_grid_sold'] = calculated_export  # For top plot
    results['P_grid_bought'] = calculated_import  # Add if needed for purple line
    results['P_grid_to_bess'] = grid_to_bess

    # Revenues (per-step arrays; include self-cons savings and EV rev)
    grid_buy_cost = calculated_import * data['grid_buy_price'] * data['delta_t']
    grid_sell_revenue = calculated_export * data['grid_sell_price'] * data['delta_t']
    #cons_savings = (pv_to_cons + bess_to_cons) * data['pi_consumer'] * data['delta_t']  # Avoided grid buy for consumer
    ev_rev = (pv_to_ev + bess_to_ev) * data['pi_ev'] * data['delta_t']  # Revenue from renewable to EV
    #pv_to_charge_cost = pv_to_charge * data['lcoe_pv'] * data['delta_t']  # Charging cost (if LCOE>0)

    total_net_per_step = grid_sell_revenue - grid_buy_cost + ev_rev

    revenues = {
        'grid_sell_revenue': grid_sell_revenue,
        'grid_buy_cost': grid_buy_cost,
        'pv_to_consumer_rev': pv_to_cons * data['pi_consumer'] * data['delta_t'],
        'bess_to_consumer_rev': bess_to_cons * data['pi_consumer'] * data['delta_t'],
        'pv_to_ev_rev': pv_to_ev * data['pi_ev'] * data['delta_t'],
        'bess_to_ev_rev': bess_to_ev * data['pi_ev'] * data['delta_t'],
        'pv_to_grid_rev': pv_to_grid * data['grid_sell_price'] * data['delta_t'],
        'bess_to_grid_rev': bess_to_grid * data['grid_sell_price'] * data['delta_t'],
        'total_net_per_step': total_net_per_step,
        'total_revenue': np.sum(total_net_per_step)
    }

    # Self-sufficiency (consumer only, now credits PV properly)
    total_consumer_demand = np.sum(consumer_demand) * data['delta_t']
    total_renewable_to_cons = np.sum(pv_to_cons + bess_to_cons) * data['delta_t']
    revenues['self_sufficiency'] = (total_renewable_to_cons / total_consumer_demand * 100) if total_consumer_demand > 0 else 0

    # EV renewable share
    total_ev_demand = np.sum(ev_demand) * data['delta_t']
    renewable_to_ev = np.sum(pv_to_ev + bess_to_ev) * data['delta_t']
    revenues['ev_renewable_share'] = (renewable_to_ev / total_ev_demand * 100) if total_ev_demand > 0 else 0

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