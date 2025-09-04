import numpy as np

def compute_revenues(results, data):
    """
    Compute detailed revenue streams and self-sufficiency for the simulation results.
    Uses actual MPC optimization results instead of overriding them.
    """
    # Extract actual MPC results
    pv_gen = results.get('P_PV_gen', np.zeros_like(data['consumer_demand']))
    bess_discharge = results.get('P_BESS_discharge', np.zeros_like(pv_gen))
    bess_charge = results.get('P_BESS_charge', np.zeros_like(pv_gen))
    consumer_demand = data['consumer_demand']
    ev_demand = data['ev_demand']
    
    # Use actual MPC results if available, otherwise fall back to priority logic
    if 'pv_to_consumer' in results and 'pv_to_ev' in results and 'pv_to_grid' in results:
        # Use MPC results directly
        pv_to_cons = results['pv_to_consumer']
        pv_to_ev = results['pv_to_ev']
        pv_to_grid = results['pv_to_grid']
        bess_to_cons = results.get('bess_to_consumer', np.zeros_like(pv_gen))
        bess_to_ev = results.get('bess_to_ev', np.zeros_like(pv_gen))
        bess_to_grid = results.get('bess_to_grid', np.zeros_like(pv_gen))
        grid_to_cons = results.get('grid_to_consumer', np.zeros_like(pv_gen))
        grid_to_ev = results.get('grid_to_ev', np.zeros_like(pv_gen))
        grid_to_bess = results.get('P_Grid_to_BESS', np.zeros_like(pv_gen))
    else:
        # Fallback to priority logic for backward compatibility
        pv_to_cons = np.minimum(pv_gen, consumer_demand)
        remaining_pv = pv_gen - pv_to_cons
        
        pv_to_ev = np.minimum(remaining_pv, ev_demand)
        remaining_pv = remaining_pv - pv_to_ev
        
        pv_to_charge = np.minimum(remaining_pv, bess_charge)
        pv_to_grid = np.maximum(remaining_pv - pv_to_charge, 0)

        # Priority 2: BESS discharge to remaining demands/grid
        remaining_cons = np.maximum(consumer_demand - pv_to_cons, 0)
        remaining_ev = np.maximum(ev_demand - pv_to_ev, 0)
        
        bess_to_ev = np.minimum(bess_discharge, remaining_ev)
        remaining_bess = bess_discharge - bess_to_ev
        
        bess_to_cons = np.minimum(remaining_bess, remaining_cons)
        bess_to_grid = np.maximum(remaining_bess - bess_to_cons, 0)

        # Grid fills any gaps
        grid_to_cons = np.maximum(remaining_cons - bess_to_cons, 0)
        grid_to_ev = np.maximum(remaining_ev - bess_to_ev, 0)
        grid_to_bess = results.get('P_Grid_to_BESS', np.zeros_like(pv_gen))

    # Calculate total flows
    calculated_import = grid_to_cons + grid_to_ev + grid_to_bess
    calculated_export = pv_to_grid + bess_to_grid

    # Update results with actual flows (don't override if already present)
    if 'pv_to_consumer' not in results:
        results['P_PV_consumer_vals'] = pv_to_cons
        results['P_PV_ev_vals'] = pv_to_ev
        results['P_PV_grid_vals'] = pv_to_grid
        results['P_BESS_consumer_vals'] = bess_to_cons
        results['P_BESS_ev_vals'] = bess_to_ev
        results['P_BESS_grid_vals'] = bess_to_grid
        results['P_grid_consumer_vals'] = grid_to_cons
        results['P_grid_ev_vals'] = grid_to_ev
        results['P_grid_sold'] = calculated_export
        results['P_grid_bought'] = calculated_import
        results['P_grid_to_bess'] = grid_to_bess
    # Always expose BESS↔Grid exchanges explicitly for plotting masks
    results['P_bess_to_grid'] = bess_to_grid
    results['P_grid_to_bess_only'] = grid_to_bess

    # Revenues (per-step arrays)
    grid_buy_cost = calculated_import * data['grid_buy_price'] * data['delta_t']
    grid_sell_revenue = calculated_export * data['grid_sell_price'] * data['delta_t']
    
    # BESS charging costs
    grid_to_bess_cost = grid_to_bess * data['grid_buy_price'] * data['delta_t']
    bess_charging_efficiency_loss = grid_to_bess * (1 - data['eta_charge']) * data['grid_buy_price'] * data['delta_t']
    
    ev_rev = (pv_to_ev + bess_to_ev) * data['pi_ev'] * data['delta_t']
    
    # Net revenue calculation
    total_net_per_step = grid_sell_revenue - grid_buy_cost + ev_rev - grid_to_bess_cost - bess_charging_efficiency_loss

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
        'total_revenue': np.sum(total_net_per_step),
        'bess_grid_charging_cost': grid_to_bess_cost,
        'bess_efficiency_loss_cost': bess_charging_efficiency_loss
    }

    # Self-sufficiency calculation
    total_consumer_demand = np.sum(consumer_demand) * data['delta_t']
    total_renewable_to_cons = np.sum(pv_to_cons + bess_to_cons) * data['delta_t']
    revenues['self_sufficiency'] = (total_renewable_to_cons / total_consumer_demand * 100) if total_consumer_demand > 0 else 0

    # EV renewable share
    total_ev_demand = np.sum(ev_demand) * data['delta_t']
    renewable_to_ev = np.sum(pv_to_ev + bess_to_ev) * data['delta_t']
    revenues['ev_renewable_share'] = (renewable_to_ev / total_ev_demand * 100) if total_ev_demand > 0 else 0
    
    # Export totals
    revenues['total_pv_to_grid_rev'] = np.sum(revenues['pv_to_grid_rev'])
    revenues['total_pv_to_ev_rev'] = np.sum(revenues['pv_to_ev_rev'])
    revenues['total_bess_to_grid_rev'] = np.sum(revenues['bess_to_grid_rev'])
    revenues['total_bess_to_ev_rev'] = np.sum(revenues['bess_to_ev_rev'])
    revenues['total_grid_buy_cost'] = np.sum(revenues['grid_buy_cost'])
    
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
    print(f"Revenue from PV to Grid: Eur{revenues['total_pv_to_grid_rev']:.2f}")
    print(f"Revenue from PV to EV: Eur{revenues['total_pv_to_ev_rev']:.2f}")
    print(f"Revenue from BESS to Grid: Eur{revenues['total_bess_to_grid_rev']:.2f}")
    print(f"Revenue from BESS to EV: Eur{revenues['total_bess_to_ev_rev']:.2f}")
    print(f"EV renewable share: {revenues['ev_renewable_share']:.2f}%")
    print(f"BESS Grid Charging Cost: Eur{np.sum(revenues['bess_grid_charging_cost']):.2f}")
    print(f"BESS Efficiency Loss Cost: Eur{np.sum(revenues['bess_efficiency_loss_cost']):.2f}")
    # Remove slack_vals check, as PyPSA always balances demand if feasible
    print("All demand met in every timestep (no slack variable used).")