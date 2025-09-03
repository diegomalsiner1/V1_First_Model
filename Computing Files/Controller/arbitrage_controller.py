import numpy as np


class ArbitrageController:
    """
    SOC-aware, PV-aware heuristic arbitrage controller on 15‑min resolution.

    Core ideas (hydro analogy):
    - Storage (SOC) behaves like a reservoir; PV is an exogenous inflow.
    - Over a finite look-ahead window (default 24 h), select the feasible number
      of charge/discharge slots (at Pmax) from the cheapest/priciest price steps.
    - Bias decisions with SOC (water value) and expected PV inflow to avoid
      emptying before big inflows or hoarding when no inflow is expected.
    - Optional minimum price gap and hysteresis reduce chattering.
    """

    def __init__(self, delta_t, bess_capacity, bess_power_limit, eta_charge, eta_discharge,
                 window_hours=24.0, gap_min=0.0, alpha_soc_charge=0.0, beta_soc_discharge=0.0,
                 gamma_pv_inflow=0.0, hold_steps=0, terminal_soc_ratio=None):
        self.delta_t = float(delta_t)
        self.bess_capacity = float(bess_capacity)
        self.bess_power_limit = float(bess_power_limit)
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        # Tunables
        self.window_hours = float(window_hours)        # look-ahead horizon (h)
        self.min_price_gap = float(gap_min)           # €/kWh minimal spread to act
        self.alpha_soc_charge = float(alpha_soc_charge)   # SOC bias for charging score
        self.beta_soc_discharge = float(beta_soc_discharge) # SOC bias for discharging score
        self.gamma_pv_inflow = float(gamma_pv_inflow)  # bias from expected PV inflow
        self.hold_steps = int(hold_steps)              # min steps to keep mode
        self.terminal_soc_ratio = terminal_soc_ratio   # soft target SOC/Cap by end of window

        # Hysteresis state across timesteps
        self._last_mode = "idle"   # one of: 'charge','discharge','idle'
        self._hold_ctr = 0

    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast,
                buy_forecast, sell_forecast, lcoe_pv, pi_ev, pi_consumer,
                horizon, start_dt):
        # Pure arbitrage with SOC & PV inflow awareness on 15‑min (delta_t) basis
        steps_look = int(min(horizon, round(self.window_hours / self.delta_t)))
        if steps_look <= 0:
            steps_24h = 1

        # Get current PV production and local demand
        pv_gen = pv_forecast[0] if len(pv_forecast) > 0 else 0.0
        consumer_demand = demand_forecast[0] if len(demand_forecast) > 0 else 0.0
        ev_demand = ev_forecast[0] if len(ev_forecast) > 0 else 0.0
        total_demand = consumer_demand + ev_demand

        window_buy = np.array(buy_forecast[:steps_look])
        window_sell = np.array(sell_forecast[:steps_look])
        window_pv = np.array(pv_forecast[:steps_look])
        window_d = np.array(demand_forecast[:steps_look])
        window_ev = np.array(ev_forecast[:steps_look])

        # Determine available slots from SOC headroom and energy
        # Each slot corresponds to max power for one time step
        slot_energy_charge = self.eta_charge * self.bess_power_limit * self.delta_t  # kWh gained per slot at Pmax
        slot_energy_discharge = (1.0 / self.eta_discharge) * self.bess_power_limit * self.delta_t  # kWh removed per slot at Pmax
        headroom_kwh = max(0.0, self.bess_capacity - soc)
        available_kwh = max(0.0, soc)
        max_charge_slots = int(np.floor(headroom_kwh / slot_energy_charge)) if slot_energy_charge > 0 else 0
        max_discharge_slots = int(np.floor(available_kwh / slot_energy_discharge)) if slot_energy_discharge > 0 else 0

        # Estimate PV inflow available to storage in window (after serving local demand)
        # This is a hydro-like “inflow pressure” that discourages emptying right before big PV hours.
        pv_surplus = np.maximum(0.0, window_pv - (window_d + window_ev))
        pv_inflow_kwh = np.sum(pv_surplus * self.delta_t)  # total kWh potentially storable
        inflow_density = (pv_inflow_kwh / max(self.bess_capacity, 1e-9))  # normalized 0..>

        # Build score vectors with optional SOC and PV-inflow bias
        soc_ratio = soc / self.bess_capacity if self.bess_capacity > 0 else 0.0
        charge_scores = -window_buy + self.alpha_soc_charge * (1.0 - soc_ratio) - self.gamma_pv_inflow * inflow_density
        discharge_scores = window_sell + self.beta_soc_discharge * soc_ratio + self.gamma_pv_inflow * inflow_density

        # Sort by scores
        charge_order = np.argsort(-charge_scores)  # higher score first
        discharge_order = np.argsort(-discharge_scores)

        # Select slots, enforcing exclusivity and optional minimum price gap
        chosen_charge = []
        chosen_discharge = []
        used = set()
        for idx in charge_order:
            if len(chosen_charge) >= max_charge_slots:
                break
            if idx in used:
                continue
            # minimal spread check vs best available sell
            if self.min_price_gap > 0 and idx < len(window_sell):
                if (np.max(window_sell) - window_buy[idx]) < self.min_price_gap:
                    continue
            chosen_charge.append(idx)
            used.add(idx)
        for idx in discharge_order:
            if len(chosen_discharge) >= max_discharge_slots:
                break
            if idx in used:
                continue
            if self.min_price_gap > 0 and idx < len(window_buy):
                if (window_sell[idx] - np.min(window_buy)) < self.min_price_gap:
                    continue
            chosen_discharge.append(idx)
            used.add(idx)

        # Current step action from slot membership with hysteresis & spread checks
        base_want_charge = (0 in chosen_charge) and (0 not in chosen_discharge)
        base_want_discharge = (0 in chosen_discharge) and (0 not in chosen_charge)
        # Apply minimal spread gate
        if self.min_price_gap > 0:
            if base_want_charge and (window_sell.max() - window_buy[0] < self.min_price_gap):
                base_want_charge = False
            if base_want_discharge and (window_sell[0] - window_buy.min() < self.min_price_gap):
                base_want_discharge = False

        # Hysteresis: prevent frequent flips unless a strong signal
        want_charge = base_want_charge
        want_discharge = base_want_discharge
        if self.hold_steps > 0:
            if self._hold_ctr < self.hold_steps:
                # If currently holding a mode, ignore the opposite signal unless strong breakout
                if self._last_mode == 'charge' and base_want_discharge and not base_want_charge:
                    if (window_sell[0] - window_buy.min()) < (self.min_price_gap * 2.0):
                        want_discharge = False
                if self._last_mode == 'discharge' and base_want_charge and not base_want_discharge:
                    if (window_sell.max() - window_buy[0]) < (self.min_price_gap * 2.0):
                        want_charge = False
            # Update hold state
            new_mode = 'charge' if want_charge else ('discharge' if want_discharge else 'idle')
            if new_mode == self._last_mode:
                self._hold_ctr += 1
            else:
                self._hold_ctr = 0
                self._last_mode = new_mode

        # Compute feasible powers respecting SOC and power limits at 15-min granularity
        max_charge_by_power = self.bess_power_limit
        max_charge_by_soc = max(0.0, (self.bess_capacity - soc) / (self.delta_t * self.eta_charge))
        charge_power = min(max_charge_by_power, max_charge_by_soc) if want_charge else 0.0

        max_discharge_by_power = self.bess_power_limit
        max_discharge_by_soc = max(0.0, soc * self.eta_discharge / self.delta_t)
        discharge_power = min(max_discharge_by_power, max_discharge_by_soc) if want_discharge else 0.0

        # Optional terminal SOC soft guidance: gently steer toward target at window end.
        # If terminal_soc_ratio is given, bias current action if far from target.
        if self.terminal_soc_ratio is not None and self.bess_capacity > 0:
            target_soc = float(self.terminal_soc_ratio) * self.bess_capacity
            gap = (soc - target_soc) / self.bess_capacity
            # If far below target, discourage discharge a bit; if far above, discourage charge a bit.
            if gap < -0.1 and want_discharge:  # low SOC → avoid discharging
                discharge_power = 0.0
            if gap > 0.1 and want_charge:      # high SOC → avoid charging
                charge_power = 0.0

        # Update SOC for next step
        soc_next = soc + self.eta_charge * charge_power * self.delta_t - (discharge_power / self.eta_discharge) * self.delta_t
        soc_next = min(max(soc_next, 0.0), self.bess_capacity)

        # Energy flow allocation with PV consideration
        # Priority: 1) PV to local demand, 2) PV to BESS (if charging), 3) PV to grid, 4) Grid to BESS, 5) BESS to grid
        
        # Step 1: PV to local demand (free energy)
        pv_to_consumer = min(pv_gen, consumer_demand)
        pv_to_ev = min(pv_gen - pv_to_consumer, ev_demand)
        pv_remaining = pv_gen - pv_to_consumer - pv_to_ev
        
        # Step 2: PV to BESS if we want to charge (free charging)
        pv_to_bess = min(pv_remaining, charge_power) if want_charge else 0.0
        charge_from_grid = charge_power - pv_to_bess  # Remaining charge from grid
        
        # Step 3: Remaining PV to grid
        pv_to_grid = pv_remaining - pv_to_bess
        
        # Step 4: BESS discharge to grid (arbitrage)
        bess_to_grid = discharge_power if want_discharge else 0.0
        
        # Step 5: Grid flows
        grid_import = charge_from_grid  # Only grid charging (PV charging is free)
        grid_export = pv_to_grid + bess_to_grid  # PV + BESS exports
        
        # Step 6: Local demand met by grid if needed
        grid_to_consumer = max(0.0, consumer_demand - pv_to_consumer)
        grid_to_ev = max(0.0, ev_demand - pv_to_ev)
        
        # Step 7: Combined flows for plotting
        pv_bess_to_consumer = pv_to_consumer  # Only PV (BESS doesn't serve local demand in arbitrage)
        pv_bess_to_ev = pv_to_ev  # Only PV
        pv_bess_to_grid = pv_to_grid + bess_to_grid  # PV + BESS exports

        # AC bus flows for PyPSA compatibility
        pv_to_ac = pv_to_consumer + pv_to_ev + pv_to_grid  # All PV goes to AC bus
        bess_to_ac = bess_to_grid  # BESS discharge goes to AC bus
        ac_to_bess = pv_to_bess + charge_from_grid  # BESS charging from AC bus (PV + grid)
        
        return {
            'P_BESS': discharge_power,  # signless kW (use separate charge/discharge fields)
            'SOC_next': soc_next,
            'P_BESS_discharge': discharge_power,
            'P_grid_to_bess': charge_from_grid,  # Only grid charging (PV charging is free)
            'P_BESS_charge': charge_power,
            'P_PV_gen': pv_gen,
            'P_link_dc_to_ac': pv_to_ac + bess_to_ac,
            'P_grid_import': grid_import,
            'P_grid_export': grid_export,
            'pv_bess_to_consumer': pv_bess_to_consumer,
            'pv_bess_to_ev': pv_bess_to_ev,
            'pv_bess_to_grid': pv_bess_to_grid,
            'grid_to_consumer': grid_to_consumer,
            'grid_to_ev': grid_to_ev,
            'ev_renewable_share': 0.0,
            'ev_revenue': 0.0,
            'slack': 0.0
        }


