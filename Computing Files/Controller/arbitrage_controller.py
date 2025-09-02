import numpy as np


class ArbitrageController:
    def __init__(self, delta_t, bess_capacity, bess_power_limit, eta_charge, eta_discharge):
        self.delta_t = float(delta_t)
        self.bess_capacity = float(bess_capacity)
        self.bess_power_limit = float(bess_power_limit)
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        # Optional tuning
        self.min_price_gap = 0.0  # Euro/kWh; set >0 to require spread for actions
        self.alpha_soc_charge = 0.0  # SOC bias weight for charging score
        self.beta_soc_discharge = 0.0  # SOC bias weight for discharging score

    def predict(self, soc, pv_forecast, demand_forecast, ev_forecast,
                buy_forecast, sell_forecast, lcoe_pv, pi_ev, pi_consumer,
                horizon, start_dt):
        # Pure arbitrage with SOC-aware allocation on 15-min (delta_t) basis
        steps_24h = int(min(horizon, round(24.0 / self.delta_t)))
        if steps_24h <= 0:
            steps_24h = 1

        window_buy = np.array(buy_forecast[:steps_24h])
        window_sell = np.array(sell_forecast[:steps_24h])

        # Determine available slots from SOC headroom and energy
        # Each slot corresponds to max power for one time step
        slot_energy_charge = self.eta_charge * self.bess_power_limit * self.delta_t  # kWh gained per slot at Pmax
        slot_energy_discharge = (1.0 / self.eta_discharge) * self.bess_power_limit * self.delta_t  # kWh removed per slot at Pmax
        headroom_kwh = max(0.0, self.bess_capacity - soc)
        available_kwh = max(0.0, soc)
        max_charge_slots = int(np.floor(headroom_kwh / slot_energy_charge)) if slot_energy_charge > 0 else 0
        max_discharge_slots = int(np.floor(available_kwh / slot_energy_discharge)) if slot_energy_discharge > 0 else 0

        # Build score vectors with optional SOC bias
        soc_ratio = soc / self.bess_capacity if self.bess_capacity > 0 else 0.0
        charge_scores = -window_buy + self.alpha_soc_charge * (1.0 - soc_ratio)
        discharge_scores = window_sell + self.beta_soc_discharge * soc_ratio

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

        # Current step action
        want_charge = (0 in chosen_charge) and (0 not in chosen_discharge)
        want_discharge = (0 in chosen_discharge) and (0 not in chosen_charge)

        # Compute feasible powers respecting SOC and power limits at 15-min granularity
        max_charge_by_power = self.bess_power_limit
        max_charge_by_soc = max(0.0, (self.bess_capacity - soc) / (self.delta_t * self.eta_charge))
        charge_power = min(max_charge_by_power, max_charge_by_soc) if want_charge else 0.0

        max_discharge_by_power = self.bess_power_limit
        max_discharge_by_soc = max(0.0, soc * self.eta_discharge / self.delta_t)
        discharge_power = min(max_discharge_by_power, max_discharge_by_soc) if want_discharge else 0.0

        # Update SOC for next step
        soc_next = soc + self.eta_charge * charge_power * self.delta_t - (discharge_power / self.eta_discharge) * self.delta_t
        soc_next = min(max(soc_next, 0.0), self.bess_capacity)

        # Map to flows (pure arbitrage: all charging from grid; all discharging to grid)
        grid_to_bess = charge_power
        pv_to_bess = 0.0
        pv_to_ac = 0.0
        bess_to_ac = 0.0
        grid_import = charge_power
        grid_export = discharge_power

        # No local consumption objective here; set allocations to zero
        pv_bess_to_consumer = 0.0
        pv_bess_to_ev = 0.0
        pv_bess_to_grid = grid_export  # for plotting consistency
        grid_to_consumer = 0.0
        grid_to_ev = 0.0

        # Do not account PV in arbitrage baseline (kept zero to isolate strategy)
        pv_gen = 0.0

        return {
            'P_BESS': discharge_power,  # signless kW (use separate charge/discharge fields)
            'SOC_next': soc_next,
            'P_BESS_discharge': discharge_power,
            'P_grid_to_bess': grid_to_bess,
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


