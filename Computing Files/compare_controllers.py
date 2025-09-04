import os
import numpy as np
import pandas as pd

from Controller.load_data import load_constants
import Controller.load_data as load_data
from Controller.mpc import MPC
from Controller.arbitrage_controller import ArbitrageController
import PostPlot.post_process as post_process
import PostPlot.plots as plots
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def pad_to_horizon(arr, horizon):
    if len(arr) < horizon:
        if len(arr) > 0:
            return np.pad(arr, (0, horizon - len(arr)), mode='edge')
        else:
            return np.zeros(horizon)
    return arr[:horizon]


def run_controller_once(data, controller_type, horizon=672):
    """
    Execute one full 15‑min rolling simulation with the selected controller.
    Returns: results (dict), revenues (dict)
    """
    # Instantiate controller
    if controller_type == 'ARBITRAGE':
        const = load_constants()
        window_hours = float(const.get('ARBI_WINDOW_HOURS', 24.0))
        gap_min = float(const.get('ARBI_GAP_MIN', 0.0))
        alpha_soc_charge = float(const.get('ARBI_ALPHA_SOC_CHARGE', 0.0))
        beta_soc_discharge = float(const.get('ARBI_BETA_SOC_DISCHARGE', 0.0))
        gamma_pv_inflow = float(const.get('ARBI_GAMMA_PV_INFLOW', 0.0))
        hold_steps = int(float(const.get('ARBI_HOLD_STEPS', 0)))
        term_soc_raw = const.get('ARBI_TERMINAL_SOC_RATIO', None)
        try:
            terminal_soc_ratio = None if term_soc_raw is None else float(term_soc_raw)
        except Exception:
            terminal_soc_ratio = None
        ctrl = ArbitrageController(
            delta_t=data['delta_t'],
            bess_capacity=data['bess_capacity'],
            bess_power_limit=data['bess_power_limit'],
            eta_charge=data['eta_charge'],
            eta_discharge=data['eta_discharge'],
            window_hours=window_hours,
            gap_min=gap_min,
            alpha_soc_charge=alpha_soc_charge,
            beta_soc_discharge=beta_soc_discharge,
            gamma_pv_inflow=gamma_pv_inflow,
            hold_steps=hold_steps,
            terminal_soc_ratio=terminal_soc_ratio,
        )
    else:
        ctrl = MPC(delta_t=data['delta_t'])

    n_steps = data['n_steps']
    delta_t = data['delta_t']
    # storage for series
    soc_actual = np.zeros(n_steps + 1)
    soc_actual[0] = data['soc_initial']
    P_PV_consumer_vals = np.zeros(n_steps)
    P_PV_ev_vals = np.zeros(n_steps)
    P_PV_grid_vals = np.zeros(n_steps)
    P_BESS_discharge_vals = np.zeros(n_steps)
    P_BESS_charge_vals = np.zeros(n_steps)
    P_grid_consumer_vals = np.zeros(n_steps)
    P_grid_ev_vals = np.zeros(n_steps)
    P_Grid_to_BESS_vals = np.zeros(n_steps)
    P_grid_import_vals = np.zeros(n_steps)
    P_grid_export_vals = np.zeros(n_steps)
    P_PV_gen = np.zeros(n_steps)

    for t in range(n_steps):
        pv_forecast = pad_to_horizon(data['pv_power'][t:t + horizon], horizon)
        demand_forecast = pad_to_horizon(data['consumer_demand'][t:t + horizon], horizon)
        ev_forecast = pad_to_horizon(data['ev_demand'][t:t + horizon], horizon)
        buy_forecast = pad_to_horizon(data['grid_buy_price'][t:t + horizon], horizon)
        sell_forecast = pad_to_horizon(data['grid_sell_price'][t:t + horizon], horizon)

        # start_dt not used here for plotting; pass something consistent
        current_start_dt = data['start_dt'] + pd.Timedelta(minutes=15*t)
        control = ctrl.predict(
            soc_actual[t], pv_forecast, demand_forecast, ev_forecast,
            buy_forecast, sell_forecast, data['lcoe_pv'], data['pi_ev'], data['pi_consumer'], horizon, current_start_dt
        )

        P_PV_consumer_vals[t] = control['pv_bess_to_consumer']
        P_PV_ev_vals[t] = control['pv_bess_to_ev']
        P_PV_grid_vals[t] = control['pv_bess_to_grid']
        P_BESS_discharge_vals[t] = control['P_BESS_discharge']
        P_BESS_charge_vals[t] = control['P_BESS_charge']
        P_grid_consumer_vals[t] = control['grid_to_consumer']
        P_grid_ev_vals[t] = control['grid_to_ev']
        P_Grid_to_BESS_vals[t] = control['P_grid_to_bess']
        P_grid_import_vals[t] = control['P_grid_import']
        P_grid_export_vals[t] = control['P_grid_export']
        soc_actual[t + 1] = control['SOC_next']
        P_PV_gen[t] = control['P_PV_gen']

    results = {
        'P_PV_consumer_vals': P_PV_consumer_vals,
        'P_PV_ev_vals': P_PV_ev_vals,
        'P_PV_grid_vals': P_PV_grid_vals,
        'P_BESS_discharge': P_BESS_discharge_vals,
        'P_BESS_charge': P_BESS_charge_vals,
        'P_grid_consumer_vals': P_grid_consumer_vals,
        'P_grid_ev_vals': P_grid_ev_vals,
        'P_Grid_to_BESS': P_Grid_to_BESS_vals,
        'P_grid_import_vals': P_grid_import_vals,
        'P_grid_export_vals': P_grid_export_vals,
        'SOC_vals': soc_actual,
        'P_PV_gen': P_PV_gen,
    }

    revenues = post_process.compute_revenues(results, data)
    return results, revenues


def compute_kpis(results, revenues, data):
    dt = data['delta_t']
    # BESS throughput & cycles
    charge_kwh = np.sum(results['P_BESS_charge']) * dt
    discharge_kwh = np.sum(results['P_BESS_discharge']) * dt
    throughput_kwh = charge_kwh + discharge_kwh
    full_cycles = throughput_kwh / (2.0 * data['bess_capacity']) if data['bess_capacity'] > 0 else 0.0

    kpis = {
        'total_revenue': revenues['total_revenue'],
        'grid_sell_revenue': float(np.sum(revenues['pv_to_grid_rev'] + revenues['bess_to_grid_rev'])),
        'grid_buy_cost': float(np.sum(revenues['grid_buy_cost'])),
        'ev_revenue': float(np.sum(revenues['pv_to_ev_rev'] + revenues['bess_to_ev_rev'])),
        'charge_kwh': charge_kwh,
        'discharge_kwh': discharge_kwh,
        'throughput_kwh': throughput_kwh,
        'full_cycles': full_cycles,
        'self_sufficiency': revenues['self_sufficiency'],
        'ev_renewable_share': revenues['ev_renewable_share'],
    }
    return kpis


def plot_comparison(data, res_mpc, rev_mpc, res_arb, rev_arb, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    t = data['time_steps']
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # 1) Cumulative revenue
    cum_mpc = np.cumsum(rev_mpc['total_net_per_step'])
    cum_arb = np.cumsum(rev_arb['total_net_per_step'])
    axes[0].plot(t, cum_mpc, label='MPC cumulative €', color='tab:blue')
    axes[0].plot(t, cum_arb, label='Arbitrage cumulative €', color='tab:orange')
    axes[0].set_title('Cumulative Revenue Comparison')
    axes[0].set_ylabel('€')
    axes[0].grid(True, linestyle=':')
    axes[0].legend()

    # 2) BESS charge/discharge overlay
    axes[1].plot(t, res_mpc['P_BESS_charge'], label='MPC charge (kW)', color='tab:blue', alpha=0.6)
    axes[1].plot(t, res_mpc['P_BESS_discharge'], label='MPC discharge (kW)', color='tab:blue', linestyle='--', alpha=0.6)
    axes[1].plot(t, res_arb['P_BESS_charge'], label='ARB charge (kW)', color='tab:orange', alpha=0.6)
    axes[1].plot(t, res_arb['P_BESS_discharge'], label='ARB discharge (kW)', color='tab:orange', linestyle='--', alpha=0.6)
    axes[1].set_title('BESS Power')
    axes[1].set_ylabel('kW')
    axes[1].grid(True, linestyle=':')
    axes[1].legend(ncol=2)

    # 3) Grid import/export overlay
    axes[2].plot(t, res_mpc['P_grid_import_vals'], label='MPC grid import', color='tab:green', alpha=0.6)
    axes[2].plot(t, res_mpc['P_grid_export_vals'], label='MPC grid export', color='tab:red', alpha=0.6)
    axes[2].plot(t, res_arb['P_grid_import_vals'], label='ARB grid import', color='tab:green', linestyle='--', alpha=0.6)
    axes[2].plot(t, res_arb['P_grid_export_vals'], label='ARB grid export', color='tab:red', linestyle='--', alpha=0.6)
    axes[2].set_title('Grid Import/Export')
    axes[2].set_ylabel('kW')
    axes[2].grid(True, linestyle=':')
    axes[2].legend(ncol=2)

    # 4) Price with total buy/sell shading (overall import/export across the site)
    ax = axes[3]
    buy_price = np.asarray(data['grid_buy_price'])
    sell_price = np.asarray(data['grid_sell_price'])
    ax.plot(t, buy_price, label='Grid Buy Price (€/kWh)', color='blue', linewidth=2, zorder=3)
    ax.plot(t, sell_price, label='Grid Sell Price (€/kWh)', color='lightblue', linestyle='--', linewidth=2, zorder=3)

    # Use link-based totals for overall site import/export
    imp_mpc = np.asarray(res_mpc['P_grid_import_vals'])
    exp_mpc = np.asarray(res_mpc['P_grid_export_vals'])
    imp_arb = np.asarray(res_arb['P_grid_import_vals'])
    exp_arb = np.asarray(res_arb['P_grid_export_vals'])
    tol = 1e-6
    # Normalize alpha by max magnitude across both controllers
    max_imp = float(max(1e-9, np.max([np.max(imp_mpc), np.max(imp_arb)])))
    max_exp = float(max(1e-9, np.max([np.max(exp_mpc), np.max(exp_arb)])))
    dt = float(data.get('delta_t', 0.25))

    # Shade for MPC (darker hues)
    for i in range(len(t)):
        x0 = t[i]
        x1 = x0 + dt
        if imp_mpc[i] > tol:
            alpha = 0.4 * (imp_mpc[i] / max_imp)
            ax.axvspan(x0, x1, color='forestgreen', alpha=alpha, linewidth=0, zorder=0)
        elif exp_mpc[i] > tol:
            alpha = 0.4 * (exp_mpc[i] / max_exp)
            ax.axvspan(x0, x1, color='firebrick', alpha=alpha, linewidth=0, zorder=0)

    # Shade for ARBITRAGE (lighter hues)
    for i in range(len(t)):
        x0 = t[i]
        x1 = x0 + dt
        if imp_arb[i] > tol:
            alpha = 0.3 * (imp_arb[i] / max_imp)
            ax.axvspan(x0, x1, color='limegreen', alpha=alpha, linewidth=0, zorder=1)
        elif exp_arb[i] > tol:
            alpha = 0.3 * (exp_arb[i] / max_exp)
            ax.axvspan(x0, x1, color='salmon', alpha=alpha, linewidth=0, zorder=1)

    ax.set_title('Market Price with Total Grid Buy/Sell Shading (Intensity ∝ Power)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('€/kWh')
    ax.grid(True, linestyle=':')
    # Legend patches explaining shading
    shade_legend = [
        Patch(facecolor='forestgreen', alpha=0.4, label='MPC: Import (darker=more kW)'),
        Patch(facecolor='firebrick', alpha=0.4, label='MPC: Export (darker=more kW)'),
        Patch(facecolor='limegreen', alpha=0.3, label='ARB: Import (darker=more kW)'),
        Patch(facecolor='salmon', alpha=0.3, label='ARB: Export (darker=more kW)'),
    ]
    price_lines = ax.legend(loc='upper right', fontsize=9)
    ax.add_artist(price_lines)
    ax.legend(handles=shade_legend, loc='upper left', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'Comparison_MPC_vs_Arbitrage.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {out_path}")


def compare_run():
    # Load identical inputs once
    data = load_data.load(reference_case=False, price_source=str(load_constants().get('PRICE_SOURCE', 'HPFC')),
                          base_forecast=100.0, peak_forecast=120.0)

    # Run MPC
    res_mpc, rev_mpc = run_controller_once(data, 'MPC')
    # Run ARBITRAGE
    res_arb, rev_arb = run_controller_once(data, 'ARBITRAGE')

    # KPIs
    kpi_mpc = compute_kpis(res_mpc, rev_mpc, data)
    kpi_arb = compute_kpis(res_arb, rev_arb, data)
    print("\n=== KPI Comparison (MPC vs ARBITRAGE) ===")
    for k in ['total_revenue','grid_sell_revenue','grid_buy_cost','ev_revenue','throughput_kwh','full_cycles','self_sufficiency','ev_renewable_share']:
        print(f"{k}: MPC={kpi_mpc[k]:.3f}  ARB={kpi_arb[k]:.3f}")

    # Plot overlay comparison
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output Files')
    plot_comparison(data, res_mpc, rev_mpc, res_arb, rev_arb, base_dir)


if __name__ == '__main__':
    compare_run()


