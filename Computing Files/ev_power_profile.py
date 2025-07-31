import numpy as np
from datetime import datetime, timedelta
import random
import pandas as pd
import os

def load_constants():
    # Cross-platform, dynamic path to constants file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONSTANTS_PATH = os.path.join(BASE_DIR, 'Input Data Files', 'Constants_Plant.csv')
    constants_df = pd.read_csv(CONSTANTS_PATH, comment='#')
    return constants_df

def generate_ev_charging_profile(start_time, end_time, num_sessions_per_day=0, session_energy=50, charger_max_power=400, load_scale=1.0):
    """
    Generates EV charging demand profile over the period.
    
    Parameters:
    - start_time: datetime - Start of the period
    - end_time: datetime - End of the period (exclusive last interval)
    - num_sessions_per_day: int - Average number of charging sessions per day (0 for no charging)
    - session_energy: float - Average energy per session in kWh (e.g., 50kWh)
    - charger_max_power: float - Max power of the station in kW (400kW total for 2 outlets)
    - load_scale: float - Scaling factor for energy load (1.0 = baseline, 0.0 = no load, >1 for increased)
    
    Returns:
    dict with 'ev_demand': array of kWh per 15-min interval
    """
    if load_scale <= 0 or num_sessions_per_day <= 0:
        # No charging scenario: return zeros for all intervals
        num_intervals = int((end_time - start_time).total_seconds() / (15 * 60))
        return {'ev_demand': np.zeros(num_intervals)}
    
    # Scale the number of sessions and energy (but cap power per session)
    effective_sessions_per_day = int(num_sessions_per_day * load_scale)
    effective_session_energy = session_energy * load_scale
    
    # Time setup: 15-min intervals
    delta_t_min = 15
    current_time = start_time
    ev_demand = []
    
    # Period probabilities and avg loads (as % of max power, scaled indirectly via sessions/energy)
    periods = [
        (0, 6, 0.15, (20, 100)),   # 00-06h: low load
        (6, 12, 0.35, (50, 200)), # 06-12h: higher
        (12, 18, 0.35, (50, 200)), # 12-18h: medium
        (18, 24, 0.15, (20, 100))  # 18-24h: medium-low
    ]
    
    # Total probability for distributing sessions
    total_prob = sum(p[2] for p in periods)
    
    # Simulate day by day
    days = (end_time - start_time).days
    for day in range(days):
        daily_demand = np.zeros(96)  # 24h * 4 intervals/h
        
        # Distribute sessions across periods
        sessions_per_period = [int(effective_sessions_per_day * p[2] / total_prob) for p in periods]
        
        for i, (start_h, end_h, _, (min_avg_kw, max_avg_kw)) in enumerate(periods):
            num_sessions = sessions_per_period[i]
            if num_sessions == 0:
                continue
            
            # Period duration in minutes
            period_min = (end_h - start_h) * 60
            interval_start = start_h * 4
            interval_end = end_h * 4
            
            for _ in range(num_sessions):
                # Random session start within period (in minutes)
                session_start_min = random.randint(0, period_min - 30)  # Assume min 30min session
                session_duration_min = random.randint(30, 90)  # 30-90 min
                # Power limited by charger max per outlet (assume 2 outlets, so per session max 200kW)
                session_power = min(charger_max_power / 2, effective_session_energy / (session_duration_min / 60))  # Fit energy, cap at 200kW
                
                # Add to 15-min intervals (kWh = power * time_fraction)
                start_interval = interval_start + (session_start_min // delta_t_min)
                num_intervals = (session_duration_min + delta_t_min - 1) // delta_t_min  # Ceiling
                
                for j in range(num_intervals):
                    interval_idx = start_interval + j
                    if interval_idx >= interval_end:
                        break
                    # Fraction of interval covered by session
                    interval_start_time_min = session_start_min + j * delta_t_min
                    interval_end_time_min = min(session_start_min + session_duration_min, interval_start_time_min + delta_t_min)
                    fraction = (interval_end_time_min - max(session_start_min, interval_start_time_min)) / delta_t_min
                    daily_demand[interval_idx] += session_power * fraction * (delta_t_min / 60)  # kWh in interval
        
        ev_demand.extend(daily_demand)
    
    # Ensure exact length (672 for 7 days)
    ev_demand = np.array(ev_demand[:672])
    
    return {'ev_demand': ev_demand}

# For testing/standalone run (similar to pv_consumer_data)
if __name__ == "__main__":
    constants_df = load_constants()
    period_start_str = constants_df[constants_df['Parameter'] == 'PERIOD_START']['Value'].iloc[0]
    period_end_str = constants_df[constants_df['Parameter'] == 'PERIOD_END']['Value'].iloc[0]
    start_dt = datetime.strptime(period_start_str, '%Y%m%d%H%M')
    end_dt = datetime.strptime(period_end_str, '%Y%m%d%H%M')
    
    # Load tuning parameters
    load_scale = float(constants_df[constants_df['Parameter'] == 'EV_LOAD_SCALE']['Value'].iloc[0])
    num_sessions_per_day = int(constants_df[constants_df['Parameter'] == 'EV_NUM_SESSIONS_PER_DAY']['Value'].iloc[0])
    session_energy = float(constants_df[constants_df['Parameter'] == 'EV_SESSION_ENERGY']['Value'].iloc[0])
    charger_max_power = float(constants_df[constants_df['Parameter'] == 'EV_CHARGER_MAX_POWER']['Value'].iloc[0])
    
    profile = generate_ev_charging_profile(start_dt, end_dt, num_sessions_per_day, session_energy, charger_max_power, load_scale)
    print(profile['ev_demand'])