import pandas as pd
import os

def compute_pv_power(week_number):
    # Cross-platform, dynamic path to data file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'Input Data Files', 'pv_FdM_2024.csv')

    df = pd.read_csv(DATA_PATH)
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S')
    df.set_index('Time', inplace=True)

    cols = ['InverterYield', 'GridFeedIn', 'PurchasedFromNet']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Resample to 15min intervals: distribute hourly energy equally over 4 intervals
    df_15min = df[cols].resample('15min').ffill() / 4

    # Calculate consumer demand in kWh per 15min
    df_15min['consumer_demand'] = df_15min['InverterYield'] + df_15min['PurchasedFromNet'] - df_15min['GridFeedIn']

    start_time = df_15min.index[0] + pd.Timedelta(days=(week_number - 1) * 7)
    end_time = start_time + pd.Timedelta(hours=167, minutes=45)
    df_week = df_15min.loc[start_time:end_time]

    if len(df_week) != 672:
        raise ValueError(f"Data slice for week {week_number} has {len(df_week)} entries, expected 672.")

    pv_production = df_week['InverterYield'].values
    consumer_demand = df_week['consumer_demand'].values

    print(f"\n--- PV Production and Consumer Demand for Week {week_number} ---")
    for i in range(len(pv_production)):
        timestamp = (start_time + pd.Timedelta(minutes=15 * i)).strftime('%Y-%m-%d %H:%M')
        print(f"{timestamp} | PV: {pv_production[i]:.2f} kWh | Demand: {consumer_demand[i]:.2f} kWh")

    return {
        'pv_production': pv_production,
        'consumer_demand': consumer_demand
    }

# Run the function directly when this script is executed
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONSTANTS_PATH = os.path.join(BASE_DIR, 'Input Data Files', 'Constants_Plant.csv')
    constants_df = pd.read_csv(CONSTANTS_PATH)
    week_number = int(constants_df[constants_df['Parameter'] == 'WEEK_NUMBER']['Value'].iloc[0])
    compute_pv_power(week_number)
