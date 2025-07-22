import pandas as pd
import numpy as np

WEEK NUMBER FETCH

def compute_pv_power(week_number=1):
    path = r'C:\Users\dell\V1_First_Model\Input Data Files\pv_FdM_2024.csv'
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S')
    df.set_index('Time', inplace=True)
    cols = ['InverterYield', 'GridFeedIn', 'PurchasedFromNet']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df_15min = df[cols].resample('15min').interpolate(method='linear')
    df_15min['consumer_demand'] = df_15min['InverterYield'] + df_15min['PurchasedFromNet'] - df_15min['GridFeedIn']
    start_time = df_15min.index[0] + pd.Timedelta(days=(week_number - 1) * 7)
    end_time = start_time + pd.Timedelta(hours=167, minutes=45)
    df_week = df_15min.loc[start_time:end_time]
    pv_production = df_week['InverterYield'].values
    consumer_demand = df_week['consumer_demand'].values
    result = {
        'pv_production': pv_production,
        'consumer_demand': consumer_demand
    }
    print("Results for Week", week_number, "of 2024:")
    print("Time Range:", start_time.strftime('%Y-%m-%d %H:%M:%S'), "to", end_time.strftime('%Y-%m-%d %H:%M:%S'))
    print("\nPV Production (15-min intervals):")
    for i, value in enumerate(pv_production):
        time = (start_time + pd.Timedelta(minutes=15 * i)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{time}: {value:.2f} kWh")
    print("\nConsumer Demand (15-min intervals):")
    for i, value in enumerate(consumer_demand):
        time = (start_time + pd.Timedelta(minutes=15 * i)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{time}: {value:.2f} kWh")
    return result

if __name__ == "__main__":
    compute_pv_power()