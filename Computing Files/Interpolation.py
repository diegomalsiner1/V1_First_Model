import pandas as pd
import numpy as np

def interpolate_to_15min(input_file, output_file, time_col='Time', value_cols=None):
    """
    Interpolate hourly data to 15-minute intervals.
    
    Parameters:
    - input_file (str): Path to input CSV with hourly data.
    - output_file (str): Path to save interpolated CSV.
    - time_col (str): Column name for time (default: 'Time').
    - value_cols (list): List of column names to interpolate (default: all numeric columns).
    """
    # Read input data
    df = pd.read_csv(input_file)
    
    # Ensure time is numeric
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    
    # Set time as index
    df.set_index(time_col, inplace=True)
    
    # If no specific columns provided, use all numeric columns
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create new time index with 15-minute intervals
    new_time = np.arange(0, 24, 0.25)  # 0 to 23.75 hours
    
    # Interpolate
    df_interpolated = df[value_cols].reindex(df.index.union(new_time)).interpolate(method='linear')
    df_interpolated = df_interpolated.loc[new_time].reset_index()
    df_interpolated.rename(columns={'index': time_col}, inplace=True)
    
    # Save to CSV
    df_interpolated.to_csv(output_file, index=False)
    print(f"Interpolated data saved to {output_file}")

# Example usage (uncomment and adjust paths when using with actual files)
# interpolate_to_15min('Input Data Files/PV_Energy_Profile_hourly.csv',
#                      'Input Data Files/PV_Energy_Profile.csv',
#                      value_cols=['PV_Power'])
# interpolate_to_15min('Input Data Files/Market_Price_hourly.csv',
#                      'Input Data Files/Market_Price.csv',
#                      value_cols=['Grid_Buy_Price', 'Grid_Sell_Price'])
