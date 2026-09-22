import pandas as pd

def isolate_recent_year(input_csv, output_csv, target_meter_id=59):
    # Load the CSV using the embedded header row
    df = pd.read_csv(input_csv)
    
    # Isolate the target building/meter
    df_bldg = df[df['meter_id'] == target_meter_id].copy()
    
    if df_bldg.empty:
        print(f"Error: No records found for Meter ID {target_meter_id}.")
        return

    # Convert the timestamp column to datetime objects
    df_bldg['timestamp'] = pd.to_datetime(df_bldg['timestamp'])
    
    # Identify the absolute newest record
    max_date = df_bldg['timestamp'].max()
    
    # Calculate the cutoff boundary exactly one year prior
    cutoff_date = max_date - pd.DateOffset(years=1)
    
    # Filter for only the records occurring within the most recent year
    df_recent = df_bldg[df_bldg['timestamp'] >= cutoff_date]
    
    # Sort chronologically to maintain sequence integrity
    df_recent = df_recent.sort_values('timestamp')
    
    # Export without the dataframe index
    df_recent.to_csv(output_csv, index=False)
    
    print(f"Extraction Complete for Meter ID {target_meter_id}")
    print(f"Date Range: {df_recent['timestamp'].min()} to {df_recent['timestamp'].max()}")
    print(f"Total Rows: {len(df_recent)}")

if __name__ == "__main__":
    # Replace these filenames with your actual container paths
    INPUT_FILE = 'CommFac/building_consumption.csv'
    OUTPUT_FILE = 'CommFac/meter_59_recent_year.csv'
    
    isolate_recent_year(INPUT_FILE, OUTPUT_FILE)