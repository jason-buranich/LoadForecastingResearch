import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_austin_data(filepath, target_dataid=None):
    """Loads, aggregates duplicate timestamps/meters, and imputes Austin microgrid data."""
    df = pd.read_csv(filepath)
    
    # 1. Identify Columns
    time_col = 'local_15min' if 'local_15min' in df.columns else df.columns[1]
    target_col = 'grid' if 'grid' in df.columns else 'Load_MW'
    
    # Optional: Filter for a single house if you do not want the community-aggregate microgrid
    if target_dataid is not None and 'dataid' in df.columns:
        df = df[df['dataid'] == target_dataid]
        
    df = df.rename(columns={time_col: 'Datetime', target_col: 'Load_MW'})
    
    # 2. Harmonize Timezones (UTC -> Central -> Naive)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df['Datetime'] = df['Datetime'].dt.tz_convert('America/Chicago').dt.tz_localize(None)
    
    # 3. Collapse duplicate timestamps by summing across meters/houses
    df = df.groupby('Datetime')['Load_MW'].sum().to_frame()
    df = df.sort_index()
    
    # (Optional) Unit conversion: Pecan Street 'grid' is in kW. 
    # If you want true MW, uncomment the line below:
    # df['Load_MW'] = df['Load_MW'] / 1000.0
    
    # 4. Reindex to strict 15-minute intervals across the timeline
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
    df = df.reindex(full_idx)
    
    # 5. Impute missing intervals using 7-day cyclical shift (672 steps) + linear interpolation
    df['Load_MW'] = df['Load_MW'].fillna(df['Load_MW'].shift(672))
    df['Load_MW'] = df['Load_MW'].interpolate(method='linear')
    
    # 6. Generate temporal covariates
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    return df

# --- Pipeline Execution ---

target_file = "Microgrid/15minute_data_austin.csv"
weather_file = "Microgrid/austin_weather_15min.csv"

# Process Microgrid and Weather Data
austin_df = process_austin_data(target_file)
weather_df = pd.read_csv(weather_file, index_col='Datetime', parse_dates=True)

# Merge on Datetime index (left join ensures we strictly keep the microgrid's timeline)
master_df = austin_df.join(weather_df, how='left')

# Impute using ONLY forward fill on the unbroken timeline (causally safe)
master_df[['Temperature_2m', 'Humidity', 'Solar_Rad']] = master_df[['Temperature_2m', 'Humidity', 'Solar_Rad']].ffill()

# Extract covariates and target
# Index 0: Month, Index 1: Load_MW (Target), Index 2: Hour, Index 3: DayOfWeek
# Index 4: Temperature_2m, Index 5: Humidity, Index 6: Solar_Rad
features = master_df[['Month', 'Load_MW', 'Hour', 'DayOfWeek', 'Temperature_2m', 'Humidity', 'Solar_Rad']].copy()

# Month-Based Split for specific seasonal evaluation
# Note: Ensure these months align with the boundaries of your specific 2023-2024 dataset
test_months = [8, 12]  # August and December
val_months = [4, 10]   # April and October

test_mask = features['Month'].isin(test_months)
val_mask = features['Month'].isin(val_months)

train_data = features[~test_mask & ~val_mask].copy()
val_data = features[val_mask].copy()
test_data = features[test_mask].copy()

# Catch any missing values at the very start of the dataset strictly within the train set to prevent leakage
train_data[['Temperature_2m', 'Humidity', 'Solar_Rad']] = train_data[['Temperature_2m', 'Humidity', 'Solar_Rad']].bfill()

# Fit the Scaler STRICTLY on the training set to prevent data leakage
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Convert back to DataFrames for slidingWindow.py
feature_cols = features.columns
train_df = pd.DataFrame(train_scaled, columns=feature_cols, index=train_data.index)
val_df = pd.DataFrame(val_scaled, columns=feature_cols, index=val_data.index)
test_df = pd.DataFrame(test_scaled, columns=feature_cols, index=test_data.index)

print(f"Data Pipeline Initialized. Target Microgrid: {target_file}")
print(f"Train shape: {train_df.shape} | Val shape: {val_df.shape} | Test shape: {test_df.shape}")