import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_austin_data(filepath, target_dataid=None):
    """Loads, imputes missing meters independently, and aggregates Austin microgrid data."""
    df = pd.read_csv(filepath)
    
    time_col = 'local_15min' if 'local_15min' in df.columns else df.columns[1]
    target_col = 'grid' if 'grid' in df.columns else 'Load_kW'
    
    if target_dataid is not None and 'dataid' in df.columns:
        df = df[df['dataid'] == target_dataid]
        
    df = df.rename(columns={time_col: 'Datetime', target_col: 'Load_kW'})
    
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df['Datetime'] = df['Datetime'].dt.tz_convert('America/Chicago').dt.tz_localize(None)
    
    # Pivot so each house/dataid is its own column
    df_pivot = df.pivot_table(index='Datetime', columns='dataid', values='Load_kW')
    
    # NEW: Drop any house missing more than 5% of its data to ensure a stable microgrid baseline
    threshold = int(len(df_pivot) * 0.95)
    df_pivot = df_pivot.dropna(thresh=threshold, axis=1)
    
    # Reindex to strict 15-minute intervals across the entire timeline
    full_idx = pd.date_range(start=df_pivot.index.min(), end=df_pivot.index.max(), freq='15min')
    df_pivot = df_pivot.reindex(full_idx)
    
    # Impute missing intervals for EACH stable house using 7-day cyclical shift + linear interpolation
    df_pivot = df_pivot.fillna(df_pivot.shift(672))
    df_pivot = df_pivot.interpolate(method='linear')
    
    # Sum across all stable houses to create the aggregate microgrid load
    df_agg = pd.DataFrame(index=df_pivot.index)
    df_agg['Load_kW'] = df_pivot.sum(axis=1)

    # Generate temporal covariates
    df_agg['Hour'] = df_agg.index.hour
    df_agg['DayOfWeek'] = df_agg.index.dayofweek
    df_agg['Month'] = df_agg.index.month
    
    return df_agg

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
# Index 0: Month, Index 1: Load_kW (Target), Index 2: Hour, Index 3: DayOfWeek
# Index 4: Temperature_2m, Index 5: Humidity, Index 6: Solar_Rad
features = master_df[['Month', 'Load_kW', 'Hour', 'DayOfWeek', 'Temperature_2m', 'Humidity', 'Solar_Rad']].copy()

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