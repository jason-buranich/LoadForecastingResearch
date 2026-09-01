import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_substation_data(filepath):
    """Loads, melts, and imputes Ausgrid substation data into a continuous 15-min time series."""
    df = pd.read_csv(filepath)
    
    # 1. Melt wide to long
    id_vars = ['year', 'Zone Substation', 'Date', 'Unit']
    df_long = df.melt(id_vars=id_vars, var_name='Interval', value_name='Load_MW')
    
    # 2. Build continuous Datetime index
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long['TimeDelta'] = pd.to_timedelta(df_long['Interval'] + ':00')
    df_long['Datetime'] = df_long['Date'] + df_long['TimeDelta']
    
    # Clean up and sort
    df_long = df_long[['Datetime', 'Zone Substation', 'Load_MW']].sort_values('Datetime')
    df_long = df_long.set_index('Datetime')
    
    # 3. Reindex to strict 15-minute intervals to expose missing days
    full_idx = pd.date_range(start=df_long.index.min(), end=df_long.index.max(), freq='15min')
    df_long = df_long.reindex(full_idx)
    
    # Forward-fill categorical metadata
    df_long['Zone Substation'] = df_long['Zone Substation'].ffill()
    
    # 4. Impute missing MW using 7-day cyclical shift (672 intervals), then linear interpolate
    df_long['Load_MW'] = df_long['Load_MW'].fillna(df_long['Load_MW'].shift(672))
    df_long['Load_MW'] = df_long['Load_MW'].interpolate(method='linear')
    
    # 5. Generate temporal covariates
    df_long['Hour'] = df_long.index.hour
    df_long['DayOfWeek'] = df_long.index.dayofweek
    df_long['Month'] = df_long.index.month
    
    return df_long

# --- Pipeline Execution ---

target_file = "Substation/Punchbowl 33_11kV FY25.csv"
weather_file = "Substation/punchbowl_weather_15min.csv"

# Process Substation and Weather Data
substation_df = process_substation_data(target_file)
weather_df = pd.read_csv(weather_file, index_col='Datetime', parse_dates=True)

# Merge on Datetime index (left join ensures we strictly keep the substation's timeline)
master_df = substation_df.join(weather_df, how='left')

# Impute any edge-case missing weather values using forward fill
master_df[['Temperature_2m', 'Humidity', 'Solar_Rad']] = master_df[['Temperature_2m', 'Humidity', 'Solar_Rad']].ffill().bfill()

# Extract covariates and target
# Index 0: Month, Index 1: Load_MW (Target), Index 2: Hour, Index 3: DayOfWeek
# Index 4: Temperature_2m, Index 5: Humidity, Index 6: Solar_Rad
features = master_df[['Month', 'Load_MW', 'Hour', 'DayOfWeek', 'Temperature_2m', 'Humidity', 'Solar_Rad']].copy()

# Month-Based Split for specific seasonal evaluation
test_months = [8, 12]  # August and December
val_months = [4, 10]   # April and October

test_mask = features['Month'].isin(test_months)
val_mask = features['Month'].isin(val_months)

train_data = features[~test_mask & ~val_mask].copy()
val_data = features[val_mask].copy()
test_data = features[test_mask].copy()

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

print(f"Data Pipeline Initialized. Target Substation: {target_file}")
print(f"Train shape: {train_df.shape} | Val shape: {val_df.shape} | Test shape: {test_df.shape}")