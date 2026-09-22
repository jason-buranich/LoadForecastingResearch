import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_latrobe_data(filepath):
    """Loads and standardizes the La Trobe Building 59 dataset."""
    df = pd.read_csv(filepath)
    
    # Format columns based on bldg_59.csv structure
    df = df.rename(columns={'timestamp': 'Datetime', 'consumption': 'Load_kW'})
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Since it's a single building, drop the static ID columns
    df = df[['Datetime', 'Load_kW']].copy()
    df = df.set_index('Datetime')
    
    # Reindex to strict 15-minute intervals across the entire timeline
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
    df = df.reindex(full_idx)
    
    # Impute missing intervals using linear interpolation
    df['Load_kW'] = df['Load_kW'].interpolate(method='linear')
    
    # Generate temporal covariates
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    return df

# --- Pipeline Execution ---

target_file = "CommFac/bldg_59.csv"
weather_file = "CommFac/latrobe_weather_15min.csv"

# Process Building Data
bldg_df = process_latrobe_data(target_file)

# Load Weather Data
weather_df = pd.read_csv(weather_file, index_col='Datetime', parse_dates=True)

# Merge on Datetime index (left join ensures we strictly keep the building's timeline)
master_df = bldg_df.join(weather_df, how='left')

# Impute using forward fill on the unbroken timeline (causally safe)
master_df[['Temperature_2m', 'Humidity', 'Solar_Rad']] = master_df[['Temperature_2m', 'Humidity', 'Solar_Rad']].ffill()

# Extract covariates and target
# Index 0: Month, Index 1: Load_kW (Target), Index 2: Hour, Index 3: DayOfWeek
# Index 4: Temperature_2m, Index 5: Humidity, Index 6: Solar_Rad
features = master_df[['Month', 'Load_kW', 'Hour', 'DayOfWeek', 'Temperature_2m', 'Humidity', 'Solar_Rad']].copy()

# Month-Based Split for specific seasonal evaluation
test_months = [8, 12]  # August and December
val_months = [4, 10]   # April and October

test_mask = features['Month'].isin(test_months)
val_mask = features['Month'].isin(val_months)

train_data = features[~test_mask & ~val_mask].copy()
val_data = features[val_mask].copy()
test_data = features[test_mask].copy()

# Catch any missing values at the very start of the dataset strictly within the train set to prevent leakage
train_data[['Temperature_2m', 'Humidity', 'Solar_Rad']] = train_data[['Temperature_2m', 'Humidity', 'Solar_Rad']].bfill()

# Fit the Scaler STRICTLY on the training set
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Convert back to DataFrames for slidingWindow.py
feature_cols = features.columns
train_df = pd.DataFrame(train_scaled, columns=feature_cols, index=train_data.index)
val_df = pd.DataFrame(val_scaled, columns=feature_cols, index=val_data.index)
test_df = pd.DataFrame(test_scaled, columns=feature_cols, index=test_data.index)

print(f"Data Pipeline Initialized. Target Building: {target_file}")
print(f"Train shape: {train_df.shape} | Val shape: {val_df.shape} | Test shape: {test_df.shape}")