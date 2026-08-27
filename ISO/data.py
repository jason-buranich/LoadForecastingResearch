import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load CAISO and Weather Data
caiso_df = pd.read_excel('ISO/CAISO_2025_combined.xlsx')
weather_df = pd.read_csv('ISO/CAISO_Regional_Weather_2025.csv')

# Ensure Date columns are datetime objects for a clean merge
caiso_df['Date'] = pd.to_datetime(caiso_df['Date'])
weather_df['Date'] = pd.to_datetime(weather_df['Date'])

# Merge on Date and Hour
df = pd.merge(caiso_df, weather_df, on=['Date', 'HR'], how='left')

# Drop messy columns
cols_to_drop = [c for c in df.columns if 'Unnamed' in c or 'DST' in c]
df = df.drop(columns=cols_to_drop).sort_values(by=['Date', 'HR']).reset_index(drop=True)

# 2. Extract Month Feature EARLY
df['Month'] = df['Date'].dt.month

# 3. Fix the missing Weather Data (No Data Leakage)
# Isolate only the weather columns that need to be interpolated
weather_cols = [c for c in df.columns if 'Temp' in c or 'Solar' in c]

# Apply the interpolation strictly to the isolated columns grouped by month
for col in weather_cols:
    df[col] = df.groupby('Month')[col].transform(lambda x: x.interpolate(method='linear').bfill().ffill())

# 4. Extract Remaining Time Features
df['hour_sin'] = np.sin(2 * np.pi * df['HR'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['HR'] / 24)

df['day_of_week'] = df['Date'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# 5. Define the Splits
val_months = [4, 10]
test_months = [8, 12]
train_months = [m for m in range(1, 13) if m not in val_months and m not in test_months]

train_raw = df[df['Month'].isin(train_months)].copy()
val_raw = df[df['Month'].isin(val_months)].copy()
test_raw = df[df['Month'].isin(test_months)].copy()

# 4. Standardize 
regional_weather_cols = [
    'PGE_Temp', 'PGE_Solar', 'SCE_Temp', 'SCE_Solar', 
    'SDGE_Temp', 'SDGE_Solar', 'VEA_Temp', 'VEA_Solar'
]
features_to_scale = ['PGE', 'SCE', 'SDGE', 'VEA', 'CAISO'] + regional_weather_cols
scaler = StandardScaler()

train_raw[features_to_scale] = scaler.fit_transform(train_raw[features_to_scale])
val_raw[features_to_scale] = scaler.transform(val_raw[features_to_scale])
test_raw[features_to_scale] = scaler.transform(test_raw[features_to_scale])

# 5. Drop non-feature columns
columns_to_keep = features_to_scale + ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'Month']
train_df = train_raw[columns_to_keep]
val_df = val_raw[columns_to_keep]
test_df = test_raw[columns_to_keep]