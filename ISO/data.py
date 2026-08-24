import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and Clean
df = pd.read_excel('ISO/CAISO_2025_combined.xlsx')
cols_to_drop = [c for c in df.columns if 'Unnamed' in c or 'DST' in c]
df = df.drop(columns=cols_to_drop).sort_values(by=['Date', 'HR']).reset_index(drop=True)

# Extract Features
df['Month'] = df['Date'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['HR'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['HR'] / 24)

df['day_of_week'] = df['Date'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# 3. Define the Splits
val_months = [4, 10]
test_months = [8, 12]
train_months = [m for m in range(1, 13) if m not in val_months and m not in test_months]

train_raw = df[df['Month'].isin(train_months)].copy()
val_raw = df[df['Month'].isin(val_months)].copy()
test_raw = df[df['Month'].isin(test_months)].copy()

# 4. Standardize (Fit ONLY on train data to prevent leakage)
features_to_scale = ['PGE', 'SCE', 'SDGE', 'VEA', 'CAISO']
scaler = StandardScaler()

train_raw[features_to_scale] = scaler.fit_transform(train_raw[features_to_scale])
val_raw[features_to_scale] = scaler.transform(val_raw[features_to_scale])
test_raw[features_to_scale] = scaler.transform(test_raw[features_to_scale])

# Drop non-feature columns (keep Month for the safe windowing step)
columns_to_keep = features_to_scale + ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'Month']
train_df = train_raw[columns_to_keep]
val_df = val_raw[columns_to_keep]
test_df = test_raw[columns_to_keep]