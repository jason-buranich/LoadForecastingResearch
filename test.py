import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import meteostat as ms
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

print(f"GPU Available: {torch.cuda.is_available()}")

# ==========================================
# STEP 1: Problem Definition & Dataset Assembly
# ==========================================
def load_and_assemble_data(filepath="15minute_data_austin.csv"):
    print("Step 1: Assembling Dataset...")
    
    # Load raw telemetry
    df = pd.read_csv(filepath, parse_dates=['local_15min'])
    df['local_15min'] = pd.to_datetime(df['local_15min'], utc=True)
    
    # Impute missing solar
    if 'solar' in df.columns:
        df['solar'] = df['solar'].fillna(0)
    
    # Aggregate VPP
    vpp_df = df.groupby('local_15min')['grid'].sum().reset_index()
    vpp_df.set_index('local_15min', inplace=True)
    
    # Weather Fetching with Fallback
    try:
        start = vpp_df.index.min().replace(tzinfo=None)
        end = vpp_df.index.max().replace(tzinfo=None)
        
        # 1. Define Austin location
        austin_point = ms.Point(30.2672, -97.7431)
        
        # 2. Fetch the top 5 closest stations
        nearby_stations = ms.stations.nearby(austin_point, limit=5)
        
        weather_data = None
        
        # 3. Loop through backup stations
        for station_id in nearby_stations.index:
            print(f"  -> Attempting weather download from Station ID {station_id}...")
            
            # Query the specific station directly
            weather_query = ms.hourly(station_id, start, end)
            temp_data = weather_query.fetch()
            
            if temp_data is not None and not temp_data.empty:
                weather_data = temp_data
                print(f"  -> Success! Station {station_id} has our data.")
                break 
                
        if weather_data is not None and not weather_data.empty:
            # Flatten index if Meteostat returns a MultiIndex
            if isinstance(weather_data.index, pd.MultiIndex):
                weather_data = weather_data.reset_index().set_index('time')
            
            weather = weather_data[['temp', 'rhum']] 
            weather_15m = weather.resample('15min').ffill()
            weather_15m.index = weather_15m.index.tz_localize('UTC')
            
            final_df = vpp_df.join(weather_15m, how='inner')
            print("  -> Weather data successfully merged!")
            return final_df.dropna()
        else:
            raise ValueError("All 5 nearby stations returned empty dataframes.")
            
    except Exception as e:
        print(f"  -> WARNING: Weather API failed ({e}).")
        print("  -> Bypassing weather. Training purely on historical load and time signals.")
        return vpp_df.dropna()
    
# ==========================================
# STEP 2 & 3: Measures of Success & Evaluation Protocol
# ==========================================
# Success metrics (RMSE, MAE) are calculated using sklearn during evaluation.
# Protocol: Chronological Split (Train: Jan-Oct, Valid: Nov, Test: Dec)

#def chronological_split(df):
#    print("Step 3: Executing Chronological Split...")
#    train_df = df[df.index.month <= 10]
#    valid_df = df[df.index.month == 11]
#    test_df  = df[df.index.month == 12]
#    return train_df, valid_df, test_df

def seasonal_block_split(df):
    print("Step 2 & 3: Executing Seasonal Block Split...")
        
    # Ensure our index is a datetime object so we can extract the month
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        
    # Define the seasonal blocks by month number
    test_months = [8, 12]   # August, December
    valid_months = [4, 10]  # April, October
    train_months = [1, 2, 3, 5, 6, 7, 9, 11] # Remaining 8 months
    
    # Slice the dataframe using pandas month masking
    train_df = df[df.index.month.isin(train_months)].copy()
    valid_df = df[df.index.month.isin(valid_months)].copy()
    test_df = df[df.index.month.isin(test_months)].copy()
    
    # Sort chronologically just to be absolutely safe
    train_df.sort_index(inplace=True)
    valid_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)
    
    print(f"  -> Train Blocks (8 months): {len(train_df)} intervals")
    print(f"  -> Valid Blocks (2 months): {len(valid_df)} intervals")
    print(f"  -> Test Blocks  (2 months): {len(test_df)} intervals")
    
    return train_df, valid_df, test_df

# ==========================================
# STEP 4: Data Preparation
# ==========================================
def prepare_data(train_df, valid_df, test_df, lookback=96):
    print("Step 4: Preparing Data (Scaling & Sliding Windows)...")
    
    # Engineer Clock-based Features
    for dataset in [train_df, valid_df, test_df]:
        dataset['hour_sin'] = np.sin(2 * np.pi * dataset.index.hour / 24)
        dataset['hour_cos'] = np.cos(2 * np.pi * dataset.index.hour / 24)
        dataset['day_sin'] = np.sin(2 * np.pi * dataset.index.dayofweek / 7)
        dataset['day_cos'] = np.cos(2 * np.pi * dataset.index.dayofweek / 7)

    # Scale data [0, 1] - Fit ONLY on training data to prevent leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    valid_scaled = scaler.transform(valid_df)
    test_scaled  = scaler.transform(test_df)

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length, 0]) # Index 0 is 'grid' (target)
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_valid, y_valid = create_sequences(valid_scaled, lookback)
    X_test, y_test   = create_sequences(test_scaled, lookback)

    # Convert to PyTorch DataLoaders 
    batch_size = 32
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid)), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)), batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, scaler

# ==========================================
# STEP 5: Beating Baselines
# ==========================================
def evaluate_persistence_baseline(test_df):
    print("\n" + "="*55)
    print(" BASELINE PERSISTENCE METRICS (Unseen Test Data)")
    print("="*55)
    
    # Extract the target variable (assuming 'grid' is the first column)
    actuals = test_df.iloc[:, 0]
    
    # 1. The 15-Minute Baseline (Shift by 1 step)
    df_15m = pd.DataFrame({'actual': actuals, 'pred': actuals.shift(1)}).dropna()
    rmse_15m = np.sqrt(mean_squared_error(df_15m['actual'], df_15m['pred']))
    mae_15m = mean_absolute_error(df_15m['actual'], df_15m['pred'])
    
    # 2. The 24-Hour Baseline (Shift by 96 steps)
    df_24h = pd.DataFrame({'actual': actuals, 'pred': actuals.shift(96)}).dropna()
    rmse_24h = np.sqrt(mean_squared_error(df_24h['actual'], df_24h['pred']))
    mae_24h = mean_absolute_error(df_24h['actual'], df_24h['pred'])
    
    print(f" 1-Step  (15-Min) Baseline  -> RMSE: {rmse_15m:.2f} kW | MAE: {mae_15m:.2f} kW")
    print(f" 96-Step (24-Hour) Baseline -> RMSE: {rmse_24h:.2f} kW | MAE: {mae_24h:.2f} kW")
    print("="*55 + "\n")

def evaluate_random_forest_baseline(train_loader, test_loader, scaler):
    from sklearn.ensemble import RandomForestRegressor    
    
    print("\n" + "="*55)
    print(" TRAINING RANDOM FOREST BASELINE (CPU)...")
    print("="*55)
    
    # 1. Extract and flatten training data from the PyTorch loader
    X_train_list, y_train_list = [], []
    for X_batch, y_batch in train_loader:
        # Flatten the (batch, 96, features) tensor into a 2D array for Scikit-Learn
        X_train_list.append(X_batch.numpy().reshape(X_batch.shape[0], -1))
        y_train_list.append(y_batch.numpy())
        
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list).squeeze()
    
    # 2. Extract and flatten test data
    X_test_list, y_test_list = [], []
    for X_batch, y_batch in test_loader:
        X_test_list.append(X_batch.numpy().reshape(X_batch.shape[0], -1))
        y_test_list.append(y_batch.numpy())
        
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list).squeeze()
    
    # 3. Train the Random Forest 
    # n_jobs=-1 tells it to use every single core on your CPU for maximum speed
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 4. Predict on the unseen test set
    preds_scaled = rf.predict(X_test)
    
    # 5. The Dummy Array Trick for Un-scaling
    input_features = scaler.n_features_in_
    dummy_preds = np.zeros((len(preds_scaled), input_features))
    dummy_actuals = np.zeros((len(y_test), input_features))
    
    dummy_preds[:, 0] = preds_scaled
    dummy_actuals[:, 0] = y_test
    
    rf_preds_kw = scaler.inverse_transform(dummy_preds)[:, 0]
    actuals_kw = scaler.inverse_transform(dummy_actuals)[:, 0]
    
    # 6. Calculate Final Real-World Metrics
    rf_rmse = np.sqrt(mean_squared_error(actuals_kw, rf_preds_kw))
    rf_mae = mean_absolute_error(actuals_kw, rf_preds_kw)
    
    print(f" Random Forest Baseline -> RMSE: {rf_rmse:.2f} kW | MAE: {rf_mae:.2f} kW")
    print("="*55 + "\n") 

    return rf_preds_kw  

# ==========================================
# STEP 6 & 7: Scaling Up & Regularization 
# ==========================================
# Step 6: LSTM architecture with enough capacity to map non-linearities
# Step 7: Dropout applied to prevent overfitting to household noise
class VPPLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(VPPLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x) # 
        out = self.fc(out[:, -1, :]) # Take the last timestep's output
        return out

def train_model(model, train_loader, valid_loader, epochs=50, lr=0.0005):
    print("Steps 6 & 7: Training with Regularization & Early Stopping...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 5 # Early stopping parameter
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                valid_loss += loss.item()
                
        val_rmse = np.sqrt(valid_loss/len(valid_loader))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Valid RMSE (Scaled): {val_rmse:.4f}")
        
        # Early Stopping Logic
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(model.state_dict(), 'best_vpp_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Assemble
    df = load_and_assemble_data()
    
    # 2 & 3. Protocol
    train_df, valid_df, test_df = seasonal_block_split(df) # chronological_split(df) if you want the original month-based split
    
    # 4. Prepare
    lookback_window = 96 # 24 hours of 15-min intervals
    train_loader, valid_loader, test_loader, scaler = prepare_data(train_df, valid_df, test_df, lookback_window)
    
    # 5. Baseline
    evaluate_persistence_baseline(test_df.copy())
    rf_predictions_kw = evaluate_random_forest_baseline(train_loader, test_loader, scaler)
    
    # 6 & 7. Build and Train
    input_features = train_df.shape[1] # Original columns
    model = VPPLSTM(input_dim=input_features, hidden_dim=128, num_layers=2)
    
    # Train the network
    train_model(model, train_loader, valid_loader)
    print("Pipeline Complete. Ready for Test Set Evaluation.")


  
    # %% Step 8: Final Evaluation & Un-scaling
   
    print("\nStep 8: Evaluating Model on December Test Set...")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # 1. Load the "Golden" weights 
    model.load_state_dict(torch.load('best_vpp_model.pth', weights_only=True))
    model.eval()
    
    actuals = []
    predictions = []
    
    # 2. Run the unseen December data through the model
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            # Squeeze and move back to CPU for numpy conversion
            preds = model(X_batch).cpu().numpy().squeeze()
            predictions.extend(preds)
            actuals.extend(y_batch.cpu().numpy().squeeze())
            
    # Reshape arrays so the scaler can read them
    import numpy as np
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # 3. The Magic Trick: Un-scale back to real-world kilowatts (kW)
    # Create fake arrays of zeros with 7 columns (matching the scaler's training shape)
    dummy_preds = np.zeros((len(predictions), 7))
    dummy_actuals = np.zeros((len(actuals), 7))
    
    # Insert our 1-column predictions and actuals into the first column (Index 0)
    dummy_preds[:, 0] = predictions.squeeze()
    dummy_actuals[:, 0] = actuals.squeeze()
    
    # Run the inverse transform on the full 7-column arrays
    unscaled_preds_full = scaler.inverse_transform(dummy_preds)
    unscaled_actuals_full = scaler.inverse_transform(dummy_actuals)
    
    # Slice out just that first column to get our real-world kilowatts!
    predictions_kw = unscaled_preds_full[:, 0]
    actuals_kw = unscaled_actuals_full[:, 0]
    
    # 4. Calculate real-world Error Metrics
    
    test_rmse = np.sqrt(mean_squared_error(actuals_kw, predictions_kw))
    test_mae = mean_absolute_error(actuals_kw, predictions_kw)
    
    print("\n" + "="*45)
    print(" FINAL RESULTS (Unseen Test Data)")
    print("="*45)
    print(f" Test RMSE: {test_rmse:.2f} kW")
    print(f" Test MAE:  {test_mae:.2f} kW")
    print("="*45)

    # 5. Generate a high-resolution plot for your research paper
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Calculate the Persistence Baseline (shifting actuals forward by 1 step)
    persistence_kw = np.roll(actuals_kw, 1)
    persistence_kw[0] = actuals_kw[0] # Handle the first interval boundary

    # plot just a 3-day slice (96 steps/day * 3 days)
    plot_window = 96
    
    # Convert the test set index to US Eastern Time
    test_df.index = test_df.index.tz_convert('America/Chicago')
    
    # --- The Midnight Shift ---
    # 8:00 PM to Midnight is 4 hours (16 intervals of 15-minutes)
    start_offset = 20 
    end_idx = start_offset + plot_window
    
    # Calculate the t-24 (24-Hour) Baseline for this exact shifted window
    persistence_24h_kw = test_df.iloc[start_offset : end_idx, 0].values
    
    # Extract the exact timestamps, offsetting by the 96-step lookback PLUS our 16-step midnight shift
    plot_dates = test_df.index[96 + start_offset : 96 + end_idx]
    act_slice = actuals_kw[start_offset : end_idx]
    lstm_slice = predictions_kw[start_offset : end_idx]
    p15_slice = persistence_kw[start_offset : end_idx]
    rf_slice = rf_predictions_kw[start_offset : end_idx]
    p24_slice = test_df.iloc[start_offset : end_idx, 0].values # 24-hour baseline
    
    # --- Create the 2x2 Subplot Grid ---
    # sharex and sharey ensure the scales perfectly match across all four panels
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle('VPP Load Forecasting Model Comparisons (24 hr period)', fontsize=18, fontweight='bold')
    
    # 1. Top Left: LSTM vs Actual
    axs[0, 0].plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    axs[0, 0].plot(plot_dates, lstm_slice, label='LSTM Forecast', color='red', linestyle='--', linewidth=1.5)
    axs[0, 0].set_title('1. LSTM vs Actual Load', fontsize=14)
    axs[0, 0].set_ylabel('Grid Load (kW)', fontsize=12)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Top Right: LSTM vs 24-Hour Baseline
    axs[0, 1].plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    axs[0, 1].plot(plot_dates, p24_slice, label='24-Hour Baseline', color='green', linestyle='-.', linewidth=1.5)
    axs[0, 1].plot(plot_dates, lstm_slice, label='LSTM', color='red', linestyle='--', linewidth=1.5)
    axs[0, 1].set_title('2. LSTM vs 24-Hour Baseline', fontsize=14)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Bottom Left: LSTM vs 15-Minute Baseline
    axs[1, 0].plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    axs[1, 0].plot(plot_dates, p15_slice, label='15-Min Baseline', color='blue', linestyle=':', linewidth=1.5)
    axs[1, 0].plot(plot_dates, lstm_slice, label='LSTM', color='red', linestyle='--', linewidth=1.5)
    axs[1, 0].set_title('3. LSTM vs 15-Min Baseline', fontsize=14)
    axs[1, 0].set_ylabel('Grid Load (kW)', fontsize=12)
    axs[1, 0].set_xlabel('Local Time (America/Central)', fontsize=12)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Bottom Right: LSTM vs Random Forest
    axs[1, 1].plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    axs[1, 1].plot(plot_dates, rf_slice, label='Random Forest', color='orange', linewidth=1.5)
    axs[1, 1].plot(plot_dates, lstm_slice, label='LSTM', color='red', linestyle='--', linewidth=1.5)
    axs[1, 1].set_title('4. LSTM vs Random Forest', fontsize=14)
    axs[1, 1].set_xlabel('Local Time (America/Central)', fontsize=12)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Auto-format the dates on the x-axis so they don't overlap
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=test_df.index.tz))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Give the main title some breathing room
    plt.savefig('vpp_forecast_subplots.png', dpi=300)
    print("\n  -> 4-Panel Subplot successfully saved as 'vpp_forecast_subplots.png'")

    # ==========================================
    # Slide 1: LSTM vs Actual
    # ==========================================
    plt.figure(figsize=(12, 5))
    plt.plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    plt.plot(plot_dates, lstm_slice, label='LSTM Forecast', color='red', linestyle='--', linewidth=1.5)
    plt.title('LSTM Load Forecast vs Actual Telemetry', fontsize=14)
    plt.ylabel('Grid Load (kW)', fontsize=12)
    plt.xlabel('Local Time (America/Central)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=test_df.index.tz))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('slide_1_lstm_actual.png', dpi=300)
    plt.close() # Close figure to free up memory

    # ==========================================
    # Slide 2: LSTM vs 24-Hour Baseline
    # ==========================================
    plt.figure(figsize=(12, 5))
    plt.plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    plt.plot(plot_dates, p24_slice, label='24-Hour Persistence', color='green', linestyle='-.', linewidth=1.5)
    plt.plot(plot_dates, lstm_slice, label='LSTM Forecast', color='red', linestyle='--', linewidth=1.5)
    plt.title('LSTM vs 24-Hour Persistence Baseline', fontsize=14)
    plt.ylabel('Grid Load (kW)', fontsize=12)
    plt.xlabel('Local Time (America/Central)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=test_df.index.tz))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('slide_2_lstm_24h.png', dpi=300)
    plt.close()

    # ==========================================
    # Slide 3: LSTM vs 15-Minute Baseline
    # ==========================================
    plt.figure(figsize=(12, 5))
    plt.plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    plt.plot(plot_dates, p15_slice, label='15-Min Persistence', color='blue', linestyle=':', linewidth=1.5)
    plt.plot(plot_dates, lstm_slice, label='LSTM Forecast', color='red', linestyle='--', linewidth=1.5)
    plt.title('LSTM vs 15-Minute Persistence Baseline', fontsize=14)
    plt.ylabel('Grid Load (kW)', fontsize=12)
    plt.xlabel('Local Time (America/Central)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=test_df.index.tz))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('slide_3_lstm_15m.png', dpi=300)
    plt.close()

    # ==========================================
    # Slide 4: LSTM vs Random Forest
    # ==========================================
    plt.figure(figsize=(12, 5))
    plt.plot(plot_dates, act_slice, label='Actual Load', color='black', linewidth=1.5)
    plt.plot(plot_dates, rf_slice, label='Random Forest', color='orange', linewidth=1.5)
    plt.plot(plot_dates, lstm_slice, label='LSTM Forecast', color='red', linestyle='--', linewidth=1.5)
    plt.title('LSTM vs Random Forest (Deep vs Shallow ML)', fontsize=14)
    plt.ylabel('Grid Load (kW)', fontsize=12)
    plt.xlabel('Local Time (America/Central)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=test_df.index.tz))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('slide_4_lstm_rf.png', dpi=300)
    plt.close()

    print("  -> Successfully generated 4 presentation-ready plots.")
   # %%
