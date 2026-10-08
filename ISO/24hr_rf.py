import random
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

def main():
    
    HORIZON = 24
    SEQ_LEN = 168
    TARGET_IDX = 4
    
    print("--- Starting 24-Model Specialized Random Forest Pipeline ---")
    
    # 1. Generate sliding window tensors
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 2. Flatten history and future weather/time covariates into 2D matrices
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    Y_train_flat = Y_train.numpy() # Shape: (N, 24)
    Y_test_flat = Y_test.numpy()   # Shape: (N, 24)
    
    # 3. Train 24 Separate Random Forest Models (Direct Multi-Model Strategy)
    print("Training 24 specialized Random Forest models (one per horizon step)...")
    preds_list = []
    
    for h in range(HORIZON):
        print(f"  -> Training model for Hour {h+1}/{HORIZON}...")
        # Isolate the target vector strictly for hour h
        y_train_h = Y_train_flat[:, h]
        
        rf_h = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            max_features=0.3,
            n_jobs=4
        )
        rf_h.fit(X_train_combined, y_train_h)
        
        # Predict for this specific hour on the test set
        pred_h = rf_h.predict(X_test_combined)
        preds_list.append(pred_h)
        
    # Stack the predictions back into an (N, 24) matrix
    preds = np.column_stack(preds_list)
    
    # 4. Inverse Scale
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
        
    preds_mw = inverse_scale(preds.flatten())
    targets_mw = inverse_scale(Y_test_flat.flatten())
    
    # 5. Calculate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- 24-Model Random Forest Evaluation Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 6. Plot Forecast
    plot_single_model_forecast(
        targets_mw, 
        preds_mw, 
        start_idx=0, 
        model_name="24-Model RF", 
        save_path='rf_specialized_forecast.png'
    )

if __name__ == "__main__":
    main()