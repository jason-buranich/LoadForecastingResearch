import random
import os
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import your core pipeline components
from data import train_df, test_df, scaler
from ISO.slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

def main():
    # 1. Hour-Ahead Specific Parameters
    HORIZON = 1
    SEQ_LEN = 24
    TARGET_IDX = 4
    
    print("--- Starting 1-Hour-Ahead LightGBM Pipeline ---")
    
    # 2. Re-slice sequences using the fixed sliding window logic
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 3. Flatten history and future weather covariates into 2D matrices
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    # Since horizon is 1, flatten the target to a 1D vector
    y_train = Y_train.numpy().ravel()
    y_test = Y_test.numpy().ravel()
    
    # 4. Initialize and train the single LightGBM model
    print(f"Training LightGBM on shape: {X_train_combined.shape} to predict t+1...")
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=15,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        verbosity=-1
    )
    lgbm_model.fit(X_train_combined, y_train)
    
    # 5. Predict and Inverse Scale
    preds_flat = lgbm_model.predict(X_test_combined)
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
        
    preds_mw = inverse_scale(preds_flat)
    targets_mw = inverse_scale(y_test)
    
    # 6. Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- Hour-Ahead LightGBM Evaluation Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 7. Plot Forecast
    plot_single_model_forecast(
        targets_mw, 
        preds_mw, 
        start_idx=0, 
        model_name="1-Hour LightGBM", 
        save_path='lgbm_hour_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()