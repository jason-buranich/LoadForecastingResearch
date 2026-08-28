import random
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import your core pipeline components
from data import train_df, test_df, scaler
from slidingWindow import create_safe_sequences

def main():
    
    # 1. Hour-Ahead Specific Parameters
    HORIZON = 1
    SEQ_LEN = 24
    TARGET_IDX = 4
    
    print("--- Starting 1-Hour-Ahead Random Forest Pipeline ---")
    
    # 2. Re-slice sequences using your existing function with the new 24h window
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
    
    # 4. Initialize and train the Random Forest
    print(f"Training Random Forest on shape: {X_train_combined.shape} to predict t+1...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        max_features=0.3,
        n_jobs=4
    )
    rf.fit(X_train_combined, y_train)
    
    # 5. Predict and Inverse Scale
    preds_flat = rf.predict(X_test_combined)
    
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
    
    print("\n--- Hour-Ahead Random Forest Evaluation Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")

if __name__ == "__main__":
    main()