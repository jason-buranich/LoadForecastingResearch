import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the configured La Trobe pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

# ==============================================================================
# 1. DATA PREPARATION: TENSOR FLATTENING
# ==============================================================================
def flatten_sequences(X_hist, X_fut):
    """
    Scikit-learn API models (including XGBoost) cannot natively process 3D temporal tensors.
    This function flattens the 96-step history and future covariates into a single 2D feature matrix.
    """
    hist_np = X_hist.numpy()
    fut_np = X_fut.numpy()
    
    # Flatten temporal and feature dimensions into a single vector per sample
    hist_flat = hist_np.reshape(hist_np.shape[0], -1)
    fut_flat = fut_np.reshape(fut_np.shape[0], -1)
    
    # Concatenate the historical sequence features with the known future covariates
    return np.hstack([hist_flat, fut_flat])

# ==============================================================================
# 2. MAIN PIPELINE
# ==============================================================================
def main():
    # 15-Minute Configuration
    HORIZON = 1              
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2  
    
    print("--- Starting 15-Minute-Ahead XGBoost Pipeline ---")
    
    # Slice sequences using the exact same masking constraints as the PyTorch models
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # Combine Train + Val to maximize the data the gradient booster can see during fitting
    X_train_full_hist = torch.cat([X_train_hist, X_val_hist], dim=0)
    X_train_full_fut = torch.cat([X_train_fut, X_val_fut], dim=0)
    
    # Flatten the 1-step target to a 1D array (N_samples,)
    Y_train_full = torch.cat([Y_train, Y_val], dim=0).numpy().flatten()
    
    # Flatten the features specifically for the tabular model
    X_train_flat = flatten_sequences(X_train_full_hist, X_train_full_fut)
    X_test_flat = flatten_sequences(X_test_hist, X_test_fut)
    Y_test_np = Y_test.numpy().flatten()
    
    # Instantiate the XGBoost Regressor (matching the approximate depth/estimators of your RF/LGBM models)
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=6,          # XGBoost typically requires slightly shallower trees than RF
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1             # Uses all available container CPU cores
    )
    
    print("Fitting XGBoost 15-Minute Predictor...")
    xgb_model.fit(X_train_flat, Y_train_full)
    
    # Predict on the test set; returns shape (N_samples,)
    preds_flat = xgb_model.predict(X_test_flat)
    
    # Inverse Scaling function
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    preds_kw = inverse_scale(preds_flat)
    targets_kw = inverse_scale(Y_test_np)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(targets_kw, preds_kw))
    mae = mean_absolute_error(targets_kw, preds_kw)
    wape = np.sum(np.abs(targets_kw - preds_kw)) / np.sum(np.abs(targets_kw)) * 100
    
    print("\n--- 15-Minute-Ahead XGBoost Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # Visualization
    plot_single_model_forecast(
        targets_kw, 
        preds_kw, 
        start_idx=0, 
        horizon=96,
        model_name="15-Min XGBoost", 
        save_path='xgboost_15min_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()