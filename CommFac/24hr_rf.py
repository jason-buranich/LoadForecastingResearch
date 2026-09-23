import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the configured La Trobe pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import get_tabular_models
from visualize import plot_single_model_forecast

# ==============================================================================
# 1. DATA PREPARATION: TENSOR FLATTENING
# ==============================================================================
def flatten_sequences(X_hist, X_fut):
    """
    Scikit-learn tree models cannot natively process 3D temporal tensors (Batch, Seq, Features).
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
    # 24-Hour (96-Step) Configuration
    HORIZON = 96              
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2  
    
    print("--- Starting 24-Hour-Ahead Random Forest Pipeline ---")
    
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
    
    # Combine Train + Val to maximize the data the tree ensemble can see during fitting
    X_train_full_hist = torch.cat([X_train_hist, X_val_hist], dim=0)
    X_train_full_fut = torch.cat([X_train_fut, X_val_fut], dim=0)
    
    # Keep target as 2D array (N_samples, 96_steps) for the MultiOutputRegressor
    Y_train_full = torch.cat([Y_train, Y_val], dim=0).numpy()
    
    # Flatten the features specifically for the tabular model
    X_train_flat = flatten_sequences(X_train_full_hist, X_train_full_fut)
    X_test_flat = flatten_sequences(X_test_hist, X_test_fut)
    Y_test_np = Y_test.numpy()
    
    # Instantiate the Random Forest model (returns mlr, rf, lgbm)
    _, rf, _ = get_tabular_models(horizon=HORIZON)
    
    print("Fitting Random Forest 24-Hour Predictor (This will train 96 distinct tree ensembles)...")
    rf.fit(X_train_flat, Y_train_full)
    
    # Predict on the test set; returns shape (N_samples, 96)
    preds_np = rf.predict(X_test_flat)
    
    # Inverse Scaling function
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    # Calculate metrics across the entire 96-step composite block
    preds_kw_flat = inverse_scale(preds_np.ravel())
    targets_kw_flat = inverse_scale(Y_test_np.ravel())
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(targets_kw_flat, preds_kw_flat))
    mae = mean_absolute_error(targets_kw_flat, preds_kw_flat)
    wape = np.sum(np.abs(targets_kw_flat - preds_kw_flat)) / np.sum(np.abs(targets_kw_flat)) * 100
    
    print("\n--- 24-Hour-Ahead Random Forest Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # Visualization: Isolate the t+96 interval (index 95) 
    t96_preds = inverse_scale(preds_np[:, 95])
    t96_targets = inverse_scale(Y_test_np[:, 95])
    
    plot_single_model_forecast(
        t96_targets, 
        t96_preds, 
        start_idx=0, 
        horizon=96,
        model_name="24-Hour Random Forest (t+96 step)", 
        save_path='rf_24hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()