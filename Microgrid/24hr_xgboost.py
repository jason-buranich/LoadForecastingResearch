import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from your updated data pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

def flatten_for_tabular(X_hist, X_fut):
    """
    Flattens 3D temporal tensors into 2D tabular matrices.
    """
    N = X_hist.shape[0]
    hist_flat = X_hist.reshape(N, -1)
    fut_flat = X_fut.reshape(N, -1)
    return np.hstack((hist_flat, fut_flat))

def main():
    # 1. 24-Hour (96-Step) Microgrid Configuration
    HORIZON = 96             
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2  
    
    print("--- Starting 24-Hour-Ahead XGBoost (Throttled Multi-Model) Predictor ---")
    
    # 2. Slice Sequences
    combined_train_df = pd.concat([train_df, val_df]).sort_index()
    
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        combined_train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 3. Flatten for scikit-learn
    X_train_flat = flatten_for_tabular(X_train_hist.numpy(), X_train_fut.numpy())
    X_test_flat = flatten_for_tabular(X_test_hist.numpy(), X_test_fut.numpy())
    
    Y_train_np = Y_train.numpy()
    Y_test_np = Y_test.numpy()
    
    # 4. Initialize and Train the 96 Independent XGBoost Models
    # Use n_jobs=1 internally to prevent threading clashes[cite: 8]
    xg_base = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist', # speed up the CPU binning
        n_jobs=1            
    )
    
    # Throttle concurrent workers to 4 to prevent OOM segmentation faults[cite: 8]
    multi_xgb = MultiOutputRegressor(xg_base, n_jobs=4)
    
    print(f"Training {HORIZON} distinct XGBoost Models (Throttled to 4 workers)...")
    multi_xgb.fit(X_train_flat, Y_train_np)
    
    # 5. Predict
    print("Evaluating on Test Set...")
    preds = multi_xgb.predict(X_test_flat)
    
    # 6. Inverse Scaling
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    preds_kw_flat = inverse_scale(preds.ravel())
    targets_kw_flat = inverse_scale(Y_test_np.ravel())
    
    # 7. Metrics
    rmse = np.sqrt(mean_squared_error(targets_kw_flat, preds_kw_flat))
    mae = mean_absolute_error(targets_kw_flat, preds_kw_flat)
    wape = np.sum(np.abs(targets_kw_flat - preds_kw_flat)) / np.sum(np.abs(targets_kw_flat)) * 100
    
    print("\n--- 24-Hour-Ahead XGBoost Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 8. Visualization
    t96_preds = inverse_scale(preds[:, 95])
    t96_targets = inverse_scale(Y_test_np[:, 95])
    
    plot_single_model_forecast(
        t96_targets, 
        t96_preds, 
        start_idx=0, 
        horizon=96, 
        model_name="24-Hour XGBoost (t+96 step)", 
        save_path='xgb_24hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()