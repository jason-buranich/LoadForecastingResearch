import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the updated data.py
from data import train_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

def main():
    # 1. 1-Hour Substation Configuration
    HORIZON = 4              # Predict 4 steps ahead (1 hour)
    SEQ_LEN = 96             # 96 intervals = 24 hours of history
    TARGET_IDX = 1           # Load_MW is at index 1 before dropping Month
    COVARIATE_START_IDX = 2  # Future covariates start at index 2
    
    print("--- Starting 1-Hour-Ahead LightGBM (Multi-Model) Pipeline ---")
    
    # 2. Slice Sequences
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 3. Flatten history and future covariates into 2D matrices for scikit-learn
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    # Keep targets as 2D arrays (N, 4) for multi-output regression
    y_train = Y_train.numpy()
    y_test = Y_test.numpy()
    
    # 4. Initialize and Train the 4 Independent LightGBM Models
    # Wrapping the base estimator trains a distinct model for each horizon step
    lgb_base = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=4,
        force_col_wise=True,
        verbose=-1
    )
    
    multi_lgb = MultiOutputRegressor(lgb_base)
    
    print(f"Training 4 distinct LightGBM Models on shape {X_train_combined.shape} to predict t+1 through t+{HORIZON}...")
    multi_lgb.fit(X_train_combined, y_train)
    print("Multi-Model LightGBM Training Complete.")
    
    # 5. Predict and Inverse Scale
    preds = multi_lgb.predict(X_test_combined)
    
    # Flatten the (N, 4) arrays to 1D for unified inverse scaling and metric calculation
    preds_flat = preds.ravel()
    y_test_flat = y_test.ravel()
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
        
    preds_mw = inverse_scale(preds_flat)
    targets_mw = inverse_scale(y_test_flat)
    
    # 6. Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- 1-Hour-Ahead LightGBM Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 7. Visualization
    # Isolate the t+4 interval (index 3) to prevent overlapping visualizations
    t4_preds = inverse_scale(preds[:, 3])
    t4_targets = inverse_scale(y_test[:, 3])
    
    plot_single_model_forecast(
        t4_targets, 
        t4_preds, 
        start_idx=0, 
        horizon=96,
        model_name="1-Hour LightGBM (t+4 step)", 
        save_path='lightgbm_1hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()