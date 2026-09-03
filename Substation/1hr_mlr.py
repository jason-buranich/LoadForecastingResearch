import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the newly updated data.py
from data import train_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

def main():
    # 1. 1-Hour Substation Configuration
    HORIZON = 4              # Predict 4 steps ahead (4 * 15 minutes = 1 hour)
    SEQ_LEN = 96             # 96 intervals = 24 hours of history
    TARGET_IDX = 1           # Load_MW is at index 1 before dropping Month
    COVARIATE_START_IDX = 2  # Future covariates start at index 2
    
    print(f"--- Starting 1-Hour-Ahead MLR (Ridge) Baseline ---")
    
    # 2. Slice Sequences
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, 
        seq_len=SEQ_LEN, 
        horizon=HORIZON, 
        target_idx=TARGET_IDX,
        covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, 
        seq_len=SEQ_LEN, 
        horizon=HORIZON, 
        target_idx=TARGET_IDX,
        covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 3. Flatten history and future covariates into 2D matrices
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    # For a multi-step horizon, Y_train is shape (N, 4). 
    # RidgeCV natively supports multi-target regression (2D y arrays).
    y_train = Y_train.numpy()
    y_test = Y_test.numpy()
    
    # 4. Initialize and train the L2-regularized baseline (Ridge)
    alphas_to_test = [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0] 
    mlr = RidgeCV(alphas=alphas_to_test)
    
    print(f"Training MLR on shape: {X_train_combined.shape} to predict t+{HORIZON} steps...")
    mlr.fit(X_train_combined, y_train)
    print(f"MLR Training Complete.")
    
    # 5. Predict and Inverse Scale
    # Flatten the (N, 4) arrays to 1D for unified inverse scaling and metric calculation
    preds = mlr.predict(X_test_combined)
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
    
    print("\n--- 1-Hour-Ahead Substation MLR Evaluation Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 7. Visualization
    # Note: Because the sliding window steps by 1 interval, flattening (N, 4) creates 
    # overlapping time predictions. To plot a true chronological day, we extract 
    # only the final horizon step (t+4) from the unflattened array.
    t4_preds = inverse_scale(preds[:, 3])
    t4_targets = inverse_scale(y_test[:, 3])
    
    plot_single_model_forecast(
        t4_targets, 
        t4_preds, 
        start_idx=0, 
        horizon=96,
        model_name="1-Hour MLR (t+4 step)", 
        save_path='mlr_1hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()