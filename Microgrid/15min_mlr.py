import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from your updated data pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import get_tabular_models
from visualize import plot_single_model_forecast

def flatten_for_tabular(X_hist, X_fut):
    """
    Flattens 3D temporal tensors into 2D tabular matrices.
    X_hist: (Samples, Seq_len, Features) -> (Samples, Seq_len * Features)
    X_fut:  (Samples, Horizon, Covariates) -> (Samples, Horizon * Covariates)
    Returns concatenated 2D array.
    """
    N = X_hist.shape[0]
    hist_flat = X_hist.reshape(N, -1)
    fut_flat = X_fut.reshape(N, -1)
    return np.hstack((hist_flat, fut_flat))

def main():
    # 1. 15-Minute (1-Step) Microgrid Configuration
    HORIZON = 1              # Predict 1 step ahead (15 minutes)
    SEQ_LEN = 96             # 96 intervals = 24 hours of history
    TARGET_IDX = 1           # Load_kW is at index 1 before dropping Month
    COVARIATE_START_IDX = 2  # Future covariates start at index 2
    
    print("--- Starting 15-Minute-Ahead MLR Baseline ---")
    
    # 2. Slice Sequences
    # We combine train and val data for the MLR since it doesn't need a validation set for early stopping
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
    
    Y_train_np = Y_train.numpy().ravel()
    Y_test_np = Y_test.numpy().ravel()
    
    # 4. Instantiate Model
    mlr, _, _ = get_tabular_models(horizon=HORIZON)
    
    # 5. Train
    print("Training MLR (Ridge Regression)...")
    mlr.fit(X_train_flat, Y_train_np)
    
    # 6. Predict
    print("Evaluating on Test Set...")
    preds = mlr.predict(X_test_flat)
    
    # 7. Inverse Scaling
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    preds_mw = inverse_scale(preds)
    targets_mw = inverse_scale(Y_test_np)
    
    # 8. Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- 15-Minute-Ahead MLR Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 9. Visualization
    # Plot a continuous 24-hour block (96 consecutive 15-min predictions)
    plot_single_model_forecast(
        targets_mw, 
        preds_mw, 
        start_idx=0, 
        horizon=96, 
        model_name="15-Min MLR", 
        save_path='mlr_15min_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()