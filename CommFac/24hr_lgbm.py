import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from tqdm import tqdm

# Import components from the configured La Trobe pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import get_tabular_models
from visualize import plot_single_model_forecast
from metrics import profile_model_efficiency, TrainingTimer

# ==============================================================================
# 1. DATA PREPARATION: TENSOR FLATTENING
# ==============================================================================
def flatten_sequences(X_hist, X_fut):
    hist_np = X_hist.numpy()
    fut_np = X_fut.numpy()
    
    hist_flat = hist_np.reshape(hist_np.shape[0], -1)
    fut_flat = fut_np.reshape(fut_np.shape[0], -1)
    
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
    
    print("--- Starting 24-Hour-Ahead LightGBM Pipeline ---")
    
    # Slice sequences
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # Combine Train + Val
    X_train_full_hist = torch.cat([X_train_hist, X_val_hist], dim=0)
    X_train_full_fut = torch.cat([X_train_fut, X_val_fut], dim=0)
    Y_train_full = torch.cat([Y_train, Y_val], dim=0).numpy()
    
    # Flatten features
    X_train_flat = flatten_sequences(X_train_full_hist, X_train_full_fut)
    X_test_flat = flatten_sequences(X_test_hist, X_test_fut)
    Y_test_np = Y_test.numpy()
    
    # Retrieve the base LightGBM configuration
    _, _, lgbm_multi = get_tabular_models(horizon=HORIZON)
    base_lgbm = lgbm_multi.estimator  # Extract the un-wrapped base model
    
    # --- MANUAL TRAINING LOOP WITH PROGRESS BAR ---
    print("\nFitting LightGBM 24-Hour Predictor...")
    trained_models = []
    
    timer = TrainingTimer(use_cuda=False)
    timer.start()
    for i in tqdm(range(HORIZON), desc="Training 96 Horizon Models", unit="model"):
        # Clone ensures a fresh model is instantiated for each time step
        step_model = clone(base_lgbm)
        step_model.fit(X_train_flat, Y_train_full[:, i])
        trained_models.append(step_model)
    timer.stop()
    hw_metrics = profile_model_efficiency(
        model=trained_models,
        test_input=X_test_flat[0:1],
        training_time_seconds=timer.training_time,
        model_type="tabular",
        save_path="temp_24hr_lgbm"
    )
    
    # --- MANUAL PREDICTION LOOP ---
    preds_np = np.zeros((X_test_flat.shape[0], HORIZON))
    for i in range(HORIZON):
        preds_np[:, i] = trained_models[i].predict(X_test_flat)
    
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
    
    print("\n--- 24-Hour-Ahead LightGBM Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # Visualization: Isolate the t+96 interval (index 95) 
    t96_preds = inverse_scale(preds_np[:, 95])
    t96_targets = inverse_scale(Y_test_np[:, 95])
    
    print("\n--- Hardware Efficiency Metrics ---")
    print(f"Inference Latency : {hw_metrics['inference_latency_seconds']:.6f} seconds")
    print(f"Training Latency  : {hw_metrics['training_latency_seconds']:.4f} seconds")
    print(f"Model Disk Size   : {hw_metrics['size_mb']:.4f} MB")

    plot_single_model_forecast(
        t96_targets, 
        t96_preds, 
        start_idx=0, 
        horizon=96,
        model_name="24-Hour LightGBM (t+96 step)", 
        save_path='lgbm_24hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()