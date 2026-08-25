import torch
from torch.utils.data import TensorDataset
from data import train_df, val_df, test_df, scaler  # Added scaler import
from slidingWindow import create_safe_sequences
from optimize import get_dataloaders
from models import DirectLSTM, get_tabular_models
from trainingLoop import train_model
from evaluate import evaluate_predictions
from visualize import plot_day_ahead_forecast
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_tabular_baseline(X_train, Y_train, X_test, Y_test, scaler, target_col_idx=4, horizon=24):
    print(f"\n--- Training Tabular ISO Baseline (MLR) ---")
    
    # 1. Flatten the 3D PyTorch tensors into 2D NumPy arrays for scikit-learn
    # X shape goes from (N, 168, 9) -> (N, 1512)
    X_train_flat = X_train.numpy().reshape(X_train.shape[0], -1)
    X_test_flat = X_test.numpy().reshape(X_test.shape[0], -1)
    
    # Y is already 2D (N, 24), just convert to NumPy
    Y_train_flat = Y_train.numpy()
    Y_test_flat = Y_test.numpy()
    
    # 2. Instantiate Tabular Models (MLR, RF, LightGBM)
    mlr, rf, lgbm = get_tabular_models(horizon=horizon)
    
    # 3. Train the MLR (Ridge Regression) model
    print("Fitting Multiple Linear Regression...")
    mlr.fit(X_train_flat, Y_train_flat)
    
    # 4. Generate Predictions
    preds = mlr.predict(X_test_flat)
    
    # 5. Inverse-Scale to Megawatts (MW)
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, target_col_idx] = data_flat
        return scaler.inverse_transform(dummy)[:, target_col_idx]
    
    preds_mw = inverse_scale(preds.flatten())
    targets_mw = inverse_scale(Y_test_flat.flatten())
    
    # 6. Calculate True Operational Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("--- MLR Baseline Evaluation Metrics (MW) ---")
    print(f"RMSE (MW): {rmse:.2f}")
    print(f"MAE (MW):  {mae:.2f}")
    print(f"WAPE (%):  {wape:.2f}")
    
    return mlr, preds_mw

def run_experiment_with_evaluation():
    HORIZON = 24
    SEQ_LEN = 168
    TARGET_IDX = 4
    
    print(f"--- Starting Full Pipeline: {HORIZON}-Hour Horizon, {SEQ_LEN}-Hour Lookback ---")
    
    # 1. Generate sliding window tensors for Train, Val, and Test
    print("Generating sliding window tensors...")
    X_train, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_val, Y_val     = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test, Y_test   = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 2. Run the ISO Baseline (MLR) First
    mlr_model, mlr_preds = run_tabular_baseline(X_train, Y_train, X_test, Y_test, scaler, TARGET_IDX, HORIZON)
    
    # 3. Create PyTorch DataLoaders (using Optuna's best batch_size of 32)
    train_loader, val_loader = get_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=32)
    
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 4. Initialize Model (using Optuna's best architecture)
    input_dim = X_train.shape[-1]
    model = DirectLSTM(input_dim=input_dim, hidden_dim=256, horizon=HORIZON, num_layers=1)
    
    model_path = f'tuned_direct_lstm_{HORIZON}h.pth'
    
    # 5. Execute Training Loop (with Early Stopping and Optuna's learning rate)
    print("Launching training loop with tuned parameters...")
    trained_model, _ = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=100, 
        lr=0.000394, 
        patience=10,
        model_save_path=model_path
    )
    
    # 6. Load Best Weights and Evaluate on Unseen Test Data
    print("Evaluating best model on test dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, weights_only=True))
    trained_model.to(device)
    
    lstm_metrics, lstm_preds, lstm_targets = evaluate_predictions(
        trained_model, 
        test_loader, 
        scaler=scaler, 
        target_col_idx=TARGET_IDX, 
        device=device
    )
    
    # 7. Generate forecast comparison graph
    print("Generating forecast comparison graph...")
    plot_day_ahead_forecast(lstm_targets, mlr_preds, lstm_preds, start_idx=0)

if __name__ == "__main__":
    run_experiment_with_evaluation()