import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import get_tabular_models

# ==========================================
# 1. Tabular Baseline Evaluation (MLR)
# ==========================================
def evaluate_tabular_baseline(X_train_hist, X_train_fut, Y_train, X_test_hist, X_test_fut, Y_test, scaler, target_col_idx=4, horizon=24, model_type='mlr'):
    print(f"\n--- Training Tabular Baseline ({model_type.upper()}) with Future Covariates ---")
    
    X_train_h_flat = X_train_hist.numpy().reshape(X_train_hist.shape[0], -1)
    X_test_h_flat = X_test_hist.numpy().reshape(X_test_hist.shape[0], -1)
    
    X_train_f_flat = X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    X_test_f_flat = X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    
    X_train_combined = np.hstack((X_train_h_flat, X_train_f_flat))
    X_test_combined = np.hstack((X_test_h_flat, X_test_f_flat))
    
    Y_train_flat = Y_train.numpy()
    Y_test_flat = Y_test.numpy()
    
    mlr, rf, lgbm = get_tabular_models(horizon=horizon)
    
    # Dynamically select the model
    if model_type.lower() == 'mlr':
        model = mlr
    elif model_type.lower() == 'rf':
        model = rf
    elif model_type.lower() == 'lgbm':
        model = lgbm
    else:
        raise ValueError("model_type must be 'mlr', 'rf', or 'lgbm'")
        
    print(f"Fitting {model_type.upper()} with historical data + 24-hour weather forecast...")
    model.fit(X_train_combined, Y_train_flat)
    
    preds = model.predict(X_test_combined)
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, target_col_idx] = data_flat
        return scaler.inverse_transform(dummy)[:, target_col_idx]
        
    preds_mw = inverse_scale(preds.flatten())
    targets_mw = inverse_scale(Y_test_flat.flatten())
    
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print(f"\n--- {model_type.upper()} Baseline (With Future Weather) Metrics (MW) ---")
    print(f"RMSE (MW): {rmse:.2f}")
    print(f"MAE (MW):  {mae:.2f}")
    print(f"WAPE (%):  {wape:.2f}")
    
    return model, preds_mw

# ==========================================
# 2. PyTorch Neural Network Evaluation
# ==========================================
def evaluate_predictions(model, test_loader, scaler, target_col_idx, device='cuda'):
    """
    Evaluates the Seq2Seq Covariate LSTM. 
    Unpacks three variables (history, future, target) from the dataloader.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x_hist, batch_x_fut, batch_y in test_loader:
            batch_x_hist = batch_x_hist.to(device)
            batch_x_fut = batch_x_fut.to(device)
            
            # Pass both history and future forecasts to the model
            preds = model(batch_x_hist, batch_x_fut)
            
            predictions.append(preds.cpu().numpy())
            targets.append(batch_y.numpy())
            
    # Flatten the outputs for scaling
    predictions_flat = np.concatenate(predictions, axis=0).flatten()
    targets_flat = np.concatenate(targets, axis=0).flatten()
    
    # Inverse scale
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, target_col_idx] = data_flat
        return scaler.inverse_transform(dummy)[:, target_col_idx]

    preds_mw = inverse_scale(predictions_flat)
    targets_mw = inverse_scale(targets_flat)
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- Seq2Seq LSTM Evaluation Metrics (MW) ---")
    print(f"RMSE (MW): {rmse:.2f}")
    print(f"MAE (MW):  {mae:.2f}")
    print(f"WAPE (%):  {wape:.2f}")
    
    return {"rmse": rmse, "mae": mae, "wape": wape}, preds_mw, targets_mw

def evaluate_direct_lstm(model, test_loader, scaler, target_col_idx, device='cuda'):
    """
    Dedicated 2-item evaluation loop for the Direct LSTM.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            predictions.append(preds.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
            
    predictions_flat = np.concatenate(predictions, axis=0).flatten()
    targets_flat = np.concatenate(targets, axis=0).flatten()
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, target_col_idx] = data_flat
        return scaler.inverse_transform(dummy)[:, target_col_idx]

    preds_mw = inverse_scale(predictions_flat)
    targets_mw = inverse_scale(targets_flat)
    
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- Direct LSTM Evaluation Metrics (MW) ---")
    print(f"RMSE (MW): {rmse:.2f}")
    print(f"MAE (MW):  {mae:.2f}")
    print(f"WAPE (%):  {wape:.2f}")
    
    return {"rmse": rmse, "mae": mae, "wape": wape}, preds_mw, targets_mw