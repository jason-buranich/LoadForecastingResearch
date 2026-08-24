import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(model, test_loader, scaler, target_col_idx=4, device=None):
    """
    Evaluates a trained PyTorch model on the test dataset loader,
    applies inverse scaling, and computes MW-scale RMSE, MAE, and WAPE.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            
            # Safely reshape without losing the batch dimension
            batch_size = batch_x.size(0)
            preds_reshaped = outputs.view(batch_size, -1)
            targets_reshaped = batch_y.view(batch_size, -1)
            
            # Safely detach and convert to numpy
            all_preds.append(preds_reshaped.detach().cpu().numpy())
            all_targets.append(targets_reshaped.detach().cpu().numpy())
            
    # Concatenate all batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Flatten arrays for global metric calculations
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Inverse transform back to Megawatts (MW)
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, target_col_idx] = data_flat
        return scaler.inverse_transform(dummy)[:, target_col_idx]
    
    preds_mw = inverse_scale(preds_flat)
    targets_mw = inverse_scale(targets_flat)
    
    # Compute operational metrics in actual MW
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    metrics = {
        "RMSE (MW)": float(rmse),
        "MAE (MW)": float(mae),
        "WAPE (%)": float(wape)
    }
    
    print("--- True Scale Evaluation Metrics (MW) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    return metrics, preds_mw, targets_mw