import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Enforce Deterministic Behavior
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Import components from the configured La Trobe pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import DirectLSTM
from visualize import plot_single_model_forecast

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    # 1-Hour (4-Step) Configuration
    HORIZON = 4              
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2
    
    # Hyperparameters
    BATCH_SIZE = 128  
    EPOCHS = 100
    PATIENCE = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting 1-Hour-Ahead DirectLSTM Pipeline on {device} ---")
    
    # 2. Extract 3D Tensors
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 3. Build DataLoaders (with pinned memory for faster CPU-to-GPU transfer)
    train_dataset = TensorDataset(X_train_hist, X_train_fut, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    val_dataset = TensorDataset(X_val_hist, X_val_fut, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # 4. Instantiate Model
    hist_input_dim = X_train_hist.shape[2]
    future_input_dim = X_train_fut.shape[2]
    
    # Guard against PyTorch crashing on single-layer dropout
    dropout_val = DROPOUT if NUM_LAYERS > 1 else 0.0
    
    model = DirectLSTM(
        hist_input_dim=hist_input_dim,
        future_input_dim=future_input_dim,
        hidden_dim=HIDDEN_DIM,
        horizon=HORIZON,
        num_layers=NUM_LAYERS,
        dropout=dropout_val
    ).to(device)
    
    # L1Loss for WAPE alignment
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 5. Training Loop with Early Stopping
    print(f"Training Model for up to {EPOCHS} Epochs (Patience: {PATIENCE})...")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'best_directlstm_1hr.pth'
    
    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for batch_hist, batch_fut, batch_y in train_loader:
            batch_hist, batch_fut, batch_y = batch_hist.to(device), batch_fut.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_hist, batch_fut)
            
            # Defensive dimension checking
            if predictions.dim() != batch_y.dim():
                predictions = predictions.view_as(batch_y)
                
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_hist.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_hist, batch_fut, batch_y in val_loader:
                batch_hist, batch_fut, batch_y = batch_hist.to(device), batch_fut.to(device), batch_y.to(device)
                
                predictions = model(batch_hist, batch_fut)
                
                if predictions.dim() != batch_y.dim():
                    predictions = predictions.view_as(batch_y)
                    
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item() * batch_hist.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss (MAE): {train_loss:.4f} | Val Loss (MAE): {val_loss:.4f}")
        
        # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered! No improvement in validation loss for {PATIENCE} epochs.")
                break
                
    # 6. Evaluation Loop
    print("\nLoading best model weights and evaluating on Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        test_dataset = TensorDataset(X_test_hist, X_test_fut)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for batch_hist, batch_fut in test_loader:
            batch_hist, batch_fut = batch_hist.to(device), batch_fut.to(device)
            preds = model(batch_hist, batch_fut)
            test_preds.append(preds.cpu().numpy())
            
    # test_preds shape: (N_samples, 4)
    preds_np = np.concatenate(test_preds, axis=0)
    Y_test_np = Y_test.numpy()
    
    # 7. Inverse Scaling & Metrics
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    # Calculate aggregate metrics across all 4 steps
    preds_kw_flat = inverse_scale(preds_np.ravel())
    targets_kw_flat = inverse_scale(Y_test_np.ravel())
    
    rmse = np.sqrt(mean_squared_error(targets_kw_flat, preds_kw_flat))
    mae = mean_absolute_error(targets_kw_flat, preds_kw_flat)
    wape = np.sum(np.abs(targets_kw_flat - preds_kw_flat)) / np.sum(np.abs(targets_kw_flat)) * 100
    
    print("\n--- 1-Hour-Ahead DirectLSTM Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 8. Visualization: Isolate the t+4 interval (index 3)
    t4_preds = inverse_scale(preds_np[:, 3])
    t4_targets = inverse_scale(Y_test_np[:, 3])
    
    plot_single_model_forecast(
        t4_targets, 
        t4_preds, 
        start_idx=0, 
        horizon=96,
        model_name="1-Hour DirectLSTM (t+4 step)", 
        save_path='directlstm_1hr_ahead_forecast.png'
    )
    
    # Cleanup temporary weight file
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

if __name__ == "__main__":
    main()