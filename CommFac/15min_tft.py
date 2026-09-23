import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Import components from the configured La Trobe pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import GridTransformer
from visualize import plot_single_model_forecast

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    # 15-Minute (1-Step) Configuration
    HORIZON = 1              
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 10
    LEARNING_RATE = 3e-3
    WEIGHT_DECAY = 5e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting 15-Minute-Ahead Grid Transformer Pipeline on {device} ---")
    
    # 1. Extract 3D Tensors
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 2. Build DataLoaders
    train_dataset = TensorDataset(X_train_hist, X_train_fut, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val_hist, X_val_fut, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Instantiate Model
    hist_input_dim = X_train_hist.shape[2]
    future_input_dim = X_train_fut.shape[2]
    
    model = GridTransformer(
        hist_input_dim=hist_input_dim,
        future_input_dim=future_input_dim,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        d_model=64,
        n_heads=2,
        num_layers=1,
        dropout=0.5
    ).to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 4. Training Loop with Early Stopping
    print(f"Training Model for up to {EPOCHS} Epochs (Patience: {PATIENCE})...")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'best_grid_transformer.pth'
    
    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for batch_hist, batch_fut, batch_y in train_loader:
            batch_hist, batch_fut, batch_y = batch_hist.to(device), batch_fut.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_hist, batch_fut)
            
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
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item() * batch_hist.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Print every epoch
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")
        
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
                
    # 5. Evaluation Loop
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
            
    preds_np = np.concatenate(test_preds, axis=0)
    Y_test_np = Y_test.numpy()
    
    # 6. Inverse Scaling & Metrics
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    preds_kw_flat = inverse_scale(preds_np.ravel())
    targets_kw_flat = inverse_scale(Y_test_np.ravel())
    
    rmse = np.sqrt(mean_squared_error(targets_kw_flat, preds_kw_flat))
    mae = mean_absolute_error(targets_kw_flat, preds_kw_flat)
    wape = np.sum(np.abs(targets_kw_flat - preds_kw_flat)) / np.sum(np.abs(targets_kw_flat)) * 100
    
    print("\n--- 15-Minute-Ahead Grid Transformer Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 7. Visualization
    plot_single_model_forecast(
        targets_kw_flat, 
        preds_kw_flat, 
        start_idx=0, 
        horizon=96,
        model_name="15-Min Grid Transformer", 
        save_path='transformer_15min_ahead_forecast.png'
    )
    
    # Cleanup temporary weight file
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

if __name__ == "__main__":
    main()