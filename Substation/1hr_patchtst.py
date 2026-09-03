import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the updated data.py
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import PatchTST
from visualize import plot_single_model_forecast

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, model_save_path='best_1hr_patchtst.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.model_save_path)

def train_patchtst(model, train_loader, val_loader, epochs=100, lr=0.000218, weight_decay=7.1838e-05, patience=10, model_save_path='best_1hr_patchtst.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience, model_save_path=model_save_path)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x_hist, batch_x_fut, batch_y in train_loader:
            batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x_hist, batch_x_fut)
            
            batch_size = batch_x_hist.size(0)
            loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x_hist, batch_x_fut, batch_y in val_loader:
                batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
                outputs = model(batch_x_hist, batch_x_fut)
                
                batch_size = batch_x_hist.size(0)
                loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train (L1): {train_loss:.4f} | Val (L1): {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return model

def main():
    # 1. 1-Hour Substation Configuration
    HORIZON = 4              # Predict 4 steps ahead (1 hour)
    SEQ_LEN = 96             # 96 intervals = 24 hours of history
    PATCH_LEN = 8          
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2  
    
    print("--- Starting 1-Hour-Ahead PatchTST Pipeline ---")
    
    # 2. Slice Sequences
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 3. DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=64, shuffle=False, num_workers=4)
    test_loader  = DataLoader(TensorDataset(X_test_hist, X_test_fut, Y_test), batch_size=64, shuffle=False, num_workers=4)
    
    model_path = 'best_1hr_patchtst.pth'
    
    # 4. Instantiate Model 
    model = PatchTST(
        hist_input_dim=X_train_hist.shape[-1],
        future_input_dim=X_train_fut.shape[-1],
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        horizon=HORIZON,
        hidden_dim=128,
        nheads=8,
        num_layers=2,
        dropout=0.224
    )
    
    trained_model = train_patchtst(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.000218,
        weight_decay=7.1838e-05,
        patience=10,
        model_save_path=model_path
    )
    
    # 5. Evaluate on Test Set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    trained_model.to(device)
    trained_model.eval()
    
    predictions, targets = [], []
    with torch.no_grad():
        for batch_x_hist, batch_x_fut, batch_y in test_loader:
            batch_x_hist, batch_x_fut = batch_x_hist.to(device), batch_x_fut.to(device)
            preds = trained_model(batch_x_hist, batch_x_fut)
            predictions.append(preds.cpu().numpy())
            targets.append(batch_y.numpy())
            
    # Shape of arrays will be (N, 4)
    preds_arr = np.concatenate(predictions, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    
    # 6. Inverse Scaling function
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    # Calculate overall metrics on flattened arrays to evaluate the entire 1-hour horizon
    preds_mw_flat = inverse_scale(preds_arr.ravel())
    targets_mw_flat = inverse_scale(targets_arr.ravel())
    
    # 7. Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw_flat, preds_mw_flat))
    mae = mean_absolute_error(targets_mw_flat, preds_mw_flat)
    wape = np.sum(np.abs(targets_mw_flat - preds_mw_flat)) / np.sum(np.abs(targets_mw_flat)) * 100
    
    print("\n--- 1-Hour-Ahead PatchTST Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # Isolate the t+4 interval (index 3) to prevent overlapping visualizations
    t4_preds = inverse_scale(preds_arr[:, 3])
    t4_targets = inverse_scale(targets_arr[:, 3])
    
    plot_single_model_forecast(
        t4_targets, 
        t4_preds, 
        start_idx=0, 
        horizon=96,
        model_name="1-Hour PatchTST (t+4 step)", 
        save_path='patchtst_1hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()