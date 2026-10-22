import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Import components from your updated data pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import Seq2Seq

# ==============================================================================
# 1. EARLY STOPPING (Seq2Seq)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, model_save_path='best_24hr_seq2seq.pth'):
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

# ==============================================================================
# 2. TRAINING LOOP (Seq2Seq)
# ==============================================================================
def train_seq2seq(model, train_loader, val_loader, epochs=100, lr=1e-3, weight_decay=1e-4, patience=15, model_save_path='best_24hr_seq2seq.pth'):
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
        print(f"Seq2Seq Epoch {epoch+1:02d}/{epochs} | Train (L1): {train_loss:.4f} | Val (L1): {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Seq2Seq Early stopping triggered.")
            break
            
    return model

# ==============================================================================
# 3. UTILITIES
# ==============================================================================
def flatten_for_tabular(X_hist, X_fut):
    N = X_hist.shape[0]
    hist_flat = X_hist.reshape(N, -1)
    fut_flat = X_fut.reshape(N, -1)
    return np.hstack((hist_flat, fut_flat))

def plot_combined_forecast(targets, preds_seq2seq, preds_xgb, start_idx=0, plot_steps=672, save_path='combined_weekly_forecast.png'):
    # Create an array of hours (each step is 15 mins or 0.25 hours)
    time_hours = np.arange(plot_steps) * 0.25
    
    plt.figure(figsize=(14, 7))
    plt.plot(time_hours, targets[start_idx:start_idx+plot_steps], label='Actual Load', color='black', linewidth=1.5)
    plt.plot(time_hours, preds_seq2seq[start_idx:start_idx+plot_steps], label='Seq2Seq (24h ahead)', color='red', linestyle='--', linewidth=1.5)
    plt.plot(time_hours, preds_xgb[start_idx:start_idx+plot_steps], label='XGBoost (24h ahead)', color='blue', linestyle='-.', linewidth=1.5)
    
    plt.title('1-Week Microgrid Load Forecast: Seq2Seq vs XGBoost')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Load (kW)')
    
    # Set x-ticks to display every 24 hours (1 day) up to 168 hours
    plt.xticks(np.arange(0, 169, 24))
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Combined graph saved to: {save_path}")

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def main():
    # Configuration
    HORIZON = 96              
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2  
    
    print("--- Starting Combined 24-Hour-Ahead Forecasting Pipeline ---")
    
    # ---------------------------------------------------------
    # Data Preparation
    # ---------------------------------------------------------
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )

    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
    
    # ---------------------------------------------------------
    # Model 1: Seq2Seq Training & Prediction
    # ---------------------------------------------------------
    print("\n[1/2] Training Deep Sequence Model (Seq2Seq)...")
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test_hist, X_test_fut, Y_test), batch_size=64, shuffle=False, num_workers=0)
    
    model_path = 'best_24hr_seq2seq.pth'
    seq_model = Seq2Seq(
        hist_input_dim=X_train_hist.shape[-1], 
        future_input_dim=X_train_fut.shape[-1],
        horizon=HORIZON,
        hidden_dim=32,  
        num_layers=1,   
        dropout=0.1     
    )
    
    trained_seq_model = train_seq2seq(
        model=seq_model, train_loader=train_loader, val_loader=val_loader,
        epochs=100, lr=1e-4, weight_decay=2.6e-6, patience=10, model_save_path=model_path
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_seq_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    trained_seq_model.to(device)
    trained_seq_model.eval()
    
    seq_predictions, seq_targets = [], []
    with torch.no_grad():
        for batch_x_hist, batch_x_fut, batch_y in test_loader:
            batch_x_hist, batch_x_fut = batch_x_hist.to(device), batch_x_fut.to(device)
            preds = trained_seq_model(batch_x_hist, batch_x_fut)
            seq_predictions.append(preds.cpu().numpy())
            seq_targets.append(batch_y.numpy())
            
    preds_seq_np = np.concatenate(seq_predictions, axis=0)
    targets_np = np.concatenate(seq_targets, axis=0) # Shared target array
    
    # Evaluate Seq2Seq
    preds_seq_kw_flat = inverse_scale(preds_seq_np.ravel())
    targets_kw_flat = inverse_scale(targets_np.ravel())
    seq_rmse = np.sqrt(mean_squared_error(targets_kw_flat, preds_seq_kw_flat))
    seq_mae = mean_absolute_error(targets_kw_flat, preds_seq_kw_flat)
    seq_wape = np.sum(np.abs(targets_kw_flat - preds_seq_kw_flat)) / np.sum(np.abs(targets_kw_flat)) * 100
    print(f"Seq2Seq Metrics -> RMSE: {seq_rmse:.2f} | MAE: {seq_mae:.2f} | WAPE: {seq_wape:.2f}%")

    # ---------------------------------------------------------
    # Model 2: XGBoost Training & Prediction
    # ---------------------------------------------------------
    print("\n[2/2] Training Tabular Tree Ensemble (XGBoost)...")
    combined_train_df = pd.concat([train_df, val_df]).sort_index()
    X_xgb_train_hist, X_xgb_train_fut, Y_xgb_train = create_safe_sequences(
        combined_train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    X_train_flat = flatten_for_tabular(X_xgb_train_hist.numpy(), X_xgb_train_fut.numpy())
    X_test_flat = flatten_for_tabular(X_test_hist.numpy(), X_test_fut.numpy())
    
    xg_base = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, tree_method='hist', n_jobs=1)
    multi_xgb = MultiOutputRegressor(xg_base, n_jobs=4)
    multi_xgb.fit(X_train_flat, Y_xgb_train.numpy())
    
    preds_xgb_np = multi_xgb.predict(X_test_flat)
    
    # Evaluate XGBoost
    preds_xgb_kw_flat = inverse_scale(preds_xgb_np.ravel())
    xgb_rmse = np.sqrt(mean_squared_error(targets_kw_flat, preds_xgb_kw_flat))
    xgb_mae = mean_absolute_error(targets_kw_flat, preds_xgb_kw_flat)
    xgb_wape = np.sum(np.abs(targets_kw_flat - preds_xgb_kw_flat)) / np.sum(np.abs(targets_kw_flat)) * 100
    print(f"XGBoost Metrics -> RMSE: {xgb_rmse:.2f} | MAE: {xgb_mae:.2f} | WAPE: {xgb_wape:.2f}%")

    # ---------------------------------------------------------
    # Visualization: Plotting the t+96 Step (24-Hour Ahead)
    # ---------------------------------------------------------
    print("\nGenerating combined forecast visualization...")
    t96_targets = inverse_scale(targets_np[:, 95])
    t96_preds_seq = inverse_scale(preds_seq_np[:, 95])
    t96_preds_xgb = inverse_scale(preds_xgb_np[:, 95])
    
    plot_combined_forecast(
        targets=t96_targets, 
        preds_seq2seq=t96_preds_seq, 
        preds_xgb=t96_preds_xgb, 
        start_idx=0, 
        plot_steps=672, 
        save_path='combined_1week_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()