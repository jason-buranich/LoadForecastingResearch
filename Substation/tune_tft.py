import os
import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import TensorDataset, DataLoader

# Import components from your existing pipeline
from data import train_df, val_df, scaler
from slidingWindow import create_safe_sequences
from models import GridTransformer

# 1-Hour Substation Configuration
HORIZON = 96              # Predict 96 steps ahead (24 hours)
SEQ_LEN = 96             # 96 intervals = 24 hours of history
TARGET_IDX = 1           
COVARIATE_START_IDX = 2  

def inverse_scale(data_flat, target_idx=TARGET_IDX):
    dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
    dummy[:, target_idx] = data_flat
    return scaler.inverse_transform(dummy)[:, target_idx]

def objective(trial):
    # 1. Hyperparameter Search Space
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    nheads = trial.suggest_categorical("nheads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NEW: Keep only the last 25% of the training data to massively speed up tuning
    split_idx = int(len(train_df) * 0.75)
    train_df_subset = train_df.iloc[split_idx:].copy()
    
    # 2. Slice Sequences (using the smaller subset)
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df_subset, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=32, shuffle=False, num_workers=4)

    # 3. Instantiate Model (Passing seq_length for positional encoding)
    model = GridTransformer(
        hist_input_dim=X_train_hist.shape[-1],
        future_input_dim=X_train_fut.shape[-1],
        hidden_dim=hidden_dim,
        horizon=HORIZON,
        nheads=nheads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=SEQ_LEN
    ).to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 1. Add this before your epoch loop
    scaler_amp = torch.amp.GradScaler('cuda')
    
    # 2. Update your training step inside the epoch loop
    for batch_x_hist, batch_x_fut, batch_y in train_loader:
        batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Cast operations to mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(batch_x_hist, batch_x_fut)
            batch_size = batch_x_hist.size(0)
            loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
        
        # Scale the loss and backpropagate
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()


    epochs = 10 # Kept relatively short for tuning
    
    for epoch in range(epochs):
        model.train()
        for batch_x_hist, batch_x_fut, batch_y in train_loader:
            batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x_hist, batch_x_fut)
            
            batch_size = batch_x_hist.size(0)
            loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # 4. Validation Loop to calculate WAPE
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch_x_hist, batch_x_fut, batch_y in val_loader:
                batch_x_hist, batch_x_fut = batch_x_hist.to(device), batch_x_fut.to(device)
                preds = model(batch_x_hist, batch_x_fut)
                predictions.append(preds.cpu().numpy())
                targets.append(batch_y.numpy())
                
        preds_arr = np.concatenate(predictions, axis=0)
        targets_arr = np.concatenate(targets, axis=0)
        
        preds_mw_flat = inverse_scale(preds_arr.ravel())
        targets_mw_flat = inverse_scale(targets_arr.ravel())
        
        val_wape = np.sum(np.abs(targets_mw_flat - preds_mw_flat)) / np.sum(np.abs(targets_mw_flat)) * 100
        
        # 5. Report to Optuna and evaluate pruning
        trial.report(val_wape, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_wape

def main():
    print("--- Starting Optuna Tuning Grid Transformer ---")
    
    # Configure MedianPruner to allow 5 startup trials before pruning, and 10 warmup epochs per trial
    # Allow 5 startup trials, but kill bad trials after just 3 epochs
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="24hr_tft_opt")
    
    # Run 20 trials
    study.optimize(objective, n_trials=20, timeout=3600)
    
    print("\n--- Tuning Complete ---")
    print(f"Best Trial Validation WAPE: {study.best_trial.value:.2f}%")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()