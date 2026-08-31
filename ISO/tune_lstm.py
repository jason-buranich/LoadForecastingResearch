import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import train_df, val_df
from ISO.slidingWindow import create_safe_sequences
from models import DirectLSTM

# ==========================================
# 1. Global Data Generation
# ==========================================
HORIZON = 24
SEQ_LEN = 168
TARGET_IDX = 4

print("Generating 3-item sliding window tensors for tuning...")
X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
X_val_hist, X_val_fut, Y_val       = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)

hist_input_dim = X_train_hist.shape[-1]
future_input_dim = X_train_fut.shape[-1]

# ==========================================
# 2. Optuna Objective Function
# ==========================================
def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Create 3-item DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the NEW 3-item DirectLSTM
    model = DirectLSTM(
        hist_input_dim=hist_input_dim, 
        future_input_dim=future_input_dim,
        hidden_dim=hidden_dim, 
        horizon=HORIZON, 
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Use L1 Loss to directly optimize for WAPE
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # Fast Training Loop (Cap at 20 epochs for tuning speed)
    for epoch in range(20):
        model.train()
        for batch_x_hist, batch_x_fut, batch_y in train_loader:
            batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x_hist, batch_x_fut)
            
            b_size = batch_x_hist.size(0)
            loss = criterion(outputs.view(b_size, -1), batch_y.view(b_size, -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x_hist, batch_x_fut, batch_y in val_loader:
                batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
                outputs = model(batch_x_hist, batch_x_fut)
                
                b_size = batch_x_hist.size(0)
                loss = criterion(outputs.view(b_size, -1), batch_y.view(b_size, -1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Optuna Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 4:
                break
                
    return best_val_loss

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    print("\n--- Launching 3-Item DirectLSTM Optuna Tuning ---")
    study.optimize(objective, n_trials=20)
    
    print("\n==================================================")
    print("Optimization Complete.")
    print(f"Best Validation Loss (L1): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==================================================")