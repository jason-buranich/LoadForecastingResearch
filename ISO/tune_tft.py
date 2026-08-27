import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import train_df, val_df
from slidingWindow import create_safe_sequences
from models import GridTransformer

# ==========================================
# 1. Global Data Generation
# ==========================================
HORIZON = 24
SEQ_LEN = 168
TARGET_IDX = 4

print("Generating 3-item sequence tensors for GridTransformer tuning...")
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
    nheads = trial.suggest_categorical("nheads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # PyTorch requirement: hidden dimension must be divisible by number of heads
    if hidden_dim % nheads != 0:
        raise optuna.exceptions.TrialPruned()

    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the GridTransformer
    model = GridTransformer(
        hist_input_dim=hist_input_dim, 
        future_input_dim=future_input_dim,
        hidden_dim=hidden_dim, 
        horizon=HORIZON, 
        nheads=nheads, 
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # Cap at 15 epochs for rapid searching
    for epoch in range(15):
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

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break
                
    return best_val_loss

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    print("\n--- Launching Grid Transformer Optuna Tuning ---")
    study.optimize(objective, n_trials=20)
    
    print("\n==================================================")
    print("Optimization Complete.")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==================================================")