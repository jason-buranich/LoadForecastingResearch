import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import your pipeline modules
from data import train_df, val_df
from ISO.slidingWindow import create_safe_sequences
from models import Seq2SeqCovariateLSTM

# ==========================================
# 1. Global Data Generation
# (Done once to save I/O overhead during tuning)
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
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqCovariateLSTM(
        hist_input_dim=hist_input_dim, 
        future_input_dim=future_input_dim, 
        hidden_dim=hidden_dim, 
        horizon=HORIZON, 
        num_layers=num_layers
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience = 4
    counter = 0

    # Fast Training Loop (Cap at 25 epochs for tuning speed)
    for epoch in range(25):
        model.train()
        for batch_x_hist, batch_x_fut, batch_y in train_loader:
            batch_x_hist = batch_x_hist.to(device)
            batch_x_fut = batch_x_fut.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x_hist, batch_x_fut, y_target=batch_y, teacher_forcing_ratio=0.5)
            
            b_size = batch_x_hist.size(0)
            loss = criterion(outputs.view(b_size, -1), batch_y.view(b_size, -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x_hist, batch_x_fut, batch_y in val_loader:
                batch_x_hist = batch_x_hist.to(device)
                batch_x_fut = batch_x_fut.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x_hist, batch_x_fut, y_target=batch_y, teacher_forcing_ratio=0.5)
                
                b_size = batch_x_hist.size(0)
                loss = criterion(outputs.view(b_size, -1), batch_y.view(b_size, -1))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)

        # Optuna Pruning: Kill the trial if it's underperforming early
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
                
    return best_val_loss

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    # Use MedianPruner to automatically stop bad trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    print("\n--- Launching Seq2Seq Optuna Tuning ---")
    study.optimize(objective, n_trials=30)  # Adjust n_trials based on your time constraints
    
    print("\n==================================================")
    print("Optimization Complete.")
    print(f"Best Validation Loss: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==================================================")