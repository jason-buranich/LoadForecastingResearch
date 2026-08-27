import optuna
import torch
from torch.utils.data import TensorDataset, DataLoader
from data import train_df, val_df, test_df
from slidingWindow import create_safe_sequences
from models import DirectLSTM
from trainingLoop import train_model

# Constants for the study
HORIZON = 24
SEQ_LEN = 168
TARGET_IDX = 4
EPOCHS_PER_TRIAL = 15

def objective(trial):
    """
    Optuna objective function to search for optimal DirectLSTM hyperparameters.
    """
    # 1. Define the Hyperparameter Search Space
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # 2. FIX: Unpack all 3 items (ignore the future covariates with '_')
    X_train_hist, _, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_val_hist, _, Y_val     = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 3. FIX: Create 2-item DataLoaders directly for the DirectLSTM
    train_dataset = TensorDataset(X_train_hist, Y_train)
    val_dataset = TensorDataset(X_val_hist, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 4. Initialize the Model with trial architectures
    input_dim = X_train_hist.shape[-1]
    model = DirectLSTM(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        horizon=HORIZON, 
        num_layers=num_layers, 
        dropout=dropout
    )
    
    # 5. Train the Model
    _, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS_PER_TRIAL,
        lr=lr,
        model_save_path=f"trial_{trial.number}_model.pth"
    )
    
    return best_val_loss

if __name__ == "__main__":
    print("--- Starting Optuna Hyperparameter Optimization ---")
    
    study = optuna.create_study(direction="minimize", study_name="CAISO_DirectLSTM_Tuning")
    study.optimize(objective, n_trials=20)
    
    print("\n--- Optuna Search Complete ---")
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")