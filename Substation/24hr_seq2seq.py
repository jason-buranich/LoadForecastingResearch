import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the updated data.py
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import Seq2SeqCovariateLSTM
from visualize import plot_single_model_forecast

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, model_save_path='best_24hr_seq2seq.pth'):
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

def train_seq2seq(model, train_loader, val_loader, epochs=100, lr=0.0003038, weight_decay=1.539e-05, patience=10, model_save_path='best_24hr_seq2seq.pth'):
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
            # Passing ground truth y_target to enable teacher forcing during training
            outputs = model(batch_x_hist, batch_x_fut, y_target=batch_y, teacher_forcing_ratio=0.5595)
            
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
                
                # Removed redundant optimizer.zero_grad() and hardcoded forcing ratio to 0.0 for evaluation
                outputs = model(batch_x_hist, batch_x_fut, y_target=None, teacher_forcing_ratio=0.0)
                
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
    # 1. 24-Hour Substation Configuration
    HORIZON = 96             # Predict 96 steps ahead (24 hours)
    SEQ_LEN = 96             # 96 intervals = 24 hours of history
    TARGET_IDX = 1           # Load_MW is at index 1 before dropping Month
    COVARIATE_START_IDX = 2  # Future covariates start at index 2
    
    print("--- Starting 24-Hour-Ahead Seq2Seq LSTM Pipeline ---")
    
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
    
    model_path = 'best_24hr_seq2seq.pth'
    
    # 4. Instantiate Model
    model = Seq2SeqCovariateLSTM(
        hist_input_dim=X_train_hist.shape[-1],
        future_input_dim=X_train_fut.shape[-1],
        hidden_dim=64,
        horizon=HORIZON,
        num_layers=2,
        target_idx=0  # Required because 'Month' is dropped, putting Load_MW at index 0 in the history tensor
    )
    
    # Note: These hyperparameters are carried over from the 1hr script. 
    # You will likely need to tune them (via tune_seq2seq.py) for the 96-step horizon.
    trained_model = train_seq2seq(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.0039,
        weight_decay=1.567e-4,
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
            preds = trained_model(batch_x_hist, batch_x_fut, y_target=None, teacher_forcing_ratio=0.0)
            predictions.append(preds.cpu().numpy())
            targets.append(batch_y.numpy())
            
    # Shape of arrays will be (N, 96)
    preds_arr = np.concatenate(predictions, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    
    # 6. Inverse Scaling
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    # Calculate overall metrics on flattened arrays to evaluate the entire 24-hour horizon
    preds_mw_flat = inverse_scale(preds_arr.ravel())
    targets_mw_flat = inverse_scale(targets_arr.ravel())
    
    # 7. Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw_flat, preds_mw_flat))
    mae = mean_absolute_error(targets_mw_flat, preds_mw_flat)
    wape = np.sum(np.abs(targets_mw_flat - preds_mw_flat)) / np.sum(np.abs(targets_mw_flat)) * 100
    
    print("\n--- 24-Hour-Ahead Seq2Seq LSTM Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # Isolate the t+96 interval (index 95) to prevent overlapping visualizations
    t96_preds = inverse_scale(preds_arr[:, 95])
    t96_targets = inverse_scale(targets_arr[:, 95])
    
    plot_single_model_forecast(
        t96_targets, 
        t96_preds, 
        start_idx=0, 
        horizon=96,
        model_name="24-Hour Seq2Seq LSTM (t+96 step)", 
        save_path='seq2seq_24hr_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()