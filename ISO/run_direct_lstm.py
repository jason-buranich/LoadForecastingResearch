import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import DirectLSTM
from visualize import plot_day_ahead_forecast

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, model_save_path='best_model.pth'):
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

def train_direct_lstm(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10, model_save_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience, model_save_path=model_save_path)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # 2-item unpacking for Direct LSTM
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            batch_size = batch_x.size(0)
            loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                batch_size = batch_x.size(0)
                loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return model

def evaluate_direct_lstm(model, test_loader, target_col_idx, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            predictions.append(preds.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
            
    predictions_flat = np.concatenate(predictions, axis=0).flatten()
    targets_flat = np.concatenate(targets, axis=0).flatten()
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, target_col_idx] = data_flat
        return scaler.inverse_transform(dummy)[:, target_col_idx]

    preds_mw = inverse_scale(predictions_flat)
    targets_mw = inverse_scale(targets_flat)
    
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- Direct LSTM Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    return preds_mw, targets_mw

def main():
    HORIZON = 24
    SEQ_LEN = 168
    TARGET_IDX = 4
    
    print(f"--- Starting Standalone Direct LSTM Pipeline ---")
    
    # 1. Generate tensors (ignoring future covariates)
    X_train_hist, _, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_val_hist, _, Y_val     = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, _, Y_test   = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 2. DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_hist, Y_train), batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(X_val_hist, Y_val), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(TensorDataset(X_test_hist, Y_test), batch_size=64, shuffle=False, num_workers=4)
    
    # 3. Initialize and Train
    model_path = f'tuned_direct_lstm_{HORIZON}h.pth'
    model = DirectLSTM(input_dim=X_train_hist.shape[-1], hidden_dim=256, horizon=HORIZON, num_layers=1)
    
    trained_model = train_direct_lstm(
        model=model, train_loader=train_loader, val_loader=val_loader, 
        epochs=100, lr=0.000394, patience=10, model_save_path=model_path
    )
    
    # 4. Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, weights_only=True))
    trained_model.to(device)
    
    lstm_preds, lstm_targets = evaluate_direct_lstm(trained_model, test_loader, TARGET_IDX, device)
    plot_day_ahead_forecast(lstm_targets, lstm_preds, lstm_preds, start_idx=0, save_path='direct_lstm_forecast.png')

if __name__ == "__main__":
    main()