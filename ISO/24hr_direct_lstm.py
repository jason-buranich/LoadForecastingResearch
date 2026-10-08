import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import DirectLSTM
from visualize import plot_single_model_forecast

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
    
    # Switched to L1Loss to align directly with WAPE
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Added Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience, model_save_path=model_save_path)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # Now unpacking 3 items
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
        print(f"Epoch {epoch+1}/{epochs} | Train (L1): {train_loss:.4f} | Val (L1): {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return model

def evaluate_direct_lstm(model, test_loader, target_col_idx, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch_x_hist, batch_x_fut, batch_y in test_loader:
            batch_x_hist, batch_x_fut, batch_y = batch_x_hist.to(device), batch_x_fut.to(device), batch_y.to(device)
            preds = model(batch_x_hist, batch_x_fut)
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
    
    print(f"--- Starting Standalone Direct LSTM Pipeline (With Future Covariates) ---")
    
    # 1. Unpack all three items 
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_val_hist, X_val_fut, Y_val       = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 2. Update DataLoaders for 3 items
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(TensorDataset(X_test_hist, X_test_fut, Y_test), batch_size=64, shuffle=False, num_workers=4)
    
    model_path = f'tuned_direct_lstm_{HORIZON}h.pth'
    
    # 3. Dynamic Dimension Mapping
    model = DirectLSTM(
        hist_input_dim=X_train_hist.shape[-1], 
        future_input_dim=X_train_fut.shape[-1],
        hidden_dim=128, 
        horizon=HORIZON, 
        num_layers=1,
        dropout=0.3214 
    )
    
    trained_model = train_direct_lstm(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=100, 
        lr=0.001836, 
        patience=10, 
        model_save_path=model_path
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, weights_only=True))
    trained_model.to(device)
    
    lstm_preds, lstm_targets = evaluate_direct_lstm(trained_model, test_loader, TARGET_IDX, device)
    
    plot_single_model_forecast(
        lstm_targets, 
        lstm_preds, 
        start_idx=0, 
        model_name="Tuned Direct LSTM", 
        save_path='direct_lstm_forecast.png'
    )

if __name__ == "__main__":
    main()