import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import os

from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from models import GridTransformer
from visualize import plot_single_model_forecast

def set_seed(seed=42):
    """Locks all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, model_save_path='best_hour_ahead_tft.pth'):
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

def train_tft(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10, model_save_path='best_hour_ahead_tft.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # L1 Loss for direct WAPE alignment
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
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
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss (L1): {train_loss:.4f} | Val Loss (L1): {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return model

def seed_worker(worker_id):
    """Ensures each dataloader worker gets a unique, but deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    set_seed(42) 
    
    # 1. Hour-Ahead Configuration
    HORIZON = 1
    SEQ_LEN = 24
    TARGET_IDX = 4
    
    print(f"--- Starting 1-Hour-Ahead Grid Transformer Pipeline ---")
    
    # 2. Reslice Sequences
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_val_hist, X_val_fut, Y_val       = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    # 3. DataLoaders
    train_dataset = TensorDataset(X_train_hist, X_train_fut, Y_train)
    val_dataset = TensorDataset(X_val_hist, X_val_fut, Y_val)
    test_dataset = TensorDataset(X_test_hist, X_test_fut, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    
    model_path = 'best_hour_ahead_tft.pth'
    
    # 4. Instantiate Grid Transformer (Passing dynamic horizon and seq_length)
    model = GridTransformer(
        hist_input_dim=X_train_hist.shape[-1], 
        future_input_dim=X_train_fut.shape[-1], 
        hidden_dim=256,   
        horizon=HORIZON,
        nheads=2,        
        num_layers=1,
        dropout=0.0514,
        seq_length=SEQ_LEN
    )
    
    # 5. Train Model
    trained_model = train_tft(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=50, 
        lr=0.0001051, 
        patience=10,
        model_save_path=model_path
    )
    
    # 6. Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, weights_only=True))
    trained_model.to(device)
    trained_model.eval()
    
    predictions, targets = [], []
    with torch.no_grad():
        for batch_x_hist, batch_x_fut, batch_y in test_loader:
            batch_x_hist, batch_x_fut = batch_x_hist.to(device), batch_x_fut.to(device)
            preds = trained_model(batch_x_hist, batch_x_fut)
            predictions.append(preds.cpu().numpy())
            targets.append(batch_y.numpy())
            
    preds_flat = np.concatenate(predictions, axis=0).flatten()
    targets_flat = np.concatenate(targets, axis=0).flatten()
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    preds_mw = inverse_scale(preds_flat)
    targets_mw = inverse_scale(targets_flat)
    
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- 1-Hour-Ahead Grid Transformer Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    plot_single_model_forecast(
        targets_mw, 
        preds_mw, 
        start_idx=0, 
        model_name="1-Hour Grid Transformer", 
        save_path='tft_hour_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()