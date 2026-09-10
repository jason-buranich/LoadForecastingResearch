import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from your updated data pipeline
from data import train_df, val_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

# ==============================================================================
# 1. EARLY STOPPING & MODEL ARCHITECTURE
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, model_save_path='best_15min_transformer.pth'):
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

class GridTransformer(nn.Module):
    def __init__(self, hist_features, fut_features, seq_len=96, d_model=64, n_heads=4, num_layers=2):
        super(GridTransformer, self).__init__()
        self.embedding = nn.Linear(hist_features, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model + fut_features, 1)

    def forward(self, x_hist, x_fut):
        x = self.embedding(x_hist) + self.pos_encoder
        encoded_seq = self.transformer(x)
        context_vector = encoded_seq[:, -1, :]
        fut_flat = x_fut.view(x_fut.size(0), -1)
        combined = torch.cat((context_vector, fut_flat), dim=1)
        return self.fc_out(combined)

# ==============================================================================
# 2. TRAINING LOOP
# ==============================================================================
def train_transformer(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, model_save_path='best_15min_transformer.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        print(f"Epoch {epoch+1:02d}/{epochs} | Train (L1): {train_loss:.4f} | Val (L1): {val_loss:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return model

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================
def main():
    # Microgrid Configuration
    HORIZON = 1              
    SEQ_LEN = 96             
    TARGET_IDX = 1           
    COVARIATE_START_IDX = 2  
    
    print("--- Starting 15-Minute-Ahead Grid Transformer Pipeline ---")
    
    # Slice Sequences (Separating train and val to trigger early stopping properly)
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_val_hist, X_val_fut, Y_val = create_safe_sequences(
        val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # DataLoaders (num_workers=0 to prevent Docker OpenMP deadlocks)
    train_loader = DataLoader(TensorDataset(X_train_hist, X_train_fut, Y_train), batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val_hist, X_val_fut, Y_val), batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test_hist, X_test_fut, Y_test), batch_size=64, shuffle=False, num_workers=0)
    
    model_path = 'best_15min_transformer.pth'
    hist_features = X_train_hist.shape[-1]
    fut_features = X_train_fut.shape[-1]
    
    # Instantiate Model
    model = GridTransformer(
        hist_features=hist_features, 
        fut_features=fut_features, 
        seq_len=SEQ_LEN
    )
    
    # Train
    trained_model = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=1e-3,
        patience=15, # Extended patience based on your request
        model_save_path=model_path
    )
    
    # Evaluate on Test Set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
    
    # Inverse Scaling
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]

    preds_kw = inverse_scale(preds_flat)
    targets_kw = inverse_scale(targets_flat)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(targets_kw, preds_kw))
    mae = mean_absolute_error(targets_kw, preds_kw)
    wape = np.sum(np.abs(targets_kw - preds_kw)) / np.sum(np.abs(targets_kw)) * 100
    
    print("\n--- 15-Minute-Ahead Grid Transformer Metrics (kW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    plot_single_model_forecast(
        targets_kw, 
        preds_kw, 
        start_idx=0, 
        horizon=96,
        model_name="15-Min Grid Transformer", 
        save_path='transformer_15min_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()