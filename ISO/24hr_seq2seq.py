import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import your pipeline modules
from data import train_df, val_df, test_df, scaler
from ISO.slidingWindow import create_safe_sequences
from models import Seq2SeqCovariateLSTM
from trainingLoop import EarlyStopping
from evaluate import evaluate_predictions
from visualize import plot_single_model_forecast

def train_seq2seq_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10, model_save_path='best_seq2seq.pth'):
    """
    Dedicated training loop for Seq2Seq architectures. 
    Unpacks three items (history, future, target) from the dataloader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience, model_save_path=model_save_path)
    
    for epoch in range(epochs):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        train_loss = 0.0
        
        # Unpack THREE items
        for batch_x_hist, batch_x_fut, batch_y in train_loader:
            batch_x_hist = batch_x_hist.to(device)
            batch_x_fut = batch_x_fut.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Pass history and future weather forecast to the model
            outputs = model(batch_x_hist, batch_x_fut, y_target=batch_y, teacher_forcing_ratio=0.5)
            
            batch_size = batch_x_hist.size(0)
            loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        # ==========================
        # Validation Phase
        # ==========================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x_hist, batch_x_fut, batch_y in val_loader:
                batch_x_hist = batch_x_hist.to(device)
                batch_x_fut = batch_x_fut.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x_hist, batch_x_fut, y_target=batch_y, teacher_forcing_ratio=0.0)
                
                batch_size = batch_x_hist.size(0)
                loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Halting training.")
            break
            
    print(f"Training complete. Best Validation Loss: {early_stopping.best_loss:.4f}. Model saved to {model_save_path}.")
    return model, early_stopping.best_loss


def main():
    HORIZON = 24
    SEQ_LEN = 168
    TARGET_IDX = 4
    
    print(f"--- Starting Seq2Seq Pipeline: {HORIZON}h Horizon, {SEQ_LEN}h Lookback ---")
    
    # 1. Generate sliding window tensors
    print("Generating sliding window tensors...")
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_val_hist, X_val_fut, Y_val       = create_safe_sequences(val_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 2. Create PyTorch DataLoaders (Pack all 3 items into the dataset)
    print("Preparing 3-Item DataLoaders...")
    train_dataset = TensorDataset(X_train_hist, X_train_fut, Y_train)
    val_dataset = TensorDataset(X_val_hist, X_val_fut, Y_val)
    test_dataset = TensorDataset(X_test_hist, X_test_fut, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. Initialize Seq2Seq Covariate LSTM Model
    hist_input_dim = X_train_hist.shape[-1]
    future_input_dim = X_train_fut.shape[-1]
    
    model = Seq2SeqCovariateLSTM(
        hist_input_dim=hist_input_dim, 
        future_input_dim=future_input_dim, 
        hidden_dim=256, 
        horizon=HORIZON, 
        num_layers=1
    )
    model_path = f'tuned_seq2seq_lstm_{HORIZON}h.pth'
    
    # 4. Execute Training Loop
    print("Launching Seq2Seq training loop...")
    trained_model, _ = train_seq2seq_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=100, 
        lr=3e-4, # You may want to run Optuna again for this new architecture later
        patience=10,
        model_save_path=model_path
    )
    
    # 5. Load Best Weights and Evaluate
    print("\nEvaluating best model on test dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load(model_path, weights_only=True))
    trained_model.to(device)
    
    # Call the 3-item evaluate function from evaluate.py
    lstm_metrics, lstm_preds, lstm_targets = evaluate_predictions(
        trained_model, 
        test_loader, 
        scaler=scaler, 
        target_col_idx=TARGET_IDX, 
        device=device
    )

    plot_single_model_forecast(
        lstm_targets, 
        lstm_preds, 
        start_idx=0, 
        model_name="Seq2Seq LSTM", 
        save_path='seq2seq_forecast.png'
    )

if __name__ == "__main__":
    main()