import torch
import torch.nn as nn

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
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
            print(f'   -> EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.model_save_path)


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10, model_save_path='best_model.pth'):
    """
    Standardized GPU training loop with AdamW optimizer, gradient clipping, 
    and Early Stopping. Unpacks two items (history, target) for Direct LSTM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=patience, model_save_path=model_save_path)
    
    for epoch in range(epochs):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        train_loss = 0.0
        
        # Unpacking TWO items: batch_x (history) and batch_y (target)
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Align shapes safely with .view()
            batch_size = batch_x.size(0)
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
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                
                batch_size = batch_x.size(0)
                loss = criterion(outputs.view(batch_size, -1), batch_y.view(batch_size, -1))
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Check Early Stopping
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Halting training.")
            break
            
    print(f"Training complete. Best Validation Loss: {early_stopping.best_loss:.4f}. Model saved to {model_save_path}.")
    
    # Return a tuple so main.py can unpack it as: trained_model, _ = train_model(...)
    return model, early_stopping.best_loss