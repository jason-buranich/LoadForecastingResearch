import torch
from torch.utils.data import TensorDataset, DataLoader

def get_dataloaders(X_train_hist, X_train_fut, Y_train, X_val_hist, X_val_fut, Y_val, batch_size=32):
    # Pack three tensors into the dataset instead of two
    train_dataset = TensorDataset(X_train_hist, X_train_fut, Y_train)
    val_dataset = TensorDataset(X_val_hist, X_val_fut, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader