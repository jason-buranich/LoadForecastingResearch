import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=64):
    """
    Converts gap-safe sequence tensors into PyTorch DataLoaders optimized 
    for a 10 CPU / 1 GPU HPC environment.
    """
    
    # 1. Wrap the raw tensors into a TensorDataset
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    # 2. Build the HPC-Optimized Training Loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,        # Leverages 8 of your 10 CPUs to parallelize data loading
        pin_memory=True,      # Drastically speeds up CPU-to-GPU memory transfer
        drop_last=True        # Drops incomplete batches to stabilize sequence learning
    )
    
    # 3. Build the Validation Loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,        # Use fewer workers for validation to save overhead
        pin_memory=True
    )
    
    return train_loader, val_loader