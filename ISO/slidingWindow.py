import torch
import numpy as np

def create_safe_sequences(df, seq_len=168, horizon=24, target_idx=4):
    """
    Builds sliding window tensors month-by-month to prevent 
    stitching discontinuous sequences across calendar gaps.
    
    Args:
        df: Pandas DataFrame containing the scaled features and a 'Month' column.
        seq_len: Number of historical hours to use as input (default 168 for 1 week).
        horizon: Number of future hours to predict (default 24).
        target_idx: The column index of the target variable after dropping 'Month' 
                    (default 4 for CAISO total load).
    """
    X, Y = [], []
    
    # Iterate through each unique month independently to avoid temporal gaps
    for month in df['Month'].unique():
        # Drop the 'Month' column before converting to a NumPy array
        month_data = df[df['Month'] == month].drop(columns=['Month']).values
        
        # Only process if the month has enough hours for at least one full sequence
        if len(month_data) > seq_len + horizon:
            for i in range(len(month_data) - seq_len - horizon):
                # Extract the historical input sequence (all features)
                x_seq = month_data[i : i + seq_len]
                
                # Extract the future target sequence (only the specific target column)
                y_seq = month_data[i + seq_len : i + seq_len + horizon, target_idx]
                
                X.append(x_seq)
                Y.append(y_seq)
                
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)