import numpy as np
import torch

def create_safe_sequences(df, seq_len=168, horizon=24, target_idx=4, covariate_start_idx=5):
    """
    Extracts sliding windows ensuring sequences do not cross month boundaries.
    Automatically adjusts column indices to account for the dropped 'Month' column.
    """
    X_hist_list, X_future_list, Y_list = [], [], []
    
    # Process each month independently to avoid gaps
    for month in df['Month'].unique():
        # Dropping a column shifts all subsequent indices left by 1
        month_data = df[df['Month'] == month].drop(columns=['Month']).values
        
        if len(month_data) < seq_len + horizon:
            continue
            
        for i in range(len(month_data) - seq_len - horizon + 1):
            # 168-hour history (all features)
            x_seq = month_data[i : i + seq_len]
            
            # 24-hour future forecast (adjusted for dropped 'Month' column)
            x_fut = month_data[i + seq_len : i + seq_len + horizon, covariate_start_idx - 1:] 
            
            # 24-hour future target (adjusted for dropped 'Month' column)
            y_seq = month_data[i + seq_len : i + seq_len + horizon, target_idx - 1]
            
            X_hist_list.append(x_seq)
            X_future_list.append(x_fut)
            Y_list.append(y_seq)
            
    X_hist_tensor = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
    X_future_tensor = torch.tensor(np.array(X_future_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)
    
    return X_hist_tensor, X_future_tensor, Y_tensor