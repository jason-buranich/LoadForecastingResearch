import numpy as np
import torch

def create_safe_sequences(df, seq_len=96, horizon=96, target_idx=1, covariate_start_idx=2):
    """
    Extracts sliding windows ensuring sequences do not cross month/gap boundaries.
    """
    X_hist_list, X_future_list, Y_list = [], [], []
    
    # 1. Create temporary unscaled columns strictly for safe masking
    df_temp = df.copy()
    df_temp['Real_Year'] = df_temp.index.year
    df_temp['Real_Month'] = df_temp.index.month
    
    # 2. Process each specific Year-Month block independently
    for year in df_temp['Real_Year'].unique():
        for month in df_temp['Real_Month'].unique():
            
            # Isolate the continuous block
            mask = (df_temp['Real_Year'] == year) & (df_temp['Real_Month'] == month)
            
            # Drop ONLY the temporary columns, keeping the scaled 'Month' covariate intact
            month_data = df_temp[mask].drop(columns=['Real_Year', 'Real_Month']).values
            
            if len(month_data) < seq_len + horizon:
                continue
                
            for i in range(len(month_data) - seq_len - horizon + 1):
                x_seq = month_data[i : i + seq_len]
                x_fut = month_data[i + seq_len : i + seq_len + horizon, covariate_start_idx:] 
                y_seq = month_data[i + seq_len : i + seq_len + horizon, target_idx]
                
                X_hist_list.append(x_seq)
                X_future_list.append(x_fut)
                Y_list.append(y_seq)
                
    X_hist_tensor = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
    X_future_tensor = torch.tensor(np.array(X_future_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)
    
    return X_hist_tensor, X_future_tensor, Y_tensor