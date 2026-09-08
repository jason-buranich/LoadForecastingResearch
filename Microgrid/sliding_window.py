import numpy as np
import torch

def create_safe_sequences(df, seq_len=96, horizon=96, target_idx=1, covariate_start_idx=2):
    """
    Extracts sliding windows ensuring sequences do not cross month/gap boundaries.
    Automatically adjusts column indices to account for the dropped 'Month' column.
    """
    X_hist_list, X_future_list, Y_list = [], [], []
    
    # 1. Create a temporary Year column from the Datetime index
    df_temp = df.copy()
    df_temp['Year'] = df_temp.index.year
    
    # 2. Process each specific Year-Month block independently to avoid train/val/test gaps
    for year in df_temp['Year'].unique():
        for month in df_temp['Month'].unique():
            
            # Isolate the continuous block and drop the temporal grouping columns
            mask = (df_temp['Year'] == year) & (df_temp['Month'] == month)
            month_data = df_temp[mask].drop(columns=['Month', 'Year']).values
            
            # Skip if the block is shorter than one full sequence + horizon
            if len(month_data) < seq_len + horizon:
                continue
                
            for i in range(len(month_data) - seq_len - horizon + 1):
                # 96-step history (all features)
                x_seq = month_data[i : i + seq_len]
                
                # 96-step future covariates (hour, dayofweek, weather)
                x_fut = month_data[i + seq_len : i + seq_len + horizon, covariate_start_idx - 1:] 
                
                # 96-step future target (Load_MW)
                y_seq = month_data[i + seq_len : i + seq_len + horizon, target_idx - 1]
                
                X_hist_list.append(x_seq)
                X_future_list.append(x_fut)
                Y_list.append(y_seq)
                
    X_hist_tensor = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
    X_future_tensor = torch.tensor(np.array(X_future_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)
    
    return X_hist_tensor, X_future_tensor, Y_tensor