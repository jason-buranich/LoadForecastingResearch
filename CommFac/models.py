import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import random

# ==============================================================================
# 1. BASELINE: SEASONAL NAIVE / PERSISTENCE
# ==============================================================================
class SeasonalNaiveBaseline:
    def __init__(self, seasonal_lag=96):
        self.seasonal_lag = seasonal_lag

    def predict(self, X, target_idx=0): 
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim == 3:
            load_history = X[:, :, target_idx]
        else:
            load_history = X
            
        return load_history[:, -self.seasonal_lag:]


# ==============================================================================
# 2. TABULAR MODELS: RANDOM FOREST & LIGHTGBM
# ==============================================================================
def get_tabular_models(horizon=96, random_state=42):
    mlr = Ridge(alpha=500.0, solver='lsqr')
    
    rf_base = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=20,
        max_features=0.3,
        random_state=random_state,
        n_jobs=4  
    )
    
    lgbm_base = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=random_state,
        n_jobs=4, 
        verbosity=-1
    )
    
    if horizon == 1:
        rf = rf_base
        lgbm = lgbm_base
    else:
        rf = MultiOutputRegressor(rf_base)
        lgbm = MultiOutputRegressor(lgbm_base)
        
    return mlr, rf, lgbm


# ==============================================================================
# 3. PYTORCH: DIRECT MULTI-STEP LSTM
# ==============================================================================
class DirectLSTM(nn.Module):
    """
    Fixed: Preserves temporal dimension of future covariates and applies a 
    time-distributed projection instead of flattening.
    """
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim=64, horizon=96, num_layers=2, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=hist_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Time-distributed projection (applied per step)
        self.fc = nn.Linear(hidden_dim + future_input_dim, 1)

    def forward(self, x_hist, x_future):
        _, (hn, _) = self.lstm(x_hist)
        last_hidden = hn[-1]  
        
        # Expand context vector across the entire horizon: (Batch, Horizon, Hidden)
        context_expanded = last_hidden.unsqueeze(1).expand(-1, self.horizon, -1)
        
        if x_future is not None and x_future.size(1) > 0:
            if x_future.size(1) < self.horizon:
                x_fut_aligned = x_future.mean(dim=1, keepdim=True).expand(-1, self.horizon, -1)
            else:
                x_fut_aligned = x_future[:, :self.horizon, :]
                
            # Fuse step-by-step
            combined = torch.cat((context_expanded, x_fut_aligned), dim=-1)
        else:
            combined = context_expanded
            
        out = self.fc(combined).squeeze(-1) # Squeeze final dimension to return (Batch, Horizon)
        return out


# ==============================================================================
# 4. PYTORCH: ENCODER-DECODER LSTM (Seq2Seq)
# ==============================================================================
class Seq2Seq(nn.Module):
    def __init__(self, hist_input_dim, future_input_dim, horizon=96, hidden_dim=64, num_layers=2, dropout=0.1, target_idx=1):
        super(Seq2Seq, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.target_idx = target_idx
        
        self.encoder = nn.LSTM(
            input_size=hist_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = nn.LSTM(
            input_size=1 + future_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist, x_fut, y_true=None, teacher_forcing_ratio=0.0):
        batch_size = x_hist.size(0)
        
        _, (hidden, cell) = self.encoder(x_hist)
        outputs = torch.zeros(batch_size, self.horizon, 1).to(x_hist.device)
        
        # NOTE: Ensure self.target_idx points to the actual load consumption feature
        dec_input = x_hist[:, -1, self.target_idx].unsqueeze(1).unsqueeze(2)
        
        for t in range(self.horizon):
            if x_fut is not None and x_fut.size(1) > 0:
                t_idx = t if t < x_fut.size(1) else -1
                covariates_t = x_fut[:, t_idx, :].unsqueeze(1)
                dec_input_combined = torch.cat((dec_input, covariates_t), dim=2)
            else:
                dec_input_combined = dec_input
            
            out, (hidden, cell) = self.decoder(dec_input_combined, (hidden, cell))
            pred = self.fc_out(out)
            outputs[:, t, :] = pred.squeeze(1)
            
            if y_true is not None and random.random() < teacher_forcing_ratio:
                if y_true.dim() == 2:
                    dec_input = y_true[:, t].unsqueeze(1).unsqueeze(2)
                else:
                    dec_input = y_true[:, t, :].unsqueeze(1)
            else:
                dec_input = pred
            
        return outputs.squeeze(-1)


# ==============================================================================
# 5. PYTORCH: GRID TRANSFORMER
# ==============================================================================
class GridTransformer(nn.Module):
    """
    Fixed: Preserves temporal dimension of future covariates and applies a 
    time-distributed projection instead of flattening.
    """
    def __init__(self, hist_input_dim, future_input_dim, seq_len=96, horizon=96, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(GridTransformer, self).__init__()
        self.horizon = horizon
        
        self.embedding = nn.Linear(hist_input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=128, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Time-distributed output layer
        self.fc_out = nn.Linear(d_model + future_input_dim, 1)

    def forward(self, x_hist, x_future, **kwargs):
        x = self.embedding(x_hist) + self.pos_encoder[:, :x_hist.size(1), :]
        encoded_seq = self.transformer(x)
        context_vector = encoded_seq[:, -1, :] # Grab final encoded state
        
        # Expand context across horizon
        context_expanded = context_vector.unsqueeze(1).expand(-1, self.horizon, -1)
        
        if x_future is not None and x_future.size(1) > 0:
            if x_future.size(1) < self.horizon:
                x_fut_aligned = x_future.mean(dim=1, keepdim=True).expand(-1, self.horizon, -1)
            else:
                x_fut_aligned = x_future[:, :self.horizon, :]
                
            combined = torch.cat((context_expanded, x_fut_aligned), dim=-1)
        else:
            combined = context_expanded
            
        return self.fc_out(combined).squeeze(-1)


# ==============================================================================
# 6. PYTORCH: PATCH TIME SERIES TRANSFORMER
# ==============================================================================
class PatchTST(nn.Module):
    """
    Fixed: Reduces parameter bloat by extracting a unified context vector from 
    the patches and mapping it to the horizon step-by-step.
    """
    def __init__(self, hist_input_dim, future_input_dim, seq_len=96, patch_len=16, horizon=96, hidden_dim=128, nheads=4, num_layers=2, dropout=0.2):
        super(PatchTST, self).__init__()
        self.horizon = horizon
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len 
        
        self.patch_embedding = nn.Linear(patch_len * hist_input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project flattened patches down to a single context vector
        self.context_proj = nn.Linear(self.num_patches * hidden_dim, hidden_dim)
        
        # Time-distributed fusion layer
        self.fc_out = nn.Linear(hidden_dim + future_input_dim, 1)

    def forward(self, x_hist, x_future, **kwargs):
        B, L, C = x_hist.shape
        
        patches = x_hist.view(B, self.num_patches, self.patch_len * C)
        x = self.patch_embedding(patches) + self.position_embedding
        
        enc_out = self.transformer_encoder(x)
        enc_flat = enc_out.view(B, -1)
        
        # Map patches to a unified state, expand across time
        context_vector = self.context_proj(enc_flat)
        context_expanded = context_vector.unsqueeze(1).expand(-1, self.horizon, -1)
        
        if x_future is not None and x_future.size(1) > 0:
            if x_future.size(1) < self.horizon:
                x_fut_aligned = x_future.mean(dim=1, keepdim=True).expand(-1, self.horizon, -1)
            else:
                x_fut_aligned = x_future[:, :self.horizon, :]
                
            combined = torch.cat([context_expanded, x_fut_aligned], dim=-1)
        else:
            combined = context_expanded
            
        out = self.fc_out(combined).squeeze(-1)
        return out