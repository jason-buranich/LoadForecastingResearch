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
    """
    Persistence baseline that repeats values from the corresponding 
    seasonal lag period. Defaulted to 96 steps (24 hours at 15-min intervals).
    """
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
    """
    Instantiates the traditional ISO baseline (Ridge Regression) alongside 
    optimized Random Forest and LightGBM models.
    """
    # 1. Linear Baseline
    mlr = Ridge(alpha=500.0, solver='lsqr')
    
    # 2. Optimized Tree Baselines (Throttled to prevent OpenMP deadlocks)
    rf_base = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=20,
        max_features=0.3,
        random_state=random_state,
        n_jobs=-1  
    )
    
    lgbm_base = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=random_state,
        n_jobs=4, 
        verbosity=-1
    )
    
    # 3. Horizon Wrapping for multi-step predictions
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
    Encodes the historical sequence and concatenates the flattened future 
    covariates before projecting directly to the entire prediction horizon.
    """
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim=64, horizon=96, num_layers=2, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.future_input_dim = future_input_dim
        
        self.lstm = nn.LSTM(
            input_size=hist_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim + (horizon * future_input_dim), horizon)

    def forward(self, x_hist, x_future):
        _, (hn, _) = self.lstm(x_hist)
        last_hidden = hn[-1]  
        
        future_flat = x_future.reshape(x_future.size(0), -1)
        combined = torch.cat((last_hidden, future_flat), dim=1)
        
        out = self.fc(combined)
        return out


# ==============================================================================
# 4. PYTORCH: ENCODER-DECODER LSTM (Seq2Seq)
# ==============================================================================
class Seq2Seq(nn.Module):
    """
    Autoregressive encoder-decoder architecture for time series forecasting.
    Includes dimensionality fixes for proper tensor slicing.
    """
    def __init__(self, hist_input_dim, future_input_dim, horizon=96, hidden_dim=64, num_layers=2, dropout=0.1):
        super(Seq2Seq, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Encoder: Processes the 96-step historical sequence
        self.encoder = nn.LSTM(
            input_size=hist_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: Processes future covariates + previous step prediction
        self.decoder = nn.LSTM(
            input_size=1 + future_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist, x_fut, **kwargs):
        batch_size = x_hist.size(0)
        
        # 1. Encode History
        _, (hidden, cell) = self.encoder(x_hist)
        
        # 2. Prepare Decoder
        outputs = torch.zeros(batch_size, self.horizon, 1).to(x_hist.device)
        dec_input = torch.zeros(batch_size, 1, 1).to(x_hist.device)
        
        # 3. Autoregressive Decoding Loop
        for t in range(self.horizon):
            covariates_t = x_fut[:, t, :].unsqueeze(1)
            dec_input_combined = torch.cat((dec_input, covariates_t), dim=2)
            
            out, (hidden, cell) = self.decoder(dec_input_combined, (hidden, cell))
            pred = self.fc_out(out)
            
            # Squeeze the middle sequence dimension to match the 2D slice [Batch, 1]
            outputs[:, t, :] = pred.squeeze(1)
            dec_input = pred
            
        return outputs.squeeze(-1)


# ==============================================================================
# 5. PYTORCH: GRID TRANSFORMER
# ==============================================================================
class GridTransformer(nn.Module):
    """
    Encoder-based Transformer that maps historical states to a single context 
    vector before fusing with future covariates for projection.
    """
    def __init__(self, hist_input_dim, future_input_dim, seq_len=96, horizon=96, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(GridTransformer, self).__init__()
        
        self.embedding = nn.Linear(hist_input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Encoder processing the sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=128, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Scale output layer for multi-step horizon
        self.fc_out = nn.Linear(d_model + (future_input_dim * horizon), horizon)

    def forward(self, x_hist, x_future, **kwargs):
        x = self.embedding(x_hist) + self.pos_encoder
        encoded_seq = self.transformer(x)
        context_vector = encoded_seq[:, -1, :]
        
        fut_flat = x_future.view(x_future.size(0), -1)
        combined = torch.cat((context_vector, fut_flat), dim=1)
        
        return self.fc_out(combined)


# ==============================================================================
# 6. PYTORCH: PATCH TIME SERIES TRANSFORMER
# ==============================================================================
class PatchTST(nn.Module):
    """
    Groups the historical history into non-overlapping patches to capture 
    local semantic meaning, processes via attention, and fuses with future covariates.
    """
    def __init__(self, hist_input_dim, future_input_dim, seq_len=96, patch_len=16, horizon=96, hidden_dim=128, nheads=4, num_layers=2, dropout=0.2):
        super(PatchTST, self).__init__()
        
        # Ensure the sequence length is cleanly divisible by the patch length
        assert seq_len % patch_len == 0, f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})"
        
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
        
        self.future_embedding = nn.Linear(horizon * future_input_dim, hidden_dim)
        self.fc_out = nn.Linear((self.num_patches * hidden_dim) + hidden_dim, horizon)

    def forward(self, x_hist, x_future, **kwargs):
        B, L, C = x_hist.shape
        
        patches = x_hist.view(B, self.num_patches, self.patch_len * C)
        x = self.patch_embedding(patches) + self.position_embedding
        
        enc_out = self.transformer_encoder(x)
        enc_flat = enc_out.view(B, -1)
        
        fut_flat = x_future.reshape(B, -1)
        fut_emb = self.future_embedding(fut_flat)
        
        combined = torch.cat([enc_flat, fut_emb], dim=1)
        out = self.fc_out(combined)
        
        return out