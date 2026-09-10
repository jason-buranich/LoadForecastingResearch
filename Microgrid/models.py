import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import random
import math

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
    mlr = Ridge(alpha=1.0, solver='lsqr')
    
    # 2. Optimized Tree Baselines
    rf_base = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=20,
        max_features=0.3,
        random_state=random_state,
        n_jobs=-1 # Ensure this is set to -1 to use all CPU cores
    )
    
    lgbm_base = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=random_state,
        n_jobs=4,
        verbosity=-1
    )
    
    # 3. Horizon Wrapping for 96 steps
    if horizon == 1:
        rf = rf_base
        lgbm = lgbm_base
    else:
        # RF natively handles multi-target arrays; only wrap LightGBM
        rf = rf_base 
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
class Seq2SeqCovariateLSTM(nn.Module):
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim, horizon=96, num_layers=2, target_idx=0):
        super(Seq2SeqCovariateLSTM, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.target_idx = target_idx
        
        self.encoder = nn.LSTM(hist_input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(1 + future_input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_hist, x_future, y_target=None, teacher_forcing_ratio=0.0):
        _, (hn, cn) = self.encoder(x_hist)
        
        current_load = x_hist[:, -1, self.target_idx].unsqueeze(1)
        predictions = []
        
        for t in range(self.horizon):
            current_covariates = x_future[:, t, :]
            
            decoder_input = torch.cat((current_load, current_covariates), dim=1).unsqueeze(1)
            decoder_out, (hn, cn) = self.decoder_lstm(decoder_input, (hn, cn))
            
            pred_load = self.fc(decoder_out.squeeze(1))
            predictions.append(pred_load)
            
            if y_target is not None and random.random() < teacher_forcing_ratio:
                current_load = y_target[:, t].unsqueeze(1)
            else:
                current_load = pred_load 
                
        return torch.stack(predictions, dim=1).squeeze(-1)

# ==============================================================================
# 5. PYTORCH: GRID TRANSFORMER
# ==============================================================================
class GridTransformer(nn.Module):
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim=128, horizon=96, nheads=4, num_layers=2, dropout=0.1, seq_length=96):
        super(GridTransformer, self).__init__()
        
        self.hist_proj = nn.Linear(hist_input_dim, hidden_dim)
        self.fut_proj = nn.Linear(future_input_dim, hidden_dim)
        
        self.pos_encoder_hist = nn.Parameter(torch.randn(1, seq_length, hidden_dim))
        self.pos_encoder_fut = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist, x_future, **kwargs):
        memory = self.hist_proj(x_hist) + self.pos_encoder_hist
        tgt = self.fut_proj(x_future) + self.pos_encoder_fut
        
        out = self.transformer_decoder(tgt, memory)
        return self.fc_out(out).squeeze(-1)
    
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

    def forward(self, x_hist, x_future):
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