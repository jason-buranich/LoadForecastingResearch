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
    seasonal lag period (e.g., 24 hours ago for 24-step day-ahead forecasts,
    or 1 hour ago for 1-step next-hour forecasts).
    """
    def __init__(self, seasonal_lag=24):
        self.seasonal_lag = seasonal_lag

    def predict(self, X, target_idx=0): # Default to 0
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim == 3:
            load_history = X[:, :, target_idx] # Use dynamic variable
        else:
            load_history = X
            
        return load_history[:, -self.seasonal_lag:]


# ==============================================================================
# 2. TABULAR MODELS: RANDOM FOREST & LIGHTGBM
# ==============================================================================
def get_tabular_models(horizon=24, random_state=42):
    """
    Instantiates the traditional ISO baseline (Ridge Regression) alongside 
    optimized Random Forest and LightGBM models.
    """
    # 1. Linear Baseline
    mlr = Ridge(alpha=500.0, solver='lsqr')
    
    # 2. Optimized Tree Baselines (using the hyperparams from our tuning)
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
        n_jobs=-1,
        verbosity=-1
    )
    
    # 3. Horizon Wrapping
    if horizon == 1:
        rf = rf_base
        lgbm = lgbm_base
    else:
        # MultiOutputRegressor automatically executes the 24-specialized-model loop natively
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
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim=64, horizon=24, num_layers=2, dropout=0.1):
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
        
        # The FC layer now takes the LSTM's hidden state PLUS all flattened future weather data
        fc_input_dim = hidden_dim + (horizon * future_input_dim)
        self.fc = nn.Linear(fc_input_dim, horizon)

    def forward(self, x_hist, x_future):
        # 1. Process history
        _, (hn, _) = self.lstm(x_hist)
        last_hidden = hn[-1]  # Shape: (Batch, hidden_dim)
        
        # 2. Flatten future covariates: (Batch, 24, features) -> (Batch, 24 * features)
        future_flat = x_future.view(x_future.size(0), -1)
        
        # 3. Concatenate historical momentum with tomorrow's forecast
        combined = torch.cat((last_hidden, future_flat), dim=1)
        
        # 4. Project to the 24-hour load horizon
        out = self.fc(combined)
        return out


# ==============================================================================
# 4. PYTORCH: ENCODER-DECODER LSTM (Seq2Seq)
# ==============================================================================
class Seq2SeqCovariateLSTM(nn.Module):
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim, horizon=24, num_layers=1, target_idx=4):
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
            
            # TEACHER FORCING LOGIC
            # If training, sometimes use the actual ground truth for the next step's input
            if y_target is not None and random.random() < teacher_forcing_ratio:
                current_load = y_target[:, t].unsqueeze(1)
            else:
                current_load = pred_load # Otherwise, use its own prediction
                
        return torch.stack(predictions, dim=1).squeeze(-1)

class GridTransformer(nn.Module):
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim=128, horizon=24, nheads=4, num_layers=2, dropout=0.1, seq_length=168):
        super(GridTransformer, self).__init__()
        
        self.hist_proj = nn.Linear(hist_input_dim, hidden_dim)
        self.fut_proj = nn.Linear(future_input_dim, hidden_dim)
        
        self.pos_encoder_hist = nn.Parameter(torch.randn(1, seq_length, hidden_dim))
        self.pos_encoder_fut = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        
        # Now uses the dynamic dropout variable instead of a hardcoded 0.1
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
    
class PatchTST(nn.Module):
    """
    An adapted Patch Time Series Transformer.
    Groups the 168-hour history into 7 daily patches (24h each) to capture 
    local semantic meaning, processes via attention, and fuses with future covariates.
    """
    def __init__(self, hist_input_dim, future_input_dim, seq_len=168, patch_len=24, horizon=24, hidden_dim=128, nheads=4, num_layers=2, dropout=0.2):
        super(PatchTST, self).__init__()
        
        self.patch_len = patch_len
        # Calculate number of non-overlapping patches (168 / 24 = 7)
        self.num_patches = seq_len // patch_len 
        
        # 1. Patch Embedding: Maps a 24-hour chunk of all features to the hidden dimension
        self.patch_embedding = nn.Linear(patch_len * hist_input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # 2. Transformer Encoder (Process the 7 patches)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Future Covariate Embedding
        self.future_embedding = nn.Linear(horizon * future_input_dim, hidden_dim)
        
        # 4. Final Projection Head
        # Output takes the flattened encoded patches + the embedded future weather
        self.fc_out = nn.Linear((self.num_patches * hidden_dim) + hidden_dim, horizon)

    def forward(self, x_hist, x_future):
        B, L, C = x_hist.shape
        
        # 1. Create Patches: Reshape (B, 168, C) -> (B, 7, 24 * C)
        patches = x_hist.view(B, self.num_patches, self.patch_len * C)
        
        # 2. Embed Patches and add Positional Encoding
        x = self.patch_embedding(patches) + self.position_embedding
        
        # 3. Apply Attention across the 7 patches
        enc_out = self.transformer_encoder(x)
        
        # 4. Flatten the encoder output: (B, 7 * hidden_dim)
        enc_flat = enc_out.view(B, -1)
        
        # 5. Process Future Covariates: (B, 24 * future_dim) -> (B, hidden_dim)
        fut_flat = x_future.view(B, -1)
        fut_emb = self.future_embedding(fut_flat)
        
        # 6. Concatenate and Project
        combined = torch.cat([enc_flat, fut_emb], dim=1)
        out = self.fc_out(combined)
        
        return out