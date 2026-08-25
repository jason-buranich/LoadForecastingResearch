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

    def predict(self, X):
        """
        X: Tensor or NumPy array of shape (N, seq_len, features) or (N, seq_len)
           where target load is at feature index 4 (or single channel).
        Returns: NumPy array of shape (N, horizon)
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # If multidimensional, extract target column (e.g., CAISO at index 4)
        if X.ndim == 3:
            load_history = X[:, :, 4]
        else:
            load_history = X
            
        # Select the last 'seasonal_lag' steps as the repeated forecast
        return load_history[:, -self.seasonal_lag:]


# ==============================================================================
# 2. TABULAR MODELS: RANDOM FOREST & LIGHTGBM
# ==============================================================================
def get_tabular_models(horizon=24, n_estimators=100, random_state=42):
    """
    Instantiates the traditional ISO baseline (Ridge Regression) alongside 
    Random Forest and LightGBM models.
    Wraps single-output estimators in MultiOutputRegressor if horizon > 1.
    """
    if horizon == 1:
        # Fast iterative solver for high-dimensional data
        mlr = Ridge(alpha=500.0, solver='lsqr')
        
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        lgbm = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, verbosity=-1)
    else:
        # Ridge natively supports multi-output. DO NOT wrap it in MultiOutputRegressor!
        mlr = Ridge(alpha=500.0, solver='lsqr')
        
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
        lgbm = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, verbosity=-1))
        
    return mlr, rf, lgbm


# ==============================================================================
# 3. PYTORCH: DIRECT MULTI-STEP LSTM
# ==============================================================================
class DirectLSTM(nn.Module):
    """
    Encodes the historical sequence and maps the final hidden state 
    directly to the entire prediction horizon via a linear projection layer.
    """
    def __init__(self, input_dim=9, hidden_dim=64, horizon=24, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (hn, _) = self.lstm(x)
        # hn[-1] shape: (batch_size, hidden_dim)
        out = self.fc(hn[-1])
        return out  # Shape: (batch_size, horizon)


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
    """
    A Transformer-based architecture capturing the core cross-attention 
    mechanics of a TFT for multi-horizon time-series forecasting.
    """
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim=64, horizon=24, nheads=4, num_layers=2):
        super(GridTransformer, self).__init__()
        
        # 1. Linear projections to align history and future dimensions
        self.hist_proj = nn.Linear(hist_input_dim, hidden_dim)
        self.fut_proj = nn.Linear(future_input_dim, hidden_dim)
        
        # 2. Learnable Positional Encodings to inject the concept of "time" 
        self.pos_encoder_hist = nn.Parameter(torch.randn(1, 168, hidden_dim))
        self.pos_encoder_fut = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        
        # 3. The Core Attention Mechanism
        # The decoder allows the future covariates to dynamically "query" the history
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. Final Output Projection to Megawatts
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist, x_future, **kwargs):
        # Project and add positional context
        memory = self.hist_proj(x_hist) + self.pos_encoder_hist
        tgt = self.fut_proj(x_future) + self.pos_encoder_fut
        
        # Transformer Cross-Attention: 
        # Future weather (tgt) searches the 168-hour history (memory) for patterns
        out = self.transformer_decoder(tgt, memory)
        
        # Map back to a single MW prediction per hour
        pred = self.fc_out(out).squeeze(-1) # Shape: (Batch, 24)
        return pred