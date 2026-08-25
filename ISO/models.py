import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb


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
class Seq2SeqLSTM(nn.Module):
    """
    Encoder-decoder LSTM architecture that compresses input dynamics 
    into latent context vectors before autoregressively generating predictions.
    """
    def __init__(self, input_dim=9, hidden_dim=64, horizon=24, num_layers=2, target_idx=4):
        super().__init__()
        self.horizon = horizon
        self.target_idx = target_idx
        
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        _, (hn, cn) = self.encoder(x)
        
        # Initialize first decoder step using the last observed target load
        dec_input = x[:, -1, self.target_idx].unsqueeze(1).unsqueeze(2)  # Shape: (batch, 1, 1)
        outputs = []
        
        for _ in range(self.horizon):
            dec_out, (hn, cn) = self.decoder(dec_input, (hn, cn))
            pred = self.fc(dec_out[:, -1, :])  # Shape: (batch, 1)
            outputs.append(pred)
            dec_input = pred.unsqueeze(1)      # Feed forward step as next input
            
        return torch.cat(outputs, dim=1)  # Shape: (batch_size, horizon)


# ==============================================================================
# 5. PYTORCH: TIME-SERIES TRANSFORMER
# ==============================================================================
class TimeSeriesTransformer(nn.Module):
    """
    Multi-head self-attention transformer head mapping sequential
    temporal dependencies directly to multi-step forecasts.
    """
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, horizon=24, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        proj = self.input_projection(x)
        enc_out = self.transformer_encoder(proj)
        # Pool across sequence dimension (mean pooling)
        pooled = enc_out.mean(dim=1)
        out = self.head(pooled)
        return out  # Shape: (batch_size, horizon)

class Seq2SeqCovariateLSTM(nn.Module):
    def __init__(self, hist_input_dim, future_input_dim, hidden_dim, horizon=24, num_layers=1):
        super(Seq2SeqCovariateLSTM, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Encoder: Processes the 168-hour history (load + weather + time)
        self.encoder = nn.LSTM(hist_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Decoder: Processes the 24-hour future weather forecast & time features
        self.decoder_lstm = nn.LSTM(future_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Final output layer to map the hidden state to a single Megawatt prediction per hour
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_hist, x_future):
        """
        x_hist shape: (Batch, 168, hist_input_dim)
        x_future shape: (Batch, 24, future_input_dim)
        """
        # 1. Encode the history
        _, (hn, cn) = self.encoder(x_hist)
        
        # 2. Decode the future forecast
        # We initialize the decoder's memory with the encoder's final state (hn, cn)
        decoder_out, _ = self.decoder_lstm(x_future, (hn, cn))
        
        # 3. Predict the load for each of the 24 future hours
        # decoder_out is (Batch, 24, hidden_dim)
        predictions = self.fc(decoder_out) # Outputs (Batch, 24, 1)
        
        # Drop the last dimension to match the target shape (Batch, 24)
        return predictions.squeeze(-1)