import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the newly updated data.py
from data import train_df, test_df, scaler
from slidingWindow import create_safe_sequences

def main():
    # 1. 15-Minute Substation Specific Parameters
    HORIZON = 1          # Predict 1 step ahead (15 minutes)
    SEQ_LEN = 96         # 96 intervals = 24 hours of historical lag
    TARGET_IDX = 3       # Load_MW is now at index 3 in the feature stack
    
    print("--- Starting 15-Min Substation MLR (Ridge) Baseline ---")
    
    # 2. Slice Sequences
    # create_safe_sequences automatically drops gaps, but data.py imputation ensures continuity
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 3. Flatten history and future covariates into 2D matrices
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    # Flatten the target to a 1D vector for HORIZON = 1
    y_train = Y_train.numpy().ravel()
    y_test = Y_test.numpy().ravel()
    
    # 4. Initialize and train the L2-regularized baseline (Ridge)
    # L2 Regularization prevents overfitting to the massive number of highly correlated 15-min lags
    alphas_to_test = [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0] 
    mlr = RidgeCV(alphas=alphas_to_test)
    
    print(f"Training MLR on shape: {X_train_combined.shape} to predict t+{HORIZON}...")
    mlr.fit(X_train_combined, y_train)
    
    print(f"Optimal Alpha Selected: {mlr.alpha_}")
    
    # 5. Predict and Inverse Scale
    preds_flat = mlr.predict(X_test_combined)
    
    def inverse_scale(data_flat):
        # Create a dummy array matching the new 4-column feature stack (Hour, DayOfWeek, Month, Load_MW)
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
        
    preds_mw = inverse_scale(preds_flat)
    targets_mw = inverse_scale(y_test)
    
    # 6. Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- 15-Minute Substation MLR Evaluation Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")

if __name__ == "__main__":
    main()