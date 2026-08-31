import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import only data and sequence generator
from data import train_df, test_df, scaler
from ISO.slidingWindow import create_safe_sequences

def main():
    HORIZON = 24
    SEQ_LEN = 168
    TARGET_IDX = 4
    
    print(f"--- Starting Standalone MLR Pipeline: {HORIZON}h Horizon, {SEQ_LEN}h Lookback ---")
    
    # 1. Generate sliding window tensors
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    X_test_hist, X_test_fut, Y_test    = create_safe_sequences(test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX)
    
    # 2. Flatten and combine historical data and future weather covariates
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    Y_train_flat = Y_train.numpy()
    Y_test_flat = Y_test.numpy()
    
    # 3. Initialize and train the L2-regularized baseline
    alphas_to_test = [500] 
    mlr = RidgeCV(alphas=alphas_to_test)
    
    print("Finding optimal alpha and fitting MLR...")
    mlr.fit(X_train_combined, Y_train_flat)
    
    print(f"Optimal Alpha Selected: {mlr.alpha_}")
    
    # 4. Predict and Inverse Scale
    preds = mlr.predict(X_test_combined)
    
    def inverse_scale(data_flat):
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
        
    preds_mw = inverse_scale(preds.flatten())
    targets_mw = inverse_scale(Y_test_flat.flatten())
    
    # 5. Calculate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- MLR Baseline Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")

if __name__ == "__main__":
    main()