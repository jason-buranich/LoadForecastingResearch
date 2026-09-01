import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import components from the updated data.py
from data import train_df, test_df, scaler
from slidingWindow import create_safe_sequences
from visualize import plot_single_model_forecast

def main():
    # 1. 15-Minute Substation Configuration
    HORIZON = 1              # Predict 1 step ahead (15 minutes)
    SEQ_LEN = 96             # 96 intervals = 24 hours of history
    TARGET_IDX = 1           # Load_MW is at index 1 before dropping Month
    COVARIATE_START_IDX = 2  # Future covariates (Hour, DayOfWeek) start at index 2
    
    print("--- Starting 15-Minute-Ahead Random Forest Pipeline ---")
    
    # 2. Slice Sequences (Safely passing covariates to prevent data leaks)
    X_train_hist, X_train_fut, Y_train = create_safe_sequences(
        train_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    X_test_hist, X_test_fut, Y_test = create_safe_sequences(
        test_df, seq_len=SEQ_LEN, horizon=HORIZON, target_idx=TARGET_IDX, covariate_start_idx=COVARIATE_START_IDX
    )
    
    # 3. Flatten history and future covariates into 2D matrices for scikit-learn
    X_train_combined = np.hstack((
        X_train_hist.numpy().reshape(X_train_hist.shape[0], -1), 
        X_train_fut.numpy().reshape(X_train_fut.shape[0], -1)
    ))
    X_test_combined = np.hstack((
        X_test_hist.numpy().reshape(X_test_hist.shape[0], -1), 
        X_test_fut.numpy().reshape(X_test_fut.shape[0], -1)
    ))
    
    # Flatten the target to a 1D vector
    y_train = Y_train.numpy().ravel()
    y_test = Y_test.numpy().ravel()
    
    # 4. Initialize and Train Random Forest
    # Using the optimized hyperparameters from your models.py
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_split=20,
        max_features=0.3,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training Random Forest on shape {X_train_combined.shape}...")
    rf.fit(X_train_combined, y_train)
    
    # 5. Predict and Inverse Scale
    preds_flat = rf.predict(X_test_combined)
    
    def inverse_scale(data_flat):
        # Create a dummy array matching the 4-column feature stack
        dummy = np.zeros((len(data_flat), scaler.mean_.shape[0]))
        dummy[:, TARGET_IDX] = data_flat
        return scaler.inverse_transform(dummy)[:, TARGET_IDX]
        
    preds_mw = inverse_scale(preds_flat)
    targets_mw = inverse_scale(y_test)
    
    # 6. Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(targets_mw, preds_mw))
    mae = mean_absolute_error(targets_mw, preds_mw)
    wape = np.sum(np.abs(targets_mw - preds_mw)) / np.sum(np.abs(targets_mw)) * 100
    
    print("\n--- 15-Minute-Ahead Random Forest Metrics (MW) ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
    
    # 7. Visualization
    plot_single_model_forecast(
        targets_mw, 
        preds_mw, 
        start_idx=0, 
        horizon=96,
        model_name="15-Min Random Forest", 
        save_path='rf_15min_ahead_forecast.png'
    )

if __name__ == "__main__":
    main()