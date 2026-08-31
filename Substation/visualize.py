import matplotlib.pyplot as plt
import numpy as np

def plot_day_ahead_forecast(targets_mw, mlr_preds_mw, lstm_preds_mw, start_idx=0, save_path='forecast_comparison.png'):
    """
    Plots a 24-hour window comparing the actual load, MLR baseline, and LSTM predictions.
    Assumes the inputs are flattened 1D arrays from the evaluation pipeline.
    """
    # Extract a single 24-hour forecast horizon
    # Since the arrays are flattened (N_samples * 24), we step by 24 to get a clean day
    slice_start = start_idx * 24
    slice_end = slice_start + 24
    
    actual = targets_mw[slice_start:slice_end]
    mlr = mlr_preds_mw[slice_start:slice_end]
    lstm = lstm_preds_mw[slice_start:slice_end]
    
    hours = np.arange(24)
    
    plt.figure(figsize=(12, 6))
    
    # Plotting the three lines
    plt.plot(hours, actual, label='Actual Load (CAISO)', color='black', linewidth=2.5, marker='o')
    plt.plot(hours, mlr, label='MLR Baseline', color='blue', linestyle='--', linewidth=2)
    plt.plot(hours, lstm, label='Tuned Direct LSTM', color='red', linestyle='-.', linewidth=2)
    
    # Formatting
    plt.title('24-Hour Day-Ahead Load Forecast: Model Comparison', fontsize=14, pad=15)
    plt.xlabel('Forecast Horizon (Hour 0 to 23)', fontsize=12)
    plt.ylabel('Grid Load (Megawatts)', fontsize=12)
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    # Save and display
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved successfully to {save_path}")
    plt.show()

def plot_single_model_forecast(targets, preds, start_idx=0, horizon=24, model_name="Model", save_path="forecast.png"):
    """
    Plots a 24-hour slice of actual grid load versus a single model's predictions.
    """
    plt.figure(figsize=(12, 6))
    
    # Slice the specific 24-hour window from the flattened arrays
    target_slice = targets[start_idx : start_idx + horizon]
    pred_slice = preds[start_idx : start_idx + horizon]
    
    plt.plot(target_slice, label='Actual Load (MW)', color='black', linewidth=2)
    plt.plot(pred_slice, label=f'{model_name} Forecast', color='blue', linestyle='--', linewidth=2)
    
    plt.title(f'24-Hour Day-Ahead Forecast: Actual vs {model_name}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Graph saved successfully to {save_path}")