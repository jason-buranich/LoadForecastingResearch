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