import matplotlib.pyplot as plt
import numpy as np

def plot_day_ahead_forecast(targets_mw, mlr_preds_mw, lstm_preds_mw, start_idx=0, save_path='forecast_comparison.png'):
    """
    Plots a 24-hour window comparing the actual load, MLR baseline, and LSTM predictions.
    Assumes the inputs are flattened 1D arrays from the evaluation pipeline.
    """
    # Extract a single 24-hour forecast horizon (96 15-minute intervals)
    slice_start = start_idx * 96
    slice_end = slice_start + 96
    
    actual = targets_mw[slice_start:slice_end]
    mlr = mlr_preds_mw[slice_start:slice_end]
    lstm = lstm_preds_mw[slice_start:slice_end]
    
    # Create an x-axis representing fractional hours for clean plotting (0.0 to 23.75)
    hours = np.arange(96) / 4.0
    
    plt.figure(figsize=(12, 6))
    
    # Plotting the three lines
    plt.plot(hours, actual, label='Actual Load (Microgrid)', color='black', linewidth=2.5)
    plt.plot(hours, mlr, label='MLR Baseline', color='blue', linestyle='--', linewidth=2)
    plt.plot(hours, lstm, label='Tuned Direct LSTM', color='red', linestyle='-.', linewidth=2)
    
    # Formatting
    plt.title('24-Hour Day-Ahead Load Forecast: Model Comparison', fontsize=14, pad=15)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Microgrid Load (MW)', fontsize=12)
    
    # Set x-ticks to display every 2 hours instead of 96 individual intervals
    plt.xticks(np.arange(0, 25, 2))
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved successfully to {save_path}")
    plt.close()

def plot_single_model_forecast(targets, preds, start_idx=0, horizon=96, model_name="Model", save_path="forecast.png"):
    """
    Plots a 24-hour slice of actual grid load versus a single model's predictions.
    """
    plt.figure(figsize=(12, 6))
    
    # Slice the specific 24-hour window (96 intervals) from the flattened arrays
    target_slice = targets[start_idx : start_idx + horizon]
    pred_slice = preds[start_idx : start_idx + horizon]
    
    # Create an x-axis representing fractional hours
    hours = np.arange(horizon) / 4.0
    
    plt.plot(hours, target_slice, label='Actual Load (MW)', color='black', linewidth=2)
    plt.plot(hours, pred_slice, label=f'{model_name} Forecast', color='blue', linestyle='--', linewidth=2)
    
    plt.title(f'24-Hour Day-Ahead Forecast: Actual vs {model_name}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (MW)')
    
    # Set x-ticks to display every 2 hours
    plt.xticks(np.arange(0, 25, 2))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Graph saved successfully to {save_path}")
    plt.close()