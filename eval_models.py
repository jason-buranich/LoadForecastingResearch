import subprocess
import re
import numpy as np
import argparse
import sys

def run_evaluation(script_path, num_runs=5):
    print(f"--- Starting {num_runs}-Run Evaluation for {script_path} ---")
    
    metrics = {'RMSE': [], 'MAE': [], 'WAPE': []}
    regex = r"RMSE:\s*([\d.]+)\s*\|\s*MAE:\s*([\d.]+)\s*\|\s*WAPE:\s*([\d.]+)%"
    
    for run in range(1, num_runs + 1):
        print(f"\n[Trial {run}/{num_runs}] Executing {script_path}...\n")
        
        # Use Popen to stream the training epochs in real time
        process = subprocess.Popen(
            [sys.executable, "-u", script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        
        captured_output = []
        
        # Print output line-by-line as PyTorch trains
        for line in process.stdout:
            print(line, end='', flush=True)  
            captured_output.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n  [!] Error executing script on trial {run}.")
            continue
            
        # Rejoin the captured lines to search for the final metrics
        full_output = "".join(captured_output)
        match = re.search(regex, full_output)
        
        if match:
            rmse, mae, wape = float(match.group(1)), float(match.group(2)), float(match.group(3))
            print(f"\n  [Trial {run} Success] -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.2f}%")
            metrics['RMSE'].append(rmse)
            metrics['MAE'].append(mae)
            metrics['WAPE'].append(wape)
        else:
            print(f"\n  [!] Warning: Could not find or parse metrics in the terminal output.")
            print("  --- Last 500 characters of terminal output ---")
            print(full_output[-500:])
            print("  ----------------------------------------------")

    if len(metrics['RMSE']) > 0:
        print("\n==================================================")
        print(f"FINAL STATISTICAL SUMMARY ({len(metrics['RMSE'])} SUCCESSFUL TRIALS)")
        print(f"Target Script: {script_path}")
        print("==================================================")
        print(f"RMSE: {np.mean(metrics['RMSE']):.2f} ± {np.std(metrics['RMSE']):.2f}")
        print(f"MAE:  {np.mean(metrics['MAE']):.2f} ± {np.std(metrics['MAE']):.2f}")
        print(f"WAPE: {np.mean(metrics['WAPE']):.2f}% ± {np.std(metrics['WAPE']):.2f}%")
        print("==================================================")
    else:
        print("\nNo successful trials to summarize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute an exterior forecasting script multiple times and record stats.")
    parser.add_argument("script", help="Path to the python script to execute (e.g., 1hr_tft.py)")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to run the script (default: 5)")
    args = parser.parse_args()
    
    run_evaluation(args.script, args.runs)