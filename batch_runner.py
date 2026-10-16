import subprocess
import sys
import datetime
import os

def run_and_stream_evaluations(script_list, log_filename="model_benchmark_results.txt"):
    with open(log_filename, 'a') as log_file:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{'='*70}\n")
        log_file.write(f"BATCH EVALUATION STARTED: {start_time}\n")
        log_file.write(f"{'='*70}\n\n")

        # Unpack the tuple (script_name, num_runs)
        for script, num_runs in script_list:
            print(f"\nEvaluating {script} ({num_runs} runs)...")
            
            log_file.write(f"--- STARTING EVALUATION: {script} ---\n")
            log_file.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not os.path.exists(script):
                error_msg = f"[ERROR] Target model file not found: {script}\n"
                print(error_msg, end='')
                log_file.write(error_msg + "\n")
                continue

            if not os.path.exists("eval_models.py"):
                print("[ERROR] eval_models.py not found in directory.")
                break

            try:
                # Use Popen to stream output in real-time, forcing unbuffered execution
                process = subprocess.Popen(
                    [sys.executable, "-u", "eval_models.py", script, "--runs", str(num_runs)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Read line-by-line and flush immediately
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log_file.write(line)
                    log_file.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    success_msg = f"\n--- SUCCESS: {script} ---\n\n"
                    print(success_msg, end='')
                    log_file.write(success_msg)
                else:
                    fail_msg = f"\n--- FAILED: {script} (Exit Code: {process.returncode}) ---\n\n"
                    print(fail_msg, end='')
                    log_file.write(fail_msg)

            except Exception as e:
                error_msg = f"[SYSTEM ERROR] Failed to execute {script}: {str(e)}\n\n"
                print(error_msg, end='')
                log_file.write(error_msg)

        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{'='*70}\n")
        log_file.write(f"BATCH EVALUATION COMPLETED: {end_time}\n")
        log_file.write(f"{'='*70}\n\n")
        
        print(f"\nAll evaluations processed. Terminal outputs and averages saved to: {log_filename}")

if __name__ == "__main__":
    # Define scripts as a list of tuples: ("script_name.py", number_of_runs)
    scripts_to_evaluate = [
        ("CommFac/15min_tft.py", 5),
        ("CommFac/15min_seq2seq.py", 5),
        ("CommFac/15min_patchtst.py", 5),
        ("CommFac/15min_lstm.py", 5),
        ("CommFac/1hr_tft.py", 5),
        ("CommFac/1hr_seq2seq.py", 5),
        ("CommFac/1hr_patchtst.py", 5),
        ("CommFac/1hr_lstm.py", 5),
        ("CommFac/24hr_tft.py", 5),
        ("CommFac/24hr_seq2seq.py", 5),
        ("CommFac/24hr_patchtst.py", 5),
        ("CommFac/24hr_lstm.py", 5)
    ]
    
    run_and_stream_evaluations(scripts_to_evaluate, log_filename="model_benchmark_results.txt")