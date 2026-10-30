import time
import os
import torch
import joblib

class TrainingTimer:
    """Context manager to cleanly measure training latency across both PyTorch and Tabular models."""
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.training_time = 0.0

    def __enter__(self):
        self.start()
        return self

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.training_time += time.perf_counter() - self.start_time

def profile_model_efficiency(model, test_input, training_time_seconds=0.0, model_type="pytorch", save_path="temp_model"):
    results = {}
    
    # 1. Log Training Latency
    results['training_latency_seconds'] = training_time_seconds
    
    # 2. Measure Inference Latency
    if model_type == "pytorch" and torch.cuda.is_available():
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            if isinstance(test_input, tuple):
                _ = model(*test_input)
            else:
                _ = model(test_input)
        torch.cuda.synchronize()
    else:
        start_time = time.perf_counter()
        if isinstance(model, (list, tuple)):
            for estimator in model:
                estimator.predict(test_input)
        else:
            _ = model.predict(test_input)
        
    results['inference_latency_seconds'] = time.perf_counter() - start_time
    
    # 3. Measure Model Size (Disk Footprint)
    if model_type == "pytorch":
        torch.save(model.state_dict(), f"{save_path}.pt")
        results['size_mb'] = os.path.getsize(f"{save_path}.pt") / (1024 * 1024)
        results['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        os.remove(f"{save_path}.pt")
    else:
        joblib.dump(model, f"{save_path}.joblib")
        results['size_mb'] = os.path.getsize(f"{save_path}.joblib") / (1024 * 1024)
        os.remove(f"{save_path}.joblib")
        
    return results