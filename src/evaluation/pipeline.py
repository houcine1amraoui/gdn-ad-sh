import numpy as np
from src.evaluation.anomaly import compute_errors

def evaluate_pipeline(model, loader_actor1, loader_actor2, device):

    # 1. Compute reconstruction errors
    errors_actor1 = compute_errors(model, loader_actor1, device)
    errors_actor2 = compute_errors(model, loader_actor2, device)

    # 2. Estimate threshold from normal data
    threshold = np.mean(errors_actor1) + 3 * np.std(errors_actor1)

    # 3. Detect anomalies in actor2
    anomaly_flags = errors_actor2 > threshold

    return {
        "errors_actor1": errors_actor1,
        "errors_actor2": errors_actor2,
        "threshold": threshold,
        "anomalies": anomaly_flags
    }