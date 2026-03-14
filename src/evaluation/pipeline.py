import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def compute_errors(model, dataloader, device="cpu"):
    """
    Compute raw errors
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            err = torch.abs(pred - y)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors, axis=0)  # [T, N]


def anomaly_score(errors, median, iqr):
    # Compute global score per timestamp
    normalized = (errors - median) / iqr
    scores = np.max(normalized, axis=1)

    # Apply SMA smoothing
    # scores_series = pd.Series(scores)
    # scores = scores_series.rolling(window=5).mean()

    return scores

def evaluate_pipeline(model, train_loader, test_loader):
    # 1. Compute raw errors
    train_errors = compute_errors(model, dataloader=train_loader)
    test_errors_actor2 = compute_errors(model, dataloader=test_loader)

    # 2. Compute normalized anomaly score
    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    iqr[iqr == 0] = 1e-6  # avoid division by zero

    # 3. Compute global anomaly scores per timestamp
    train_scores = anomaly_score(train_errors, median, iqr)
    test_scores_actor2 = anomaly_score(test_errors_actor2, median, iqr)

    # 4. Estimate threshold from normal data (Method 1)
    # threshold = np.percentile(train_scores, 99)

    # 4. Estimate threshold from normal data (Method 2)
    median = np.median(train_scores)
    q75 = np.percentile(train_scores, 75)
    q25 = np.percentile(train_scores, 25)
    iqr = q75 - q25
    threshold = median + 3 * iqr

    # # 5. Detect anomalies
    train_anomaly_flags = train_scores > threshold
    test_anomaly_flags = test_scores_actor2 > threshold

    # Extract Detected Anomaly Points
    train_anomaly_indices = np.where(train_scores > threshold)[0]
    test_anomaly_indices = np.where(test_scores_actor2 > threshold)[0]
    print(len(train_anomaly_indices))
    print(len(test_anomaly_indices))

    # Combine With Sensor Attribution
    timestep = 0
    sensor_scores = test_scores_actor2[timestep]
    top_sensor = np.argmax(sensor_scores)
    print(top_sensor)

    # Visualize Threshold on the Score Plot
    plt.figure(figsize=(15,5))
    plt.plot(test_scores_actor2, label="Anomaly Score")
    plt.axhline(threshold, color='r', linestyle='--', label="Threshold")
    plt.legend()
    plt.title("Anomaly Detection")
    plt.show()

    return {
        "errors_actor1": train_errors,
        "errors_actor2": test_errors_actor2,
        "threshold": threshold,
        "anomalies": test_anomaly_flags
    }