import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def compute_prediction_errors(model, dataloader, device="cpu"):
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

def compute_anomaly_score(train_errors, actor2_test_errors, actor1_test_errors):

    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    iqr[iqr == 0] = 1e-6

    # Normalize BOTH
    train_norm = np.abs((train_errors - median) / iqr)
    actor2_test_norm  = np.abs((actor2_test_errors  - median) / iqr)
    actor1_test_norm  = np.abs((actor1_test_errors  - median) / iqr)

    # Use TOP-K instead of max
    k = max(1, int(0.1 * train_norm.shape[1]))

    train_topk = np.sort(train_norm, axis=1)[:, -k:]
    actor2_test_topk  = np.sort(actor2_test_norm, axis=1)[:, -k:]
    actor1_test_topk  = np.sort(actor1_test_norm, axis=1)[:, -k:]

    train_scores = np.mean(train_topk, axis=1)
    actor2_test_scores  = np.mean(actor2_test_topk, axis=1)
    actor1_test_scores  = np.mean(actor1_test_topk, axis=1)

    return train_scores, actor2_test_scores, actor1_test_scores
    
def evaluate_pipeline(model, train_loader, test_loader):
    # 1. Compute raw errors
    train_errors = compute_prediction_errors(model, dataloader=train_loader)
    test_errors = compute_prediction_errors(model, dataloader=test_loader)

    # 2. Compute global (normalized) anomaly scores per timestamp
    train_scores = compute_anomaly_score(train_errors, test_errors=train_errors)
    test_scores = compute_anomaly_score(train_errors, test_errors=test_errors)
    
    np.save("train_scores.npy", train_scores)
    np.save("test_scores.npy", test_scores)

    # weak labeling
    # y_true_actor1 = np.zeros(len(train_scores))
    # y_true_actor2 = np.ones(len(test_scores))
    # y_true = np.concatenate([y_true_actor1, y_true_actor2])
    # y_score = np.concatenate([train_scores, test_scores])

    # auc = roc_auc_score(y_true, y_score)
    # print("AUC:", auc)

    # 4. Estimate threshold from normal data (Method 1)
    # threshold = np.percentile(train_scores, 99)

    # 4. Estimate threshold from normal data (Method 2)
    # median = np.median(train_scores)
    # q75 = np.percentile(train_scores, 75)
    # q25 = np.percentile(train_scores, 25)
    # iqr = q75 - q25
    # threshold = median + 3 * iqr

    # # 5. Detect anomalies
    # train_anomaly_flags = train_scores > threshold
    # test_anomaly_flags = test_scores > threshold

    # plt.hist(train_scores, bins=50, alpha=0.5, label="Actor1")
    # plt.hist(test_scores, bins=50, alpha=0.5, label="Actor2")
    # plt.legend()
    # plt.title("Score Distribution")
    # plt.show()
    # Extract Detected Anomaly Points
    # train_anomaly_indices = np.where(train_scores > threshold)[0]
    # test_anomaly_indices = np.where(test_scores > threshold)[0]
    # print("Anomalies in train", len(train_anomaly_indices)/len(train_scores)*100, "%")
    # print("Anomalies in train",len(test_anomaly_indices)/len(test_scores)*100, "%")

    # # Combine With Sensor Attribution
    # timestep = 0
    # devices_scores = test_scores[timestep]
    # top_device = np.argmax(devices_scores)
    # print("Top device: ", top_device)

    # # Visualize Threshold on the Score Plot
    # plt.figure(figsize=(15,5))
    # plt.plot(train_scores, label="Train Score")
    # plt.plot(test_scores, label="Anomaly Score")
    # plt.axhline(threshold, color='r', linestyle='--', label="Threshold")
    # plt.legend()
    # plt.title("Anomaly Detection")
    # plt.show()

    return {
        train_scores,
        test_scores,
    #     "threshold": threshold,
    #     "anomalies": test_anomaly_flags
    }