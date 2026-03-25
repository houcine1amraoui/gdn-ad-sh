import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.models.builders import build_gdn_model
from src.evaluation.load_checkpoint import load_checkpoint

def errors_computation_pipeline(config, best_exp_path, exp_dir):
    # 1. Model Initialization
    model_arch = build_gdn_model(config)

    # 2. load best checkpoint
    optimizer = optim.Adam(model_arch.parameters(), lr=config["training"]["lr"])
    model, optimizer, _ = load_checkpoint(
        model_arch,
        f"{best_exp_path}/best.pth",
        optimizer
    )

    # 4. Dataset/Loaders Create
    data_loaders = create_evaluation_dataloaders(config)
    
    # 5. Compute raw prediction errors
    compute_prediction_errors(model, data_loaders, exp_dir)

def compute_prediction_errors_per_loader(model, dataloader, device="cpu"):
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

def compute_prediction_errors(model, data_loaders, exp_dir):
    train_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["train_loader"])
    val_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["val_loader"])
    actor2_test_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["actor2_test_loader"])
    actor1_test_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["actor1_test_loader"])
    
    np.save(f"{exp_dir}/train_errors.npy", train_errors)
    np.save(f"{exp_dir}/val_errors.npy", val_errors)
    np.save(f"{exp_dir}/actor2_test_errors.npy", actor2_test_errors)
    np.save(f"{exp_dir}/actor1_test_errors.npy", actor1_test_errors)

def compute_anomaly_scores(exp_dir):
    # --- Load errors ---
    train_errors = np.load(f"{exp_dir}/train_errors.npy")
    val_errors = np.load(f"{exp_dir}/val_errors.npy")
    actor2_test_errors = np.load(f"{exp_dir}/actor2_test_errors.npy")
    actor1_test_errors = np.load(f"{exp_dir}/actor1_test_errors.npy")

    # Robust Stats
    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    iqr[iqr == 0] = 1e-6  # avoid division by zero

    # --- Normalize ---
    train_norm = (train_errors - median) / iqr
    val_norm = (val_errors - median) / iqr
    actor2_test_norm = (actor2_test_errors - median) / iqr
    actor1_test_norm = (actor1_test_errors - median) / iqr


    train_scores = np.max(train_norm, axis=1)
    val_scores = np.max(val_norm, axis=1)
    actor2_test_scores = np.max(actor2_test_norm, axis=1)
    actor1_test_scores = np.max(actor1_test_norm, axis=1)
    
    scores = {
        "train": train_scores,
        "val": val_scores,
        "actor2_test": actor2_test_scores,
        "actor1_test": actor1_test_scores
    }

    return scores

def compute_anomaly_score_and_detection(
    topk_ratio=0.4,        # fraction of sensors for top-k
    threshold_percentile=99 # percentile threshold on train scores
):
    """
    Compute anomaly scores with weighted top-k aggregation and detection rates.

    Returns:
    - scores dict: train, val, actor2, actor1
    - detection_rate: Actor2
    - false_positive_rate: Actor1
    """

    # --- Load errors ---
    train_errors = np.load("train_errors.npy")
    val_errors = np.load("val_errors.npy")
    actor2_test_errors = np.load("actor2_test_errors.npy")
    actor1_test_errors = np.load("actor1_test_errors.npy")

    # --- Robust stats ---
    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)

    # Stabilize IQR to avoid division by very small values
    iqr = np.maximum(iqr, 0.05)

    # --- Normalize ---
    train_norm = np.abs((train_errors - median) / iqr)
    val_norm   = np.abs((val_errors - median) / iqr)
    actor2_norm = np.abs((actor2_test_errors - median) / iqr)
    actor1_norm = np.abs((actor1_test_errors - median) / iqr)

    # Clip extreme values
    train_norm = np.clip(train_norm, 0, 10)
    val_norm   = np.clip(val_norm, 0, 10)
    actor2_norm = np.clip(actor2_norm, 0, 10)
    actor1_norm = np.clip(actor1_norm, 0, 10)

    # --- Weighted Top-k aggregation ---
    n_sensors = train_norm.shape[1]
    k = max(1, int(topk_ratio * n_sensors))

    # Inverted IQR weighting (stable sensors contribute more)
    weights = 1 / (iqr + 1e-6)
    weights = weights / np.sum(weights)

    def aggregate_topk(norm_errors):
        sorted_idx = np.argsort(norm_errors, axis=1)[:, -k:]  # indices of top-k
        topk_values = np.take_along_axis(norm_errors, sorted_idx, axis=1)
        topk_weights = np.take_along_axis(weights[None, :], sorted_idx, axis=1)
        return np.sum(topk_values * topk_weights, axis=1)

    train_scores  = aggregate_topk(train_norm)
    val_scores    = aggregate_topk(val_norm)
    actor2_scores = aggregate_topk(actor2_norm)
    actor1_scores = aggregate_topk(actor1_norm)

    # --- Log compression ---
    train_scores  = np.log1p(train_scores)
    val_scores    = np.log1p(val_scores)
    actor2_scores = np.log1p(actor2_scores)
    actor1_scores = np.log1p(actor1_scores)

    # --- Threshold ---
    threshold = np.percentile(train_scores, threshold_percentile)

    # --- Detection rates ---
    detection_rate = np.mean(actor2_scores > threshold)
    false_positive_rate = np.mean(actor1_scores > threshold)

    # --- Debug prints ---
    print("Train mean:", np.mean(train_scores))
    print("Actor2 mean:", np.mean(actor2_scores))
    print("Actor1 mean:", np.mean(actor1_scores))
    print(f"Threshold ({threshold_percentile}th percentile):", threshold)
    print("Detection rate:", detection_rate)
    print("False positive rate:", false_positive_rate)

    scores = {
        "train": train_scores,
        "val": val_scores,
        "actor2": actor2_scores,
        "actor1": actor1_scores
    }

    return scores, detection_rate, false_positive_rate    

def plot_anomaly_score_distributions(scores):
    """
    Plot distribution of anomaly scores for train, val, Actor2, Actor1.
    
    Parameters:
    - scores: dict with keys 'train', 'val', 'actor2', 'actor1'
    """

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Plot each distribution
    sns.kdeplot(scores['train'], label='Train', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(scores['val'], label='Validation', color='green', fill=True, alpha=0.3)
    sns.kdeplot(scores['actor2'], label='Actor2 Test (Anomaly)', color='red', fill=True, alpha=0.3)
    sns.kdeplot(scores['actor1'], label='Actor1 Test (Normal)', color='orange', fill=True, alpha=0.3)

    plt.xlabel("Anomaly Score (log-scaled)")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def compute_detection_rates(train_scores, actor2_test_scores, actor1_test_scores):
    """
    Detection rate (Actor2) Expect: HIGH (~1.0)
    False positive rate (Actor1 Test) LOW (~0.05)
    """
    threshold = np.percentile(train_scores, 99)
    detection_rate = np.mean(actor2_test_scores > threshold)
    fp_rate = np.mean(actor1_test_scores > threshold)
    return detection_rate, fp_rate

def create_evaluation_dataloaders(config):
    # load config
    processed_data_folder = config["dataset"]["processed_folder"]
    window_size = config["dataset"]["window_size"]
    batch_size = config["training"]["batch_size"]

    train_array = np.load(f"{processed_data_folder}/train_array.npy")
    val_array = np.load(f"{processed_data_folder}/val_array.npy")
    actor2_test_array = np.load(f"{processed_data_folder}/actor2_test_array.npy")
    actor1_test_array = np.load(f"{processed_data_folder}/actor1_test_array.npy")
    
    train_dataset = TimeSeriesDataset(train_array, window_size)
    val_dataset = TimeSeriesDataset(val_array, window_size)
    actor2_test_dataset = TimeSeriesDataset(actor2_test_array, window_size)
    actor1_test_dataset = TimeSeriesDataset(actor1_test_array, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    actor2_test_loader = DataLoader(actor2_test_dataset, batch_size, shuffle=True)
    actor1_test_loader = DataLoader(actor1_test_dataset, batch_size, shuffle=True)

    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "actor2_test_loader": actor2_test_loader,
        "actor1_test_loader": actor1_test_loader
    }
    return data_loaders

def evaluate_pipeline(model, train_loader, test_loader):
    # 1. Compute raw errors
    train_errors = compute_prediction_errors(model, dataloader=train_loader)
    test_errors = compute_prediction_errors(model, dataloader=test_loader)

    # 2. Compute global (normalized) anomaly scores per timestamp
    # train_scores = compute_anomaly_score(train_errors, test_errors=train_errors)
    # test_scores = compute_anomaly_score(train_errors, test_errors=test_errors)
    
    # np.save("train_scores.npy", train_scores)
    # np.save("test_scores.npy", test_scores)

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

    # return {
    #     train_scores,
    #     test_scores,
    # #     "threshold": threshold,
    # #     "anomalies": test_anomaly_flags
    # }