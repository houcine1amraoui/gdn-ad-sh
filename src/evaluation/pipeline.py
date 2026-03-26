import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import os
import yaml
from scipy.stats import ks_2samp

from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from src.models.builders import build_gdn_model
from src.evaluation.load_checkpoint import load_checkpoint
from src.utils.experiment import get_best_experiment
from src.utils.device import get_device

def compute_metrics(scores, config):
    """
    Detection rate (Actor2) Expect: HIGH (~1.0)
    False positive rate (Actor1 Test) LOW (~0.05)
    """
    threshold_percentile = config["evaluation"]["threshold_percentile"]
    eval_results_folder = config["evaluation"]["eval_results_folder"]

    # --- Threshold ---
    threshold = np.percentile(scores["train"], threshold_percentile)
    # actor2_test_threshold = np.percentile(scores["actor2_test"], threshold_percentile)
    # actor1_test_threshold = np.percentile(scores["actor1_test"], threshold_percentile)
    # stat, p = ks_2samp(scores["train"], scores["actor2_test"])

    detection_rate = np.mean(scores["actor2_test"] > threshold)
    false_positive_rate = np.mean(scores["actor1_test"] > threshold)

    with open(os.path.join(f"{eval_results_folder}/metrics.yaml"), "w") as f:
         yaml.dump({"detection_rate": float(detection_rate)}, f)
         yaml.dump({"false_positive_rate": float(false_positive_rate)}, f)

def errors_computation_pipeline(config):
    train_experiments_main_folder = config["training"]["train_experiments_main_folder"]
    eval_results_folder = config["evaluation"]["eval_results_folder"]

    best_exp_path, _ = get_best_experiment(train_experiments_main_folder)

    # 1. Model Initialization
    model_arch = build_gdn_model("gdn", config)

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
    compute_prediction_errors(model, data_loaders, eval_results_folder)

def compute_prediction_errors_per_loader(model, dataloader):
    """
    Compute raw errors
    """
    device = get_device()
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

def compute_prediction_errors(model, data_loaders, eval_results_folder):
    train_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["train_loader"])
    val_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["val_loader"])
    actor2_test_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["actor2_test_loader"])
    actor1_test_errors = compute_prediction_errors_per_loader(model, dataloader=data_loaders["actor1_test_loader"])
    
    errors_folder = f"{eval_results_folder}/errors"
    os.makedirs(errors_folder, exist_ok=True)

    np.save(f"{errors_folder}/train_errors.npy", train_errors)
    np.save(f"{errors_folder}/val_errors.npy", val_errors)
    np.save(f"{errors_folder}/actor2_test_errors.npy", actor2_test_errors)
    np.save(f"{errors_folder}/actor1_test_errors.npy", actor1_test_errors)

def compute_anomaly_scores(config):
    eval_results_folder = config["evaluation"]["eval_results_folder"]

    # --- Load errors ---
    train_errors = np.load(f"{eval_results_folder}/errors/train_errors.npy")
    val_errors = np.load(f"{eval_results_folder}/errors/val_errors.npy")
    actor2_test_errors = np.load(f"{eval_results_folder}/errors/actor2_test_errors.npy")
    actor1_test_errors = np.load(f"{eval_results_folder}/errors/actor1_test_errors.npy")

    # Robust Stats
    median = np.median(train_errors, axis=0)
    iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    iqr[iqr == 0] = 1e-6  # avoid division by zero

    # --- Normalize ---
    train_norm = (train_errors - median) / iqr
    val_norm = (val_errors - median) / iqr
    actor2_test_norm = (actor2_test_errors - median) / iqr
    actor1_test_norm = (actor1_test_errors - median) / iqr

    train_scores = np.mean(train_norm, axis=1)
    val_scores = np.mean(val_norm, axis=1)
    actor2_test_scores = np.mean(actor2_test_norm, axis=1)
    actor1_test_scores = np.mean(actor1_test_norm, axis=1)
    
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

def create_evaluation_dataloaders(config):
    # load config
    processed_data_folder = config["dataset"]["processed_data_folder"]
    window_size = config["dataset"]["window_size"]
    batch_size = config["evaluation"]["batch_size"]

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

    
