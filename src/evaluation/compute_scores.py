import numpy as np
import os
from scipy.stats import ks_2samp

def compute_final_error(split_errors, alpha=0.5):
    f = split_errors["forecast"]
    r = split_errors["reconstruction"]

    if r is None:
        return f  # GDN case
    return alpha * f + (1 - alpha) * r  # MTAD-GAT case
    
def load_errors_all_splits(config):
    eval_results_folder = config["evaluation"]["eval_results_folder"]
    model_name = config["evaluation"]["model"]

    splits = ["train", "val", "actor2_test", "actor1_test"]

    errors = {}

    for split in splits:
        path = os.path.join(eval_results_folder, model_name, "errors", f"{split}.npz")
        data = np.load(path)
        errors[split] = compute_final_error(data, alpha=0.5)

    return errors

# def normalize(train_scores, scores):
#     min_v = train_scores.min()
#     max_v = train_scores.max()
#     return (scores - min_v) / (max_v - min_v + 1e-8)

def normalize_errors_all_splits(errors):
    """
    Normalize errors using train statistics only
    """
    median = np.median(errors["train"], axis=0)
    iqr = np.percentile(errors["train"], 75, axis=0) - np.percentile(errors["train"], 25, axis=0)
    # stabilize
    iqr = np.maximum(iqr, 0.05)

    # norm and clip
    train_norm = np.abs((errors["train"] - median) / iqr)
    val_norm = np.abs((errors["val"] - median) / iqr)
    actor2_test_norm = np.abs((errors["actor2_test"] - median) / iqr)
    actor1_test_norm = np.abs((errors["actor1_test"] - median) / iqr)
    
    # clip
    errors["train"] = np.clip(train_norm, 0, 10)
    errors["val"] = np.clip(val_norm, 0, 10)
    errors["actor2_test"] = np.clip(actor2_test_norm, 0, 10)
    errors["actor1_test"] = np.clip(actor1_test_norm, 0, 10)

    return errors, median, iqr

def compute_scores(norm_errors, iqr, topk_ratio=0.4):
    n_sensors = norm_errors.shape[1]
    k = max(1, int(topk_ratio * n_sensors))

    # weight stable sensors more
    weights = 1 / (iqr + 1e-6)
    weights = weights / np.sum(weights)

    # top-k selection
    idx = np.argsort(norm_errors, axis=1)[:, -k:]

    topk_vals = np.take_along_axis(norm_errors, idx, axis=1)
    topk_weights = np.take_along_axis(weights[None, :], idx, axis=1)

    scores = np.sum(topk_vals * topk_weights, axis=1)

    # log compression
    scores = np.log1p(scores)

    return scores

def evalutation_pipeline(config):
    errors = load_errors_all_splits(config)
    errors_norm, median, iqr = normalize_errors_all_splits(errors)

    train_scores = compute_scores(errors_norm["train"], iqr)
    val_scores = compute_scores(errors_norm["val"], iqr)
    actor2_scores = compute_scores(errors_norm["actor2_test"], iqr)
    actor1_scores = compute_scores(errors_norm["actor1_test"], iqr)
    
    threshold = np.percentile(train_scores, 95)
    
    # KS Test
    # ks_actor2 = ks_2samp(train_scores, actor2_scores)
    # ks_actor1 = ks_2samp(train_scores, actor1_scores)
    # print("KS Train vs Actor2:", ks_actor2)
    # print("KS Train vs Actor1:", ks_actor1)

    detection_rate = np.mean(actor2_scores > threshold)
    fp_rate = np.mean(actor1_scores > threshold)
    print("Detection rate:", detection_rate)
    print("False positive rate:", fp_rate)

    # Aggregation
    # train_scores = errors["train"].mean(axis=1)
    # val_scores = errors["val"].mean(axis=1)
    # actor2_test_scores = errors["actor2_test"].mean(axis=1)
    # actor1_test_scores = errors["actor1_test"].mean(axis=1)
    
    # Normalization
    # val_scores = normalize(train_scores, val_scores)
    # actor2_test_scores = normalize(train_scores, actor2_test_scores)
    # actor1_test_scores = normalize(train_scores, actor1_test_scores)

    # scores = {
    #     "train": train_scores,
    #     "val": val_scores,
    #     "actor2_test": actor2_test_scores,
    #     "actor1_test": actor1_test_scores
    # }

    # return scores
    # Aggregate errors
    # forecast = 
    # Normalize
    # train_scores = normalize()
    # # Robust Stats
    # median = np.median(train_errors, axis=0)
    # iqr = np.percentile(train_errors, 75, axis=0) - np.percentile(train_errors, 25, axis=0)
    # iqr[iqr == 0] = 1e-6  # avoid division by zero

    # # --- Normalize ---
    # train_norm = (train_errors - median) / iqr
    # val_norm = (val_errors - median) / iqr
    # actor2_test_norm = (actor2_test_errors - median) / iqr
    # actor1_test_norm = (actor1_test_errors - median) / iqr

    # train_scores = np.mean(train_norm, axis=1)
    # val_scores = np.mean(val_norm, axis=1)
    # actor2_test_scores = np.mean(actor2_test_norm, axis=1)
    # actor1_test_scores = np.mean(actor1_test_norm, axis=1)
    
    # scores = {
    #     "train": train_scores,
    #     "val": val_scores,
    #     "actor2_test": actor2_test_scores,
    #     "actor1_test": actor1_test_scores
    # }

    # return scores