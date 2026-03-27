import numpy as np
import os

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

def normalize(train_scores, scores):
    min_v = train_scores.min()
    max_v = train_scores.max()
    return (scores - min_v) / (max_v - min_v + 1e-8)

def compute_scores(config):
    errors = load_errors_all_splits(config)

    # Aggregation
    train_scores = errors["train"].mean(axis=1)
    val_scores = errors["val"].mean(axis=1)
    actor2_test_scores = errors["actor2_test"].max(axis=1)
    actor1_test_scores = errors["actor1_test"].max(axis=1)
    
    # Normalization
    val_scores = normalize(train_scores, val_scores)
    actor2_test_scores = normalize(train_scores, actor2_test_scores)
    actor1_test_scores = normalize(train_scores, actor1_test_scores)

    scores = {
        "train": train_scores,
        "val": val_scores,
        "actor2_test": actor2_test_scores,
        "actor1_test": actor1_test_scores
    }

    return scores
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