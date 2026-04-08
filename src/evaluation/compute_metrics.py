import numpy as np
import os
import yaml
import numpy as np
from scipy.stats import genpareto

from src.utils.get_folders_utils import get_evaluation_results_main_folder

def compute_metrics(config):
    """
    Detection rate (Actor2) → expect HIGH
    False positive rate (Actor1) → expect LOW
    """

    print("Computing metrics...")

    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores/scores.npz"

    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    threshold_percentile = config["evaluation"]["threshold_percentile"]

    # 🔹 choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")

    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]
    actor1_scores = scores["actor1_test"][score_type]

    # 🔹 threshold from NORMAL data only
    threshold = np.percentile(train_scores, threshold_percentile)

    # 🔹 metrics
    detection_rate = np.mean(actor2_scores > threshold)
    false_positive_rate = np.mean(actor1_scores > threshold)

    print(f"Threshold: {threshold:.4f}")
    print(f"Detection Rate (Actor2): {detection_rate:.4f}")
    print(f"False Positive Rate (Actor1): {false_positive_rate:.4f}")

    # 🔹 save properly
    metrics = {
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "score_type": score_type,
        "detection_rate": float(detection_rate),
        "false_positive_rate": float(false_positive_rate),
    }

    with open(os.path.join(eval_results_folder, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)

    return metrics

def compute_segment_metrics(pred, start, end):
    segment = pred[start:end]

    # SDR
    SDR = int(np.any(segment))

    # Coverage
    coverage = np.mean(segment)

    # Delay
    if SDR:
        delay = np.argmax(segment)
    else:
        delay = np.inf

    return SDR, coverage, delay

def fit_pot_threshold(train_scores, q=0.98, alpha=1e-3):
    """
    Fit POT (Peaks Over Threshold)

    Args:
        train_scores: np.array (normal training scores)
        q: initial threshold quantile (e.g., 0.98)
        alpha: risk level (smaller = stricter threshold)

    Returns:
        final_threshold
    """

    train_scores = np.asarray(train_scores)

    # Step 1: initial threshold u
    u = np.quantile(train_scores, q)

    # Step 2: excesses over threshold
    excesses = train_scores[train_scores > u] - u

    if len(excesses) < 10:
        raise ValueError("Not enough tail samples for POT. Increase dataset or lower q.")

    # Step 3: fit GPD
    # shape (xi), loc, scale (beta)
    xi, loc, beta = genpareto.fit(excesses, floc=0)

    # Step 4: compute final threshold τ
    n = len(train_scores)
    nu = len(excesses)

    # POT formula
    tau = u + (beta / xi) * (((n / nu) * alpha) ** (-xi) - 1)

    return tau, {"u": u, "xi": xi, "beta": beta, "n_tail": nu}

def compute_metrics_with_pot_thresholding(scores):
    # Fit POT
    threshold, info = fit_pot_threshold(scores["train"], q=0.98, alpha=1e-3)

    print("Threshold:", threshold)
    print("GPD params:", info)

    detection_rate = np.mean(scores["actor2_test"] > threshold)
    false_positive_rate = np.mean(scores["actor1_test"] > threshold)

    print(detection_rate, false_positive_rate)
