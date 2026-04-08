import numpy as np
import os

from src.utils.get_folders_utils import get_evaluation_results_main_folder

def compute_scores(config, topk_ratio=0.4, combine_errors=True, alpha=0.5):
    """
    Compute anomaly scores from normalized errors.

    Parameters:
    - combine_errors: if True, combine forecast + reconstruction
    - alpha: weight for forecast when combining
    """
    print("Computing scores...")

    eval_results_folder = get_evaluation_results_main_folder(config)

    data = np.load(f"{eval_results_folder}/errors/norm_errors.npz", allow_pickle=True)
    norm_errors = data["arr_0"].item()  # dict

    scores = {}
    iqr_dict = {}  # optional: return for analysis

    # 🔹 compute IQR from TRAIN ONLY
    for error_type in norm_errors["train"].keys():
        train_err = norm_errors["train"][error_type]

        iqr = np.percentile(train_err, 75, axis=0) - np.percentile(train_err, 25, axis=0)

        # stabilization
        iqr_floor = 0.1 * np.median(iqr)
        iqr = np.maximum(iqr, iqr_floor)

        iqr_dict[error_type] = iqr

    # 🔹 compute scores per split
    for split in ["train", "val", "actor2_test", "actor1_test"]:

        split_scores = {}

        for error_type in norm_errors[split].keys():

            e = norm_errors[split][error_type]  # (T, n_sensors)
            iqr = iqr_dict[error_type]

            n_sensors = e.shape[1]
            k = max(1, int(topk_ratio * n_sensors))

            # weights (stable sensors ↑)
            weights = 1 / (iqr + 1e-6)
            weights = weights / np.sum(weights)

            # top-k
            idx = np.argsort(e, axis=1)[:, -k:]

            topk_vals = np.take_along_axis(e, idx, axis=1)
            topk_weights = np.take_along_axis(weights[None, :], idx, axis=1)

            s = np.sum(topk_vals * topk_weights, axis=1)

            # log compression
            s = np.log1p(s)

            split_scores[error_type] = s

        # 🔥 optional combination
        # combine_errors = config["evaluation"].get("combine_errors", False)
        if combine_errors and "reconstruction" in split_scores:
            combined = (
                alpha * split_scores["forecast"]
                + (1 - alpha) * split_scores["reconstruction"]
            )
        else:
            combined = split_scores["forecast"]

        scores[split] = {
            "forecast": split_scores.get("forecast"),
            "reconstruction": split_scores.get("reconstruction"),
            "combined": combined,
        }

    # create errors folders
    scores_folder = f"{eval_results_folder}/scores"
    os.makedirs(scores_folder, exist_ok=True)
    
    np.savez(
        f"{scores_folder}/scores.npz",
            scores=scores,
        )