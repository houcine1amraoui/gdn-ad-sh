import numpy as np

def compute_anomaly_score_gdn(batch, output):
    pass

def compute_anomaly_score_mtad(batch, output):
    pass

def compute_anomaly_score(model, batch, output):
    if hasattr(model, "recon_model"):
        return compute_anomaly_score_mtad(batch, output)
    else:
        return compute_anomaly_score_gdn(batch, output)

def compute_scores(config):
    eval_results_folder = config["evaluation"]["eval_results_folder"]
    model_name = config["evaluation"]["model"]

    # --- Load errors ---
    train_errors = np.load(f"{eval_results_folder}/{model_name}/errors/train_errors.npz")
    val_errors = np.load(f"{eval_results_folder}/{model_name}/errors/val_errors.npz")
    actor2_test_errors = np.load(f"{eval_results_folder}/{model_name}/errors/actor2_test_errors.npz")
    actor1_test_errors = np.load(f"{eval_results_folder}/{model_name}/errors/actor1_test_errors.npz")

    forecast = train_errors["forecast"]
    recon = train_errors["reconstruction"] if "reconstruction" in train_errors else None

    print(forecast)
    print(recon)
    

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