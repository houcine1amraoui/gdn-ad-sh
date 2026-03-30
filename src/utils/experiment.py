import os
from datetime import datetime
import yaml
import json
import os
import yaml

def create_experiment_folder(config, train_exeriments_per_model_folder):
    """
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    train_experiments_sub_folder = os.path.join(train_exeriments_per_model_folder, f"exp_{timestamp}")

    os.makedirs(train_experiments_sub_folder, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(train_experiments_sub_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    return train_experiments_sub_folder

def get_best_experiment(train_experiments_per_model_folder, metric_name="best_val_loss", mode="min"):
    best_value = float("inf") if mode == "min" else -float("inf")
    best_exp_path = None

    for exp_name in os.listdir(train_experiments_per_model_folder):
        exp_path = os.path.join(train_experiments_per_model_folder, exp_name)

        metrics_path = os.path.join(exp_path, "metrics.yaml")

        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = yaml.safe_load(f)

        value = metrics.get(metric_name)
        if value is None:
            continue

        if (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
            best_value = value
            best_exp_path = exp_path

    return best_exp_path, best_value

def save_experiment_config(config, flat_config, args, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Save full config (FINAL)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # 2. Save flattened config (useful for quick inspection)
    with open(os.path.join(exp_dir, "config_flat.json"), "w") as f:
        json.dump(flat_config, f, indent=4)

    # 3. Save ONLY CLI overrides
    cli_args = {
        k: v for k, v in vars(args).items() if v is not None
    }

    with open(os.path.join(exp_dir, "cli_args.json"), "w") as f:
        json.dump(cli_args, f, indent=4)