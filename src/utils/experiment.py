import os
from datetime import datetime
import yaml
import json
import argparse
from types import SimpleNamespace
from src.utils.device import get_device
from src.utils.seed import set_seed

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

import os
import yaml

def get_best_experiment(train_experiments_main_folder, model_name, metric_name="best_val_loss", mode="min"):
    best_value = float("inf") if mode == "min" else -float("inf")
    best_exp_path = None

    model_exp_dir = f"{train_experiments_main_folder}/{model_name}"

    for exp_name in os.listdir(model_exp_dir):
        exp_path = os.path.join(model_exp_dir, exp_name)

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

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items

def set_nested_value(config, key, value, sep="."):
    keys = key.split(sep)
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def load_config_and_setup(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    flat_config = flatten_dict(config)

    parser = argparse.ArgumentParser()

    for key, value in flat_config.items():
        arg_type = type(value)

        if arg_type is bool:
            parser.add_argument(f"--{key}", type=str)
        else:
            parser.add_argument(f"--{key}", type=arg_type)

    args = parser.parse_args()
    print("args: ", args)
    # Apply overrides
    for key in flat_config.keys():
        print("key:", key)
        arg_name = key.replace(".", "_")
        arg_value = getattr(args, arg_name, None)
        
        if arg_value is not None:
            if isinstance(flat_config[key], bool):
                arg_value = arg_value.lower() in ["true", "1", "yes"]

            set_nested_value(config, key, arg_value)

    # Setup
    device = get_device()

    raw_data_path = config["dataset"]["raw_data_path"]
    processed_data_folder = config["dataset"]["processed_data_folder"]
    experiments_folder = config["experiments_folder"]

    exp_dir = create_experiment_folder(config, experiments_folder)

    with open(os.path.join(processed_data_folder, "devices.json"), "r") as f:
        devices = json.load(f)
    
    # after config is finalized (after CLI override)
    seed = config.get("seed", 42)
    set_seed(seed)

    # Save experiment config
    final_flat_config = flatten_dict(config)
    save_experiment_config(config, final_flat_config, args, exp_dir)


    setup = {
        "raw_data_path": raw_data_path,
        "processed_data_folder": processed_data_folder,
        "experiments_folder": experiments_folder,
        "device": device,
        "exp_dir": exp_dir,
        "devices": devices,
        "window_size": config["dataset"]["window_size"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
        "lr": config["training"]["lr"],
    }

    return SimpleNamespace(**setup)
