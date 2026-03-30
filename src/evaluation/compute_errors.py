from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import os
import torch.optim as optim
import torch

from src.utils.experiment import get_best_experiment
from src.utils.device import get_device
from src.models.builder import build_model

def load_checkpoint(model, path, optimizer):
    device = get_device()
    # if training was done on GPU but evaluation will be on CPU
    # use map_location=torch.device('cpu')
    checkpoint = torch.load(path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch

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

def compute_errors_per_loader(model, dataloader):
    device = get_device()
    model.eval()

    forecast_errors = []
    recon_errors = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            # 🔵 Case 1: MTAD-GAT (dict output)
            if isinstance(output, dict):
                pred = output["pred"]
                recon = output.get("recon", None)
            else:
                # 🔵 Case 2: GDN (tensor output)
                pred = output
                recon = None

            # --- Forecast error ---
            f_err = torch.abs(pred - y)   # (B, k)
            forecast_errors.append(f_err.cpu().numpy())

            # --- Reconstruction error (if exists) ---
            if recon is not None:
                r_err = torch.abs(recon - x)      # (B, n, k)
                r_err_last = r_err[:, -1, :]      # align with prediction
                recon_errors.append(r_err_last.cpu().numpy())

    forecast_errors = np.concatenate(forecast_errors, axis=0)

    if len(recon_errors) > 0:
        recon_errors = np.concatenate(recon_errors, axis=0)
    else:
        recon_errors = None

    return {
        "forecast": forecast_errors,   # shape [T, k]
        "reconstruction": recon_errors # shape [T, k] or None
    }

def compute_errors_all_loaders(model, eval_results_per_model_folder, config):
    data_loaders = create_evaluation_dataloaders(config)

    train_errors = compute_errors_per_loader(model, data_loaders["train_loader"])
    val_errors = compute_errors_per_loader(model, data_loaders["val_loader"])
    actor2_test_errors = compute_errors_per_loader(model, data_loaders["actor2_test_loader"])
    actor1_test_errors = compute_errors_per_loader(model, data_loaders["actor1_test_loader"])

    errors_folder = f"{eval_results_per_model_folder}/errors"
    os.makedirs(errors_folder, exist_ok=True)

    # Check once
    has_recon = train_errors["reconstruction"] is not None

    def save_split(name, errors):
        if has_recon:
            np.savez(
                f"{errors_folder}/{name}.npz",
                forecast=errors["forecast"],
                reconstruction=errors["reconstruction"]
            )
        else:
            np.savez(
                f"{errors_folder}/{name}.npz",
                forecast=errors["forecast"]
            )

    save_split("train", train_errors)
    save_split("val", val_errors)
    save_split("actor2_test", actor2_test_errors)
    save_split("actor1_test", actor1_test_errors)

def compute_errors(config):
    project_root_dir = config["project_root_dir"]
    model_name = config["evaluation"]["model"]

    train_experiments_main_folder = f"{project_root_dir}/train_experiments"
    train_experiments_per_model_folder = f"{train_experiments_main_folder}/{model_name}"
    
    eval_results_folder = f"{project_root_dir}/eval_results"
    eval_results_per_model_folder = f"{eval_results_folder}/{model_name}"
    # Create a folder if it doesn't exist
    os.makedirs(eval_results_per_model_folder, exist_ok=True)

    best_exp_path, _ = get_best_experiment(train_experiments_per_model_folder)

    # Intitialize model
    model_arch = build_model(model_name, config)

    # load best checkpoint
    optimizer = optim.Adam(model_arch.parameters(), lr=config["training"]["lr"])
    model, optimizer, _ = load_checkpoint(
        model_arch,
        f"{best_exp_path}/best.pth",
        optimizer
    )

    # Compute errors for all loaders
    compute_errors_all_loaders(model, eval_results_per_model_folder, config)