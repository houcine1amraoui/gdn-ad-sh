import torch.optim as optim
import yaml

from src.dataset.preprocessing import prepare_data, create_dataloaders
from src.models.builders import build_gdn_model
from src.training.trainer import train
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.evaluation.load_checkpoint import load_checkpoint
from src.evaluation.anomaly import compute_errors
from src.visualization.plot_scores import plot_scores
from src.utils.experiment import create_experiment_folder
from src.utils.io import save_scores

def main():
    # 1. Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # 2. Set seed
    set_seed(config["seed"])

    device = get_device()
    # 
    exp_dir = create_experiment_folder(config)

    # 1. Data Preparation
    data_path = config["dataset"]["path"]
    train_array, test_array_actor1, test_array_actor2, sensor_columns = prepare_data(data_path)
  
    # 2. Dataset/DataLoader creation
    window_size = config["dataset"]["window_size"]
    train_loader, test_loader_actor1, test_loader_actor2 = create_dataloaders(
        train_array, test_array_actor1, test_array_actor2, window_size)

    # 3. Model Initialization
    model = build_gdn_model(len(sensor_columns), config, device)

    # 4. Train or load checkpoint
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    if config["training"]["train"]:
        train(model, train_loader, optimizer, config["training"]["epochs"], exp_dir, device)
    else:
        model, optimizer, _ = load_checkpoint(
            model,
            config["checkpoint"]["path"],
            optimizer
        )

    # 5. Evaluation
    test_errors_actor1 = compute_errors(model, test_loader_actor1, device)
    test_errors_actor2 = compute_errors(model, test_loader_actor2, device)
    save_scores(test_errors_actor1, exp_dir)
    save_scores(test_errors_actor2, exp_dir)

    # 6. Visualization
    plot_scores(test_errors_actor1, exp_dir)
    # plot_scores(test_errors_actor1, test_errors_actor2, exp_dir)


if __name__ == "__main__":
    main()