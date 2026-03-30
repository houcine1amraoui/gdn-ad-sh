import os
from datetime import datetime
import yaml

def create_time_folder(config, parent_folder):
    """
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    time_folder = os.path.join(parent_folder, f"exp_{timestamp}")

    os.makedirs(time_folder, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(time_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    return time_folder

def create_train_experiments_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["dataset"]["dataset_name"]
    merge_bre_cu = config["dataset"]["merge_bre_cu"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    train_experiments_main_folder = f"{project_root_dir}/train_experiments/{name}"
    # Create a folder if it doesn't exist
    os.makedirs(train_experiments_main_folder, exist_ok=True)

    model_name = config["training"]["model"]
    train_exeriments_per_model_folder = f"{train_experiments_main_folder}/{model_name}"
    # Create a folder if it doesn't exist
    os.makedirs(train_exeriments_per_model_folder, exist_ok=True)    

    time_folder = create_time_folder(config, train_exeriments_per_model_folder)
    
    pass