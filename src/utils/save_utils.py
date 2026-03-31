import numpy as np
import joblib
import json
import os
from src.utils.get_folders_utils import get_dataset_name

def save_processed_data(train_array, val_array, actor2_test_array, 
                   actor1_test_array, scaler, devices, 
                   timestamps_train, timestamps_val, 
                   timestamps_actor2_test, timestamps_actor1_test,
                   config):
    
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]
    dataset_name = get_dataset_name(config)
    
    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed/{dataset_name}"

    # Create folder if it doesn't exist
    os.makedirs(processed_data_folder, exist_ok=True)

    np.save(f"{processed_data_folder}/train_array.npy", train_array)
    np.save(f"{processed_data_folder}/val_array.npy", val_array)
    np.save(f"{processed_data_folder}/actor2_test_array.npy", actor2_test_array)
    np.save(f"{processed_data_folder}/actor1_test_array.npy", actor1_test_array)

    np.save(f"{processed_data_folder}/train_timestamps.npy", timestamps_train)
    np.save(f"{processed_data_folder}/val_timestamps.npy", timestamps_val)
    np.save(f"{processed_data_folder}/actor2_test_timestamps.npy", timestamps_actor2_test)
    np.save(f"{processed_data_folder}/actor1_test_timestamps.npy", timestamps_actor1_test)

    joblib.dump(scaler, f"{processed_data_folder}/scaler.pkl")

    with open(f"{processed_data_folder}/devices.json", "w") as f:
        json.dump(devices, f)