import numpy as np
import joblib
import json
import yaml

def save_processed(train_array, val_array, actor2_test_array, actor1_test_array, scaler, devices, config):
    processed_data_path = config["dataset"]["processed_folder"]
    np.save(f"{processed_data_path}/train_array.npy", train_array)
    np.save(f"{processed_data_path}/val_array.npy", val_array)
    np.save(f"{processed_data_path}/actor2_test_array.npy", actor2_test_array)
    np.save(f"{processed_data_path}/actor1_test_array.npy", actor1_test_array)

    joblib.dump(scaler, f"{processed_data_path}/scaler.pkl")

    with open(f"{processed_data_path}/devices.json", "w") as f:
        json.dump(devices, f)