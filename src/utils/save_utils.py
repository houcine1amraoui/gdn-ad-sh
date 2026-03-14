import numpy as np
import joblib
import json
import yaml

def save_processed(train_array, test_array_actor1, test_array_actor2, scaler, sensor_columns):

    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    processed_data_path = config["dataset"]["processed_folder"]

    np.save(f"{processed_data_path}/train_array.npy", train_array)
    np.save(f"{processed_data_path}/test_array_actor1.npy", test_array_actor1)
    np.save(f"{processed_data_path}/test_array_actor2.npy", test_array_actor2)

    joblib.dump(scaler, f"{processed_data_path}/scaler.pkl")

    with open(f"{processed_data_path}/sensors.json", "w") as f:
        json.dump(sensor_columns, f)