import numpy as np
import joblib
import json

def save_processed(train_array, test_array_actor1, test_array_actor2, scaler, sensor_columns):

    np.save("data/processed/train.npy", train_array)
    np.save("data/processed/test_actor1.npy", test_array_actor1)
    np.save("data/processed/test_actor2.npy", test_array_actor2)

    joblib.dump(scaler, "data/processed/scaler.pkl")

    with open("data/processed/sensors.json", "w") as f:
        json.dump(sensor_columns, f)