from src.preprocessing.preprocessing import data_preprocessing
from src.utils.save_utils import save_processed
import yaml
import argparse

def main_preprocess():
    # 1. Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    # override dataset path
    if args.data_path:
        config["dataset"]["path"] = args.data_path
        
    data_path = config["dataset"]["path"]
    train_array, test_array_actor1, test_array_actor2, scaler, sensors = data_preprocessing(data_path)

    save_processed(
        train_array,
        test_array_actor1,
        test_array_actor2,
        scaler,
        sensors
    )
    print("Preprocessing Done.")

if __name__ == "__main__":
    main_preprocess()