from src.preprocessing.preprocessing import data_preprocessing
from src.utils.save_utils import save_processed
import yaml
import argparse

def main_preprocess():
    """
    Typical preprocessing pipeline:
    1s-raw data -> actors split -> noise filtering -> 5s-downsampling 
    -> normalization -> sliding windows
    """
    # 1. Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--processed_folder", type=str)
    args = parser.parse_args()

    # override dataset path
    if args.data_path:
        config["dataset"]["path"] = args.data_path

    # override processed data folder path
    if args.processed_folder:
        config["dataset"]["processed_folder"] = args.processed_folder
        
    data_path = config["dataset"]["path"]
    
    train_array, val_array, test_array, scaler, devices = data_preprocessing(data_path)
    print(len(train_array), len(val_array), len(test_array))
    save_processed(
        train_array,
        val_array,
        test_array,
        scaler,
        devices,
        config
    )
    print("Preprocessing Done.")

if __name__ == "__main__":
    main_preprocess()