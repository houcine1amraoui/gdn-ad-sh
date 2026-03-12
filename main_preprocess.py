from src.dataset.preprocessing import data_preprocessing
from src.utils.save_utils import save_processed
import yaml

def main_preprocess():
    # 1. Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

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