from src.preprocessing.preprocessing import data_preprocessing
from src.utils.save_utils import save_processed_data
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

    root = config["project_root_dir"]
    print(root)
    # # parse CLI args
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--project_root_dir", type=str)
    # parser.add_argument("--dataset_folder", type=str)
    # parser.add_argument("--dataset_name", type=str)
    # parser.add_argument("--merge_bre_cu", type=bool)
    # args = parser.parse_args()

    # # override project_root_directory
    # if args.project_root_dir:
    #     config["project_root_dir"] = args.project_root_dir

    # # override dataset folder
    # if args.dataset_folder:
    #     config["dataset"]["dataset_folder"] = args.dataset_folder

    # # override dataset name
    # if args.dataset_name:
    #     config["dataset"]["dataset_name"] = args.dataset_name

    # # override merge_bre_cu
    # if args.merge_bre_cu:
    #     config["dataset"]["merge_bre_cu"] = args.merge_bre_cu

    (train_array, val_array, actor2_test_array, actor1_test_array, 
     scaler, devices, 
     timestamps_train, timestamps_val, 
     timestamps_actor2_test, timestamps_actor1_test) = (data_preprocessing(config))

    print(len(train_array), len(val_array), len(actor2_test_array), len(actor1_test_array))
    
    save_processed_data(
        train_array,
        val_array,
        actor2_test_array,
        actor1_test_array,
        scaler,
        devices,
        timestamps_train, 
        timestamps_val, 
        timestamps_actor2_test, 
        timestamps_actor1_test,
        config
    )
    print("Preprocessing Done.")

if __name__ == "__main__":
    main_preprocess()