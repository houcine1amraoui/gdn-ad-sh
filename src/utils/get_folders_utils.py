from pathlib import Path

def get_dataset_path(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["dataset"]["dataset_name"]
    dataset_folder = config["dataset"]["dataset_folder"]
    data_path = f"{project_root_dir}/{dataset_folder}/{dataset_name}Master.csv"

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Data file does not exist. Please, place datasets into data folder first.")

    return data_path

def get_processed_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]
    dataset_name = config["dataset"]["dataset_name"]
    
    merge_bre_cu = config["dataset"]["merge_bre_cu"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed/{name}"
    
    path = Path(processed_data_folder)
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] Processed data folder does not exist. \
                                Please, make sure to run data preprocessing first.")


    return processed_data_folder

def get_train_experiments_main_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["dataset"]["dataset_name"]
    model_name = config["training"]["model"]
    merge_bre_cu = config["dataset"]["merge_bre_cu"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    train_experiments_main_folder = f"{project_root_dir}/train_experiments/{name}/{model_name}"

    path = Path(train_experiments_main_folder)
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] Processed data folder does not exist. \
                                Please, make sure to run model training first.")
    
    return train_experiments_main_folder

def get_evaluation_results_main_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["dataset"]["dataset_name"]
    model_name = config["training"]["model"]
    merge_bre_cu = config["dataset"]["merge_bre_cu"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    evaluation_results_main_folder = f"{project_root_dir}/eval_results/{name}/{model_name}"

    path = Path(evaluation_results_main_folder)
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] Processed data folder does not exist. \
                                Please, make sure to run evaluation first.")
    
    return evaluation_results_main_folder