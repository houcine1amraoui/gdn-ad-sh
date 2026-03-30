def get_processed_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]
    dataset_name = config["dataset"]["dataset_name"]
    
    merge_bre_cu = config["dataset"]["merge_bre_cu"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed/{name}"
    
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

    return train_experiments_main_folder