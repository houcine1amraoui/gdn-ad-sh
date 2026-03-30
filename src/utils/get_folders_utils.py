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
