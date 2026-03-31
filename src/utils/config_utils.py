def get_dataset_name(config):
    merge_bre_cu = config["dataset"]["merge_bre_cu"]
    dataset_name = config["dataset"]["dataset_name"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    return name