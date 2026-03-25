import json
import yaml
from src.models.gdn import GDN
from src.utils.device import get_device

def build_gdn_model(config):
    processed_data_folder = config["dataset"]["processed_folder"]

    with open(f"{processed_data_folder}/devices.json") as f:
        devices = json.load(f)

    device = get_device()

    model = GDN(
        number_nodes=len(devices),
        in_dim=config["dataset"]["window_size"],
        hid_dim=config["model"]["hidden_dim"],
        topk=config["model"]["topk"],
        heads=config["model"]["heads"]
    ).to(device)
    return model