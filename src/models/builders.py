import json
from src.models.gdn import GDN
from src.models.mtad_gat import MTAD_GAT
from src.utils.device import get_device 
    
def build_gdn_model(name, config):
    processed_data_folder = config["dataset"]["processed_data_folder"]

    with open(f"{processed_data_folder}/devices.json") as f:
        devices = json.load(f)

    device = get_device()
    model = None

    if name == "gdn":
        params = config["models"][0]["params"]
        model = GDN(
            number_nodes=len(devices),
            in_dim=config["dataset"]["window_size"],
            hid_dim=params["hidden_dim"],
            topk=params["topk"],
            heads=params["heads"]
        ).to(device)
    elif name == "mtad_gat":
        params = config["models"][0]["params"]
        model = MTAD_GAT().to(device)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    return model