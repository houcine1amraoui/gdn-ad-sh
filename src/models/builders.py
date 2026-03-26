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
        model = GDN(
            number_nodes=len(devices),
            in_dim=config["dataset"]["window_size"],
            hid_dim=config["models"][0]["params"]["hidden_dim"],
            topk=config["models"][0]["params"]["topk"],
            heads=config["models"][0]["params"]["heads"]
        )
    elif name == "mtad_gat":
        model = MTAD_GAT()
    else:
        raise ValueError(f"Unknown model: {name}")
    
    model.to(device)
    return model