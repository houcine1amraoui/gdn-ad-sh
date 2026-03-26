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

        model = MTAD_GAT(
            n_features=len(devices),
            window_size=config["dataset"]["window_size"],
            out_dim=len(devices),

            kernel_size=params.get("kernel_size", 7),
            feat_gat_embed_dim=params.get("feat_gat_embed_dim", None),
            time_gat_embed_dim=params.get("time_gat_embed_dim", None),

            gru_n_layers=params.get("gru_n_layers", 1),
            gru_hid_dim=params.get("gru_hid_dim", 150),

            forecast_n_layers=params.get("forecast_n_layers", 1),
            forecast_hid_dim=params.get("forecast_hid_dim", 150),

            recon_n_layers=params.get("recon_n_layers", 1),
            recon_hid_dim=params.get("recon_hid_dim", 150),

            dropout=params.get("dropout", 0.2),
            alpha=params.get("alpha", 0.2)
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    return model