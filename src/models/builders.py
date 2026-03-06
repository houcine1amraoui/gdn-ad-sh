from src.models.gdn import GDN

def build_gdn_model(num_nodes, config, device):
    model = GDN(
        number_nodes=num_nodes,
        in_dim=config["dataset"]["window_size"],
        hid_dim=config["model"]["hidden_dim"],
        topk=config["model"]["topk"],
        heads=config["model"]["heads"]
    ).to(device)
    return model