import torch

def compute_anomaly_scores(model, dataloader, device="cuda"):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.abs(output - batch.mean(dim=2))
            scores.append(error.cpu())

    return torch.cat(scores)