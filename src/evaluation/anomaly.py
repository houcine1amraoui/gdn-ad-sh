import torch
import numpy as np
from tqdm import tqdm

def compute_errors(model, dataloader, device="cpu"):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            
            err = torch.abs(pred - y)
            errors.append(err.cpu().numpy())
    
    return np.concatenate(errors, axis=0)  # [T, N]

def compute_anomaly_scores(model, dataloader, device="cpu"):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.abs(output - batch.mean(dim=2))
            scores.append(error.cpu())

    return torch.cat(scores)