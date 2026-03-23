import torch
import torch.optim as optim
from src.utils.device import get_device

def load_checkpoint(model, path, optimizer):
    device = get_device()
    # if training was done on GPU but evaluation will be on CPU
    # use map_location=torch.device('cpu')
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch