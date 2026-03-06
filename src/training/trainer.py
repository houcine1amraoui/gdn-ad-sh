import torch
from tqdm import tqdm
import os

def train(model, train_loader, optimizer, epochs, exp_dir, device="cpu"):
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = model.loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")
    
    torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pth"))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "gdn_checkpoint50.pth")