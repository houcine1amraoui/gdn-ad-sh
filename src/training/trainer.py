import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, device="cpu"):
    # 6. Training Loop
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
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
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "gdn_checkpoint50.pth")
    # torch.save(model.state_dict(), "gdn_actor1.pth")

# def train2(model, dataloader, epochs=50, lr=1e-3, device="cpu"):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0

#         for batch in dataloader:
#             batch = batch.to(device)

#             optimizer.zero_grad()
#             output = model(batch)

#             loss = criterion(output, batch.mean(dim=2))
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")