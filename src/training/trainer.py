import torch
import torch.nn as nn

def train(model, dataloader, epochs=50, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            output = model(batch)

            loss = criterion(output, batch.mean(dim=2))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")