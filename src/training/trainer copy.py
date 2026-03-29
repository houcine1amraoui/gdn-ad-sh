import torch
from tqdm import tqdm
import os
import yaml
import torch.optim as optim

from src.utils.device import get_device

def train(model, train_loader, val_loader, train_experiments_sub_folder, config):
    
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    device = get_device()

    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            batch = {"x": x, "y": y}
            output = model(x)
            loss = model.loss(batch, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                batch = {"x": x, "y": y}
                output = model(x)
                loss = model.loss(batch, output)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save BEST model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, f"{train_experiments_sub_folder}/best.pth")

            # Save metric
            with open(os.path.join(f"{train_experiments_sub_folder}/metrics.yaml"), "w") as f:
                yaml.dump({"best_val_loss": float(best_val_loss)}, f)

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save LAST model (optional)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{train_experiments_sub_folder}/last_model.pth")
