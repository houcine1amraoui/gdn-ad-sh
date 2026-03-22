import torch
from tqdm import tqdm
import os

# def train(model, train_loader, optimizer, epochs, exp_dir, device="cpu"):
#     for epoch in tqdm(range(epochs)):
#         model.train()
#         total_loss = 0
        
#         for x, y in train_loader:
#             x = x.to(device)
#             y = y.to(device)
            
#             pred = model(x)
#             loss = model.loss(pred, y)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")
    
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, f"{exp_dir}/gdn_checkpoint{epoch+1}.pth")


def train(model, train_loader, val_loader, optimizer, epochs, exp_dir,
          device="cpu", patience=10):
    
    os.makedirs(exp_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = model.loss(pred, y)

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

                pred = model(x)
                loss = model.loss(pred, y)

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
            }, f"{exp_dir}/best_model.pth")

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
    }, f"{exp_dir}/last_model.pth")
