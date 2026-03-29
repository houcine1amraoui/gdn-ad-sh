import torch
from tqdm import tqdm
import os
import yaml
import torch.optim as optim

from src.utils.device import get_device

def train(
    model,
    train_loader,
    val_loader,
    train_experiments_sub_folder,
    config,
    grad_accum_steps=1  # set >1 if you want effective larger batch size
):

    device = get_device()
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    base_lr = config["training"]["lr"]

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # LR Scheduler (Reduce LR on plateau)
    # lder versions of PyTorch don’t accept verbose in the ReduceLROnPlateau constructor.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        # verbose=True,
        min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss = 0

        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            batch = {"x": x, "y": y}
            output = model(x)
            loss = model.loss(batch, output)

            # Gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps  # scale back for logging

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
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

        # -------------------------
        # LR Scheduler step
        # -------------------------
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # -------------------------
        # Save BEST model
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, f"{train_experiments_sub_folder}/best.pth")

            with open(
                os.path.join(train_experiments_sub_folder, "metrics.yaml"), "w"
            ) as f:
                yaml.dump({
                    "best_val_loss": float(best_val_loss),
                    "lr": float(current_lr)
                }, f)

        else:
            patience_counter += 1

        # -------------------------
        # Early stopping
        # -------------------------
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # -------------------------
    # Save LAST model
    # -------------------------
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{train_experiments_sub_folder}/last_model.pth")