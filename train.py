import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitClassifierCNN


# Runs a single training epoch over the dataset
def train_epoch(model, train_loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        # Every 64 batch goes through this cycle:
        # 1. Clear gradients from previous iteration
        # 2. Forward pass: compute predictions
        # 3. Calculate loss
        # 4. Backward pass: compute gradients
        # 5. Optimizer step: update weights
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Track statistics of this epoch
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main():
    # Pure CPU training for now
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Standard MNIST data preprocessing.
    # Convert pixel ints [0 - 255] to floats [0.0 - 1.0].
    # Subtracts the mean and then divide by standart deviation to get inputs
    # near 0, which helps to converge faster and more reliably
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    print("Loading MNIST dataset...")
    data_dir = "./data"
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Feeds images in batches of 64 and shuffles them
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = DigitClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5 # MODIFY EPOCHS HERE but 5 seems like a sweet spot
    print(f"\nStarting training for {epochs} epochs. Do not exit...\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} complete | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%"
        )

    # Save the model weights
    save_path = "model.pt"
    torch.save(model.state_dict(), f"{save_path}")
    print(f"\nTraining complete! Model weights saved to '{save_path}'")


if __name__ == "__main__":
    main()
