# src/train_classifier.py
import torch
from torch import nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
import time
from tqdm.auto import tqdm
from src import config # Import centralized configuration

def create_dataloaders(root_dir, batch_size, num_workers):
    """Creates training and testing DataLoaders for Food101."""
    print(f"Creating DataLoaders with batch size {batch_size}...")

    weights = models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()

    train_dataset = datasets.Food101(root=root_dir, split="train", download=True, transform=auto_transforms)
    test_dataset = datasets.Food101(root=root_dir, split="test", download=True, transform=auto_transforms)
    class_names = train_dataset.classes

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("✅ DataLoaders created successfully.")
    return train_dataloader, test_dataloader, class_names

def build_model(num_classes, device):
    """Builds an EfficientNet_B0 model with a custom classifier head."""
    print("Building model...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes)
    ).to(device)

    print(f"✅ Model built and moved to {device}.")
    return model

def train_step(model, dataloader, loss_fn, optimizer, device):
    """Performs a single training step."""
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    return train_loss / len(dataloader), train_acc / len(dataloader)

def test_step(model, dataloader, loss_fn, device):
    """Performs a single testing step."""
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    return test_loss / len(dataloader), test_acc / len(dataloader)

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, save_path):
    """Trains and evaluates the model."""
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best test accuracy: {best_test_acc:.4f}. Saving model to {save_path}...")
            torch.save(model.state_dict(), save_path)

    return results

if __name__ == '__main__':
    print(f"--- Starting Food101 Classification Training ---")
    print(f"Using device: {config.DEVICE}")

    train_loader, test_loader, class_names = create_dataloaders(
        root_dir=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    model = build_model(num_classes=len(class_names), device=config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CLASSIFICATION_LEARNING_RATE)

    start_time = time.time()
    train(model=model,
          train_dataloader=train_loader,
          test_dataloader=test_loader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          epochs=config.CLASSIFICATION_EPOCHS,
          device=config.DEVICE,
          save_path=config.CLASSIFICATION_MODEL_SAVE_PATH)
    end_time = time.time()

    print(f"✅ Total training time: {end_time - start_time:.2f} seconds")
