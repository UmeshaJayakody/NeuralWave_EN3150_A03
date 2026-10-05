"""
Shared training/evaluation helpers used by the optimizer tuning, transfer learning, and final report notebooks.
All models here end their forward pass in Softmax (per the assignment spec),
so training uses NLLLoss on log(probs) -- equivalent to CrossEntropyLoss on logits.
"""

import io
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_score, recall_score,
)
from torchinfo import summary


def train_model(model, train_loader, val_loader, optimizer, epochs=20, device="cpu"):
    """Trains `model` and returns (history, avg_seconds_per_epoch)."""
    model.to(device)
    criterion = nn.NLLLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    epoch_times = []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = model(images)
            loss = criterion(torch.log(probs.clamp(min=1e-12)), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss, train_acc = running_loss / total, correct / total
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        epoch_times.append(time.time() - start)

        print(f"epoch {epoch + 1}/{epochs}  train_loss {train_loss:.4f} train_acc {train_acc:.3f}  "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.3f}")

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    return history, avg_epoch_time


def evaluate_loss_acc(model, loader, criterion, device="cpu"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs = model(images)
            loss = criterion(torch.log(probs.clamp(min=1e-12)), labels)
            running_loss += loss.item() * images.size(0)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def predict(model, loader, device="cpu"):
    """Returns (y_true, y_pred) as numpy arrays for a full pass over `loader`."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = model(images.to(device))
            y_pred.extend(probs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def evaluate_classification(model, loader, device="cpu", class_names=None):
    """Runs `model` over `loader` and returns a dict of test-set metrics.
    `report` is sklearn's per-class precision/recall/F1 table (as text)."""
    y_true, y_pred = predict(model, loader, device)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model):
    """Actual on-disk size (KB) of the model's state_dict, the way it would be shipped."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024


def count_macs(model, img_size=64):
    """Multiply-accumulate operations for one forward pass of a single RGB image (torchinfo).
    Hardware-independent compute cost -- frozen layers count too, they still run at inference."""
    stats = summary(model, input_size=(1, 3, img_size, img_size), device="cpu", verbose=0)
    return stats.total_mult_adds


def inference_latency_ms(model, batch_size=1, img_size=64, warmup=20, runs=200):
    """Mean CPU wall-clock time (ms) of one forward pass over a batch, measured after warm-up."""
    model.to("cpu").eval()
    images = torch.rand(batch_size, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(warmup):
            model(images)
        start = time.perf_counter()
        for _ in range(runs):
            model(images)
    return (time.perf_counter() - start) / runs * 1000


if __name__ == "__main__":
    from data_loader import get_dataloaders
    from models import ModelA

    train_loader, val_loader, _ = get_dataloaders(batch_size=32)
    model = ModelA()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history, avg_epoch_time = train_model(model, train_loader, val_loader, optimizer, epochs=1)
    assert len(history["train_loss"]) == 1
    assert 0.0 <= history["train_acc"][0] <= 1.0
    assert avg_epoch_time > 0
    assert model_size_kb(model) > 0
    assert count_macs(model) > 0
    assert inference_latency_ms(model, runs=5) > 0

    metrics = evaluate_classification(model, val_loader)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"].shape == (9, 9)
    print("OK")
