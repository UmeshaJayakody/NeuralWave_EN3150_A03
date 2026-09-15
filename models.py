"""
Model A (Standard CNN) and Model B (Lightweight/Depthwise-Separable CNN)
for the RealWaste 9-class classification task on 64x64 RGB images.
"""

import torch
import torch.nn as nn

IMG_SIZE = 64
NUM_CLASSES = 9


class ModelA(nn.Module):
    """Standard CNN: plain Conv2D + ReLU + MaxPool blocks, then Dense + Softmax."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 64x64x3  -> 32x32x16
        x = self.pool(self.relu(self.conv2(x)))   # 32x32x16 -> 16x16x32
        x = self.pool(self.relu(self.conv3(x)))   # 16x16x32 -> 8x8x64
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class SeparableConv2d(nn.Module):
    """Depthwise Separable Conv = depthwise 3x3 (per-channel) + pointwise 1x1 (mixing)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ModelB(nn.Module):
    """Lightweight CNN: depthwise separable convs. Must stay under 100k params."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # first layer kept standard, 3 input channels are too few to split
        self.sep2 = SeparableConv2d(16, 32)
        self.sep3 = SeparableConv2d(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 64x64x3  -> 32x32x16
        x = self.pool(self.relu(self.sep2(x)))    # 32x32x16 -> 16x16x32
        x = self.pool(self.relu(self.sep3(x)))    # 16x16x32 -> 8x8x64
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    model_a = ModelA()
    out_a = model_a(dummy)
    assert out_a.shape == (1, NUM_CLASSES)
    assert torch.allclose(out_a.sum(dim=1), torch.tensor([1.0]), atol=1e-5)  # softmax sums to 1

    model_b = ModelB()
    out_b = model_b(dummy)
    assert out_b.shape == (1, NUM_CLASSES)
    assert torch.allclose(out_b.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    assert count_params(model_b) < 100_000, "Model B exceeds the 100k parameter budget"

    print(f"Model A params: {count_params(model_a):,}")
    print(f"Model B params: {count_params(model_b):,}")
    print("All checks passed.")
