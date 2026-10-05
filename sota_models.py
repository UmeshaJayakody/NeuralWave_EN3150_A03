"""
Transfer-learning wrappers around two lightweight, edge-friendly SOTA backbones:
MobileNetV2 and SqueezeNet1_1, trained in two stages:

1. Transfer learning: the ImageNet backbone is frozen and only a new Dense/Conv
   classification head for the 9 RealWaste classes is trained.
2. Fine-tuning: `unfreeze_last_blocks` makes the last few backbone blocks
   trainable again, and they are trained together with the head at a lower
   learning rate so the pretrained features adapt without being destroyed.

Pretrained backbones expect ImageNet-normalized input, so `imagenet_normalize`
is applied on top of the [0, 1] images the shared data pipeline produces.
"""

import torch
import torch.nn as nn
from torchvision import models as tvm

NUM_CLASSES = 9

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(images):
    """images: NCHW float tensor in [0, 1] -> ImageNet-normalized."""
    return (images - IMAGENET_MEAN.to(images.device)) / IMAGENET_STD.to(images.device)


def unfreeze_last_blocks(model, n_blocks=None):
    """Stage 2 (fine-tuning): make the last `n_blocks` children of `model.features` trainable.
    Defaults to the per-architecture choice in `model.FINE_TUNE_BLOCKS`."""
    n_blocks = n_blocks or model.FINE_TUNE_BLOCKS
    for block in list(model.features.children())[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


class MobileNetV2Classifier(nn.Module):
    # features[16..18]: the last two inverted-residual blocks + the final 1x1 conv (320 -> 1280)
    FINE_TUNE_BLOCKS = 3

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base.last_channel, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = imagenet_normalize(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return self.softmax(x)


class SqueezeNetClassifier(nn.Module):
    # features[11..12]: the last two Fire modules (512 output channels each)
    FINE_TUNE_BLOCKS = 2

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = tvm.squeezenet1_1(weights=tvm.SqueezeNet1_1_Weights.DEFAULT)
        self.features = base.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = imagenet_normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return self.softmax(x)


if __name__ == "__main__":
    from train_utils import count_params

    dummy = torch.rand(2, 3, 64, 64)

    mnet = MobileNetV2Classifier()
    out = mnet(dummy)
    assert out.shape == (2, NUM_CLASSES)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4)
    trainable_mnet = count_params(mnet)
    assert trainable_mnet < sum(p.numel() for p in mnet.parameters()), "backbone should be frozen"

    sqz = SqueezeNetClassifier()
    out = sqz(dummy)
    assert out.shape == (2, NUM_CLASSES)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4)
    trainable_sqz = count_params(sqz)

    print(f"MobileNetV2 trainable (head only): {trainable_mnet:,}")
    print(f"SqueezeNet trainable (head only): {trainable_sqz:,}")

    unfreeze_last_blocks(mnet)
    unfreeze_last_blocks(sqz)
    assert count_params(mnet) > trainable_mnet and count_params(sqz) > trainable_sqz
    assert count_params(mnet) < sum(p.numel() for p in mnet.parameters()), "early blocks stay frozen"
    print(f"MobileNetV2 trainable (fine-tuning): {count_params(mnet):,}")
    print(f"SqueezeNet trainable (fine-tuning): {count_params(sqz):,}")
    print("OK")
