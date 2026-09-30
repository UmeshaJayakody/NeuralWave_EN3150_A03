
import torch
import torch.nn as nn
from torchvision import models as tvm

NUM_CLASSES = 9

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(images):
    """images: NCHW float tensor in [0, 1] -> ImageNet-normalized."""
    return (images - IMAGENET_MEAN.to(images.device)) / IMAGENET_STD.to(images.device)


class MobileNetV2Classifier(nn.Module):
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

    print(f"MobileNetV2 trainable (head only): {trainable_mnet:,}")
    print(f"SqueezeNet trainable (head only): {count_params(sqz):,}")
    print("OK")
