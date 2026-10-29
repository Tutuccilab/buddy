"""
The network of the bud model: a U-Net with a ResNet-34 encoder, plus a classification head on the deepest features.

This is segmentation_models_pytorch's Unet("resnet34", classes=2, aux_params={"classes": 3, "dropout": 0.2})
written out in plain PyTorch (torchvision's resnet34 as the encoder), with the same module names, so the released
weights (trained with segmentation_models_pytorch 0.5.0) load unchanged, and the outputs are identical.
segmentation_models_pytorch itself pulled in timm, huggingface-hub and friends for this one architecture.

    net = Unet(in_channels=3, classes=2, aux_classes=3)
    maps, cell_class = net(x)            # x: (batch, in_channels, H, W) -> (batch, 2, H, W), (batch, 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

ENCODER_CHANNELS = (3, 64, 64, 128, 256, 512)    # input, after conv1, layer1, layer2, layer3, layer4
DECODER_CHANNELS = (256, 128, 64, 32, 16)


class ResNet34Encoder(nn.Module):
    """torchvision's resnet34 without the classifier, returning the features of every resolution."""

    def __init__(self, in_channels=3, pretrained=False):
        super().__init__()
        net = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        if in_channels != 3:
            net.conv1 = _patch_first_conv(net.conv1, in_channels, pretrained)
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

    def forward(self, x):
        features = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.layer1(self.maxpool(x))
        features.append(x)
        for layer in (self.layer2, self.layer3, self.layer4):
            x = layer(x)
            features.append(x)
        return features                                   # resolutions H, H/2, H/4, H/8, H/16, H/32


def _patch_first_conv(conv, in_channels, pretrained):
    """A first convolution with in_channels inputs; pretrained weights are repeated over the extra channels."""
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    if pretrained:
        w = conv.weight.data
        new.weight.data = torch.stack([w[:, i % 3] for i in range(in_channels)], dim=1) * (3 / in_channels)
    return new


def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    """Upsample x2 (nearest), concatenate the skip features, two 3x3 conv + BN + ReLU."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = conv_bn_relu(in_channels + skip_channels, out_channels)
        self.conv2 = conv_bn_relu(out_channels, out_channels)

    def forward(self, x, size, skip=None):
        x = F.interpolate(x, size=size, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UnetDecoder(nn.Module):

    def __init__(self, encoder_channels=ENCODER_CHANNELS, decoder_channels=DECODER_CHANNELS):
        super().__init__()
        enc = list(encoder_channels[1:])[::-1]                            # deepest first
        in_channels = [enc[0]] + list(decoder_channels[:-1])
        skip_channels = enc[1:] + [0]
        self.blocks = nn.ModuleList(DecoderBlock(i, s, o) for i, s, o in zip(in_channels, skip_channels, decoder_channels))

    def forward(self, features):
        sizes = [f.shape[2:] for f in features][::-1]
        features = features[1:][::-1]                                     # drop the input, deepest first
        x, skips = features[0], features[1:]
        for i, block in enumerate(self.blocks):
            x = block(x, sizes[i + 1], skips[i] if i < len(skips) else None)
        return x


class Unet(nn.Module):
    """encoder -> decoder -> segmentation_head (maps); encoder's deepest features -> classification_head (classes)."""

    def __init__(self, in_channels=3, classes=2, aux_classes=3, dropout=0.2, pretrained_encoder=False):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels, pretrained_encoder)
        self.decoder = UnetDecoder()
        self.segmentation_head = nn.Sequential(nn.Conv2d(DECODER_CHANNELS[-1], classes, 3, padding=1),
                                               nn.Identity(), nn.Identity())
        self.classification_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout, inplace=True),
                                                 nn.Linear(ENCODER_CHANNELS[-1], aux_classes), nn.Identity())

    def forward(self, x):
        features = self.encoder(x)
        maps = self.segmentation_head(self.decoder(features))
        return maps, self.classification_head(features[-1])
