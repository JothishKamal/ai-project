"""
Neural network layers for style transfer.
"""

import torch
import torch.nn as nn


def gram_matrix(image):
    """
    Compute the Gram matrix for style loss calculation.

    Args:
        image: Input tensor of shape (N, C, H, W)

    Returns:
        torch.Tensor: Gram matrix
    """
    N, C, H, W = image.size()
    features = image.view(N * C, H * W)
    gram = torch.mm(features, features.t())
    return torch.div(gram, N * C * H * W)


class NormalizationLayer(nn.Module):
    """
    Normalization layer using ImageNet statistics.
    """

    def __init__(self):
        super().__init__()

    def forward(self, image):
        dev = image.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(dev)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(dev)
        return (image - mean) / std


class ContentLossLayer(nn.Module):
    """
    Content loss layer for neural style transfer.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, pred):
        self.loss = torch.mean((pred - self.target) ** 2)
        return pred


class StyleLossLayer(nn.Module):
    """
    Style loss layer for neural style transfer.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, pred):
        pred_gram = gram_matrix(pred)
        target_gram = gram_matrix(self.target)
        self.loss = torch.mean((pred_gram - target_gram) ** 2)
        return pred
