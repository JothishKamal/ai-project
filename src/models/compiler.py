"""
Model compiler for neural style transfer.
"""

import torch
import torch.nn as nn
from .layers import NormalizationLayer, ContentLossLayer, StyleLossLayer


class StyleTransferCompiler:
    """
    Compiles a VGG model for neural style transfer by inserting loss layers.
    """

    def __init__(
        self, base_model, content_layer_names, style_layer_names, device="cuda:0"
    ):
        """
        Initialize the compiler.

        Args:
            base_model: Pre-trained VGG model
            content_layer_names: List of layer names for content loss
            style_layer_names: List of layer names for style loss
            device: Device to run on
        """
        self.base_model = base_model.to(device)
        self.content_layer_names = content_layer_names
        self.style_layer_names = style_layer_names

    def compile(self, content_image, style_image, device="cuda:0"):
        """
        Compile the model with loss layers.

        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            device: Device to run on

        Returns:
            tuple: (model, content_loss_layers, style_loss_layers)
        """
        content_image = content_image.to(device)
        style_image = style_image.to(device)
        content_layers = []
        style_layers = []

        model = nn.Sequential()
        model.add_module("norm", NormalizationLayer())

        i = 0
        for layer in self.base_model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv{}".format(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu{}".format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool{}".format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn{}".format(i)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in self.content_layer_names:
                target = model(content_image).detach()
                loss_layer = ContentLossLayer(target)
                model.add_module("content{}".format(i), loss_layer)
                content_layers.append(loss_layer)

            if name in self.style_layer_names:
                target = model(style_image).detach()
                loss_layer = StyleLossLayer(target)
                model.add_module("style{}".format(i), loss_layer)
                style_layers.append(loss_layer)

        # Trim the model after the last loss layer
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLossLayer, StyleLossLayer)):
                break
        model = model[: (i + 1)]

        return model, content_layers, style_layers
