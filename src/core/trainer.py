"""
Trainer for neural style transfer.
"""

import torch


class StyleTransferTrainer:
    """
    Trainer for neural style transfer optimization.
    """

    def __init__(self, model, content_layers, style_layers, device="cuda:0"):
        """
        Initialize the trainer.

        Args:
            model: Compiled model with loss layers
            content_layers: List of content loss layers
            style_layers: List of style loss layers
            device: Device to run on
        """
        self.model = model.to(device)
        self.content_layers = content_layers
        self.style_layers = style_layers

    def train(
        self,
        input_image,
        epochs=10,
        content_weight=1.0,
        style_weight=1e6,
        device="cuda:0",
    ):
        """
        Train the style transfer model.

        Args:
            input_image: Input image tensor to optimize
            epochs: Number of optimization epochs
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            device: Device to run on

        Returns:
            tuple: (losses_dict, final_image)
        """
        input_image = input_image.to(device)
        optimizer = torch.optim.LBFGS([input_image.requires_grad_(True)])

        content_losses = []
        style_losses = []
        total_losses = []

        for epoch in range(1, epochs + 1):

            def closure():
                # Use non-in-place operation to avoid gradient issues
                with torch.no_grad():
                    input_image.clamp_(0, 1)
                optimizer.zero_grad()
                self.model(input_image)

                content_loss = torch.tensor(0.0, device=device, requires_grad=True)
                style_loss = torch.tensor(0.0, device=device, requires_grad=True)

                for content_layer in self.content_layers:
                    content_loss = content_loss + content_layer.loss
                for style_layer in self.style_layers:
                    style_loss = style_loss + style_layer.loss

                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()

                content_losses.append((content_weight * content_loss).item())
                style_losses.append((style_weight * style_loss).item())
                total_losses.append(total_loss.item())

                return total_loss

            optimizer.step(closure)

            print(
                "Epoch {}/{} --- Total Loss: {:.4f}".format(
                    epoch, epochs, total_losses[-1]
                )
            )
            print(
                "Content Loss: {:.6f} --- Style Loss: {:.6f}".format(
                    content_losses[-1], style_losses[-1]
                )
            )
            print("---" * 17)

        losses = {
            "total": total_losses,
            "content": content_losses,
            "style": style_losses,
        }

        return losses, torch.clamp(input_image, 0, 1)
