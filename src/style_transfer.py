"""
Main style transfer module.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from .utils.image_utils import load_image
from .models.compiler import StyleTransferCompiler
from .core.trainer import StyleTransferTrainer


class NeuralStyleTransfer:
    """
    Main class for neural style transfer.
    """

    def __init__(self, device=None):
        """
        Initialize the style transfer model.

        Args:
            device: Device to run on (auto-detected if None)
        """
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Load pre-trained VGG19 model
        self.base_model = (
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            .features.eval()
            .to(self.device)
        )

        # Define layers for content and style loss
        self.content_layers = ["conv4"]
        self.style_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

    def transfer_style(
        self,
        content_image,
        style_image,
        epochs=10,
        content_weight=1.0,
        style_weight=1e6,
    ):
        """
        Perform style transfer.

        Args:
            content_image: PIL Image object for content
            style_image: PIL Image object for style
            epochs: Number of optimization epochs
            content_weight: Weight for content loss
            style_weight: Weight for style loss

        Returns:
            PIL.Image: Stylized image
        """
        print(f"Using device: {self.device}")
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")

        print(f"Content image size: {content_image.size}")
        print(f"Style image size: {style_image.size}")

        # Preprocess images - preserve content image resolution, resize style to match
        content_tensor = load_image(content_image, max_size=1024).to(self.device)

        # Get the processed content image size from tensor
        _, _, h, w = content_tensor.shape
        print(f"Processing at resolution: {w}x{h}")

        # Resize style image to match content image dimensions
        style_tensor = load_image(style_image, size=(h, w)).to(self.device)

        # Compile model with loss layers
        compiler = StyleTransferCompiler(
            self.base_model,
            self.content_layers,
            self.style_layers,
            device=str(self.device),
        )
        model, content_loss_layers, style_loss_layers = compiler.compile(
            content_tensor, style_tensor, device=str(self.device)
        )

        # Initialize input image as a clone of content image
        input_img = content_tensor.clone().to(self.device)

        # Train the model
        trainer = StyleTransferTrainer(
            model, content_loss_layers, style_loss_layers, device=str(self.device)
        )
        _, output = trainer.train(
            input_img,
            epochs=epochs,
            content_weight=content_weight,
            style_weight=style_weight,
            device=str(self.device),
        )

        # Convert tensor to PIL Image
        output = output.cpu().detach().squeeze(0)
        output_img = transforms.ToPILImage()(output)
        return output_img
