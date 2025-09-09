"""
Image utilities for neural style transfer.
"""

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def load_image(image_input, size=None, max_size=1024):
    """
    Load and preprocess an image for neural style transfer.

    Args:
        image_input: Either a file path (str) or PIL Image object
        size: Optional tuple (height, width) to resize to
        max_size: Maximum dimension to limit memory usage

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if isinstance(image_input, str):
        # If input is a string, treat it as a file path
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        # If input is already a PIL Image, use it directly
        image = image_input
    else:
        raise ValueError(
            "image_input must be either a file path (str) or PIL Image object"
        )

    # If size is specified, use it
    if size is not None:
        h, w = size
        image = image.resize((w, h))
    elif max_size is not None:
        # Preserve aspect ratio but limit maximum dimension
        original_width, original_height = image.size
        if max(original_width, original_height) > max_size:
            if original_width > original_height:
                new_width = max_size
                new_height = int(original_height * max_size / original_width)
            else:
                new_height = max_size
                new_width = int(original_width * max_size / original_height)
            image = image.resize((new_width, new_height))

    image = transforms.ToTensor()(image).unsqueeze(0)
    return image


def show_image(tensor, title, size=(10, 8), save=False):
    """
    Display a tensor as an image.

    Args:
        tensor: Image tensor to display
        title: Title for the image
        size: Figure size tuple
        save: Whether to save the image
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    if save:
        image.save(title + ".jpg")
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.title(title)
    plt.show()
