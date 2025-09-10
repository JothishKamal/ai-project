"""
FastAPI Neural Style Transfer API

A clean, self-contained neural style transfer API using PyTorch and FastAPI.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import torch
import threading
from typing import Optional

# Import our custom style transfer module
from src import NeuralStyleTransfer

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_lock = threading.Lock()
style_transfer_model = None

# Initialize FastAPI app
app = FastAPI(
    title="Neural Style Transfer API",
    description="Neural style transfer using PyTorch with GPU acceleration",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_style_transfer_model():
    """Get or create the style transfer model instance."""
    global style_transfer_model
    if style_transfer_model is None:
        style_transfer_model = NeuralStyleTransfer(device=device)
    return style_transfer_model


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Neural Style Transfer API!",
        "version": "2.0.0",
        "docs": "/docs",
        "gpu_available": torch.cuda.is_available(),
        "features": {
            "self_contained": True,
            "gpu_acceleration": torch.cuda.is_available(),
            "preserves_resolution": True,
            "max_resolution": "1024px",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_cached": f"{torch.cuda.memory_cached(0) / 1024**3:.1f} GB",
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "features": {
            "preserves_original_resolution": True,
            "max_resolution_limit": "1024px",
            "gpu_acceleration": torch.cuda.is_available(),
            "self_contained": True,
        },
        "gpu_info": gpu_info,
    }


@app.post("/stylize")
async def stylize_image(
    content_image: UploadFile = File(..., description="Content image file"),
    style_image: UploadFile = File(..., description="Style image file"),
    epochs: Optional[int] = 10,
    content_weight: Optional[float] = 1.0,
    style_weight: Optional[float] = 1000000.0,
):
    """
    API endpoint to perform style transfer.

    Args:
        content_image: Content image file
        style_image: Style image file
        epochs: Number of optimization epochs (default: 10)
        content_weight: Weight for content loss (default: 1.0)
        style_weight: Weight for style loss (default: 1000000.0)
    """
    # Validate file types
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

    if content_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Content image must be one of: {', '.join(allowed_types)}",
        )

    if style_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Style image must be one of: {', '.join(allowed_types)}",
        )

    try:
        # Read and process images
        content_bytes = await content_image.read()
        style_bytes = await style_image.read()

        content_img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
        style_img = Image.open(io.BytesIO(style_bytes)).convert("RGB")

        print("Processing images...")

        # Use thread lock for model access
        with model_lock:
            model = get_style_transfer_model()
            stylized_img = model.transfer_style(
                content_img,
                style_img,
                epochs=epochs or 10,
                content_weight=content_weight or 1.0,
                style_weight=style_weight or 1000000.0,
            )

        print("Style transfer completed.")

        # Save stylized image to a BytesIO object
        img_io = io.BytesIO()
        stylized_img.save(img_io, "JPEG", quality=95)
        img_io.seek(0)

        return StreamingResponse(
            io.BytesIO(img_io.getvalue()),
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=stylized_image.jpg"},
        )

    except Exception as e:
        print(f"Error during style transfer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True, log_level="info")
