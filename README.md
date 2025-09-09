# Neural Style Transfer API - Project Structure

## Overview

This is a clean, self-contained neural style transfer API built with FastAPI and PyTorch. The project has been refactored to remove external dependencies and follows a proper Python package structure.

## Project Structure

```
c:\Projects\test\
├── main.py                 # FastAPI application entry point
├── pyproject.toml          # Project configuration
├── README.md              # This file
├── uv.lock                # Dependencies lock file
├── src/                   # Main package
│   ├── __init__.py        # Package initialization
│   ├── style_transfer.py  # Main NeuralStyleTransfer class
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   └── trainer.py     # StyleTransferTrainer class
│   ├── models/            # Neural network models
│   │   ├── __init__.py
│   │   ├── compiler.py    # StyleTransferCompiler class
│   │   └── layers.py      # Custom neural network layers
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── image_utils.py # Image processing utilities
└── neural-style-transfer/ # Legacy folder (can be removed)
```

## Features

- ✅ Self-contained architecture (no external neural-style-transfer dependency)
- ✅ GPU acceleration with CUDA support
- ✅ Preserves original image resolution (up to 1024px max dimension)
- ✅ Clean, modular code structure
- ✅ FastAPI with automatic documentation
- ✅ Configurable training parameters
- ✅ Thread-safe model operations

## API Endpoints

### GET /

- Root endpoint with API information

### GET /health

- Health check with GPU information and feature status

### POST /stylize

- Main style transfer endpoint
- Parameters:
  - `content_image`: Content image file
  - `style_image`: Style image file
  - `epochs`: Number of optimization epochs (default: 10)
  - `content_weight`: Weight for content loss (default: 1.0)
  - `style_weight`: Weight for style loss (default: 1000000.0)

## Usage

### Start the server:

```bash
uv run main.py
```

### Access the API:

- API Documentation: http://localhost:5002/docs
- Health Check: http://localhost:5002/health
- Style Transfer: POST to http://localhost:5002/stylize

## Technical Details

### GPU Support

- Automatically detects and uses CUDA-enabled GPUs
- Falls back to CPU if GPU is not available
- Uses PyTorch 2.5.1+cu121 for optimal performance

### Image Processing

- Supports JPEG, PNG, WebP formats
- Automatically resizes images while preserving aspect ratio
- Maximum dimension limited to 1024px for memory efficiency

### Neural Network Architecture

- Uses pre-trained VGG19 as the base model
- Content loss from conv4 layer
- Style loss from conv1, conv2, conv3, conv4, conv5 layers
- LBFGS optimizer for style transfer optimization

## Dependencies

- FastAPI: Web framework
- PyTorch: Deep learning framework
- PIL (Pillow): Image processing
- torchvision: Pre-trained models and transforms
- uvicorn: ASGI server
- python-multipart: File upload support

## Recent Fixes

- ✅ Fixed in-place operation error with gradient computation
- ✅ Removed deprecated VGG19 pretrained parameter warnings
- ✅ Improved type safety and error handling
- ✅ Self-contained architecture without external dependencies
