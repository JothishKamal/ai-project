# Neural Style Transfer - Quick Start Guide

## What is Neural Style Transfer?

Neural Style Transfer is an AI technique that combines the **content** of one image with the **artistic style** of another image. Think of it as teaching a computer to paint like Van Gogh or Picasso!

## How It Works (Simple Explanation)

```
Your Photo + Famous Painting = Your Photo in the Painting's Style
```

### The Process:

1. **Content Image**: Your photo (what objects/scenes to keep)
2. **Style Image**: An artwork (what artistic style to apply)
3. **AI Processing**: VGG19 neural network analyzes both images
4. **Result**: New image with your content + the artwork's style

## VGG19 Neural Network Explained

VGG19 is like a **very smart image analyzer** with 19 layers that understand images at different levels:

### Layer Types:

- **Early Layers (conv1-2)**: See basic patterns, textures, colors
- **Middle Layers (conv3-4)**: Recognize shapes, objects
- **Later Layers (conv5)**: Understand complex scenes, compositions

### Why These Layers?

- **Content Loss** uses **conv4**: Best balance of detail and abstraction
- **Style Loss** uses **conv1,2,3,4,5**: Captures style at multiple scales

## Real Example

**Input:**

- Content: Photo of your dog
- Style: Van Gogh's "Starry Night"

**Process:**

1. VGG19 analyzes your dog photo → extracts content features
2. VGG19 analyzes Starry Night → extracts style patterns
3. AI creates new image that looks like your dog painted by Van Gogh

**Result:** Your dog rendered in swirling, colorful Van Gogh style!

## API Usage Examples

### Basic Usage:

```bash
# Upload your content and style images
POST http://localhost:5002/stylize
- content_image: your_photo.jpg
- style_image: artwork.jpg
```

### Advanced Settings:

```bash
# Fine-tune the result
POST http://localhost:5002/stylize
- content_image: your_photo.jpg
- style_image: artwork.jpg
- epochs: 20                     # More = better quality, slower
- content_weight: 1.0            # How much to preserve original
- style_weight: 1000000.0        # How strong the style effect
```

## Parameter Effects

| Setting            | Low Value        | High Value          |
| ------------------ | ---------------- | ------------------- |
| **epochs**         | Fast, OK quality | Slow, great quality |
| **content_weight** | More stylized    | More like original  |
| **style_weight**   | Subtle style     | Very stylized       |

## Tips for Best Results

### Good Content Images:

- ✅ Clear subjects (people, objects, landscapes)
- ✅ Good lighting and contrast
- ✅ Not too cluttered

### Good Style Images:

- ✅ Strong artistic patterns (brushstrokes, textures)
- ✅ Distinctive color palettes
- ✅ Famous artworks often work well

### Avoid:

- ❌ Very small or blurry images
- ❌ Images with too much detail
- ❌ Very similar content and style images

## Common Use Cases

1. **Artistic Photos**: Turn vacation photos into paintings
2. **Social Media**: Create unique profile pictures
3. **Art Projects**: Combine multiple art styles
4. **Education**: Understand different artistic techniques
5. **Fun**: See yourself painted by famous artists!

## Performance Guide

| Image Size | Quality | Time (GPU) | Time (CPU) |
| ---------- | ------- | ---------- | ---------- |
| 512×512    | Good    | ~45 sec    | ~10 min    |
| 768×768    | Better  | ~75 sec    | ~20 min    |
| 1024×1024  | Best    | ~120 sec   | ~45 min    |

## Try It Now!

1. **Start the API**: `uv run main.py`
2. **Open docs**: http://localhost:5002/docs
3. **Upload images** and experiment!

---

_For detailed technical information, see [DOCUMENTATION.md](DOCUMENTATION.md)_
