# VAE Comparison Script
# ---------------------
# This script compares two different VAE models by processing an input image through both:
# 1. FLUX.1-schnell VAE from black-forest-labs
# 2. SDXL VAE from StabilityAI
#
# The script creates a combined output image showing:
# - Original image
# - FLUX VAE reconstruction
# - SDXL VAE reconstruction

import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import json
from safetensors.torch import load_file
import os
from typing import Tuple


def load_image(image_path: str, device: str) -> Tuple[torch.Tensor, tuple]:
    """
    Load and prepare an image for processing by the VAE.
    
    Args:
        image_path (str): Path to the input image
        device (str): Device to load the image on ('cuda' or 'cpu')
        
    Returns:
        tuple: (image tensor, original image size)
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Store original size for later resizing
    
    # Transform image to tensor with range 0-1
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Add batch dimension and move to the specified device
    image = transform(image).unsqueeze(0).to(device)
    return image, original_size


def normalize_image(image_np: np.ndarray) -> np.ndarray:
    """
    Apply robust normalization to ensure proper color range.
    This function uses percentile-based clipping to handle outliers.
    
    Args:
        image_np (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Normalized image array
    """
    # Cut off extreme outliers using percentiles
    low, high = np.percentile(image_np, (0.5, 99.5))
    image_np = np.clip(image_np, low, high)  # Clip values to the calculated range
    image_np = (image_np - low) / (high - low)  # Rescale to 0-1 range
    return image_np


def load_flux_vae() -> AutoencoderKL:
    """
    Download and initialize the FLUX VAE model from Hugging Face.
    
    Returns:
        AutoencoderKL: Initialized VAE model
    """
    repo_id = "black-forest-labs/FLUX.1-schnell"
    
    # Download the VAE weights
    vae_weights_path = hf_hub_download(repo_id, "vae/diffusion_pytorch_model.safetensors")
    
    # Download the configuration file
    vae_config_path = hf_hub_download(repo_id, "vae/config.json")
    
    # Load the configuration file to initialize the model
    with open(vae_config_path, "r") as f:
        config = json.load(f)
    
    # Create an AutoencoderKL object with the correct configuration
    vae = AutoencoderKL.from_config(config)
    
    # Load the model weights
    vae.load_state_dict(load_file(vae_weights_path), strict=False)
    
    return vae


def load_sdxl_vae() -> AutoencoderKL:
    """
    Load the SDXL VAE model from Hugging Face.
    
    Returns:
        AutoencoderKL: Initialized VAE model
    """
    return AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")


def process_image_with_vae(image: torch.Tensor, vae: AutoencoderKL, original_size: tuple, 
                           apply_gamma: bool = False) -> Image.Image:
    """
    Process an image through a VAE model and return the reconstructed image.
    
    Args:
        image (torch.Tensor): Input image tensor
        vae (AutoencoderKL): VAE model to use for processing
        original_size (tuple): Original image dimensions for resizing
        apply_gamma (bool): Whether to apply gamma correction (used for SDXL)
        
    Returns:
        PIL.Image.Image: Reconstructed image
    """
    # Process the image through the VAE
    with torch.no_grad():  # Disable gradient computation for inference
        # Encode the image to latent space
        latents = vae.encode(image).latent_dist.sample()
        # Decode the latents back to image space
        reconstructed_image = vae.decode(latents).sample
    
    # Post-process the reconstructed image
    reconstructed_image = reconstructed_image.clamp(-1, 1)  # Ensure values are in [-1, 1]
    reconstructed_image = (reconstructed_image + 1) / 2  # Convert to range [0, 1]
    
    # Convert from torch tensor to numpy array
    reconstructed_image = reconstructed_image.cpu().permute(0, 2, 3, 1).numpy()[0]  # Remove batch dimension
    reconstructed_image = normalize_image(reconstructed_image)  # Apply percentile-based normalization
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Convert to 8-bit format
    
    # Apply gamma correction for SDXL VAE if specified
    if apply_gamma:
        gamma = 1.4  # Adjust brightness slightly
        reconstructed_image = np.clip(((reconstructed_image / 255.0) ** (1 / gamma)) * 255, 0, 255).astype(np.uint8)
    
    # Convert numpy array back to PIL Image and resize to original dimensions
    reconstructed_pil = Image.fromarray(reconstructed_image)
    reconstructed_pil = reconstructed_pil.resize(original_size, Image.LANCZOS)
    
    return reconstructed_pil


def process_image_with_both_vaes(image_path: str, output_path: str = "vae_comparison.png"):
    """
    Process an image through both FLUX and SDXL VAEs and create a comparison image.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path where to save the output comparison image
    """
    # Determine device based on CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load both VAE models and move them to the appropriate device
    print("Loading FLUX VAE model...")
    flux_vae = load_flux_vae().to(device)
    flux_vae.eval()  # Set model to evaluation mode
    
    print("Loading SDXL VAE model...")
    sdxl_vae = load_sdxl_vae().to(device)
    sdxl_vae.eval()  # Set model to evaluation mode
    
    # Load the image
    print(f"Processing image: {image_path}")
    image, original_size = load_image(image_path, device)
    
    # Process the image through both VAEs
    print("Processing with FLUX VAE...")
    flux_reconstructed = process_image_with_vae(image, flux_vae, original_size)
    
    print("Processing with SDXL VAE...")
    sdxl_reconstructed = process_image_with_vae(image, sdxl_vae, original_size, apply_gamma=True)
    
    # Load the original input image
    input_pil = Image.open(image_path).convert("RGB")
    
    # Add labels above the images
    from PIL import ImageDraw, ImageFont
    
    # Try to get a font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", 36)
    except IOError:
        font = ImageFont.load_default()
    
    # Set the label height
    label_height = 50
    
    # Create a combined image with space for labels above
    total_width = input_pil.width * 3
    total_height = input_pil.height + label_height
    combined_image = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    
    # Add images to the combined image (below the label area)
    combined_image.paste(input_pil, (0, label_height))
    combined_image.paste(flux_reconstructed, (input_pil.width, label_height))
    combined_image.paste(sdxl_reconstructed, (input_pil.width * 2, label_height))
    
    # Create a drawing object
    draw = ImageDraw.Draw(combined_image)
    
    # Add the labels in the space above each image
    # Original image label
    label_text = "Original Image"
    text_width = draw.textlength(label_text, font=font) if hasattr(draw, 'textlength') else font.getsize(label_text)[0]
    text_x = (input_pil.width - text_width) // 2
    draw.text((text_x, (label_height - 36) // 2), label_text, fill=(0, 0, 0), font=font)
    
    # FLUX VAE label
    label_text = "FLUX VAE"
    text_width = draw.textlength(label_text, font=font) if hasattr(draw, 'textlength') else font.getsize(label_text)[0]
    text_x = input_pil.width + (input_pil.width - text_width) // 2
    draw.text((text_x, (label_height - 36) // 2), label_text, fill=(0, 0, 0), font=font)
    
    # SDXL VAE label
    label_text = "SDXL VAE"
    text_width = draw.textlength(label_text, font=font) if hasattr(draw, 'textlength') else font.getsize(label_text)[0]
    text_x = input_pil.width * 2 + (input_pil.width - text_width) // 2
    draw.text((text_x, (label_height - 36) // 2), label_text, fill=(0, 0, 0), font=font)
    
    # Add separation lines between the sections
    draw.line([(input_pil.width, 0), (input_pil.width, total_height)], fill=(200, 200, 200), width=2)
    draw.line([(input_pil.width * 2, 0), (input_pil.width * 2, total_height)], fill=(200, 200, 200), width=2)
    draw.line([(0, label_height), (total_width, label_height)], fill=(200, 200, 200), width=2)
    
    # Save the comparison image
    combined_image.save(output_path)
    print(f"Comparison image saved as '{output_path}'")


if __name__ == "__main__":
    # Get the image path from command line argument or use default
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image path - change this to your image path
        image_path = "image_path.jpg"  # Change this to the correct path

        
        # Check if the default path exists, otherwise prompt for a path
        if not os.path.exists(image_path):
            image_path = input("Please enter the path to your image: ")
    
    # Process the image
    process_image_with_both_vaes(image_path, "vae_comparison_result.jpg")
