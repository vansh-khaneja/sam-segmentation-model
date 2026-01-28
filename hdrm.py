import tempfile
import numpy as np
from PIL import Image
import HDRutils  # pip install HDRutils

def merge_hdr_locally(raw_file_paths: list, output_path: str, gamma: float = 1.1):
    """
    Merge multiple RAW files into an HDR image locally.
    
    Args:
        raw_file_paths: List of paths to RAW files (min 2)
        output_path: Where to save the result
        gamma: Gamma correction value (default 1.1)
    """
    if len(raw_file_paths) < 2:
        raise ValueError("Need at least 2 files for HDR merge")
    
    # Step 1: Merge RAW files using HDRutils
    # This performs exposure fusion and returns HDR radiance map
    hdr, mask = HDRutils.merge(raw_file_paths)
    
    # Step 2: Reinhard tonemapping (converts HDR to displayable LDR)
    hdr = np.maximum(hdr, 0)  # Remove negative values
    
    # Calculate luminance (weighted RGB based on human perception)
    luminance = (
        0.2126 * hdr[..., 0] +   # Red weight
        0.7152 * hdr[..., 1] +   # Green weight  
        0.0722 * hdr[..., 2]     # Blue weight
    )
    
    # Log-average luminance (key value for scene)
    L_avg = np.exp(np.mean(np.log(1e-6 + luminance)))
    
    # Reinhard operator: L / (1 + L)
    mapped = hdr / (hdr + L_avg)
    mapped = np.clip(mapped, 0, 1)
    
    # Gamma correction
    mapped = mapped ** (1 / gamma)
    
    # Convert to 8-bit
    ldr = (mapped * 255).astype(np.uint8)
    
    # Step 3: Save as image
    img = Image.fromarray(ldr)
    img.save(output_path, quality=95)
    
    return img


# Usage:
if __name__ == "__main__":
    raw_files = ["DSC05885.ARW", "DSC05886.ARW"] 
    result = merge_hdr_locally(raw_files, "merged_hdr.jpg")