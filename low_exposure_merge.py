import os
import numpy as np
import rawpy  # pip install rawpy
from hdrm import merge_hdr_locally


def get_image_brightness(raw_path: str) -> float:
    """
    Calculate average brightness of a RAW file.
    Lower values indicate underexposed/darker images.
    """
    with rawpy.imread(raw_path) as raw:
        # Get a quick thumbnail for faster processing
        thumb = raw.postprocess(half_size=True, no_auto_bright=True)
        # Calculate mean brightness across all channels
        brightness = np.mean(thumb)
    return brightness


def select_low_exposure_images(raw_file_paths: list, threshold_percentile: float = 40) -> list:
    """
    Select images with low exposure from the list.

    Args:
        raw_file_paths: List of paths to RAW files
        threshold_percentile: Images below this percentile of brightness are considered low exposure

    Returns:
        List of paths to low exposure images
    """
    # Calculate brightness for each image
    brightness_values = []
    for path in raw_file_paths:
        brightness = get_image_brightness(path)
        brightness_values.append((path, brightness))
        print(f"{os.path.basename(path)}: brightness = {brightness:.2f}")

    # Sort by brightness
    brightness_values.sort(key=lambda x: x[1])

    # Calculate threshold
    all_brightnesses = [b for _, b in brightness_values]
    threshold = np.percentile(all_brightnesses, threshold_percentile)

    # Select images below threshold
    low_exposure = [path for path, b in brightness_values if b <= threshold]

    print(f"\nThreshold brightness: {threshold:.2f}")
    print(f"Selected {len(low_exposure)} low exposure images")

    return low_exposure


def merge_low_exposure_images(raw_file_paths: list, output_path: str,
                               threshold_percentile: float = 40, gamma: float = 1.1):
    """
    Select low exposure images and merge them into a separate HDR image.

    Args:
        raw_file_paths: List of paths to all RAW files
        output_path: Where to save the low exposure merge result
        threshold_percentile: Images below this percentile are considered low exposure
        gamma: Gamma correction value

    Returns:
        Merged image or None if not enough low exposure images
    """
    low_exposure_files = select_low_exposure_images(raw_file_paths, threshold_percentile)

    if len(low_exposure_files) < 2:
        print("Not enough low exposure images to merge (need at least 2)")
        return None

    print(f"\nMerging low exposure images: {[os.path.basename(f) for f in low_exposure_files]}")
    return merge_hdr_locally(low_exposure_files, output_path, gamma)


def analyze_exposures(raw_file_paths: list):
    """Print brightness analysis to understand exposure spread."""
    brightnesses = [(os.path.basename(p), get_image_brightness(p)) for p in raw_file_paths]
    brightnesses.sort(key=lambda x: x[1])
    
    print("Exposure analysis:")
    for name, b in brightnesses:
        bar = "█" * int(b / 5)
        print(f"  {name}: {b:6.1f} {bar}")
    
    spread = brightnesses[-1][1] - brightnesses[0][1]
    print(f"\nExposure spread: {spread:.1f} (higher = better for HDR)")


# Usage:
if __name__ == "__main__":
    raw_files = ["DSC05885.ARW", "DSC05886.ARW", "DSC05887.ARW", "DSC05888.ARW", "DSC05889.ARW"]

    # Analyze exposures first
    analyze_exposures(raw_files)

    # 1. Merge ALL images
    print("\n--- Merging ALL images ---")
    full_result = merge_hdr_locally(raw_files, "merged_all.jpg")
    print("Saved: merged_all.jpg")

    # 2. Merge only DARK images (bottom 3 by brightness)
    print("\n--- Merging DARK images only ---")
    dark_files = ["DSC05888.ARW", "DSC05886.ARW", "DSC05885.ARW"]  # brightness: 4.8, 15.6, 42.7
    dark_result = merge_hdr_locally(dark_files, "merged_dark.jpg")
    print("Saved: merged_dark.jpg")
