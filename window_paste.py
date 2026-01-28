import numpy as np
from PIL import Image
from ultralytics.models.sam.predict import SAM3SemanticPredictor


def segment_and_paste_windows(source_image: str, target_image: str, output_path: str):
    """
    Segment windows from source image and paste them onto target image.

    Args:
        source_image: Path to image to segment windows FROM
        target_image: Path to image to paste windows ONTO
        output_path: Path to save the result
    """
    # Initialize SAM3 predictor
    overrides = dict(
        conf=0.35,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,
        boxes=False,
        project=".",
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Segment windows from source image
    print(f"Segmenting windows from: {source_image}")
    predictor.set_image(source_image)
    results = predictor(text=["glass window"], save=False)

    if results[0].masks is None or len(results[0].masks.xy) == 0:
        print("No windows detected!")
        return None

    print(f"Found {len(results[0].masks.xy)} window(s)")

    # Load both images
    source_img = Image.open(source_image).convert("RGBA")
    target_img = Image.open(target_image).convert("RGBA")
    source_array = np.array(source_img)

    # Get the combined mask from all detected windows
    mask_data = results[0].masks.data.cpu().numpy()  # Shape: (num_masks, H, W)
    combined_mask = np.any(mask_data, axis=0)  # Combine all masks

    # Create RGBA image with only the window regions from source
    window_layer = np.zeros_like(source_array)
    window_layer[combined_mask] = source_array[combined_mask]
    window_layer[..., 3] = (combined_mask * 255).astype(np.uint8)  # Set alpha

    window_img = Image.fromarray(window_layer)

    # Paste windows onto target image
    result = target_img.copy()
    result.paste(window_img, (0, 0), window_img)

    # Save result
    result = result.convert("RGB")
    result.save(output_path, quality=95)
    print(f"Saved: {output_path}")

    return result


# Usage:
if __name__ == "__main__":
    segment_and_paste_windows(
        source_image="DSC05885.png",  # Segment windows from this (darker)
        target_image="DSC05887.png",  # Paste onto this (brighter)
        output_path="windows_composited.jpg"
    )
