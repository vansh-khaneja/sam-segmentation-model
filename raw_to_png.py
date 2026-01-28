import rawpy  # pip install rawpy
from PIL import Image


def convert_raw_to_png(raw_path: str, output_path: str = None):
    """
    Convert a RAW file to PNG.

    Args:
        raw_path: Path to the RAW file
        output_path: Path for output PNG (defaults to same name with .png extension)
    """
    if output_path is None:
        output_path = raw_path.rsplit(".", 1)[0] + ".png"

    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess()

    img = Image.fromarray(rgb)
    img.save(output_path)
    print(f"Saved: {output_path}")
    return img


# Usage:
if __name__ == "__main__":
    convert_raw_to_png("DSC05887.ARW")
