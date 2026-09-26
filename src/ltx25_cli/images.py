"""Local start-frame loading without importing torch."""
from PIL import Image, ImageOps


def load_start_image(path, width, height, fit="contain"):
    """Apply camera orientation, flatten transparency and preserve aspect ratio."""
    with Image.open(path) as source:
        source.seek(0)
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", oriented.size, "black")
        background.alpha_composite(oriented)
        rgb = background.convert("RGB")
    if fit == "cover":
        return ImageOps.fit(rgb, (width, height), method=Image.Resampling.LANCZOS)
    return ImageOps.pad(rgb, (width, height), method=Image.Resampling.LANCZOS, color="black")
