"""Convert the two approved Image 2 v19 schematics to tightly cropped PDFs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parent.parent


def tight_crop(image: Image.Image, padding: int = 10) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    box = difference.point(lambda value: 255 if value > 8 else 0).getbbox()
    if box is None:
        return rgb
    left, upper, right, lower = box
    return rgb.crop((
        max(0, left - padding), max(0, upper - padding),
        min(rgb.width, right + padding), min(rgb.height, lower + padding),
    ))


def convert(source: Path, destination: Path) -> None:
    image = tight_crop(Image.open(source))
    width = 7.05
    height = width * image.height / image.width
    figure = plt.figure(figsize=(width, height), frameon=False)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(image)
    axis.axis("off")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def main() -> None:
    figure_dir = ROOT / "paper_Q1/figures"
    for stem in ("figure1_image2_v19", "figure2_image2_v19"):
        convert(figure_dir / f"{stem}.png", figure_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
