"""Create readable contact sheets from rendered manuscript pages for QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pages_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-sheet", type=int, default=4)
    args = parser.parse_args()

    pages = sorted(args.pages_dir.glob("page-*.png"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pages), args.per_sheet):
        selected = pages[start : start + args.per_sheet]
        thumbnails = []
        for page_path in selected:
            image = Image.open(page_path).convert("RGB")
            image.thumbnail((850, 1100), Image.Resampling.LANCZOS)
            thumbnails.append((page_path, image.copy()))
        canvas = Image.new("RGB", (1750, 2250), "#D7DCE2")
        draw = ImageDraw.Draw(canvas)
        positions = [(15, 15), (885, 15), (15, 1135), (885, 1135)]
        for (page_path, image), (x, y) in zip(thumbnails, positions):
            canvas.paste(image, (x, y + 25))
            draw.text((x, y), page_path.stem, fill="black")
        end = start + len(selected)
        canvas.save(args.output_dir / f"contact-{start + 1:03d}-{end:03d}.png")
    print(f"sheets={(len(pages) + args.per_sheet - 1) // args.per_sheet}")


if __name__ == "__main__":
    main()
