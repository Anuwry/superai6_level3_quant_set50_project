"""Render every PDF page to a PNG for local visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.pdf)
    zoom = args.dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(args.output_dir / f"page-{index:03d}.png")
    print(f"pages={len(document)} output={args.output_dir}")


if __name__ == "__main__":
    main()
