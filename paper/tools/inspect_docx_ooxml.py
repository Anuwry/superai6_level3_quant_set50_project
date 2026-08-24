"""Read the body order, styles, tables, and drawing markers from a DOCX."""

from __future__ import annotations

import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"


def paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag == W + "t":
            parts.append(node.text or "")
        elif node.tag == W + "tab":
            parts.append("\t")
        elif node.tag == W + "br":
            parts.append("\n")
    return "".join(parts)


def paragraph_style(p: ET.Element) -> str:
    ppr = p.find(W + "pPr")
    if ppr is None:
        return ""
    style = ppr.find(W + "pStyle")
    return "" if style is None else style.attrib.get(W + "val", "")


def main() -> None:
    source = Path(sys.argv[1])
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    body = root.find(W + "body")
    assert body is not None
    p_index = 0
    t_index = 0
    for child in body:
        if child.tag == W + "p":
            text = paragraph_text(child)
            style = paragraph_style(child)
            drawings = sum(1 for _ in child.iter(WP + "inline")) + sum(1 for _ in child.iter(WP + "anchor"))
            print(f"P{p_index:04d}\tstyle={style!r}\tdrawings={drawings}\t{text!r}")
            p_index += 1
        elif child.tag == W + "tbl":
            rows = child.findall(W + "tr")
            print(f"T{t_index:03d}\trows={len(rows)}")
            for ri, row in enumerate(rows):
                values = []
                for cell in row.findall(W + "tc"):
                    values.append(" | ".join(paragraph_text(p) for p in cell.findall(W + "p")))
                print(f"  R{ri:03d}\t" + " || ".join(repr(v) for v in values))
            t_index += 1


if __name__ == "__main__":
    main()
