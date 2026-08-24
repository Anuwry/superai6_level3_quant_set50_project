"""Apply consistent publication-sized typography to displayed equations.

The edit is performed at OOXML level so the existing OMML equations, Zotero
fields, figures, tables, and document relationships remain intact.
"""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper" / "SET_direction_manuscript_journal_formatted_v1.docx"
OUTPUT = ROOT / "paper" / "SET_direction_manuscript_journal_formatted_v2.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}
W = f"{{{W_NS}}}"

EQUATION_SIZE = 24  # 12 pt
NUMBER_SIZE = 22  # 11 pt


def ensure_child(parent: etree._Element, tag: str, *, first: bool = False) -> etree._Element:
    child = parent.find(f"{W}{tag}")
    if child is None:
        child = etree.Element(f"{W}{tag}")
        if first:
            parent.insert(0, child)
        else:
            parent.append(child)
    return child


def set_run_size(run_properties: etree._Element, half_points: int) -> None:
    for tag in ("sz", "szCs"):
        node = ensure_child(run_properties, tag)
        node.set(f"{W}val", str(half_points))


def set_math_font(run_properties: etree._Element) -> None:
    fonts = ensure_child(run_properties, "rFonts", first=True)
    if fonts.get(f"{W}ascii") is None:
        fonts.set(f"{W}ascii", "Cambria Math")
    if fonts.get(f"{W}hAnsi") is None:
        fonts.set(f"{W}hAnsi", "Cambria Math")


def format_math_object(math: etree._Element) -> None:
    for math_run in math.xpath(".//m:r", namespaces=NS):
        run_properties = math_run.find(f"{W}rPr")
        if run_properties is None:
            run_properties = etree.Element(f"{W}rPr")
            math_run.insert(0, run_properties)
        set_math_font(run_properties)
        set_run_size(run_properties, EQUATION_SIZE)
    for control in math.xpath(".//m:ctrlPr", namespaces=NS):
        run_properties = control.find(f"{W}rPr")
        if run_properties is None:
            run_properties = etree.SubElement(control, f"{W}rPr")
        set_math_font(run_properties)
        set_run_size(run_properties, EQUATION_SIZE)
    # Cover any additional run-property nodes nested inside the math object.
    for run_properties in math.xpath(".//w:rPr", namespaces=NS):
        set_run_size(run_properties, EQUATION_SIZE)


def format_equation_paragraph(paragraph: etree._Element) -> None:
    paragraph_properties = ensure_child(paragraph, "pPr", first=True)
    paragraph_run_properties = ensure_child(paragraph_properties, "rPr")
    set_run_size(paragraph_run_properties, EQUATION_SIZE)

    spacing = ensure_child(paragraph_properties, "spacing")
    spacing.set(f"{W}before", "100")
    spacing.set(f"{W}after", "100")
    spacing.set(f"{W}line", "280")
    spacing.set(f"{W}lineRule", "auto")
    ensure_child(paragraph_properties, "keepLines")

    for run in paragraph.xpath(".//w:r[not(ancestor::m:oMath)]", namespaces=NS):
        run_properties = ensure_child(run, "rPr", first=True)
        visible_text = "".join(run.xpath("./w:t/text()", namespaces=NS)).strip()
        size = NUMBER_SIZE if re.fullmatch(r"\(\d+\)", visible_text) else EQUATION_SIZE
        set_run_size(run_properties, size)


def patch_document(document_xml: bytes) -> tuple[bytes, int]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(document_xml, parser)
    equations = root.xpath(".//m:oMath", namespaces=NS)
    if len(equations) != 14:
        raise ValueError(f"Expected 14 displayed equations, found {len(equations)}")
    for equation in equations:
        format_math_object(equation)
        paragraph = equation.xpath("ancestor::w:p[1]", namespaces=NS)[0]
        format_equation_paragraph(paragraph)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes"), len(equations)


def main() -> None:
    with ZipFile(SOURCE, "r") as source_zip:
        document_xml, equation_count = patch_document(source_zip.read("word/document.xml"))
        with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as output_zip:
            for entry in source_zip.infolist():
                data = document_xml if entry.filename == "word/document.xml" else source_zip.read(entry.filename)
                output_zip.writestr(entry, data)
    print(OUTPUT)
    print(f"display_equations_formatted={equation_count}")
    print("equation_size_pt=12")
    print("equation_number_size_pt=11")


if __name__ == "__main__":
    main()
