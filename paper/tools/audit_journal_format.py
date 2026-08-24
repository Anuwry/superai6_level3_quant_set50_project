from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
DOCX = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else ROOT / "paper" / "SET_direction_manuscript_journal_formatted_v1.docx"
)
SOURCE = ROOT / "paper" / "newest_original_manuscript_results_visuals_integrated_cited.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
}


EXPECTED_HEADINGS = [
    "1. Introduction",
    "2. Related Work",
    "2.1 Deep Sequence Models",
    "2.2 Denoising and Multimodal Data",
    "2.3 LLM Role Systems",
    "2.4 Market Regimes and Explainability",
    "2.5 Evaluation Reliability",
    "3. Methods",
    "3.1 Study Design and Data Governance",
    "3.2 News and Sentiment Data",
    "3.3 Point-in-Time Evaluation",
    "3.4 Numerical Features and VMD",
    "3.5 Neural Architectures",
    "3.6 Multimodal and LLM Evaluation",
    "3.7 Market Regimes and SHAP",
    "3.8 Metrics and Inference",
    "4. Results",
    "4.1 VMD",
    "4.2 Predicted News",
    "4.3 LLM Sentiment",
    "4.4 Regime-SHAP",
    "4.5 Partial-2026 Robustness",
    "4.6 SET100 Transfer",
    "5. Discussion",
    "6. Conclusion",
    "Limitations",
    "Future Work",
    "Acknowledgements",
    "Reproducibility and Data Availability",
    "References",
]


def text_of(node: etree._Element) -> str:
    return "".join(node.xpath(".//w:t/text()", namespaces=NS)).strip()


def style_of(paragraph: etree._Element) -> str:
    values = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else "Normal"


def count_fields(root: etree._Element) -> Counter[str]:
    return Counter(root.xpath(".//w:fldChar/@w:fldCharType", namespaces=NS))


def citation_instructions(root: etree._Element) -> list[str]:
    return [
        value
        for value in root.xpath(".//w:instrText/text()", namespaces=NS)
        if "CSL_CITATION" in value
    ]


def validate_citation_json(instructions: list[str]) -> int:
    valid = 0
    for instruction in instructions:
        payload = instruction.split("CSL_CITATION", 1)[1].strip()
        json.loads(payload)
        valid += 1
    return valid


def package_counts(path: Path) -> dict[str, int]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return {
        "media": sum(name.startswith("word/media/") for name in names),
        "embeddings": sum(name.startswith("word/embeddings/") for name in names),
        "customXml": sum(name.startswith("customXml/") for name in names),
    }


def main() -> int:
    failures: list[str] = []
    with zipfile.ZipFile(DOCX) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
        styles = etree.fromstring(archive.read("word/styles.xml"))
        core = etree.fromstring(archive.read("docProps/core.xml"))

    paragraphs = document.xpath(".//w:body/w:p", namespaces=NS)
    all_paragraphs = document.xpath(".//w:p", namespaces=NS)
    title_text = text_of(paragraphs[0])
    correspondence = next((text_of(p) for p in paragraphs if text_of(p).startswith("Correspondence:")), "")
    headings = [text_of(p) for p in paragraphs if style_of(p) in {"Heading1", "Heading2"}]
    heading_counts = Counter(style_of(p) for p in paragraphs)
    captions = [
        p
        for p in all_paragraphs
        if re.match(r"^(Figure|Table)\s+\d+[A-Z]?\.", text_of(p))
    ]
    notes = [p for p in all_paragraphs if text_of(p).startswith("Note:")]

    if title_text != "Evaluating Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting":
        failures.append(f"title mismatch: {title_text!r}")
    if correspondence != "Correspondence: Arsanchai.su@wu.ac.th":
        failures.append(f"correspondence mismatch: {correspondence!r}")
    if headings != EXPECTED_HEADINGS:
        failures.append("heading text/order mismatch")
    if heading_counts["Heading1"] != 9 or heading_counts["Heading2"] != 21:
        failures.append(f"heading style counts mismatch: {heading_counts}")
    if len(captions) != 17:
        failures.append(f"caption count mismatch: {len(captions)}")
    if len(notes) != 6:
        failures.append(f"note count mismatch: {len(notes)}")

    for paragraph in captions + notes:
        sizes = set(paragraph.xpath(".//w:r[w:t]/w:rPr/w:sz/@w:val", namespaces=NS))
        if sizes != {"20"}:
            failures.append(f"non-10pt caption/note: {text_of(paragraph)[:60]!r} sizes={sizes}")

    for style_id, expected_size, expect_keep_next in (
        ("Heading1", "24", True),
        ("Heading2", "24", True),
        ("Caption", "20", False),
    ):
        node = styles.xpath(f".//w:style[@w:styleId='{style_id}']", namespaces=NS)[0]
        sizes = node.xpath("./w:rPr/w:sz/@w:val", namespaces=NS)
        keep_next = bool(node.xpath("./w:pPr/w:keepNext", namespaces=NS))
        if sizes != [expected_size] or keep_next != expect_keep_next:
            failures.append(f"style {style_id} mismatch: size={sizes}, keepNext={keep_next}")

    fields = count_fields(document)
    instructions = citation_instructions(document)
    valid_json = validate_citation_json(instructions)
    if fields != Counter({"begin": 68, "separate": 68, "end": 68}):
        failures.append(f"field structure mismatch: {fields}")
    if len(instructions) != 67 or valid_json != 67:
        failures.append(f"citation fields mismatch: total={len(instructions)}, valid_json={valid_json}")

    references_index = next(
        (index for index, paragraph in enumerate(paragraphs) if text_of(paragraph) == "References"),
        len(paragraphs),
    )
    body_before_refs = "\n".join(text_of(p) for p in paragraphs[:references_index])
    for token in ("N_t", "n_t", "R^2", "alpha =", "tau =", "omega_(", "sigma_(", "lambda_f"):
        if token in body_before_refs:
            failures.append(f"unformatted notation remains: {token}")
    if re.search(r"\b(?:TODO|TBD)\b|Error!", body_before_refs, flags=re.IGNORECASE):
        failures.append("placeholder or Word error text remains")

    core_title = core.xpath("string(./dc:title)", namespaces=NS)
    if core_title != title_text:
        failures.append(f"core title mismatch: {core_title!r}")

    source_counts = package_counts(SOURCE)
    output_counts = package_counts(DOCX)
    if source_counts != output_counts:
        failures.append(f"package object counts changed: source={source_counts}, output={output_counts}")

    summary = {
        "paragraphs": len(paragraphs),
        "tables": len(document.xpath(".//w:tbl", namespaces=NS)),
        "drawings": len(document.xpath(".//w:drawing", namespaces=NS)),
        "headings": dict(heading_counts),
        "captions": len(captions),
        "notes": len(notes),
        "citation_fields": len(instructions),
        "valid_citation_json": valid_json,
        "field_chars": dict(fields),
        "subscripts": len(document.xpath(".//w:vertAlign[@w:val='subscript']", namespaces=NS)),
        "superscripts": len(document.xpath(".//w:vertAlign[@w:val='superscript']", namespaces=NS)),
        "package_objects": output_counts,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
