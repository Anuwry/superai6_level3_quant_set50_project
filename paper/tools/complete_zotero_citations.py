"""Complete live Zotero citations and remove punctuation after citations.

The source manuscript contains seven bold full-stop markers denoting citation
insertion points and two author-year citations that were inserted as plain text.
This script replaces all nine locations with valid Zotero CSL_CITATION fields,
using item metadata already embedded in the document, and removes a full stop
that immediately follows any live Zotero citation as requested by the author.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper" / "newest_original_manuscript_results_visuals_integrated.docx"
OUTPUT = ROOT / "paper" / "newest_original_manuscript_results_visuals_integrated_cited.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"


MARKER_CITATIONS = [
    (
        "Table 2 reports the corrected point-in-time numerical ablation.",
        "(T. Liu et al., 2022; Y. Liu et al., 2024)",
        [
            "A Stock Price Prediction Method Based on Meta-Learning",
            "A Stock Series Prediction Model Based on Variational Mode Decomposition",
        ],
    ),
    (
        "The error and direction endpoints also diverged.",
        "(Olorunnimbe & Viktor, 2023; Sezer et al., 2020)",
        [
            "Deep Learning in the Stock Market",
            "Financial Time Series Forecasting with Deep Learning",
        ],
    ),
    (
        "The observed-news effect relative to the market-only arm",
        "(W.-J. Liu et al., 2024; M. Wang et al., 2024)",
        [
            "News-Driven Stock Market Index Prediction",
            "Modeling News Interactions and Influence",
        ],
    ),
    (
        "The negative-control result should not be interpreted",
        "(Luo et al., 2023; Sawhney et al., 2020; H. Wang et al., 2020)",
        [
            "Causality-Guided Multi-Memory Interaction Network",
            "Deep Attentive Learning for Stock Movement Prediction",
            "Incorporating Expert-Based Investment Opinion Signals",
        ],
    ),
    (
        "The local sentiment classifier achieved out-of-sample accuracies",
        "(Uthayopas et al., 2025)",
        ["Thai-English Target-Based Stock Sentiment Dataset"],
    ),
    (
        "LIME did not provide an independent validation of SHAP.",
        "(Ribeiro et al., 2016; Adebayo et al., 2018; Yeo et al., 2025)",
        [
            "Why Should I Trust You?",
            "Sanity Checks for Saliency Maps",
            "A Comprehensive Review on Financial Explainable AI",
        ],
    ),
    (
        "The 138-row partial-2026 stress test provided",
        "(Brodersen et al., 2010)",
        ["The Balanced Accuracy and Its Posterior Distribution"],
    ),
    (
        "Mean balanced accuracy declined on SET100",
        "(Bergmeir et al., 2018; Olorunnimbe & Viktor, 2023; Sezer et al., 2020)",
        [
            "A Note on the Validity of Cross-Validation",
            "Deep Learning in the Stock Market",
            "Financial Time Series Forecasting with Deep Learning",
        ],
    ),
    (
        "Note: Matched SET50 is the frozen 122-feature global numerical comparator",
        "(The Stock Exchange of Thailand, n.d.-a, n.d.-b)",
        [
            "SET50 Index and SET50FF Index Profile",
            "SET100 Index and SET100FF Index Profile",
        ],
    ),
]

PLAIN_CITATIONS = [
    (
        "Taken model by model, LSTM was strongest at level calibration",
        "(Lundberg & Lee, 2017; Ribeiro et al., 2016; Yeo et al., 2025)",
        [
            "A Unified Approach to Interpreting Model Predictions",
            "Why Should I Trust You?",
            "A Comprehensive Review on Financial Explainable AI",
        ],
    ),
    (
        "The prediction-level diagnostics further clarify",
        "(Fischer & Krauss, 2018; Hoseinzade & Haratizadeh, 2019; Sezer et al., 2020)",
        [
            "Deep Learning with Long Short-Term Memory Networks",
            "CNNpred: CNN-Based Stock Market Prediction",
            "Financial Time Series Forecasting with Deep Learning",
        ],
    ),
]


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def parse_csl_instruction(instruction: str) -> dict | None:
    marker = "CSL_CITATION "
    if marker not in instruction:
        return None
    payload = instruction.split(marker, 1)[1].strip()
    return json.loads(payload)


def collect_item_registry(root: etree._Element) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for node in root.xpath(".//w:instrText[contains(text(), 'CSL_CITATION')]", namespaces=NS):
        citation = parse_csl_instruction(node.text or "")
        if not citation:
            continue
        for item in citation.get("citationItems", []):
            title = item.get("itemData", {}).get("title")
            if title:
                registry.setdefault(title, deepcopy(item))
    return registry


def find_item(registry: dict[str, dict], title_fragment: str) -> dict:
    matches = [item for title, item in registry.items() if title_fragment.casefold() in title.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one Zotero item for {title_fragment!r}, found {len(matches)}")
    return deepcopy(matches[0])


def find_template_field_runs(root: etree._Element) -> list[etree._Element]:
    instruction = root.xpath(
        ".//w:instrText[contains(text(), 'ADDIN ZOTERO_ITEM CSL_CITATION')]", namespaces=NS
    )[0]
    instruction_run = instruction.getparent()
    paragraph = instruction_run.getparent()
    children = list(paragraph)
    instruction_index = children.index(instruction_run)

    start = instruction_index
    while start >= 0:
        field_types = children[start].xpath("./w:fldChar/@w:fldCharType", namespaces=NS)
        if "begin" in field_types:
            break
        start -= 1
    end = instruction_index
    while end < len(children):
        field_types = children[end].xpath("./w:fldChar/@w:fldCharType", namespaces=NS)
        if "end" in field_types:
            break
        end += 1
    if start < 0 or end >= len(children):
        raise ValueError("Could not isolate a Zotero citation field template")
    return [deepcopy(child) for child in children[start : end + 1]]


def build_field_runs(
    template_runs: list[etree._Element],
    display: str,
    items: list[dict],
    identity: str,
) -> list[etree._Element]:
    runs = [deepcopy(run) for run in template_runs]
    citation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    citation = {
        "citationID": citation_id,
        "properties": {
            "unsorted": False,
            "formattedCitation": display,
            "plainCitation": display,
            "noteIndex": 0,
        },
        "citationItems": items,
        "schema": "https://github.com/citation-style-language/schema/raw/master/csl-citation.json",
    }
    instruction_text = (
        " ADDIN ZOTERO_ITEM CSL_CITATION "
        + json.dumps(citation, ensure_ascii=False, separators=(",", ":"))
        + " "
    )

    instruction_nodes = []
    result_nodes = []
    after_separator = False
    for run in runs:
        field_types = run.xpath("./w:fldChar/@w:fldCharType", namespaces=NS)
        if "separate" in field_types:
            after_separator = True
        if "end" in field_types:
            after_separator = False
        instruction_nodes.extend(run.xpath("./w:instrText", namespaces=NS))
        if after_separator and not field_types:
            result_nodes.extend(run.xpath("./w:t", namespaces=NS))

    if len(instruction_nodes) != 1 or not result_nodes:
        raise ValueError("Unexpected Zotero field template structure")
    instruction_nodes[0].text = instruction_text
    instruction_nodes[0].set(f"{{{XML_NS}}}space", "preserve")
    result_nodes[0].text = display
    for node in result_nodes[1:]:
        node.text = ""
    return runs


def set_run_text(run: etree._Element, text: str) -> None:
    for child in list(run):
        if child.tag != f"{W}rPr":
            run.remove(child)
    text_node = etree.SubElement(run, f"{W}t")
    text_node.text = text
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(f"{{{XML_NS}}}space", "preserve")


def strip_bold(run: etree._Element) -> None:
    properties = run.find(f"{W}rPr")
    if properties is None:
        return
    for tag in (f"{W}b", f"{W}bCs"):
        for node in properties.findall(tag):
            properties.remove(node)


def insert_field_at_marker(
    paragraph: etree._Element,
    field_runs: list[etree._Element],
) -> None:
    children = list(paragraph)
    marker = None
    for child in reversed(children):
        if child.tag != f"{W}r":
            continue
        text = "".join(child.xpath("./w:t/text()", namespaces=NS))
        is_bold = bool(child.xpath("./w:rPr/w:b", namespaces=NS))
        if text == "." and is_bold:
            marker = child
            break
    if marker is None:
        raise ValueError(f"Bold citation marker not found: {paragraph_text(paragraph)[:100]}")

    marker_index = paragraph.index(marker)
    strip_bold(marker)
    set_run_text(marker, " ")
    for offset, run in enumerate(field_runs, start=1):
        paragraph.insert(marker_index + offset, run)


def replace_plain_citation(
    paragraph: etree._Element,
    target: str,
    field_runs: list[etree._Element],
) -> None:
    # Word may wrap a visible run in revision/proofing containers. Search all
    # descendant runs and insert the field beside the matching run in its
    # immediate container so that those wrappers remain valid.
    # One paragraph inherited an earlier misspelling ("Hoseinzadeh") in the
    # plain-text placeholder. Match it, but render the canonical Zotero form
    # ("Hoseinzade") stored in ``target``.
    target_variants = [target, target.replace("Hoseinzade ", "Hoseinzadeh ")]
    for run in paragraph.xpath(".//w:r", namespaces=NS):
        text = "".join(run.xpath("./w:t/text()", namespaces=NS))
        matched_target = next((candidate for candidate in target_variants if candidate in text), None)
        if matched_target is None:
            continue
        before, after = text.split(matched_target, 1)
        if after.startswith("."):
            after = after[1:]
        container = run.getparent()
        run_index = container.index(run)
        set_run_text(run, before)
        for offset, field_run in enumerate(field_runs, start=1):
            container.insert(run_index + offset, field_run)
        if after:
            suffix_run = deepcopy(run)
            set_run_text(suffix_run, after)
            container.insert(run_index + len(field_runs) + 1, suffix_run)
        return
    raise ValueError(f"Plain citation not found: {target}")


def is_zotero_field_group(children: list[etree._Element], start: int, end: int) -> bool:
    instruction = "".join(
        text
        for child in children[start : end + 1]
        for text in child.xpath("./w:instrText/text()", namespaces=NS)
    )
    return "ADDIN ZOTERO_ITEM CSL_CITATION" in instruction


def remove_periods_after_zotero_fields(root: etree._Element) -> int:
    removed = 0
    for paragraph in root.xpath(".//w:body/w:p", namespaces=NS):
        children = list(paragraph)
        index = 0
        while index < len(children):
            field_types = children[index].xpath("./w:fldChar/@w:fldCharType", namespaces=NS)
            if "begin" not in field_types:
                index += 1
                continue
            end = index + 1
            while end < len(children):
                end_types = children[end].xpath("./w:fldChar/@w:fldCharType", namespaces=NS)
                if "end" in end_types:
                    break
                end += 1
            if end >= len(children) or not is_zotero_field_group(children, index, end):
                index = end + 1
                continue

            next_index = end + 1
            while next_index < len(children):
                next_run = children[next_index]
                if next_run.tag != f"{W}r":
                    next_index += 1
                    continue
                text = "".join(next_run.xpath("./w:t/text()", namespaces=NS))
                if not text:
                    next_index += 1
                    continue
                if text.startswith("."):
                    set_run_text(next_run, text[1:])
                    if not text[1:]:
                        paragraph.remove(next_run)
                        children.pop(next_index)
                    removed += 1
                break
            children = list(paragraph)
            index = end + 1
    return removed


def find_paragraph(paragraphs: list[etree._Element], prefix: str) -> etree._Element:
    matches = [p for p in paragraphs if paragraph_text(p).strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph starting {prefix!r}, found {len(matches)}")
    return matches[0]


def patch_document_xml(document_xml: bytes) -> tuple[bytes, dict[str, int]]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(document_xml, parser)
    registry = collect_item_registry(root)
    template_runs = find_template_field_runs(root)
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)

    inserted_markers = 0
    for prefix, display, title_fragments in MARKER_CITATIONS:
        paragraph = find_paragraph(paragraphs, prefix)
        items = [find_item(registry, fragment) for fragment in title_fragments]
        fields = build_field_runs(template_runs, display, items, f"marker:{prefix}:{display}")
        insert_field_at_marker(paragraph, fields)
        inserted_markers += 1

    converted_plain = 0
    for prefix, display, title_fragments in PLAIN_CITATIONS:
        paragraph = find_paragraph(paragraphs, prefix)
        items = [find_item(registry, fragment) for fragment in title_fragments]
        fields = build_field_runs(template_runs, display, items, f"plain:{prefix}:{display}")
        replace_plain_citation(paragraph, display, fields)
        converted_plain += 1

    removed_periods = remove_periods_after_zotero_fields(root)
    output = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
    return output, {
        "marker_fields_inserted": inserted_markers,
        "plain_citations_converted": converted_plain,
        "periods_removed_after_citations": removed_periods,
    }


def write_docx() -> dict[str, int]:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    with ZipFile(SOURCE, "r") as source_zip:
        patched_xml, stats = patch_document_xml(source_zip.read("word/document.xml"))
        with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as output_zip:
            for entry in source_zip.infolist():
                data = patched_xml if entry.filename == "word/document.xml" else source_zip.read(entry.filename)
                output_zip.writestr(entry, data)
    return stats


def main() -> None:
    stats = write_docx()
    print(OUTPUT)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
