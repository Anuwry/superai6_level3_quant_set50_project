"""Apply journal-style typography and inline mathematical notation.

The patch is intentionally OOXML-level so live Zotero fields and all embedded
figures/tables remain untouched. Only front matter, section headings, captions,
table notes, selected inline notation, paragraph styles, and core title
metadata are changed.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper" / "newest_original_manuscript_results_visuals_integrated_cited.docx"
OUTPUT = ROOT / "paper" / "SET_direction_manuscript_journal_formatted_v1.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
DC_NS = "http://purl.org/dc/elements/1.1/"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"

TITLE_LINE_1 = "Evaluating Multimodal and Regime-Aware Deep Learning for"
TITLE_LINE_2 = "Next-Day SET Index Direction Forecasting"
TITLE = f"{TITLE_LINE_1} {TITLE_LINE_2}"
CORRESPONDENCE = "Correspondence: Arsanchai.su@wu.ac.th"


HEADING_MAP = {
    "1. Introduction": ("1. Introduction", "Heading1"),
    "2. Related works": ("2. Related Work", "Heading1"),
    "2.1 Deep Sequence models for financial direction": ("2.1 Deep Sequence Models", "Heading2"),
    "2.2 Denoising and multimodal financial information": ("2.2 Denoising and Multimodal Data", "Heading2"),
    "2.3 LLM role systems and compute matching": ("2.3 LLM Role Systems", "Heading2"),
    "2.4 Market regimes and explanation reliability": ("2.4 Market Regimes and Explainability", "Heading2"),
    "2.5 Evaluation reliability": ("2.5 Evaluation Reliability", "Heading2"),
    "3. Materials and Methods": ("3. Methods", "Heading1"),
    "3.1 Study design, data governance and evidence hierarchy": ("3.1 Study Design and Data Governance", "Heading2"),
    "3.2 News data and out-of-sample sentiment construction": ("3.2 News and Sentiment Data", "Heading2"),
    "3.3 Point-in-time contract, temporal splits and frozen windows": ("3.3 Point-in-Time Evaluation", "Heading2"),
    "3.4 Numerical features and causal VMD": ("3.4 Numerical Features and VMD", "Heading2"),
    "3.5 Neural architectures and training": ("3.5 Neural Architectures", "Heading2"),
    "3.6 Multimodal falsification and Bull/Bear/Leader evaluation": ("3.6 Multimodal and LLM Evaluation", "Heading2"),
    "3.7 Causal market regimes and SHAP selection": ("3.7 Market Regimes and SHAP", "Heading2"),
    "3.8 Metrics and statistical inference": ("3.8 Metrics and Inference", "Heading2"),
    "4. Results": ("4. Results", "Heading1"),
    "4.1 Numerical denoising": ("4.1 VMD", "Heading2"),
    "4.2 Predicted news": ("4.2 Predicted News", "Heading2"),
    "4.3 The LLM role system sentiment": ("4.3 LLM Sentiment", "Heading2"),
    "4.4 Regime-SHAP selection": ("4.4 Regime-SHAP", "Heading2"),
    "4.5 Partial performance exposed majority-side collapse": ("4.5 Partial-2026 Robustness", "Heading2"),
    "4.6 Frozen SET100 transfer": ("4.6 SET100 Transfer", "Heading2"),
    "5. Discussion": ("5. Discussion", "Heading1"),
    "6. Conclusion": ("6. Conclusion", "Heading1"),
    "Acknowledgements": ("Acknowledgements", "Heading1"),
    "Reproducibility and data availability": ("Reproducibility and Data Availability", "Heading1"),
    "References": ("References", "Heading1"),
}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def ensure_child(parent: etree._Element, tag: str, first: bool = False) -> etree._Element:
    child = parent.find(f"{W}{tag}")
    if child is None:
        child = etree.Element(f"{W}{tag}")
        if first:
            parent.insert(0, child)
        else:
            parent.append(child)
    return child


def remove_children(parent: etree._Element, tags: tuple[str, ...]) -> None:
    wanted = {f"{W}{tag}" for tag in tags}
    for child in list(parent):
        if child.tag in wanted:
            parent.remove(child)


def set_paragraph_style(paragraph: etree._Element, style_id: str) -> etree._Element:
    ppr = ensure_child(paragraph, "pPr", first=True)
    pstyle = ensure_child(ppr, "pStyle", first=True)
    pstyle.set(f"{W}val", style_id)
    return ppr


def set_paragraph_spacing(
    paragraph: etree._Element,
    *,
    before: int | None = None,
    after: int | None = None,
    line: int | None = None,
    line_rule: str = "auto",
) -> None:
    ppr = ensure_child(paragraph, "pPr", first=True)
    spacing = ensure_child(ppr, "spacing")
    if before is not None:
        spacing.set(f"{W}before", str(before))
    if after is not None:
        spacing.set(f"{W}after", str(after))
    if line is not None:
        spacing.set(f"{W}line", str(line))
        spacing.set(f"{W}lineRule", line_rule)


def set_keep_next(paragraph: etree._Element, enabled: bool) -> None:
    ppr = ensure_child(paragraph, "pPr", first=True)
    existing = ppr.find(f"{W}keepNext")
    if enabled and existing is None:
        ppr.append(etree.Element(f"{W}keepNext"))
    elif not enabled and existing is not None:
        ppr.remove(existing)


def set_alignment(paragraph: etree._Element, value: str) -> None:
    ppr = ensure_child(paragraph, "pPr", first=True)
    jc = ensure_child(ppr, "jc")
    jc.set(f"{W}val", value)


def run_properties(
    run: etree._Element,
    *,
    size_half_points: int | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    vertical: str | None = None,
    font: str = "Times New Roman",
) -> etree._Element:
    rpr = ensure_child(run, "rPr", first=True)
    fonts = ensure_child(rpr, "rFonts", first=True)
    for attr in ("ascii", "hAnsi", "cs"):
        fonts.set(f"{W}{attr}", font)
    if size_half_points is not None:
        for tag in ("sz", "szCs"):
            node = ensure_child(rpr, tag)
            node.set(f"{W}val", str(size_half_points))
    if bold is not None:
        remove_children(rpr, ("b", "bCs"))
        if bold:
            rpr.append(etree.Element(f"{W}b"))
            rpr.append(etree.Element(f"{W}bCs"))
    if italic is not None:
        remove_children(rpr, ("i", "iCs"))
        if italic:
            rpr.append(etree.Element(f"{W}i"))
            rpr.append(etree.Element(f"{W}iCs"))
    remove_children(rpr, ("vertAlign",))
    if vertical:
        node = etree.SubElement(rpr, f"{W}vertAlign")
        node.set(f"{W}val", vertical)
    # Remove temporary authoring highlight from rebuilt title/caption runs.
    remove_children(rpr, ("highlight",))
    return rpr


def append_run(
    paragraph: etree._Element,
    text: str,
    *,
    size_half_points: int = 24,
    bold: bool = False,
    italic: bool = False,
    vertical: str | None = None,
    line_break_after: bool = False,
) -> etree._Element:
    run = etree.SubElement(paragraph, f"{W}r")
    run_properties(
        run,
        size_half_points=size_half_points,
        bold=bold,
        italic=italic,
        vertical=vertical,
    )
    text_node = etree.SubElement(run, f"{W}t")
    text_node.text = text
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    if line_break_after:
        etree.SubElement(run, f"{W}br")
    return run


def clear_paragraph_content(paragraph: etree._Element) -> None:
    for child in list(paragraph):
        if child.tag != f"{W}pPr":
            paragraph.remove(child)


def rebuild_plain_paragraph(
    paragraph: etree._Element,
    text: str,
    *,
    style_id: str,
    size_half_points: int,
    bold: bool = False,
    italic: bool = False,
) -> None:
    clear_paragraph_content(paragraph)
    set_paragraph_style(paragraph, style_id)
    append_run(
        paragraph,
        text,
        size_half_points=size_half_points,
        bold=bold,
        italic=italic,
    )


def find_exact(paragraphs: list[etree._Element], text: str) -> etree._Element:
    matches = [p for p in paragraphs if paragraph_text(p).strip() == text]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact paragraph {text!r}, found {len(matches)}")
    return matches[0]


def find_prefix(paragraphs: list[etree._Element], prefix: str) -> etree._Element:
    matches = [p for p in paragraphs if paragraph_text(p).strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph starting {prefix!r}, found {len(matches)}")
    return matches[0]


def configure_heading(paragraph: etree._Element, style_id: str, text: str) -> None:
    rebuild_plain_paragraph(
        paragraph,
        text,
        style_id=style_id,
        size_half_points=24,
        bold=True,
    )
    if style_id == "Heading1":
        set_paragraph_spacing(paragraph, before=240, after=120, line=240)
    else:
        set_paragraph_spacing(paragraph, before=180, after=60, line=240)
    set_alignment(paragraph, "left")
    set_keep_next(paragraph, True)


def split_inline_section(paragraph: etree._Element, label: str) -> None:
    full = paragraph_text(paragraph).strip()
    source_label, separator, remainder = full.partition(":")
    if not separator or source_label.casefold() != label.casefold():
        raise ValueError(f"Expected inline section {label!r}, found {source_label!r}")
    prefix = f"{source_label}:"
    body = remainder.strip()
    heading = etree.Element(f"{W}p")
    configure_heading(heading, "Heading2", label)
    parent = paragraph.getparent()
    parent.insert(parent.index(paragraph), heading)
    rebuild_plain_paragraph(
        paragraph,
        body,
        style_id="Normal",
        size_half_points=24,
        bold=False,
    )
    set_paragraph_spacing(paragraph, after=120, line=360)
    set_alignment(paragraph, "both")


def format_caption(paragraph: etree._Element, label: str, description: str, kind: str) -> None:
    clear_paragraph_content(paragraph)
    set_paragraph_style(paragraph, "Caption")
    append_run(paragraph, label, size_half_points=20, bold=True)
    if description:
        append_run(paragraph, f" {description}", size_half_points=20, bold=False)
    set_paragraph_spacing(paragraph, before=60, after=120, line=240)
    set_keep_next(paragraph, kind == "Table")


def format_note(paragraph: etree._Element, text: str) -> None:
    # Keep live Zotero fields and any intentional inline emphasis in table
    # notes. Rebuilding the paragraph as plain text would discard those fields.
    set_paragraph_style(paragraph, "Caption")
    set_paragraph_spacing(paragraph, before=40, after=120, line=240)
    set_keep_next(paragraph, False)
    visible_runs = paragraph.xpath(".//w:r[w:t]", namespaces=NS)
    for run in visible_runs:
        run_properties(run, size_half_points=20)
    if not visible_runs:
        return
    first_run = visible_runs[0]
    first_text_nodes = first_run.xpath("./w:t", namespaces=NS)
    if len(first_text_nodes) != 1:
        return
    first_text = first_text_nodes[0].text or ""
    if first_text == "Note:":
        run_properties(first_run, size_half_points=20, italic=True)
    elif first_text.startswith("Note:"):
        remainder = first_text[len("Note:") :]
        parent = first_run.getparent()
        index = parent.index(first_run)
        parent.remove(first_run)
        label_run = deepcopy(first_run)
        for child in list(label_run):
            if child.tag != f"{W}rPr":
                label_run.remove(child)
        run_properties(label_run, size_half_points=20, italic=True)
        label_text = etree.SubElement(label_run, f"{W}t")
        label_text.text = "Note:"
        parent.insert(index, label_run)
        if remainder:
            body_run = deepcopy(first_run)
            for child in list(body_run):
                if child.tag != f"{W}rPr":
                    body_run.remove(child)
            run_properties(body_run, size_half_points=20, italic=False)
            body_text = etree.SubElement(body_run, f"{W}t")
            body_text.text = remainder
            if remainder.startswith(" "):
                body_text.set(f"{{{XML_NS}}}space", "preserve")
            parent.insert(index + 1, body_run)


def plain_component(text: str, *, italic: bool | None = None, vertical: str | None = None) -> dict:
    return {"text": text, "italic": italic, "vertical": vertical}


TOKEN_COMPONENTS: dict[str, list[dict]] = {
    "N_t": [plain_component("N", italic=True), plain_component("t", italic=True, vertical="subscript")],
    "n_t": [plain_component("n", italic=True), plain_component("t", italic=True, vertical="subscript")],
    "C_(t+1)": [
        plain_component("C", italic=True),
        plain_component("t", italic=True, vertical="subscript"),
        plain_component("+1", italic=False, vertical="subscript"),
    ],
    "Label_Date_t": [
        plain_component("LabelDate", italic=False),
        plain_component("t", italic=True, vertical="subscript"),
    ],
    "Label_Date": [plain_component("LabelDate", italic=False)],
    "u_(k,t)": [
        plain_component("u", italic=True),
        plain_component("k", italic=True, vertical="subscript"),
        plain_component(",", italic=False, vertical="subscript"),
        plain_component("t", italic=True, vertical="subscript"),
    ],
    "omega_(k,t)": [
        plain_component("ω", italic=True),
        plain_component("k", italic=True, vertical="subscript"),
        plain_component(",", italic=False, vertical="subscript"),
        plain_component("t", italic=True, vertical="subscript"),
    ],
    "sigma_(t,v(h))": [
        plain_component("σ", italic=True),
        plain_component("t", italic=True, vertical="subscript"),
        plain_component(",", italic=False, vertical="subscript"),
        plain_component("v", italic=True, vertical="subscript"),
        plain_component("(", italic=False, vertical="subscript"),
        plain_component("h", italic=True, vertical="subscript"),
        plain_component(")", italic=False, vertical="subscript"),
    ],
    "ADX_(14,t)": [
        plain_component("ADX", italic=False),
        plain_component("14,", italic=False, vertical="subscript"),
        plain_component("t", italic=True, vertical="subscript"),
    ],
    "EWMA_3": [plain_component("EWMA", italic=False), plain_component("3", italic=False, vertical="subscript")],
    "lambda_f": [plain_component("λ", italic=True), plain_component("f", italic=True, vertical="subscript")],
    "z_t": [plain_component("z", italic=True), plain_component("t", italic=True, vertical="subscript")],
    "T_r": [plain_component("T", italic=True), plain_component("r", italic=True, vertical="subscript")],
    "S_r(k)": [
        plain_component("S", italic=True),
        plain_component("r", italic=True, vertical="subscript"),
        plain_component("(", italic=False),
        plain_component("k", italic=True),
        plain_component(")", italic=False),
    ],
    "R^2": [plain_component("R", italic=True), plain_component("2", italic=False, vertical="superscript")],
    "alpha = 1000": [plain_component("α", italic=True), plain_component(" = 1000", italic=False)],
    "tau = 0": [plain_component("τ", italic=True), plain_component(" = 0", italic=False)],
    "10^-7": [plain_component("10", italic=False), plain_component("−7", italic=False, vertical="superscript")],
    "v(h)=20": [
        plain_component("v", italic=True), plain_component("(", italic=False), plain_component("h", italic=True),
        plain_component(") = 20", italic=False),
    ],
    "h=60": [plain_component("h", italic=True), plain_component(" = 60", italic=False)],
    "t+1": [plain_component("t", italic=True), plain_component(" + 1", italic=False)],
    "t-59": [plain_component("t", italic=True), plain_component(" − 59", italic=False)],
    "H": [plain_component("H", italic=True)],
    "c": [plain_component("c", italic=True)],
    "t": [plain_component("t", italic=True)],
    "f": [plain_component("f", italic=True)],
    "r": [plain_component("r", italic=True)],
    "j": [plain_component("j", italic=True)],
    "k": [plain_component("k", italic=True)],
    "h": [plain_component("h", italic=True)],
}


NOTATION_JOBS = [
    ("For each article-ticker pair", ["N_t", "n_t", "c", "t"]),
    ("For feature date t", ["C_(t+1)", "Label_Date_t", "Label_Date", "t"]),
    ("Causal rolling VMD adds six variables", ["alpha = 1000", "tau = 0", "10^-7", "t-59", "t"]),
    ("Equations (3)-(5) formalize", ["u_(k,t)", "omega_(k,t)", "k", "t"]),
    ("Daily regimes are assigned", ["sigma_(t,v(h))", "v(h)=20", "h=60", "ADX_(14,t)", "EWMA_3", "H", "h", "t"]),
    ("For outer fold f", ["lambda_f", "z_t", "t+1", "f", "t"]),
    ("Within regime r", ["T_r", "S_r(k)", "r", "j", "k"]),
    ("Seven outer arms separated", ["R^2"]),
    ("LIME did not provide", ["R^2"]),
    ("Figure 7 places", ["R^2"]),
    ("Figure 7. Explainability audit.", ["R^2"]),
]


def token_pattern(tokens: list[str]) -> re.Pattern[str]:
    parts = []
    for token in sorted(tokens, key=len, reverse=True):
        escaped = re.escape(token)
        if token.isalpha():
            parts.append(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        else:
            parts.append(escaped)
    return re.compile("|".join(parts))


def make_segment_run(template: etree._Element, component: dict) -> etree._Element:
    run = deepcopy(template)
    for child in list(run):
        if child.tag != f"{W}rPr":
            run.remove(child)
    existing_sizes = template.xpath("./w:rPr/w:sz/@w:val", namespaces=NS)
    size_half_points = int(existing_sizes[0]) if existing_sizes else 24
    run_properties(
        run,
        size_half_points=size_half_points,
        italic=component.get("italic"),
        vertical=component.get("vertical"),
    )
    text_node = etree.SubElement(run, f"{W}t")
    text_node.text = component["text"]
    if component["text"].startswith(" ") or component["text"].endswith(" "):
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    return run


def format_spanning_token(paragraph: etree._Element, token: str) -> int:
    """Format a notation token split across adjacent Word runs.

    Word often divides strings such as ``u_(k,t)`` at spell-check or edit
    boundaries. This routine replaces only direct text runs and leaves nearby
    Zotero field runs untouched.
    """
    changed = 0
    pattern = token_pattern([token])
    while True:
        run_entries: list[tuple[etree._Element, str, int, int]] = []
        cursor = 0
        for child in paragraph:
            if child.tag != f"{W}r":
                continue
            if child.xpath("./w:fldChar|./w:instrText|./w:drawing|./w:pict", namespaces=NS):
                continue
            text_nodes = child.xpath("./w:t", namespaces=NS)
            if len(text_nodes) != 1:
                continue
            value = text_nodes[0].text or ""
            run_entries.append((child, value, cursor, cursor + len(value)))
            cursor += len(value)
        combined = "".join(value for _, value, _, _ in run_entries)
        match = pattern.search(combined)
        if not match:
            break
        first_entry = next(entry for entry in run_entries if entry[2] <= match.start() < entry[3])
        last_entry = next(entry for entry in run_entries if entry[2] < match.end() <= entry[3])
        first_run, first_text, first_start, _ = first_entry
        last_run, last_text, last_start, _ = last_entry
        first_index = run_entries.index(first_entry)
        last_index = run_entries.index(last_entry)
        involved = [entry[0] for entry in run_entries[first_index : last_index + 1]]
        prefix = first_text[: match.start() - first_start]
        suffix = last_text[match.end() - last_start :]
        insert_at = paragraph.index(first_run)
        for run in involved:
            paragraph.remove(run)
        segments: list[dict] = []
        if prefix:
            segments.append(plain_component(prefix))
        segments.extend(TOKEN_COMPONENTS[token])
        if suffix:
            segments.append(plain_component(suffix))
        for offset, component in enumerate(segments):
            paragraph.insert(insert_at + offset, make_segment_run(first_run, component))
        changed += 1
    return changed


def format_tokens_in_paragraph(paragraph: etree._Element, tokens: list[str]) -> int:
    pattern = token_pattern(tokens)
    changed = 0
    # Copy the list because matching runs are replaced during iteration.
    for run in list(paragraph.xpath(".//w:r", namespaces=NS)):
        if run.xpath("./w:fldChar|./w:instrText|./w:drawing|./w:pict", namespaces=NS):
            continue
        text_nodes = run.xpath("./w:t", namespaces=NS)
        if len(text_nodes) != 1:
            continue
        text = text_nodes[0].text or ""
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        segments: list[dict] = []
        cursor = 0
        for match in matches:
            if match.start() > cursor:
                segments.append(plain_component(text[cursor : match.start()]))
            segments.extend(TOKEN_COMPONENTS[match.group(0)])
            cursor = match.end()
        if cursor < len(text):
            segments.append(plain_component(text[cursor:]))
        parent = run.getparent()
        index = parent.index(run)
        parent.remove(run)
        for offset, component in enumerate(segments):
            parent.insert(index + offset, make_segment_run(run, component))
        changed += len(matches)
    # Process any complex notation that Word divided across multiple runs only
    # after the single-run pass, so newly created subscripts are not matched a
    # second time by shorter tokens such as ``t`` or ``k``.
    changed += sum(
        format_spanning_token(paragraph, token)
        for token in tokens
        if "_" in token
    )
    return changed


def configure_style(
    styles_root: etree._Element,
    style_id: str,
    *,
    name: str,
    size_half_points: int,
    bold: bool,
    before: int,
    after: int,
    outline_level: int | None,
    keep_next: bool = True,
) -> None:
    matches = styles_root.xpath(f".//w:style[@w:styleId='{style_id}']", namespaces=NS)
    if matches:
        style = matches[0]
    else:
        style = etree.SubElement(styles_root, f"{W}style")
        style.set(f"{W}type", "paragraph")
        style.set(f"{W}styleId", style_id)
        style.set(f"{W}customStyle", "1")
        name_node = etree.SubElement(style, f"{W}name")
        name_node.set(f"{W}val", name)
        based = etree.SubElement(style, f"{W}basedOn")
        based.set(f"{W}val", "Normal")
        next_node = etree.SubElement(style, f"{W}next")
        next_node.set(f"{W}val", "Normal")
        etree.SubElement(style, f"{W}qFormat")
    ppr = ensure_child(style, "pPr")
    remove_children(ppr, ("spacing", "keepNext", "keepLines", "outlineLvl"))
    if keep_next:
        etree.SubElement(ppr, f"{W}keepNext")
    etree.SubElement(ppr, f"{W}keepLines")
    spacing = etree.SubElement(ppr, f"{W}spacing")
    spacing.set(f"{W}before", str(before))
    spacing.set(f"{W}after", str(after))
    spacing.set(f"{W}line", "240")
    spacing.set(f"{W}lineRule", "auto")
    if outline_level is not None:
        outline = etree.SubElement(ppr, f"{W}outlineLvl")
        outline.set(f"{W}val", str(outline_level))
    rpr = ensure_child(style, "rPr")
    remove_children(rpr, ("rFonts", "b", "bCs", "i", "iCs", "color", "sz", "szCs"))
    fonts = etree.SubElement(rpr, f"{W}rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        fonts.set(f"{W}{attr}", "Times New Roman")
    if bold:
        etree.SubElement(rpr, f"{W}b")
        etree.SubElement(rpr, f"{W}bCs")
    color = etree.SubElement(rpr, f"{W}color")
    color.set(f"{W}val", "000000")
    for tag in ("sz", "szCs"):
        node = etree.SubElement(rpr, f"{W}{tag}")
        node.set(f"{W}val", str(size_half_points))


def patch_styles(styles_xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(styles_xml, parser)
    configure_style(
        root,
        "Heading1",
        name="heading 1",
        size_half_points=24,
        bold=True,
        before=240,
        after=120,
        outline_level=0,
    )
    configure_style(
        root,
        "Heading2",
        name="heading 2",
        size_half_points=24,
        bold=True,
        before=180,
        after=60,
        outline_level=1,
    )
    configure_style(
        root,
        "Caption",
        name="Caption",
        size_half_points=20,
        bold=False,
        before=60,
        after=120,
        outline_level=None,
        keep_next=False,
    )
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def patch_core(core_xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(core_xml, parser)
    title = root.find(f"{{{DC_NS}}}title")
    if title is None:
        title = etree.SubElement(root, f"{{{DC_NS}}}title")
    title.text = TITLE
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def patch_document(document_xml: bytes) -> tuple[bytes, dict[str, int]]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(document_xml, parser)
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)

    title = paragraphs[0]
    clear_paragraph_content(title)
    set_paragraph_style(title, "Normal")
    append_run(title, f"{TITLE_LINE_1} ", size_half_points=28, bold=True, line_break_after=True)
    append_run(title, TITLE_LINE_2, size_half_points=28, bold=True)
    set_alignment(title, "center")
    set_paragraph_spacing(title, before=0, after=160, line=280)
    set_keep_next(title, True)

    correspondence = find_exact(paragraphs, "* Correspondence: tanawat.run@ku.th")
    rebuild_plain_paragraph(
        correspondence,
        CORRESPONDENCE,
        style_id="Normal",
        size_half_points=20,
    )
    set_alignment(correspondence, "center")
    set_paragraph_spacing(correspondence, after=160, line=240)

    heading_count = 0
    for old_text, (new_text, style_id) in HEADING_MAP.items():
        paragraph = find_exact(paragraphs, old_text)
        configure_heading(paragraph, style_id, new_text)
        heading_count += 1

    # Abstract is a front-matter label rather than a numbered section, but it
    # receives consistent heading typography without entering the outline.
    abstract = find_exact(paragraphs, "Abstract")
    rebuild_plain_paragraph(abstract, "Abstract", style_id="Normal", size_half_points=24, bold=True)
    set_paragraph_spacing(abstract, before=120, after=60, line=240)
    set_keep_next(abstract, True)

    split_inline_section(find_prefix(paragraphs, "Limitations:"), "Limitations")
    split_inline_section(find_prefix(paragraphs, "Future work:"), "Future Work")

    # Refresh paragraph list after inserting the two subsection headings.
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)

    caption_count = 0
    note_count = 0
    for paragraph in paragraphs:
        text = paragraph_text(paragraph).strip()
        match = re.match(r"^((Figure|Table)\s+\d+[A-Z]?\.)\s*(.*)$", text, flags=re.DOTALL)
        if match:
            format_caption(paragraph, match.group(1), match.group(3), match.group(2))
            caption_count += 1
        elif text.startswith("Note:"):
            format_note(paragraph, text)
            note_count += 1

    notation_replacements = 0
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
    for prefix, tokens in NOTATION_JOBS:
        paragraph = find_prefix(paragraphs, prefix)
        notation_replacements += format_tokens_in_paragraph(paragraph, tokens)

    output = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
    return output, {
        "headings_formatted": heading_count + 2,
        "captions_formatted": caption_count,
        "table_notes_formatted": note_count,
        "inline_notation_replacements": notation_replacements,
    }


def write_docx() -> dict[str, int]:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    with ZipFile(SOURCE, "r") as source_zip:
        patched_document, stats = patch_document(source_zip.read("word/document.xml"))
        patched_styles = patch_styles(source_zip.read("word/styles.xml"))
        patched_core = patch_core(source_zip.read("docProps/core.xml"))
        with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as output_zip:
            for entry in source_zip.infolist():
                if entry.filename == "word/document.xml":
                    data = patched_document
                elif entry.filename == "word/styles.xml":
                    data = patched_styles
                elif entry.filename == "docProps/core.xml":
                    data = patched_core
                else:
                    data = source_zip.read(entry.filename)
                output_zip.writestr(entry, data)
    return stats


def main() -> None:
    stats = write_docx()
    print(OUTPUT)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
