from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "output" / "pdf" / "first_pipeline_revised.pdf"

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
INK = HexColor("#17212B")
MUTED = HexColor("#52606D")
PANEL = HexColor("#F7F9FB")
PANEL_BORDER = HexColor("#D7DEE5")
GREEN_FILL = HexColor("#E9F5EF")
GREEN_BORDER = HexColor("#2F7D5B")
AMBER_FILL = HexColor("#FFF3D6")
AMBER_BORDER = HexColor("#B7791F")
GRAY_FILL = HexColor("#F0F2F4")
GRAY_BORDER = HexColor("#7A8793")
BLUE = HexColor("#245B8A")


def wrapped_lines(
    text: str,
    *,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def draw_text_block(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    width: float,
    lines: list[str],
    font_size: float = 7.3,
    leading: float = 9.4,
) -> None:
    cursor = y
    for source in lines:
        bullet = source.startswith("- ")
        text = source[2:] if bullet else source
        wrapped = wrapped_lines(
            text,
            font_name="Helvetica",
            font_size=font_size,
            max_width=width - (9 if bullet else 0),
        )
        for index, line in enumerate(wrapped):
            canvas.setFont("Helvetica", font_size)
            canvas.setFillColor(INK)
            if bullet and index == 0:
                canvas.circle(x + 2.0, cursor + 2.2, 1.1, fill=1, stroke=0)
            canvas.drawString(x + (8 if bullet else 0), cursor, line)
            cursor -= leading
        cursor -= 1.8


def draw_status_pill(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    label: str,
    fill_color: HexColor,
    border_color: HexColor,
) -> None:
    width = stringWidth(label, "Helvetica-Bold", 6.5) + 12
    canvas.setFillColor(fill_color)
    canvas.setStrokeColor(border_color)
    canvas.roundRect(x, y, width, 13, 6, fill=1, stroke=1)
    canvas.setFillColor(border_color)
    canvas.setFont("Helvetica-Bold", 6.5)
    canvas.drawCentredString(x + width / 2, y + 4.0, label)


def draw_box(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    fill_color: HexColor,
    border_color: HexColor,
    status: str,
) -> None:
    canvas.setFillColor(fill_color)
    canvas.setStrokeColor(border_color)
    canvas.setLineWidth(1.1)
    canvas.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    canvas.setFillColor(border_color)
    canvas.setFont("Helvetica-Bold", 8.2)
    canvas.drawString(x + 8, y + height - 16, title)
    draw_status_pill(
        canvas,
        x=x + width - 8 - (stringWidth(status, "Helvetica-Bold", 6.5) + 12),
        y=y + height - 20,
        label=status,
        fill_color=white,
        border_color=border_color,
    )
    canvas.setStrokeColor(border_color)
    canvas.setLineWidth(0.45)
    canvas.line(x + 8, y + height - 25, x + width - 8, y + height - 25)
    draw_text_block(
        canvas,
        x=x + 9,
        y=y + height - 38,
        width=width - 18,
        lines=lines,
    )


def draw_phase_panel(
    canvas: Canvas,
    *,
    x: float,
    width: float,
    number: int,
    title: str,
) -> None:
    canvas.setFillColor(PANEL)
    canvas.setStrokeColor(PANEL_BORDER)
    canvas.roundRect(x, 93, width, 417, 9, fill=1, stroke=1)
    canvas.setFillColor(BLUE)
    canvas.setFont("Helvetica-Bold", 7.0)
    canvas.drawString(x + 8, 520, f"PHASE {number}")
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 9.0)
    canvas.drawString(x + 8, 507, title)


def draw_flow_arrow(canvas: Canvas, *, x1: float, x2: float, y: float) -> None:
    canvas.setStrokeColor(BLUE)
    canvas.setFillColor(BLUE)
    canvas.setLineWidth(1.4)
    canvas.line(x1, y, x2 - 6, y)
    canvas.line(x2 - 6, y, x2 - 11, y + 3.5)
    canvas.line(x2 - 6, y, x2 - 11, y - 3.5)


def generate() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUTPUT_PATH), pagesize=landscape(A4))
    canvas.setTitle("Registered SET50 Pipeline Before SHAP")
    canvas.setAuthor("SET50 Direction Prediction Project")

    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(18, PAGE_HEIGHT - 28, "Leakage-Aware SET50 Forecasting Pipeline")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8.6)
    canvas.drawString(
        18,
        PAGE_HEIGHT - 43,
        "Registered state before SHAP execution - protocol freeze 2026-07-31",
    )

    draw_status_pill(
        canvas,
        x=596,
        y=PAGE_HEIGHT - 43,
        label="COMPLETED",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
    )
    draw_status_pill(
        canvas,
        x=674,
        y=PAGE_HEIGHT - 43,
        label="REGISTERED NEXT",
        fill_color=AMBER_FILL,
        border_color=AMBER_BORDER,
    )
    draw_status_pill(
        canvas,
        x=778,
        y=PAGE_HEIGHT - 43,
        label="NOT CLAIMED",
        fill_color=GRAY_FILL,
        border_color=GRAY_BORDER,
    )

    xs = (15, 178, 341, 514, 677)
    widths = (153, 153, 163, 153, 150)
    phase_titles = (
        "DATA & FEATURES",
        "CAUSAL TREATMENTS",
        "REGIME & SHAP",
        "MODELS & ABLATION",
        "EVALUATION",
    )
    for index, (x, width, title) in enumerate(
        zip(xs, widths, phase_titles, strict=True),
        start=1,
    ):
        draw_phase_panel(
            canvas,
            x=x,
            width=width,
            number=index,
            title=title,
        )
    for left, right, width in zip(xs[:-1], xs[1:], widths[:-1], strict=True):
        draw_flow_arrow(canvas, x1=left + width + 1, x2=right - 1, y=526)

    draw_box(
        canvas,
        x=24,
        y=350,
        width=135,
        height=137,
        title="SET50 MARKET",
        status="DONE",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
        lines=[
            "- 2012-05-03 to 2025-12-18",
            "- 2025 is explicitly partial",
            "- Daily / weekly / monthly",
            "- 116 causal TA features",
            "- Target: next closing level",
        ],
    )
    draw_box(
        canvas,
        x=24,
        y=174,
        width=135,
        height=153,
        title="FINANCIAL NEWS",
        status="DONE",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
        lines=[
            "- OOS daily features: 2019-2025",
            "- Local relevance + sentiment",
            "- Terra debate retained as ablation",
            "- Eight locked daily features",
            "- Source shift is reported",
        ],
    )

    draw_box(
        canvas,
        x=187,
        y=350,
        width=135,
        height=137,
        title="TRACK A",
        status="DONE",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
        lines=[
            "- Full TA control",
            "- +6 rolling VMD features",
            "- 60-day past-only extraction",
            "- Paired dates / models / seeds",
            "- Mixed effects reported",
        ],
    )
    draw_box(
        canvas,
        x=187,
        y=174,
        width=135,
        height=153,
        title="TRACK B",
        status="DONE",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
        lines=[
            "- Technical vs +News",
            "- Strict next-trading-day mapping",
            "- Identical eligible dates",
            "- LLM vs local benchmark",
            "- News stays a separate treatment",
        ],
    )

    draw_box(
        canvas,
        x=350,
        y=350,
        width=145,
        height=137,
        title="DAILY REGIME v2",
        status="DONE",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
        lines=[
            "- Current regime R(t)",
            "- Bull / Sideway / Bear",
            "- 1,3,5,10,20,60-day trend + ADX",
            "- Threshold fit on training only",
            "- HMM v1 is an ablation",
        ],
    )
    draw_box(
        canvas,
        x=350,
        y=174,
        width=145,
        height=153,
        title="CONSENSUS SHAP",
        status="NEXT",
        fill_color=AMBER_FILL,
        border_color=AMBER_BORDER,
        lines=[
            "- All five locked model-window pairs",
            "- Temporal train / rank / validate",
            "- k: 10,20,30,40,60,80,100,122",
            "- One-SE + metric guardrails",
            "- No outer-test tuning",
        ],
    )

    draw_box(
        canvas,
        x=523,
        y=318,
        width=135,
        height=169,
        title="FIVE REGRESSORS",
        status="LOCKED",
        fill_color=GREEN_FILL,
        border_color=GREEN_BORDER,
        lines=[
            "- LSTM W5",
            "- CNN W5",
            "- LSTM-CNN W20",
            "- LSTM-Attention W20",
            "- LSTM-CNN-Attention W20",
            "- MSE next-close training",
        ],
    )
    draw_box(
        canvas,
        x=523,
        y=174,
        width=135,
        height=121,
        title="FOUR ARMS",
        status="NEXT",
        fill_color=AMBER_FILL,
        border_color=AMBER_BORDER,
        lines=[
            "- Global-All",
            "- Global-SHAP",
            "- Regime-All",
            "- Regime-SHAP",
        ],
    )

    draw_box(
        canvas,
        x=686,
        y=318,
        width=132,
        height=169,
        title="FORECAST TEST",
        status="NEXT",
        fill_color=AMBER_FILL,
        border_color=AMBER_BORDER,
        lines=[
            "- DA primary",
            "- Balanced Accuracy + MCC",
            "- RMSE + MAE",
            "- Seeds averaged within fold",
            "- Four temporal fold units",
            "- Paired uncertainty + Holm",
        ],
    )
    draw_box(
        canvas,
        x=686,
        y=174,
        width=132,
        height=121,
        title="SCOPE LIMITS",
        status="NO CLAIM",
        fill_color=GRAY_FILL,
        border_color=GRAY_BORDER,
        lines=[
            "- Track C is post-hoc robustness",
            "- No profitability claim",
            "- No live execution claim",
            "- Backtest needs a tradeable vehicle",
        ],
    )

    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 8.0)
    canvas.drawString(18, 72, "Controlling documents")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.4)
    canvas.drawString(
        18,
        59,
        "test/pre_shap_experiment_manifest.md  |  "
        "test/pre_shap_reviewer_risk_resolution.md  |  "
        "pipeline/pipeline8.md",
    )
    canvas.drawRightString(
        PAGE_WIDTH - 18,
        30,
        "Forecasting evidence only; economic and live-trading claims are excluded.",
    )
    canvas.setStrokeColor(PANEL_BORDER)
    canvas.line(18, 47, PAGE_WIDTH - 18, 47)

    canvas.showPage()
    canvas.save()
    return OUTPUT_PATH


if __name__ == "__main__":
    print(generate())
