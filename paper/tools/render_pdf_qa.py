from pathlib import Path

import fitz
from PIL import Image, ImageDraw


PDF = Path(r"D:\SET50_direction_prediction_paper\paper\qa\newest_original_manuscript_llm_benchmark_scope_corrected.pdf")
OUT = Path(r"D:\SET50_direction_prediction_paper\paper\qa\render_llm_benchmark_scope_corrected")
OUT.mkdir(parents=True, exist_ok=True)

document = fitz.open(PDF)
page_files: list[Path] = []
for index, page in enumerate(document):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False)
    target = OUT / f"page_{index + 1:02d}.png"
    pixmap.save(target)
    page_files.append(target)

thumb_width = 350
thumbs: list[Image.Image] = []
for page_number, page_file in enumerate(page_files, start=1):
    image = Image.open(page_file).convert("RGB")
    height = int(image.height * thumb_width / image.width)
    image = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (thumb_width + 20, height + 42), "white")
    canvas.paste(image, (10, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), f"Page {page_number}", fill="black")
    thumbs.append(canvas)

per_sheet = 6
for sheet_index in range(0, len(thumbs), per_sheet):
    group = thumbs[sheet_index : sheet_index + per_sheet]
    rows = 2
    cols = 3
    cell_width = max(image.width for image in group)
    cell_height = max(image.height for image in group)
    sheet = Image.new("RGB", (cols * cell_width, rows * cell_height), "#d8d8d8")
    for item_index, image in enumerate(group):
        x = (item_index % cols) * cell_width
        y = (item_index // cols) * cell_height
        sheet.paste(image, (x, y))
    sheet.save(OUT / f"contact_{sheet_index // per_sheet + 1:02d}.png", quality=95)

print(f"pages={len(page_files)}")
print(f"contact_sheets={(len(thumbs) + per_sheet - 1) // per_sheet}")
