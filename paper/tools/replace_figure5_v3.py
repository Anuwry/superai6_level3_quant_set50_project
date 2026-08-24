from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(r"D:\SET50_direction_prediction_paper")
SOURCE = ROOT / "paper" / "SET_direction_manuscript_journal_formatted_v2.docx"
OUTPUT = ROOT / "paper" / "SET_direction_manuscript_journal_formatted_v3.docx"
NEW_FIGURE = ROOT / "paper" / "assets" / "figure5_separated_audits_v2.png"
MEDIA_MEMBER = "word/media/image5.png"


def main():
    for path in (SOURCE, NEW_FIGURE):
        if not path.exists():
            raise FileNotFoundError(path)

    figure_bytes = NEW_FIGURE.read_bytes()

    with ZipFile(SOURCE, "r") as source_zip:
        names = source_zip.namelist()
        if MEDIA_MEMBER not in names:
            raise RuntimeError(f"Missing expected manuscript member: {MEDIA_MEMBER}")

        with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as output_zip:
            for item in source_zip.infolist():
                payload = figure_bytes if item.filename == MEDIA_MEMBER else source_zip.read(item.filename)
                output_zip.writestr(item, payload)

    with ZipFile(OUTPUT, "r") as check_zip:
        embedded = check_zip.read(MEDIA_MEMBER)
        if embedded != figure_bytes:
            raise RuntimeError("Figure 5 replacement verification failed")

    print(OUTPUT)
    print(f"Replaced {MEDIA_MEMBER} with {NEW_FIGURE.name} ({len(figure_bytes):,} bytes)")


if __name__ == "__main__":
    main()
