from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.manuscript_artifacts import build_manuscript_artifacts

DEFAULT_DESTINATION = PROJECT_ROOT / "outputs" / "manuscript_tables_v1"


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    result = build_manuscript_artifacts(PROJECT_ROOT, args.destination)
    print(f"Built {len(result['generated'])} manuscript artifacts at {args.destination}")
    return result

if __name__ == "__main__":
    main()
