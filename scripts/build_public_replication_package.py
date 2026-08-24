from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.public_replication_package import (
    build_public_replication_package,
    collect_default_public_sources,
)

DEFAULT_DESTINATION = PROJECT_ROOT / "release" / "public_replication_package_v3"


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    sources = collect_default_public_sources(PROJECT_ROOT)
    result = build_public_replication_package(
        PROJECT_ROOT,
        args.destination,
        sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
