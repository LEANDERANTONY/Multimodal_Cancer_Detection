from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Find notebook cell ids containing a given substring")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("needle", type=str)
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    hits: list[tuple[int, str, str]] = []
    for idx, cell in enumerate(cells, start=1):
        src = normalize_source(cell)
        if args.needle in src:
            cell_id = cell.get("id") or cell.get("metadata", {}).get("id") or ""
            first_line = ""
            if isinstance(cell.get("source"), list) and cell["source"]:
                first_line = str(cell["source"][0]).strip()
            hits.append((idx, cell_id, first_line))

    print(f"Needle: {args.needle!r}")
    print(f"Matches: {len(hits)}")
    for idx, cell_id, first_line in hits:
        print(f"- cell#{idx} id={cell_id} first={first_line!r}")


if __name__ == "__main__":
    main()
