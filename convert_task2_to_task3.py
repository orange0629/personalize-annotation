#!/usr/bin/env python3
"""
One-time conversion: generate *_task3.jsonl from *_task2.jsonl.

Mapping:
  necessary_explicit | necessary_implicit | optional  →  relevant: true
  unnecessary | forbidden | none | (missing)          →  relevant: false (skipped)

Only attributes with a non-empty, non-"none" rating in Task 2 are included.
Existing *_task3.jsonl files are NOT overwritten (use --force to overwrite).

Usage:
    python3 convert_task2_to_task3.py
    python3 convert_task2_to_task3.py --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ANNOT_DIR = Path(__file__).parent / "data" / "annotations"
YES_RATINGS = {"necessary_explicit", "necessary_implicit", "optional"}


def convert(src: Path, dst: Path) -> int:
    """Convert one task2 file to task3 format. Returns number of records written."""
    records = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("index")
            if idx is None:
                continue
            judgments = []
            for j in rec.get("relevance_judgments", []):
                attr   = j.get("attribute", "")
                rating = j.get("rating", "")
                if not attr or not rating or rating in ("none", ""):
                    continue
                judgments.append({
                    "attribute": attr,
                    "relevant":  rating in YES_RATINGS,
                })
            records[idx] = {
                "task":                "3",
                "index":               idx,
                "user_id":             rec.get("user_id", ""),
                "prompt_index":        rec.get("prompt_index", -1),
                "relevance_judgments": judgments,
                "note":                rec.get("note", "") + " [converted from task2]",
                "flagged":             rec.get("flagged", False),
                "annotator":           rec.get("annotator", ""),
                "timestamp":           rec.get("timestamp", ""),
            }

    with open(dst, "w", encoding="utf-8") as f:
        for rec in sorted(records.values(), key=lambda r: r["index"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing _task3.jsonl files.")
    args = parser.parse_args()

    src_files = sorted(ANNOT_DIR.glob("*_task2.jsonl"))
    if not src_files:
        print("No *_task2.jsonl files found in", ANNOT_DIR)
        return

    for src in src_files:
        name = src.name[: -len("_task2.jsonl")]
        dst  = ANNOT_DIR / f"{name}_task3.jsonl"

        if dst.exists() and not args.force:
            print(f"  SKIP  {dst.name}  (already exists; use --force to overwrite)")
            continue

        n = convert(src, dst)
        print(f"  WROTE {dst.name}  ({n} records)")


if __name__ == "__main__":
    main()
