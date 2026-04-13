#!/usr/bin/env python3
"""
Build byte-offset indexes for large JSONL files used by the annotation tool,
and optionally extract the needed records into small portable files for deployment.

Indexes are stored in SQLite databases in data/indexes/.
Extracted records (AWS-ready, no large files needed) go to data/extracted/.

Run this once before starting the annotation tool:
    python3 build_index.py                  # build indexes only
    python3 build_index.py --extract        # build indexes + extract for deployment
    python3 build_index.py --extract-only   # extract from existing indexes (skip rebuild)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

DATA_DIR    = Path(__file__).parent / "data" / "indexes"
EXTRACT_DIR = Path(__file__).parent / "data" / "extracted"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TASK1_ITEMS = EXTRACT_DIR / "task1_items.jsonl"
TASK2_ITEMS = EXTRACT_DIR / "task2_items.jsonl"

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/shared/0/projects/research-jam-summer-2024/tmp")
# ATTRS_JSONL       = BASE / "attrs_per_conversation_final_merged_agglo_0.7_v2.jsonl"
ATTRS_JSONL       = "/home/leczhang/research/llm-personalization/data/attrs_per_conversation_final_merged_agglo_0.7_v2_sampled_1000.jsonl"
CHECKLIST_JSONL   = BASE / "checklist_final_merged_agglo_0.7_v2_LIMA.jsonl"
CONVS_JSONL       = BASE / "wildchat_long_conversations_15_assistant_like_confidence_final_english_v1.jsonl"

ATTRS_INDEX_DB    = DATA_DIR / "attrs_index.db"
CHECKLIST_INDEX_DB = DATA_DIR / "checklist_index.db"
CONVS_INDEX_DB    = DATA_DIR / "convs_index.db"


def _progress(msg: str) -> None:
    print(msg, flush=True)


def build_attrs_index(force: bool = False) -> None:
    """Index attrs JSONL: user_id -> byte_offset, line_index."""
    if ATTRS_INDEX_DB.exists() and not force:
        _progress(f"[attrs] Index already exists at {ATTRS_INDEX_DB}. Use --force to rebuild.")
        return

    _progress(f"[attrs] Building index for {ATTRS_JSONL} ...")
    conn = sqlite3.connect(str(ATTRS_INDEX_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS attrs_index;
        CREATE TABLE attrs_index (
            line_index INTEGER PRIMARY KEY,
            user_id    TEXT NOT NULL,
            byte_offset INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_attrs_user ON attrs_index(user_id);
    """)

    count = 0
    with open(ATTRS_JSONL, "rb") as f:
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                user_id = d.get("user_id", "")
                if user_id:
                    cur.execute(
                        "INSERT INTO attrs_index (line_index, user_id, byte_offset) VALUES (?, ?, ?)",
                        (count, user_id, offset),
                    )
                    count += 1
                    if count % 5000 == 0:
                        conn.commit()
                        _progress(f"  [attrs] {count} records indexed...")
            except (json.JSONDecodeError, Exception):
                continue

    conn.commit()
    conn.close()
    _progress(f"[attrs] Done. {count} records indexed.")


def build_checklist_index(force: bool = False) -> None:
    """
    Index checklist JSONL: only records with items.
    Stores sequential annotation_index -> byte_offset.
    Also stores user_id and prompt_index for reference.
    """
    if CHECKLIST_INDEX_DB.exists() and not force:
        _progress(f"[checklist] Index already exists at {CHECKLIST_INDEX_DB}. Use --force to rebuild.")
        return

    _progress(f"[checklist] Building index for {CHECKLIST_JSONL} ...")
    _progress("  (This may take a while for the 70GB file...)")
    conn = sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS checklist_index;
        CREATE TABLE checklist_index (
            annotation_index INTEGER PRIMARY KEY,
            user_id          TEXT NOT NULL,
            prompt_index     INTEGER NOT NULL,
            byte_offset      INTEGER NOT NULL,
            prompt_snippet   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cl_user ON checklist_index(user_id);
    """)

    count = 0
    total_lines = 0
    with open(CHECKLIST_JSONL, "rb") as f:
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            raw = raw.strip()
            if not raw:
                continue
            total_lines += 1
            try:
                d = json.loads(raw)
                items = d.get("items")
                if not items:
                    continue
                user_id = d.get("user_id", "")
                prompt_index = d.get("prompt_index", -1)
                prompt_snippet = (d.get("prompt_text") or "")[:120]
                cur.execute(
                    "INSERT INTO checklist_index (annotation_index, user_id, prompt_index, byte_offset, prompt_snippet) VALUES (?, ?, ?, ?, ?)",
                    (count, user_id, prompt_index, offset, prompt_snippet),
                )
                count += 1
                if count % 1000 == 0:
                    conn.commit()
                    _progress(f"  [checklist] {count} annotation records (from {total_lines} total lines)...")
            except (json.JSONDecodeError, Exception):
                continue

    conn.commit()
    conn.close()
    _progress(f"[checklist] Done. {count} annotation records indexed (from {total_lines} total lines).")


def build_convs_index(force: bool = False) -> None:
    """Index conversations JSONL: hashed_ip -> list of (byte_offset, conversation_hash)."""
    if CONVS_INDEX_DB.exists() and not force:
        _progress(f"[convs] Index already exists at {CONVS_INDEX_DB}. Use --force to rebuild.")
        return

    _progress(f"[convs] Building index for {CONVS_JSONL} ...")
    conn = sqlite3.connect(str(CONVS_INDEX_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS convs_index;
        CREATE TABLE convs_index (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            hashed_ip         TEXT NOT NULL,
            conversation_hash TEXT,
            byte_offset       INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conv_ip ON convs_index(hashed_ip);
    """)

    count = 0
    filtered = 0
    with open(CONVS_JSONL, "rb") as f:
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                # Keep only conversations with at most 1 non-English turn (skip if field absent)
                is_english = d.get("is_english")
                if is_english is not None and sum(1 for v in is_english if not v) > 1:
                    filtered += 1
                    continue
                hashed_ip = d.get("hashed_ip", "")
                conv_hash_raw = d.get("conversation_hash", "")
                # conversation_hash may be a list or a string
                if isinstance(conv_hash_raw, list):
                    conv_hash = json.dumps(conv_hash_raw)
                else:
                    conv_hash = str(conv_hash_raw) if conv_hash_raw else ""
                if hashed_ip:
                    cur.execute(
                        "INSERT INTO convs_index (hashed_ip, conversation_hash, byte_offset) VALUES (?, ?, ?)",
                        (hashed_ip, conv_hash, offset),
                    )
                    count += 1
                    if count % 5000 == 0:
                        conn.commit()
                        _progress(f"  [convs] {count} records indexed...")
            except (json.JSONDecodeError, Exception):
                continue

    conn.commit()
    conn.close()
    _progress(f"[convs] Done. {count} records indexed, {filtered} filtered (non-English).")


def build_checklist_sample(
    n_per_user: int = 10,
    force: bool = False,
    skip_prompt_indexes: Optional[set] = None,
) -> None:
    """
    Create checklist_sample table inside the checklist index DB.
    Randomly selects up to n_per_user records per user_id, assigns
    a sequential sample_index used by the app for navigation.
    Requires the checklist_index table to exist first.

    skip_prompt_indexes: set of prompt_index integers to exclude from the sample.
    """
    if not CHECKLIST_INDEX_DB.exists():
        _progress("[sample] Checklist index DB not found. Run build_checklist_index first.")
        return

    conn = sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    # Check if already built
    existing = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='checklist_sample'"
    ).fetchone()
    if existing and not force:
        count = cur.execute("SELECT COUNT(*) FROM checklist_sample").fetchone()[0]
        _progress(f"[sample] Already exists ({count} records). Use --force to rebuild.")
        conn.close()
        return

    skip_set = set(skip_prompt_indexes) if skip_prompt_indexes else set()
    if skip_set:
        _progress(f"[sample] Skipping prompt_indexes: {sorted(skip_set)}")

    # Build a WHERE clause to exclude unwanted prompt_indexes
    skip_clause = ""
    if skip_set:
        placeholders = ",".join("?" * len(skip_set))
        skip_clause = f"WHERE prompt_index NOT IN ({placeholders})"

    _progress(f"[sample] Sampling up to {n_per_user} users per prompt from checklist_index...")
    cur.executescript("""
        DROP TABLE IF EXISTS checklist_sample;
        CREATE TABLE checklist_sample (
            sample_index     INTEGER PRIMARY KEY,
            annotation_index INTEGER NOT NULL
        );
    """)

    query = f"""
        SELECT annotation_index
        FROM (
            SELECT annotation_index,
                   ROW_NUMBER() OVER (PARTITION BY prompt_index ORDER BY RANDOM()) AS rn
            FROM checklist_index
            {skip_clause}
        )
        WHERE rn <= {n_per_user}
        ORDER BY annotation_index
    """
    rows = conn.execute(query, sorted(skip_set)).fetchall()

    for sample_idx, (ann_idx,) in enumerate(rows):
        cur.execute(
            "INSERT INTO checklist_sample (sample_index, annotation_index) VALUES (?, ?)",
            (sample_idx, ann_idx),
        )

    conn.commit()
    conn.close()
    _progress(f"[sample] Done. {len(rows)} records sampled ({n_per_user} users per prompt, {len(skip_set)} prompt indexes skipped).")


def _strip_embeddings(obj):
    if isinstance(obj, dict):
        return {k: _strip_embeddings(v) for k, v in obj.items() if k != "embedding"}
    if isinstance(obj, list):
        return [_strip_embeddings(x) for x in obj]
    return obj


def extract_data(force: bool = False) -> None:
    """
    Extract the sampled records from the large source files into small portable JSONLs.
    After running this, the app can run on AWS with only data/extracted/ — no large files needed.

    Outputs:
      data/extracted/checklist_sample.jsonl  — 9,961 checklist records (embeddings stripped)
      data/extracted/attrs.jsonl             — all attrs records (embeddings stripped)
      data/extracted/convs.jsonl             — conversations for users present in attrs
    """
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    def _needs_write(path: Path) -> bool:
        """Return True if the file is missing, empty, or force-overwrite is set."""
        return force or not path.exists() or path.stat().st_size == 0

    # ── checklist ──────────────────────────────────────────────────────────────
    out = EXTRACT_DIR / "checklist_sample.jsonl"
    if not _needs_write(out):
        _progress(f"[extract] {out.name} already exists, skipping.")
    else:
        _progress(f"[extract] Extracting checklist sample -> {out}")
        with sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=30) as conn:
            rows = conn.execute(
                """SELECT cs.sample_index, ci.byte_offset
                   FROM checklist_sample cs
                   JOIN checklist_index ci ON cs.annotation_index = ci.annotation_index
                   ORDER BY cs.sample_index"""
            ).fetchall()
        n = len(rows)
        if n == 0:
            raise RuntimeError("[extract] checklist_sample JOIN returned 0 rows — checklist_sample may be stale. Run with --force.")
        tmp = out.with_suffix(".jsonl.tmp")
        with open(CHECKLIST_JSONL, "rb") as fh, open(tmp, "w", encoding="utf-8") as out_f:
            for sample_index, offset in rows:
                fh.seek(offset)
                rec = _strip_embeddings(json.loads(fh.readline()))
                rec["sample_index"] = sample_index
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (sample_index + 1) % 1000 == 0:
                    _progress(f"  [extract/checklist] {sample_index + 1}/{n}")
        tmp.replace(out)
        _progress(f"[extract] checklist done: {n} records, {out.stat().st_size // 1024 // 1024} MB")

    # ── attrs ──────────────────────────────────────────────────────────────────
    out = EXTRACT_DIR / "attrs.jsonl"
    if not _needs_write(out):
        _progress(f"[extract] {out.name} already exists, skipping.")
    else:
        _progress(f"[extract] Extracting attrs -> {out}")
        with sqlite3.connect(str(ATTRS_INDEX_DB), timeout=30) as conn:
            rows = conn.execute(
                "SELECT line_index, byte_offset FROM attrs_index ORDER BY line_index"
            ).fetchall()
        n = len(rows)
        if n == 0:
            raise RuntimeError("[extract] attrs_index returned 0 rows.")
        tmp = out.with_suffix(".jsonl.tmp")
        with open(ATTRS_JSONL, "rb") as fh, open(tmp, "w", encoding="utf-8") as out_f:
            for line_index, offset in rows:
                fh.seek(offset)
                rec = _strip_embeddings(json.loads(fh.readline()))
                rec["line_index"] = line_index
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (line_index + 1) % 2000 == 0:
                    _progress(f"  [extract/attrs] {line_index + 1}/{n}")
        tmp.replace(out)
        _progress(f"[extract] attrs done: {n} records, {out.stat().st_size // 1024 // 1024} MB")

    # ── convs ──────────────────────────────────────────────────────────────────
    out = EXTRACT_DIR / "convs.jsonl"
    if not _needs_write(out):
        _progress(f"[extract] {out.name} already exists, skipping.")
    else:
        _progress(f"[extract] Extracting conversations -> {out}")
        with sqlite3.connect(str(ATTRS_INDEX_DB), timeout=30) as conn:
            user_ids = set(r[0] for r in conn.execute("SELECT user_id FROM attrs_index").fetchall())
        with sqlite3.connect(str(CONVS_INDEX_DB), timeout=30) as conn:
            all_rows = conn.execute(
                "SELECT hashed_ip, byte_offset FROM convs_index ORDER BY id"
            ).fetchall()
        relevant = [(ip, off) for ip, off in all_rows if ip in user_ids]
        n = len(relevant)
        _progress(f"  [extract/convs] {n} records for {len(user_ids)} users")
        if n == 0:
            raise RuntimeError("[extract] convs: no matching users found.")
        tmp = out.with_suffix(".jsonl.tmp")
        with open(CONVS_JSONL, "rb") as fh, open(tmp, "w", encoding="utf-8") as out_f:
            for i, (hashed_ip, offset) in enumerate(relevant):
                fh.seek(offset)
                out_f.write(fh.readline().decode("utf-8", errors="replace"))
                if (i + 1) % 2000 == 0:
                    _progress(f"  [extract/convs] {i + 1}/{n}")
        tmp.replace(out)
        _progress(f"[extract] convs done: {n} records, {out.stat().st_size // 1024 // 1024} MB")

    _progress(
        f"\n[extract] All done. Copy annotation_tool/data/extracted/ to AWS — "
        f"no large source files needed."
    )


TASK1_WILDCHAT_N = 50   # first N Wildchat users (after english filtering)
TASK1_EXTRA_N    = 10   # random users per extra source
TASK1_RANDOM_SEED = 42

# Extra sources: (source_name, attrs_jsonl, personas_jsonl)
EXTRA_SOURCES = [
    ("cupid",
     Path("/home/leczhang/research/llm-personalization/data/cupid_attrs_merged_agglo_0.7.jsonl"),
     Path("/home/leczhang/research/llm-personalization/data/cupid_100personas.jsonl")),
#     ("personamem",
#     Path("/home/leczhang/research/llm-personalization/data/personamem_attrs_merged_agglo_0.7.jsonl"),
#     Path("/home/leczhang/research/llm-personalization/data/personamem_100personas.jsonl")),
    ("prefeval",
     Path("/home/leczhang/research/llm-personalization/data/prefeval_attrs_merged_agglo_0.7.jsonl"),
     Path("/home/leczhang/research/llm-personalization/data/prefeval_100personas.jsonl")),
]


def _load_extra_source(source_name: str, attrs_path: Path, personas_path: Path, n: int, rng) -> list:
    """Load n random users from an extra source, returning task1-format items (without item_index)."""
    # Load attrs: user_id -> merged_attributes
    attrs_map = {}
    with open(attrs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _strip_embeddings(json.loads(line))
            attrs_map[str(rec["user_id"])] = rec.get("merged_attributes", [])

    # Load personas: hashed_ip (== user_id) -> list of conversation records
    convs_map = {}
    with open(personas_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("hashed_ip", ""))
            if uid:
                convs_map.setdefault(uid, []).append(rec)

    # Only keep users present in both files
    valid_users = [uid for uid in attrs_map if uid in convs_map]
    sampled = rng.sample(valid_users, min(n, len(valid_users)))

    items = []
    for uid in sampled:
        items.append({
            "user_id": uid,
            "source": source_name,
            "merged_attributes": attrs_map[uid],
            "conversations": convs_map[uid],
        })
    return items


def build_task1_items(force: bool = False) -> None:
    """
    Build task1_items.jsonl combining:
      - First TASK1_WILDCHAT_N Wildchat users (after English filtering)
      - TASK1_EXTRA_N random users each from cupid, personamem, prefeval
    Items are shuffled with TASK1_RANDOM_SEED and tagged with a 'source' field.
    """
    import random
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if TASK1_ITEMS.exists() and not force:
        _progress(f"[task1] {TASK1_ITEMS.name} already exists. Use --force to rebuild.")
        return
    if not ATTRS_INDEX_DB.exists():
        _progress("[task1] attrs index not found — run without --extract-only to build it first.")
        return
    if not CONVS_INDEX_DB.exists():
        _progress("[task1] convs index not found — run without --extract-only to build it first.")
        return

    _progress("[task1] Building task1_items.jsonl ...")

    # ── Wildchat: first TASK1_WILDCHAT_N users with valid convs ──────────────
    with sqlite3.connect(str(ATTRS_INDEX_DB), timeout=30) as conn:
        attrs_rows = conn.execute(
            "SELECT line_index, user_id, byte_offset FROM attrs_index ORDER BY line_index"
        ).fetchall()

    with sqlite3.connect(str(CONVS_INDEX_DB), timeout=30) as conn:
        conv_rows = conn.execute(
            "SELECT hashed_ip, byte_offset FROM convs_index ORDER BY id"
        ).fetchall()
    convs_map: dict = {}
    for ip, off in conv_rows:
        convs_map.setdefault(ip, []).append(off)

    wildchat_items = []
    skipped = 0
    with open(ATTRS_JSONL, "rb") as attrs_fh, open(CONVS_JSONL, "rb") as convs_fh:
        for line_index, user_id, attrs_offset in attrs_rows:
            if len(wildchat_items) >= TASK1_WILDCHAT_N:
                break
            if user_id not in convs_map:
                skipped += 1
                continue
            attrs_fh.seek(attrs_offset)
            rec = _strip_embeddings(json.loads(attrs_fh.readline()))
            conversations = []
            for conv_offset in convs_map[user_id]:
                convs_fh.seek(conv_offset)
                try:
                    raw = json.loads(convs_fh.readline())
                    conv_field = raw.get("conversation", [])
                    if conv_field and isinstance(conv_field[0], list):
                        # Nested list of sub-conversations — split into one record each
                        for sub_conv in conv_field:
                            entry = dict(raw)
                            entry["conversation"] = sub_conv
                            conversations.append(entry)
                    else:
                        conversations.append(raw)
                except Exception:
                    continue
            if not conversations:
                skipped += 1
                continue
            wildchat_items.append({
                "user_id": user_id,
                "source": "wildchat",
                "merged_attributes": rec.get("merged_attributes", []),
                "conversations": conversations,
            })
    _progress(f"  [task1] {len(wildchat_items)} Wildchat users collected ({skipped} skipped).")

    # ── Extra sources ─────────────────────────────────────────────────────────
    rng = random.Random(TASK1_RANDOM_SEED)
    all_items = list(wildchat_items)
    for source_name, attrs_path, personas_path in EXTRA_SOURCES:
        items = _load_extra_source(source_name, attrs_path, personas_path, TASK1_EXTRA_N, rng)
        _progress(f"  [task1] {len(items)} users from {source_name}.")
        all_items.extend(items)

    # ── Shuffle and write ─────────────────────────────────────────────────────
    rng.shuffle(all_items)
    tmp = TASK1_ITEMS.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as out_f:
        for item_index, item in enumerate(all_items):
            item["item_index"] = item_index
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(TASK1_ITEMS)
    _progress(f"[task1] Done. {len(all_items)} items -> {TASK1_ITEMS.name} "
              f"(wildchat={len(wildchat_items)}, "
              + ", ".join(f"{s}={TASK1_EXTRA_N}" for s, _, _ in EXTRA_SOURCES) + ")")


def build_task2_items(force: bool = False) -> None:
    """
    Build task2_items.jsonl: one record per checklist sample entry (used by Task 2 and Task 3).
    Uses the SQLite checklist index as cache (must be built first).
    """
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if TASK2_ITEMS.exists() and not force:
        _progress(f"[task2] {TASK2_ITEMS.name} already exists. Use --force to rebuild.")
        return
    if not CHECKLIST_INDEX_DB.exists():
        _progress("[task2] checklist index not found — run without --extract-only to build it first.")
        return

    _progress("[task2] Building task2_items.jsonl ...")

    with sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=30) as conn:
        has_sample = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checklist_sample'"
        ).fetchone() is not None
        if has_sample:
            rows = conn.execute("""
                SELECT cs.sample_index, ci.byte_offset
                FROM checklist_sample cs
                JOIN checklist_index ci ON cs.annotation_index = ci.annotation_index
                ORDER BY cs.sample_index
            """).fetchall()
        else:
            rows = conn.execute(
                "SELECT annotation_index, byte_offset FROM checklist_index ORDER BY annotation_index"
            ).fetchall()

    if not rows:
        _progress("[task2] No checklist records found.")
        return

    n = len(rows)
    tmp = TASK2_ITEMS.with_suffix(".jsonl.tmp")
    with open(CHECKLIST_JSONL, "rb") as fh, open(tmp, "w", encoding="utf-8") as out_f:
        for item_index, (_, offset) in enumerate(rows):
            fh.seek(offset)
            rec = _strip_embeddings(json.loads(fh.readline()))
            rec["item_index"] = item_index
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (item_index + 1) % 1000 == 0:
                _progress(f"  [task2] {item_index + 1}/{n}")
    tmp.replace(TASK2_ITEMS)
    _progress(f"[task2] Done. {n} items -> {TASK2_ITEMS.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build byte-offset indexes for annotation tool.")
    parser.add_argument("--skip-checklist",  action="store_true")
    parser.add_argument("--skip-attrs",      action="store_true")
    parser.add_argument("--skip-convs",      action="store_true")
    parser.add_argument("--skip-sample",     action="store_true", help="Skip building checklist sample")
    parser.add_argument("--n-per-user",      type=int, default=5, help="Max users per prompt in sample")
    parser.add_argument("--skip-prompt-indexes", type=str, default="60,148,420,244,312,353,240,31,108,79,46,1",
                        help="Comma-separated prompt_index values to exclude from sample, e.g. '3,17,42'")
    parser.add_argument("--force",           action="store_true", help="Rebuild even if index exists")
    parser.add_argument("--extract",         action="store_true", help="Also extract portable data files after indexing")
    parser.add_argument("--extract-only",    action="store_true", help="Only extract (skip index rebuild)", default=True)
    args = parser.parse_args()

    if not args.extract_only:
        if not args.skip_attrs:
            build_attrs_index(force=args.force)
        if not args.skip_convs:
            build_convs_index(force=args.force)
        if not args.skip_checklist:
            build_checklist_index(force=args.force)
        if not args.skip_sample and not args.skip_checklist:
            skip_set = (
                {int(x.strip()) for x in args.skip_prompt_indexes.split(",") if x.strip()}
                if args.skip_prompt_indexes else set()
            )
            build_checklist_sample(n_per_user=args.n_per_user, force=args.force, skip_prompt_indexes=skip_set)

    if args.extract or args.extract_only:
        skip_set = (
            {int(x.strip()) for x in args.skip_prompt_indexes.split(",") if x.strip()}
            if args.skip_prompt_indexes else set()
        )

        def _table_exists(db_path: Path, table: str) -> bool:
            """Return True only if the DB file exists AND contains the named table."""
            if not db_path.exists():
                return False
            with sqlite3.connect(str(db_path), timeout=30) as _c:
                return _c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone() is not None

        # Only rebuild index tables if truly missing (these are slow — never force-rebuild here)
        if not _table_exists(ATTRS_INDEX_DB, "attrs_index"):
            build_attrs_index(force=True)
        if not _table_exists(CONVS_INDEX_DB, "convs_index"):
            build_convs_index(force=True)
        if not _table_exists(CHECKLIST_INDEX_DB, "checklist_index"):
            build_checklist_index(force=True)
        # Sample is fast to rebuild — force it when --force is set or when it's missing
        if args.force or not _table_exists(CHECKLIST_INDEX_DB, "checklist_sample"):
            build_checklist_sample(n_per_user=args.n_per_user, force=True, skip_prompt_indexes=skip_set)
        build_task1_items(force=args.force)
        build_task2_items(force=args.force)

    _progress("All done.")


if __name__ == "__main__":
    main()
