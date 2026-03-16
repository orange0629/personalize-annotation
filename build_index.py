#!/usr/bin/env python3
"""
Build byte-offset indexes for large JSONL files used by the annotation tool.

Indexes are stored in SQLite databases in data/indexes/.
Run this once before starting the annotation tool.

Usage:
    python3 build_index.py [--skip-checklist] [--skip-attrs] [--skip-convs]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "indexes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/shared/0/projects/research-jam-summer-2024/tmp")
ATTRS_JSONL       = BASE / "attrs_per_conversation_final_merged_agglo_0.7_v2.jsonl"
CHECKLIST_JSONL   = BASE / "checklist_final_merged_agglo_0.7_v2_LIMA.jsonl"
CONVS_JSONL       = BASE / "wildchat_long_conversations_15_assistant_like_confidence_final.jsonl"

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
    _progress(f"[convs] Done. {count} records indexed.")


def build_checklist_sample(n_per_user: int = 10, force: bool = False) -> None:
    """
    Create checklist_sample table inside the checklist index DB.
    Randomly selects up to n_per_user records per user_id, assigns
    a sequential sample_index used by the app for navigation.
    Requires the checklist_index table to exist first.
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

    _progress(f"[sample] Sampling up to {n_per_user} records per user from checklist_index...")
    cur.executescript("""
        DROP TABLE IF EXISTS checklist_sample;
        CREATE TABLE checklist_sample (
            sample_index     INTEGER PRIMARY KEY,
            annotation_index INTEGER NOT NULL
        );
    """)

    # Use window function to pick n_per_user random records per user
    rows = conn.execute(f"""
        SELECT annotation_index
        FROM (
            SELECT annotation_index,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY RANDOM()) AS rn
            FROM checklist_index
        )
        WHERE rn <= {n_per_user}
        ORDER BY annotation_index
    """).fetchall()

    for sample_idx, (ann_idx,) in enumerate(rows):
        cur.execute(
            "INSERT INTO checklist_sample (sample_index, annotation_index) VALUES (?, ?)",
            (sample_idx, ann_idx),
        )

    conn.commit()
    conn.close()
    _progress(f"[sample] Done. {len(rows)} records sampled ({n_per_user} per user).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build byte-offset indexes for annotation tool.")
    parser.add_argument("--skip-checklist", action="store_true")
    parser.add_argument("--skip-attrs",     action="store_true")
    parser.add_argument("--skip-convs",     action="store_true")
    parser.add_argument("--skip-sample",    action="store_true", help="Skip building checklist sample")
    parser.add_argument("--n-per-user",     type=int, default=10, help="Records per user in sample")
    parser.add_argument("--force",          action="store_true", help="Rebuild even if index exists")
    args = parser.parse_args()

    if not args.skip_attrs:
        build_attrs_index(force=args.force)
    if not args.skip_convs:
        build_convs_index(force=args.force)
    if not args.skip_checklist:
        build_checklist_index(force=args.force)
    if not args.skip_sample and not args.skip_checklist:
        build_checklist_sample(n_per_user=args.n_per_user, force=args.force)

    _progress("All done.")


if __name__ == "__main__":
    main()
