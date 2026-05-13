#!/usr/bin/env python3
"""
Build task4_items.jsonl and task5_items.jsonl for the annotation tool.

task4 (70 items total) — task1 format + prompt assignment fields:
  - 50 wildchat users sampled from the checklist (prompt_index 0-100, items>=2)
    Each record: user_id, source, merged_attributes, conversations, item_index,
                 + prompt_index, prompt_text, items  (from the checklist assignment)
  - 20 users from extra sources (10 cupid + 10 prefeval), same format as task1

task5 — task2 format (checklist record + item_index):
  - Additional prompt assignments for the same 50 wildchat task4 users
  - Each user gets up to (MAX_PROMPTS_PER_USER - 1) more prompts beyond their task4 prompt
  - Caps: <=MAX_PROMPTS_PER_USER prompts/user total, <=MAX_USERS_PER_PROMPT users/prompt

Run:
    python3 build_index_final.py
    python3 build_index_final.py --force
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "data" / "indexes"
EXTRACT_DIR = Path(__file__).parent / "data" / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

TASK4_ITEMS = EXTRACT_DIR / "task4_items.jsonl"
TASK5_ITEMS = EXTRACT_DIR / "task5_items.jsonl"

BASE               = Path("/shared/0/projects/research-jam-summer-2024/tmp")
CHECKLIST_JSONL    = BASE / "checklist_final_merged_agglo_0.7_v2_LIMA.jsonl"
CONVS_JSONL        = BASE / "wildchat_long_conversations_15_assistant_like_confidence_final_english_v1.jsonl"
ATTRS_JSONL        = Path("/home/leczhang/research/llm-personalization/data/attrs_per_conversation_final_merged_agglo_0.7_v2_sampled_5000.jsonl")

CHECKLIST_INDEX_DB = DATA_DIR / "checklist_index.db"
ATTRS_INDEX_DB     = DATA_DIR / "attrs_index.db"
CONVS_INDEX_DB     = DATA_DIR / "convs_index.db"

PROMPT_MIN           = 0
PROMPT_MAX           = 42   # inclusive — first 100 prompts are indices 0-100
TASK4_WILDCHAT_N     = 50    # wildchat users in task4
TASK4_EXTRA_N        = 10    # users per extra source (10 cupid + 10 prefeval = 20)
MIN_PROMPTS_PER_USER = 2
MAX_PROMPTS_PER_USER = 5
MIN_USERS_PER_PROMPT = 2
MAX_USERS_PER_PROMPT = 4
MIN_ITEMS            = 2
SAMPLE_SEED          = 42
# Known bad prompt indexes — excluded before any mapping is done (same list as build_index.py).
DEFAULT_SKIP_PROMPT_INDEXES = "60,148,420,244,312,353,240,31,108,79,46,1"

EXTRA_SOURCES = [
    ("cupid",
     Path("/home/leczhang/research/llm-personalization/data/cupid_attrs_merged_agglo_0.7.jsonl"),
     Path("/home/leczhang/research/llm-personalization/data/cupid_100personas.jsonl")),
    ("prefeval",
     Path("/home/leczhang/research/llm-personalization/data/prefeval_attrs_merged_agglo_0.7.jsonl"),
     Path("/home/leczhang/research/llm-personalization/data/prefeval_100personas.jsonl")),
     ("personalens",
     Path("/home/leczhang/research/llm-personalization/data/personalens_attrs_merged_agglo_0.7.jsonl"),
     Path("/home/leczhang/research/llm-personalization/data/personalens_100personas.jsonl")),
]


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _strip_embeddings(obj):
    if isinstance(obj, dict):
        return {k: _strip_embeddings(v) for k, v in obj.items() if k != "embedding"}
    if isinstance(obj, list):
        return [_strip_embeddings(x) for x in obj]
    return obj


def _load_extra_source(
    source_name: str,
    attrs_path: Path,
    personas_path: Path,
    n: int,
    rng,
    exclude: set | None = None,
) -> list:
    """Load n random users from an extra source (same logic as task1)."""
    attrs_map = {}
    with open(attrs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _strip_embeddings(json.loads(line))
            attrs_map[str(rec["user_id"])] = rec.get("merged_attributes", [])

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

    exclude = exclude or set()
    valid_users = [uid for uid in attrs_map if uid in convs_map and uid not in exclude]
    sampled = rng.sample(valid_users, min(n, len(valid_users)))

    return [
        {
            "user_id": uid,
            "source": source_name,
            "merged_attributes": attrs_map[uid],
            "conversations": convs_map[uid],
        }
        for uid in sampled
    ]


def _scan_attrs(target_ids: set[str]) -> dict[str, list]:
    """
    Scan ATTRS_JSONL and return merged_attributes for every user_id in target_ids.
    Avoids byte-offset index so it works correctly when the file is replaced.
    """
    result: dict[str, list] = {}
    with open(ATTRS_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = _strip_embeddings(json.loads(line))
                uid = str(rec.get("user_id", ""))
                if uid in target_ids:
                    result[uid] = rec.get("merged_attributes", [])
                    if len(result) == len(target_ids):
                        break
            except Exception:
                continue
    return result


def _load_wildchat_convs(user_ids: list[str]) -> dict[str, list]:
    """Load conversations for wildchat user_ids via the convs index."""
    user_convs: dict[str, list] = {}
    if not user_ids or not CONVS_INDEX_DB.exists():
        if not CONVS_INDEX_DB.exists():
            _progress("  Warning: convs_index.db not found — conversations will be empty.")
        return user_convs

    placeholders = ",".join("?" * len(user_ids))
    with sqlite3.connect(str(CONVS_INDEX_DB), timeout=30) as conn:
        conv_rows = conn.execute(
            f"SELECT hashed_ip, byte_offset FROM convs_index WHERE hashed_ip IN ({placeholders})",
            user_ids,
        ).fetchall()

    convs_offsets: dict[str, list[int]] = defaultdict(list)
    for hashed_ip, offset in conv_rows:
        convs_offsets[hashed_ip].append(offset)

    with open(CONVS_JSONL, "rb") as fh:
        for uid in user_ids:
            conversations = []
            for offset in convs_offsets.get(uid, []):
                fh.seek(offset)
                try:
                    raw = json.loads(fh.readline())
                    conv_field = raw.get("conversation", [])
                    if conv_field and isinstance(conv_field[0], list):
                        for sub_conv in conv_field:
                            entry = dict(raw)
                            entry["conversation"] = sub_conv
                            conversations.append(entry)
                    else:
                        conversations.append(raw)
                except Exception:
                    continue
            user_convs[uid] = conversations

    return user_convs


def build_task4_task5_items(
    force: bool = False,
    from_task1_n: int = 0,
    skip_prompt_indexes: set[int] | None = None,
) -> None:
    if TASK4_ITEMS.exists() and TASK5_ITEMS.exists() and not force:
        _progress(
            f"[task4/5] Both files already exist. Use --force to rebuild.\n"
            f"  {TASK4_ITEMS}\n  {TASK5_ITEMS}"
        )
        return

    if not CHECKLIST_INDEX_DB.exists():
        _progress("[task4/5] checklist_index.db not found — build it first.")
        return

    # ── Optionally seed task4 with first N items from task1_items.jsonl ───────
    fixed_items: list[dict] = []
    fixed_user_ids: set[str] = set()
    if from_task1_n > 0:
        task1_path = EXTRACT_DIR / "task1_items.jsonl"
        if not task1_path.exists():
            _progress(f"  Warning: task1_items.jsonl not found — ignoring --from-task1-n.")
        else:
            with open(task1_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= from_task1_n:
                        break
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        fixed_items.append(item)
                        fixed_user_ids.add(str(item.get("user_id", "")))
            _progress(f"  Loaded {len(fixed_items)} fixed items from task1_items.jsonl.")

    # Adjust how many more wildchat / extra users to sample
    wildchat_fixed = sum(1 for it in fixed_items if it.get("source") == "wildchat")
    wildchat_to_sample = max(0, TASK4_WILDCHAT_N - wildchat_fixed)
    extra_to_sample = {
        src: max(0, TASK4_EXTRA_N - sum(1 for it in fixed_items if it.get("source") == src))
        for src, _, _ in EXTRA_SOURCES
    }

    # ── Load checklist index rows for prompts 0-100 (excluding bad prompts) ────
    skip_set = skip_prompt_indexes or set()
    if skip_set:
        _progress(f"[task4/5] Skipping prompt_indexes: {sorted(skip_set)}")
    skip_clause = (
        f"AND prompt_index NOT IN ({','.join('?' * len(skip_set))})"
        if skip_set else ""
    )
    _progress(f"[task4/5] Loading checklist index for prompt_index {PROMPT_MIN}–{PROMPT_MAX}...")
    with sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=30) as conn:
        rows = conn.execute(
            f"""
            SELECT annotation_index, user_id, prompt_index, byte_offset
            FROM checklist_index
            WHERE prompt_index BETWEEN ? AND ?
            {skip_clause}
            ORDER BY annotation_index
            """,
            (PROMPT_MIN, PROMPT_MAX, *sorted(skip_set)),
        ).fetchall()
    _progress(f"  {len(rows)} index rows in prompt range.")

    # ── Filter: keep only records with len(items) >= MIN_ITEMS ────────────────
    _progress(f"  Verifying item count >= {MIN_ITEMS}...")
    valid_pairs: list[tuple] = []  # (user_id, prompt_idx, ann_idx, byte_offset)
    with open(CHECKLIST_JSONL, "rb") as fh:
        for i, (ann_idx, user_id, prompt_idx, offset) in enumerate(rows):
            if i % 5000 == 0 and i > 0:
                _progress(f"    {i}/{len(rows)} checked, {len(valid_pairs)} valid so far...")
            fh.seek(offset)
            try:
                rec = json.loads(fh.readline())
                if len(rec.get("items") or []) >= MIN_ITEMS:
                    valid_pairs.append((user_id, prompt_idx, ann_idx, offset))
            except Exception:
                continue

    user_prompts: dict[str, list] = defaultdict(list)
    for user_id, prompt_idx, ann_idx, offset in valid_pairs:
        user_prompts[user_id].append((prompt_idx, ann_idx, offset))

    # Drop users who can't meet the minimum-prompts-per-user requirement.
    # Must happen before attrs scanning so those users never enter the eligible pool.
    user_prompts = {uid: pairs for uid, pairs in user_prompts.items()
                    if len(pairs) >= MIN_PROMPTS_PER_USER}
    _progress(
        f"  {len(valid_pairs)} valid pairs; {len(user_prompts)} users "
        f"with >= {MIN_PROMPTS_PER_USER} valid prompts."
    )

    # ── Assign prompts to fixed wildchat users first (updates caps for new sampling) ──
    rng = random.Random(SAMPLE_SEED)

    prompt_user_count: dict[int, int] = defaultdict(int)
    fixed_wildchat_prompts: dict[str, tuple] = {}  # user_id -> (prompt_idx, ann_idx, offset)

    for item in fixed_items:
        user_id = str(item.get("user_id", ""))
        if item.get("source") != "wildchat" or user_id not in user_prompts:
            continue
        available = [
            (p, a, o) for p, a, o in user_prompts[user_id]
            if prompt_user_count[p] < MAX_USERS_PER_PROMPT
        ]
        if not available:
            _progress(f"  Warning: fixed wildchat user {user_id} has no available prompts.")
            continue
        rng.shuffle(available)
        chosen = available[0]
        fixed_wildchat_prompts[user_id] = chosen
        prompt_user_count[chosen[0]] += 1

    _progress(f"  Assigned prompts to {len(fixed_wildchat_prompts)} fixed wildchat users.")

    # ── Scan attrs file once for all checklist users ──────────────────────────
    # This is the single source of truth for attrs eligibility and data.
    # The byte-offset index (attrs_index.db) is intentionally not used here
    # because it becomes stale when ATTRS_JSONL is replaced with a newer file.
    _progress("  Scanning attrs file for checklist users (this may take a moment)...")
    all_checklist_users = set(user_prompts.keys())
    attrs_data: dict[str, list] = _scan_attrs(all_checklist_users)
    _progress(f"  {len(attrs_data)} checklist users found in attrs file.")

    # ── Build eligible set: must have attrs AND convs ─────────────────────────
    if CONVS_INDEX_DB.exists():
        with sqlite3.connect(str(CONVS_INDEX_DB), timeout=30) as conn:
            convs_user_ids = {
                row[0] for row in conn.execute("SELECT DISTINCT hashed_ip FROM convs_index").fetchall()
            }
    else:
        convs_user_ids = all_checklist_users
        _progress("  Warning: convs_index.db not found — skipping convs filter.")

    eligible_user_ids = set(attrs_data.keys()) & convs_user_ids
    _progress(
        f"  {len(eligible_user_ids)} users have both attrs and convs data "
        f"(out of {len(user_prompts)} checklist users with valid pairs)."
    )

    # ── Unified prompt-centric assignment (task4/task5 split applied at the end) ─
    # user_assigned: uid -> [(prompt_idx, ann_idx, offset), ...]
    # Fixed wildchat users are seeded with their already-chosen task4 prompt.
    # prompt_users tracks the set of uids assigned to each prompt (for O(1) lookup).

    user_assigned: dict[str, list] = {}
    prompt_users: dict[int, set] = defaultdict(set)
    for uid, (prompt_idx, ann_idx, offset) in fixed_wildchat_prompts.items():
        user_assigned[uid] = [(prompt_idx, ann_idx, offset)]
        prompt_users[prompt_idx].add(uid)
        # prompt_user_count[prompt_idx] already incremented when fixed_wildchat_prompts was built.

    # Select new wildchat users from the eligible pool (not already fixed).
    candidate_users = [
        u for u in user_prompts
        if u in eligible_user_ids and u not in fixed_user_ids
    ]
    rng.shuffle(candidate_users)
    new_wildchat_users = candidate_users[:wildchat_to_sample]
    for uid in new_wildchat_users:
        user_assigned[uid] = []

    all_wildchat_users = set(user_assigned.keys())
    _progress(
        f"  {len(new_wildchat_users)} new wildchat users selected "
        f"(target {wildchat_to_sample}); {len(all_wildchat_users)} total wildchat users."
    )

    # Build per-prompt candidate list (each user appears at most once per prompt).
    prompt_eligible: dict[int, list] = defaultdict(list)
    seen_per_prompt: dict[int, set] = defaultdict(set)
    for uid in sorted(all_wildchat_users):
        for prompt_idx, ann_idx, offset in user_prompts.get(uid, []):
            if uid not in seen_per_prompt[prompt_idx]:
                prompt_eligible[prompt_idx].append((uid, ann_idx, offset))
                seen_per_prompt[prompt_idx].add(uid)
    del seen_per_prompt

    def _assign(uid: str, prompt_idx: int, ann_idx: int, offset: int) -> None:
        user_assigned[uid].append((prompt_idx, ann_idx, offset))
        prompt_users[prompt_idx].add(uid)
        prompt_user_count[prompt_idx] += 1

    # Phase 1: guarantee MIN_USERS_PER_PROMPT for every viable prompt.
    all_prompt_idxs = sorted(prompt_eligible.keys())
    rng.shuffle(all_prompt_idxs)
    for prompt_idx in all_prompt_idxs:
        already = prompt_user_count[prompt_idx]
        if already >= MIN_USERS_PER_PROMPT or already >= MAX_USERS_PER_PROMPT:
            continue
        need = MIN_USERS_PER_PROMPT - already
        candidates = [
            (uid, a, o) for uid, a, o in prompt_eligible[prompt_idx]
            if uid not in prompt_users[prompt_idx]
            and len(user_assigned[uid]) < MAX_PROMPTS_PER_USER
        ]
        if already + len(candidates) < MIN_USERS_PER_PROMPT:
            continue  # cannot satisfy minimum — skip this prompt
        rng.shuffle(candidates)
        for uid, a, o in candidates[:need]:
            _assign(uid, prompt_idx, a, o)

    # Phase 2: fill remaining capacity (only for prompts that already meet minimum).
    rng.shuffle(all_prompt_idxs)
    for prompt_idx in all_prompt_idxs:
        if prompt_user_count[prompt_idx] < MIN_USERS_PER_PROMPT:
            continue  # couldn't satisfy minimum — exclude entirely
        cap = MAX_USERS_PER_PROMPT - prompt_user_count[prompt_idx]
        if cap <= 0:
            continue
        candidates = [
            (uid, a, o) for uid, a, o in prompt_eligible[prompt_idx]
            if uid not in prompt_users[prompt_idx]
            and len(user_assigned[uid]) < MAX_PROMPTS_PER_USER
        ]
        if not candidates:
            continue
        rng.shuffle(candidates)
        for uid, a, o in candidates[:cap]:
            _assign(uid, prompt_idx, a, o)

    # task4: one item per new wildchat user (their first assignment).
    # task5: ALL assignments for every wildchat user (task4 items are a subset of task5).
    task4_wildchat: list[tuple] = []  # (uid, prompt_idx, ann_idx, offset) — new users only
    task5_pairs: list[tuple] = []     # (uid, prompt_idx, ann_idx, offset) — all assignments

    for uid in sorted(all_wildchat_users):
        assignments = user_assigned[uid]
        if not assignments and uid not in fixed_wildchat_prompts:
            _progress(f"  Warning: user {uid} got 0 assignments — skipping.")
            continue
        # task5 gets all assignments for this user.
        for p_idx, a_idx, off in assignments:
            task5_pairs.append((uid, p_idx, a_idx, off))
        # task4 gets only the first assignment for new (non-fixed) users.
        if uid not in fixed_wildchat_prompts:
            p_idx, a_idx, off = assignments[0]
            task4_wildchat.append((uid, p_idx, a_idx, off))

    task4_users = {u for u, *_ in task4_wildchat} | set(fixed_wildchat_prompts.keys())
    task5_users = {u for u, *_ in task5_pairs}
    assert task5_users <= task4_users, "BUG: task5 contains users not in task4"
    _progress(
        f"  task4 wildchat: {len(task4_wildchat)} new + {len(fixed_wildchat_prompts)} fixed = "
        f"{len(task4_users)} total users\n"
        f"  task5: {len(task5_pairs)} items across {len(task5_users)} users "
        f"(includes task4 assignments)"
    )

    # ── Load conversations for newly sampled wildchat users ───────────────────
    # attrs are already in attrs_data (scanned above); fixed items carry their own.
    selected_user_ids = [u for u, *_ in task4_wildchat]
    _progress("  Loading conversations for new wildchat users...")
    user_convs = _load_wildchat_convs(selected_user_ids)

    # ── Load extra source items for task4 (20 items: 10 cupid + 10 prefeval) ──
    extra_items: list[dict] = []
    for source_name, attrs_path, personas_path in EXTRA_SOURCES:
        n = extra_to_sample[source_name]
        items = _load_extra_source(source_name, attrs_path, personas_path, n, rng, exclude=fixed_user_ids)
        _progress(f"  {len(items)} users from {source_name}.")
        extra_items.extend(items)

    # ── Build task4 records ────────────────────────────────────────────────────
    # Fixed items go first (in original order), with prompt fields added for wildchat.
    # Newly sampled items (wildchat + extra) are shuffled and appended after.

    fixed_records: list[dict] = []
    with open(CHECKLIST_JSONL, "rb") as fh:
        for item in fixed_items:
            user_id = str(item.get("user_id", ""))
            rec = {k: v for k, v in item.items() if k != "item_index"}
            if user_id in fixed_wildchat_prompts:
                prompt_idx, ann_idx, offset = fixed_wildchat_prompts[user_id]
                fh.seek(offset)
                checklist_rec = _strip_embeddings(json.loads(fh.readline()))
                rec["prompt_index"] = prompt_idx
                rec["prompt_text"] = checklist_rec.get("prompt_text", "")
                rec["items"] = checklist_rec.get("items", [])
            fixed_records.append(rec)

    new_records: list[dict] = []
    with open(CHECKLIST_JSONL, "rb") as fh:
        for user_id, prompt_idx, ann_idx, offset in task4_wildchat:
            fh.seek(offset)
            checklist_rec = _strip_embeddings(json.loads(fh.readline()))
            new_records.append({
                "user_id": user_id,
                "source": "wildchat",
                "merged_attributes": attrs_data.get(user_id, []),
                "conversations": user_convs.get(user_id, []),
                "prompt_index": prompt_idx,
                "prompt_text": checklist_rec.get("prompt_text", ""),
                "items": checklist_rec.get("items", []),
            })
    for item in extra_items:
        new_records.append(item)

    rng.shuffle(new_records)
    task4_records = fixed_records + new_records

    # ── Write task4_items.jsonl ────────────────────────────────────────────────
    _progress(f"[task4] Writing {TASK4_ITEMS.name} ({len(task4_records)} items)...")
    tmp4 = TASK4_ITEMS.with_suffix(".jsonl.tmp")
    with open(tmp4, "w", encoding="utf-8") as out_f:
        for item_index, rec in enumerate(task4_records):
            rec["item_index"] = item_index
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp4.replace(TASK4_ITEMS)
    _progress(f"[task4] Done → {TASK4_ITEMS}")

    # ── Write task5_items.jsonl ────────────────────────────────────────────────
    # Format: task2 format — raw checklist record + item_index
    _progress(f"[task5] Writing {TASK5_ITEMS.name} ({len(task5_pairs)} items)...")
    tmp5 = TASK5_ITEMS.with_suffix(".jsonl.tmp")
    with open(CHECKLIST_JSONL, "rb") as fh, open(tmp5, "w", encoding="utf-8") as out_f:
        for item_index, (user_id, prompt_idx, ann_idx, offset) in enumerate(task5_pairs):
            fh.seek(offset)
            rec = _strip_embeddings(json.loads(fh.readline()))
            rec["item_index"] = item_index
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp5.replace(TASK5_ITEMS)
    _progress(f"[task5] Done → {TASK5_ITEMS}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build task4_items.jsonl and task5_items.jsonl."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if files exist")
    parser.add_argument(
        "--from-task1-n", type=int, default=0, metavar="N",
        help="Seed task4 with the first N items from task1_items.jsonl (default: 0)",
    )
    parser.add_argument(
        "--skip-prompt-indexes", type=str, default=DEFAULT_SKIP_PROMPT_INDEXES,
        help="Comma-separated prompt_index values to exclude (default: %(default)s)",
    )
    args = parser.parse_args()
    skip_set = (
        {int(x.strip()) for x in args.skip_prompt_indexes.split(",") if x.strip()}
        if args.skip_prompt_indexes else set()
    )
    build_task4_task5_items(force=args.force, from_task1_n=args.from_task1_n, skip_prompt_indexes=skip_set)
    _progress("All done.")


if __name__ == "__main__":
    main()
