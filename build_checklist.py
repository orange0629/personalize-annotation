#!/usr/bin/env python3
"""
Build evaluation-ready checklist JSONL from task2_items.jsonl.

Two modes:
  llm        — majority vote over LLM_VOTERS relevance files
  annotator  — majority vote over ANNOTATORS task3 annotation files

Hard-coded constants (edit at the top of this file as needed):

  LLM_VOTERS    — stems of files under data/relevance/
  ANNOTATORS    — stems of annotator IDs (looks for {id}_task3.jsonl in data/annotations/)
  CONVS_JSONL   — path to the wildchat conversations JSONL (required; error if missing)

Behaviors are loaded from data/behaviors_simple/{tag}.jsonl (field: expected_behavior).
If not found there, falls back to data/behaviors/{tag}.jsonl (field: explicit_behavior).
A warning is printed when the fallback is used.

Conversations are always included in the output (loaded from CONVS_JSONL via convs_index.db).

Output line format:
  {
    "user_id": "...",
    "prompt_id": "<user_id>_<prompt_index>",
    "prompt_text": "...",
    "profile_attributes": [...],
    "items": [
      {
        "attribute": "...",
        "expected_behavior": "The response should ...",
        "relevance": true
      },
      ...
    ],
    "conversations": [...],
    "meta": {
      "source": "llm|annotator",
      "behavior_tag": "...",
      "yes_votes": {"<attribute>": N, ...},
      "no_votes":  {"<attribute>": N, ...}
    }
  }

Usage:
    python3 build_checklist.py --mode llm --behavior-tag gpt-5.4
    python3 build_checklist.py --mode annotator --behavior-tag gpt-5.4
    python3 build_checklist.py --mode llm --behavior-tag gpt-5.4 \\
        --output data/checklists/llm_gpt54.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR       = Path(__file__).parent
TASK2_FILE     = BASE_DIR / "data" / "extracted" / "task2_items.jsonl"
RELEVANCE_DIR  = BASE_DIR / "data" / "relevance"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
BEHAVIORS_SIMPLE_DIR = BASE_DIR / "data" / "behaviors_simple"
BEHAVIORS_DIR  = BASE_DIR / "data" / "behaviors"
CONVS_INDEX_DB = BASE_DIR / "data" / "indexes" / "convs_index.db"
CHECKLIST_DIR  = BASE_DIR / "data" / "checklists"

# Hard-coded path to the wildchat conversations JSONL.
# Conversations are always included in the output — this file must exist.
CONVS_JSONL = Path(
    "/projects/bfuj/lzhang49/llm-personalization/data"
    "/wildchat_long_conversations_15_assistant_like_confidence_final.jsonl"
)

# ─── Hard-coded voter lists ────────────────────────────────────────────────────
# Edit these lists as voters are added or removed.

LLM_VOTERS: List[str] = [
    "arn_aws_bedrock_us-east-2_421746314685_inference-profile_us.anthropic.claude-sonnet-4-6",
    "google_gemma-4-31b-it",
    "gpt-5.4",
    "meta-llama_llama-3.3-70b-instruct",
    "qwen_qwen3.5-27b",
]

ANNOTATORS: List[str] = [
    "6928a474af049220b469f829",
    "6983b7ab2a0f33094b17b688",
    "6986319d4cdbba3cdf60ef30",
    "699395d4db097a35f5fde573",
    "Lechen",
    "Lechen2",
]

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_task2_items(max_samples: int) -> List[Dict]:
    records = []
    with open(TASK2_FILE, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["sample_index"] = line_idx
            records.append(rec)
            if len(records) >= max_samples:
                break
    return records


def load_llm_votes() -> Dict[int, Dict[str, Tuple[int, int]]]:
    """Returns {sample_index: {attribute: (yes_count, no_count)}}."""
    votes: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    loaded = []
    for voter in LLM_VOTERS:
        path = RELEVANCE_DIR / f"{voter}.jsonl"
        if not path.exists():
            print(f"  [warn] LLM voter file not found: {path.name}", flush=True)
            continue
        loaded.append(voter)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sidx = rec["sample_index"]
                for item in rec["items"]:
                    attr = item["attribute"]
                    relevant = item["relevant"]
                    if relevant is True:
                        votes[sidx][attr][0] += 1
                    elif relevant is False:
                        votes[sidx][attr][1] += 1
    print(f"  Loaded {len(loaded)}/{len(LLM_VOTERS)} LLM voter files.", flush=True)
    return {sidx: {attr: (v[0], v[1]) for attr, v in attrs.items()} for sidx, attrs in votes.items()}


def load_annotator_votes() -> Dict[int, Dict[str, Tuple[int, int]]]:
    """Returns {sample_index: {attribute: (yes_count, no_count)}}."""
    votes: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    loaded = []
    for annotator in ANNOTATORS:
        path = ANNOTATIONS_DIR / f"{annotator}_task3.jsonl"
        if not path.exists():
            print(f"  [warn] Annotator task3 file not found: {path.name}", flush=True)
            continue
        loaded.append(annotator)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sidx = rec["index"]
                for j in rec["relevance_judgments"]:
                    attr = j["attribute"]
                    relevant = j["relevant"]
                    if relevant is True:
                        votes[sidx][attr][0] += 1
                    elif relevant is False:
                        votes[sidx][attr][1] += 1
    print(f"  Loaded {len(loaded)}/{len(ANNOTATORS)} annotator task3 files.", flush=True)
    return {sidx: {attr: (v[0], v[1]) for attr, v in attrs.items()} for sidx, attrs in votes.items()}


def load_behaviors(tag: str) -> Dict[int, Dict[str, str]]:
    """Returns {sample_index: {attribute: expected_behavior}}."""
    simple_path = BEHAVIORS_SIMPLE_DIR / f"{tag}.jsonl"
    fallback_path = BEHAVIORS_DIR / f"{tag}.jsonl"

    if simple_path.exists():
        path = simple_path
        field = "expected_behavior"
        print(f"  Loading behaviors from {simple_path.relative_to(BASE_DIR)}", flush=True)
    elif fallback_path.exists():
        path = fallback_path
        field = "explicit_behavior"
        print(
            f"  WARNING: behaviors_simple/{tag}.jsonl not found — falling back to "
            f"behaviors/{tag}.jsonl and using 'explicit_behavior' instead of 'expected_behavior'. "
            f"Run generate_expected_behaviors_simple.py to produce the preferred file.",
            flush=True,
        )
    else:
        print(f"ERROR: No behavior file found for tag '{tag}'.", file=sys.stderr)
        print(f"  Tried (preferred): {simple_path}", file=sys.stderr)
        print(f"  Tried (fallback):  {fallback_path}", file=sys.stderr)
        sys.exit(1)

    behaviors: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sidx = rec.get("sample_index")
            if sidx is None:
                continue
            behaviors[sidx] = {
                item.get("attribute", ""): item.get(field, "")
                for item in rec.get("items", [])
            }
    return behaviors


# ─── Conversation loading ──────────────────────────────────────────────────────

def load_convs_lookup() -> Dict[str, List[Dict]]:
    """Returns {user_id: [conv_record, ...]} using CONVS_JSONL + convs_index.db."""
    if not CONVS_JSONL.exists():
        print(f"ERROR: Conversations JSONL not found: {CONVS_JSONL}", file=sys.stderr)
        print(
            "  Update the CONVS_JSONL constant at the top of this file to point to the correct path.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not CONVS_INDEX_DB.exists():
        print(f"ERROR: Conversation index DB not found: {CONVS_INDEX_DB}", file=sys.stderr)
        print("  Run build_index.py first to build the index.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading conversation index from {CONVS_INDEX_DB.name} ...", flush=True)
    with sqlite3.connect(str(CONVS_INDEX_DB), timeout=30) as conn:
        rows = conn.execute(
            "SELECT hashed_ip, byte_offset FROM convs_index ORDER BY id"
        ).fetchall()

    convs_map: Dict[str, List[int]] = defaultdict(list)
    for hashed_ip, offset in rows:
        convs_map[hashed_ip].append(offset)

    convs_lookup: Dict[str, List[Dict]] = {}
    print(f"  Reading conversations from {CONVS_JSONL.name} ...", flush=True)
    with open(CONVS_JSONL, "rb") as fh:
        for user_id, offsets in convs_map.items():
            recs = []
            for offset in offsets:
                fh.seek(offset)
                raw = fh.readline()
                if raw.strip():
                    try:
                        recs.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
            if recs:
                convs_lookup[user_id] = recs
    return convs_lookup


# ─── Checklist building ────────────────────────────────────────────────────────

def build_checklist(
    records: List[Dict],
    vote_map: Dict[int, Dict[str, Tuple[int, int]]],
    behaviors: Dict[int, Dict[str, str]],
    convs_lookup: Dict[str, List[Dict]],
    mode: str,
    behavior_tag: str,
    min_votes: int,
) -> List[Dict]:
    results = []
    skipped_no_votes = 0
    skipped_no_behaviors = 0

    for rec in records:
        sidx = rec["sample_index"]
        user_id = rec["user_id"]
        prompt_index = rec["prompt_index"]
        prompt_text = rec["prompt_text"]
        profile_attributes = rec["profile_attributes"]

        attr_votes = vote_map.get(sidx, {})
        attr_behaviors = behaviors.get(sidx, {})

        items_out = []        # only relevant attributes (yes > no)
        yes_votes_meta: Dict[str, int] = {}
        no_votes_meta: Dict[str, int] = {}

        for pa in profile_attributes:
            attr = pa["attribute"]
            yes, no = attr_votes.get(attr, (0, 0))
            total = yes + no

            if total < min_votes:
                skipped_no_votes += 1
                continue

            if yes <= no:
                # Not relevant — keep in profile_attributes but skip items
                continue

            behavior = attr_behaviors.get(attr, "")
            if not behavior:
                print(
                    f"  WARNING: no behavior for attribute in sample {sidx}: {attr[:60]}",
                    flush=True,
                )
                skipped_no_behaviors += 1
                continue

            items_out.append({
                "attribute": attr,
                "expected_behavior": behavior,
                "relevance": True,
            })
            yes_votes_meta[attr] = yes
            no_votes_meta[attr] = no

        if not items_out:
            continue

        out_rec: Dict = {
            "user_id": user_id,
            "prompt_id": f"{user_id}_{prompt_index}",
            "prompt_text": prompt_text,
            "profile_attributes": profile_attributes,
            "items": items_out,
        }

        user_convs = convs_lookup.get(user_id, [])
        if not user_convs:
            print(
                f"  WARNING: no conversations found for user {user_id[:16]}... "
                f"(sample_index={sidx}). 'conversations' will be an empty list.",
                flush=True,
            )
        out_rec["conversations"] = user_convs

        out_rec["meta"] = {
            "source": mode,
            "behavior_tag": behavior_tag,
            "sample_index": sidx,
            "yes_votes": yes_votes_meta,
            "no_votes": no_votes_meta,
        }

        results.append(out_rec)

    if skipped_no_votes:
        print(f"  Skipped {skipped_no_votes} attributes with < {min_votes} vote(s).", flush=True)
    if skipped_no_behaviors:
        print(f"  Skipped {skipped_no_behaviors} attributes with no behavior.", flush=True)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evaluation-ready checklist JSONL from task2_items.jsonl."
    )
    parser.add_argument("--mode", required=True, choices=["llm", "annotator"],
                        help="Vote source: 'llm' or 'annotator'")
    parser.add_argument("--behavior-tag", required=True,
                        help="Model tag for behavior file (e.g. 'gpt-5.4')")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/checklists/<mode>_<behavior_tag>.jsonl)")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--min-votes", type=int, default=1,
                        help="Minimum non-null votes required to include an attribute")
    args = parser.parse_args()

    if not TASK2_FILE.exists():
        print(f"ERROR: {TASK2_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    # Print voter lists
    print("=" * 60, flush=True)
    print(f"Mode: {args.mode}", flush=True)
    if args.mode == "llm":
        print(f"LLM voters ({len(LLM_VOTERS)}):", flush=True)
        for v in LLM_VOTERS:
            print(f"  - {v}", flush=True)
    else:
        print(f"Annotators ({len(ANNOTATORS)}):", flush=True)
        for a in ANNOTATORS:
            print(f"  - {a}", flush=True)
    print(f"Behavior tag: {args.behavior_tag}", flush=True)
    print(f"Min votes: {args.min_votes}", flush=True)
    print(f"Conversations JSONL: {CONVS_JSONL}", flush=True)
    print("=" * 60, flush=True)

    print(f"Loading task2 items (max {args.max_samples}) ...", flush=True)
    records = load_task2_items(args.max_samples)
    print(f"  Loaded {len(records)} samples.", flush=True)

    print(f"Loading {args.mode} votes ...", flush=True)
    if args.mode == "llm":
        vote_map = load_llm_votes()
    else:
        vote_map = load_annotator_votes()
    print(f"  Vote map covers {len(vote_map)} samples.", flush=True)

    print("Loading behaviors ...", flush=True)
    behaviors = load_behaviors(args.behavior_tag)
    print(f"  Behaviors loaded for {len(behaviors)} samples.", flush=True)

    print("Loading conversations ...", flush=True)
    convs_lookup = load_convs_lookup()
    print(f"  Conversations loaded for {len(convs_lookup)} users.", flush=True)

    print("Building checklist ...", flush=True)
    results = build_checklist(
        records, vote_map, behaviors, convs_lookup,
        mode=args.mode,
        behavior_tag=args.behavior_tag,
        min_votes=args.min_votes,
    )
    print(f"  Built {len(results)} checklist entries.", flush=True)

    tag_safe = args.behavior_tag.replace("/", "_").replace(":", "_").replace(" ", "_").lower()
    out_path = Path(args.output) if args.output else CHECKLIST_DIR / f"{args.mode}_{tag_safe}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)} records written -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
