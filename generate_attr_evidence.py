#!/usr/bin/env python3
"""
For each attribute of the first 100 Task-1 users, ask vLLM to identify which
conversation and turn(s) best support that attribute.

One LLM prompt is built per (user, attribute) pair. All prompts are batched
into a single llm.generate() call for efficiency.

Cache format — data/attr_evidence_cache.jsonl, one line per user:
  {"line_index": N, "user_id": "...", "evidence": [
      {"conv_idx": 0, "turn_idxs": [1, 2]},   # index matches merged_attributes order
      {"conv_idx": null, "turn_idxs": []},     # no evidence found
      ...
  ]}

Re-running is safe — already-cached users are skipped unless --force is used.

Usage:
    python3 generate_attr_evidence.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tensor-parallel-size 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR   = Path(__file__).parent
ATTRS_FILE = BASE_DIR / "data" / "extracted" / "attrs.jsonl"
CONVS_FILE = BASE_DIR / "data" / "extracted" / "convs.jsonl"
CACHE_FILE = BASE_DIR / "data" / "attr_evidence_cache.jsonl"

# One prompt per attribute: give full conv context + single attribute + reason.
# Ask for conv_idx and turn_idxs only.
PROMPT_TEMPLATE = """
You are given a user's conversation history and one inferred attribute about them.
Identify the conversation and turn(s) that most directly support this attribute.

Conversations are 0-indexed. Turns within each conversation are 0-indexed.

Return ONLY a single JSON object — no other text:
{{"conv_idx": <int or null>, "turn_idxs": [<int>, ...]}}

Use null and [] if no relevant turn exists.

---

Conversation history:
{conv_context}

---

Attribute: {attribute}
Reason it was inferred: {reason}

Return the JSON object now.
""".strip()


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_attrs(max_users: int) -> List[Dict]:
    records = []
    with open(ATTRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for a in rec.get("merged_attributes", []):
                a.pop("embedding", None)
            records.append(rec)
            if len(records) >= max_users:
                break
    records.sort(key=lambda r: r.get("line_index", 0))
    return records


def build_convs_map(user_ids: set) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    with open(CONVS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                uid = rec.get("hashed_ip", "")
                if uid not in user_ids:
                    continue
                convs = rec.get("conversation", [])
                if isinstance(convs, list) and convs and isinstance(convs[0], dict):
                    convs = [convs]
                result[uid] = convs
            except Exception:
                continue
    return result


def build_conv_context(convs: List[Any]) -> str:
    """Numbered transcript so the LLM can reference conversations and turns by index."""
    lines = []
    for c_idx, conv in enumerate(convs):
        if not conv:
            continue
        if len(convs) > 1:
            lines.append(f"[Conversation {c_idx}]")
        transcript = ""
        for t_idx, turn in enumerate(conv):
            if not isinstance(turn, dict):
                continue
            role    = (turn.get("role") or "").upper()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"  Turn {t_idx} [{role}]: {content}")
            transcript += content
            if len(transcript) > 100000:
                break
    return "\n".join(lines)


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def coerce_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    return json.loads(stripped)


def parse_single(output: str) -> Dict:
    null_entry = {"conv_idx": None, "turn_idxs": []}
    try:
        parsed = coerce_json(output)
        if not isinstance(parsed, dict):
            return null_entry
        return {
            "conv_idx":  parsed.get("conv_idx"),
            "turn_idxs": parsed.get("turn_idxs") or [],
        }
    except Exception:
        return null_entry


# ─── Cache ────────────────────────────────────────────────────────────────────

def load_cached_indexes() -> set:
    done: set = set()
    if not CACHE_FILE.exists():
        return done
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx = rec.get("line_index")
                if idx is not None:
                    done.add(idx)
            except Exception:
                continue
    return done


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-attribute evidence pointers via vLLM.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path (default: %(default)s)")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path in [ATTRS_FILE, CONVS_FILE]:
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading attrs for first {args.max_users} users...", flush=True)
    attrs_records = load_attrs(args.max_users)

    user_ids = {r["user_id"] for r in attrs_records}
    print("Loading conversations...", flush=True)
    convs_map = build_convs_map(user_ids)
    print(f"  Found for {len(convs_map)}/{len(user_ids)} users.", flush=True)

    already_done = set() if args.force else load_cached_indexes()
    todo = [r for r in attrs_records if r.get("line_index") not in already_done]
    print(f"  {len(already_done)} cached, {len(todo)} users to generate.", flush=True)

    if not todo:
        print("Nothing to do. Use --force to regenerate.", flush=True)
        return

    if args.force and CACHE_FILE.exists():
        CACHE_FILE.unlink()

    # ── Load vLLM ──────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    print(f"\nLoading {args.model} ...", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len)
    tokenizer = llm.get_tokenizer()
    sampling  = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)

    # ── Build one prompt per (user, attribute) ────────────────────────────────
    print("\nBuilding prompts (one per attribute)...", flush=True)
    prompts:  List[str]  = []
    # meta tracks which user/attr each prompt corresponds to
    # meta[i] = (line_index, user_id, attr_idx, n_attrs)
    meta: List[tuple] = []

    for rec in todo:
        attrs     = rec.get("merged_attributes", [])
        convs     = convs_map.get(rec["user_id"], [])
        conv_ctx  = build_conv_context(convs)

        for a_idx, attr in enumerate(attrs):
            prompt_text = PROMPT_TEMPLATE.format(
                conv_context=conv_ctx,
                attribute=attr.get("attribute", ""),
                reason=(attr.get("reason") or "").strip()[:300],
            )
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(chat_prompt)
            meta.append((rec.get("line_index", 0), rec["user_id"], a_idx, len(attrs)))

    total_prompts = len(prompts)
    print(f"  {total_prompts} prompts for {len(todo)} users.", flush=True)

    # ── Generate (all batched) ────────────────────────────────────────────────
    print(f"Generating...", flush=True)
    generations = llm.generate(prompts, sampling)

    # ── Reassemble per-user evidence lists ────────────────────────────────────
    # user_evidence[line_index] = list of size n_attrs, filled by attr_idx
    user_evidence: Dict[int, List[Optional[Dict]]] = {}
    user_id_map:   Dict[int, str] = {}

    for gen, (line_index, user_id, a_idx, n_attrs) in zip(generations, meta):
        if line_index not in user_evidence:
            user_evidence[line_index] = [{"conv_idx": None, "turn_idxs": []}] * n_attrs
            user_id_map[line_index] = user_id
        output = gen.outputs[0].text if gen.outputs else ""
        user_evidence[line_index][a_idx] = parse_single(output)

    # ── Write cache ────────────────────────────────────────────────────────────
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a", encoding="utf-8") as cache_out:
        for line_index, evidence in user_evidence.items():
            cache_out.write(json.dumps({
                "line_index": line_index,
                "user_id":    user_id_map[line_index],
                "evidence":   evidence,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(user_evidence)} users written, {len(already_done)} skipped.", flush=True)
    print(f"Cache: {CACHE_FILE}", flush=True)


if __name__ == "__main__":
    main()
