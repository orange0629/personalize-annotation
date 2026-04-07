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
Given a user's conversation history, an inferred attribute, and the reason it was inferred, find the conversation and turn(s) whose content best corresponds to both the attribute and the reason.
Note: the attribute describes the USER, so the supporting evidence will primarily come from turns where the user speaks (Turn [USER]).

## Indexing
- Conversations are 0-indexed (conv_idx).
- Turns within each conversation are 0-indexed (turn_idxs).

## Output format
Output ONLY a single-line JSON object — no explanation, no markdown:
{{"conv_idx": <int>, "turn_idxs": [<int>, ...]}}

Examples of valid output:
{{"conv_idx": 0, "turn_idxs": [2, 3]}}
{{"conv_idx": 1, "turn_idxs": [0]}}

## Conversation history

{conv_context}

## Attribute and reason

Attribute: {attribute}
Reason it was inferred: {reason}

Find the turn(s) whose content best matches this attribute and the above reason. Output the JSON now:
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


def build_conv_context(convs: List[Any], max_chars: int = 80000, c_idx_start: int = 0) -> str:
    """Numbered transcript so the LLM can reference conversations and turns by index."""
    lines = []
    total_chars = 0
    show_conv_label = (len(convs) + c_idx_start) > 1
    for i, conv in enumerate(convs):
        c_idx = c_idx_start + i
        if not conv:
            continue
        if total_chars >= max_chars:
            break
        if show_conv_label:
            lines.append(f"[Conversation {c_idx}]")
        for t_idx, turn in enumerate(conv):
            if not isinstance(turn, dict):
                continue
            role    = (turn.get("role") or "").upper()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            line = f"  Turn {t_idx} [{role}]: {content}"
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(line) > remaining:
                lines.append(line[:remaining])
                total_chars = max_chars
                break
            lines.append(line)
            total_chars += len(line)
    return "\n".join(lines)


def get_conv_context_chunks(
    convs: List[Any],
    tokenizer: Any,
    max_ctx_tokens: int,
) -> List[tuple]:
    """
    Return a list of (c_idx_start, conv_ctx_string) chunks such that each chunk's
    context fits within max_ctx_tokens (checked against the template with empty attr/reason).
    Splits by conversation count until each chunk fits. For a single oversized conversation,
    reduces max_chars iteratively (1.5 chars/token worst case for code/math content).
    """
    def _ctx_and_tokens(sub_convs: List[Any], offset: int, max_chars: int) -> tuple:
        ctx = build_conv_context(sub_convs, max_chars=max_chars, c_idx_start=offset)
        test = PROMPT_TEMPLATE.format(conv_context=ctx, attribute="", reason="")
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": test}],
            tokenize=False, add_generation_prompt=True,
        )
        return ctx, len(tokenizer.encode(chat))

    def _split(sub_convs: List[Any], offset: int) -> List[tuple]:
        # Use 1.5 chars/token as a conservative initial char budget
        init_max_chars = int(max_ctx_tokens * 1.5)
        ctx, token_len = _ctx_and_tokens(sub_convs, offset, init_max_chars)
        if token_len <= max_ctx_tokens:
            return [(offset, ctx)]
        if len(sub_convs) <= 1:
            # Cannot split further by conversations; reduce char budget until it fits
            max_chars = int(init_max_chars * max_ctx_tokens / token_len * 0.9)
            while max_chars > 50:
                ctx, token_len = _ctx_and_tokens(sub_convs, offset, max_chars)
                if token_len <= max_ctx_tokens:
                    break
                max_chars = int(max_chars * 0.8)
            return [(offset, ctx)]
        mid = len(sub_convs) // 2
        return _split(sub_convs[:mid], offset) + _split(sub_convs[mid:], offset + mid)

    return _split(convs, 0)


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def coerce_json(text: str) -> Any:
    # Strip thinking tokens produced by reasoning models
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Handle unclosed <think> blocks (model cut off before finishing reasoning)
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if stripped.startswith("```"):
        match = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    # Try direct parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back: find first {...} blob in the output
    match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in output: {stripped[:200]!r}")


def parse_single(output: str, debug: bool = False) -> Dict:
    null_entry = {"conv_idx": None, "turn_idxs": []}
    try:
        parsed = coerce_json(output)
        if not isinstance(parsed, dict):
            if debug:
                print(f"  [parse_single] not a dict: {output[:200]!r}", flush=True)
            return null_entry
        return {
            "conv_idx":  parsed.get("conv_idx"),
            "turn_idxs": parsed.get("turn_idxs") or [],
        }
    except Exception as e:
        if debug:
            print(f"  [parse_single] failed ({e}): {output[:200]!r}", flush=True)
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
    parser.add_argument("--max-new-tokens", type=int, default=8192)
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

    # Reserve tokens for the template boilerplate, attribute/reason, and generation
    max_ctx_tokens = args.max_model_len - args.max_new_tokens - 512

    empty_conv_users = 0
    for rec in todo:
        attrs  = rec.get("merged_attributes", [])
        convs  = convs_map.get(rec["user_id"], [])
        chunks = get_conv_context_chunks(convs, tokenizer, max_ctx_tokens)
        if not any(ctx.strip() for _, ctx in chunks):
            empty_conv_users += 1
            if empty_conv_users <= 3:
                print(f"  WARNING: empty conv_ctx for user {rec['user_id']} "
                      f"(convs length={len(convs)})", flush=True)
        if len(chunks) > 1:
            print(f"  INFO: user {rec['user_id']} split into {len(chunks)} chunks", flush=True)

        for a_idx, attr in enumerate(attrs):
            for c_idx_offset, conv_ctx in chunks:
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
                meta.append((rec.get("line_index", 0), rec["user_id"], a_idx, len(attrs), c_idx_offset))

    total_prompts = len(prompts)
    print(f"  {total_prompts} prompts for {len(todo)} users.", flush=True)

    # ── Generate (all batched) ────────────────────────────────────────────────
    print(f"Generating...", flush=True)
    generations = llm.generate(prompts, sampling)

    # ── Reassemble per-user evidence lists ────────────────────────────────────
    # user_evidence[line_index] = list of size n_attrs, filled by attr_idx
    # When a user has multiple chunks, each chunk produces a result; we keep the first
    # non-null result (in chunk order) for each attribute.
    user_evidence: Dict[int, List[Optional[Dict]]] = {}
    user_id_map:   Dict[int, str] = {}

    debug_shown = 0
    for gen, (line_index, user_id, a_idx, n_attrs, c_idx_offset) in zip(generations, meta):
        if line_index not in user_evidence:
            user_evidence[line_index] = [{"conv_idx": None, "turn_idxs": []}] * n_attrs
            user_id_map[line_index] = user_id
        output = gen.outputs[0].text if gen.outputs else ""
        # Print first 3 raw outputs so we can see what the model is generating
        if debug_shown < 3:
            print(f"\n  [DEBUG sample {debug_shown+1}] raw output: {output[:300]!r}", flush=True)
            debug_shown += 1
        result = parse_single(output, debug=(debug_shown <= 3))
        # Apply conversation index offset (for chunked users)
        if result["conv_idx"] is not None:
            result = {"conv_idx": result["conv_idx"] + c_idx_offset, "turn_idxs": result["turn_idxs"]}
        # Only overwrite if we don't yet have a non-null result for this attr
        if user_evidence[line_index][a_idx]["conv_idx"] is None:
            user_evidence[line_index][a_idx] = result

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
