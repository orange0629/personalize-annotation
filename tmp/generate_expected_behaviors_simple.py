#!/usr/bin/env python3
"""
Simplified version of generate_expected_behaviors.py.

Instead of asking for separate explicit and implicit behaviors, generates a
single expected_behavior per (attribute, prompt) pair.  The model is free to
decide whether the personalization should be explicit (directly referencing the
attribute) or implicit (adjusting tone, depth, framing, etc.) — whichever is
more natural given the prompt.

Output goes to data/behaviors_simple/<model_tag>.jsonl.

Output line format:
  {
    "model": "gpt-4o",
    "sample_index": N,
    "prompt_index": N,
    "user_id": "...",
    "items": [
      {
        "attribute": "...",
        "expected_behavior": "The response should ..."
      },
      ...
    ]
  }

Usage:
    python3 tmp/generate_expected_behaviors_simple.py --backend openai --model gpt-4o
    python3 tmp/generate_expected_behaviors_simple.py --backend anthropic \\
        --model claude-3-5-sonnet-20241022
    python3 tmp/generate_expected_behaviors_simple.py --backend vllm \\
        --model meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR       = Path(__file__).parent.parent
TASK2_FILE     = BASE_DIR / "data" / "extracted" / "task2_items.jsonl"
BEHAVIORS_DIR  = BASE_DIR / "data" / "behaviors_simple"

# ─── Prompt template ──────────────────────────────────────────────────────────

BEHAVIOR_PROMPT = """You are helping design a personalized AI assistant.

**Task:** Given a user attribute and a prompt they sent, write the expected behavior of an AI response that appropriately incorporates this attribute when personalizing its reply.

The personalization may be explicit (e.g., directly referencing the attribute: "Since you are X..." or "As someone who Y...") or implicit (e.g., subtly adjusting tone, vocabulary, examples, or depth to match the attribute without naming it) — choose whichever is more natural for this prompt and attribute.

The expected behavior must describe what specifically changes due to this attribute — not just how to answer the prompt in general. Ask yourself: what would be different compared to a generic reply, because of this attribute?

Output a single JSON object and nothing else (expected_behavior should be one concise sentence under 30 words):
{{"expected_behavior": "The response should ..."}}

User Attribute: {attribute}
Prompt: {prompt}""".strip()


def build_prompt(attribute: str, prompt_text: str) -> str:
    return BEHAVIOR_PROMPT.format(attribute=attribute, prompt=prompt_text[:800])


# ─── Data helpers ──────────────────────────────────────────────────────────────

def load_task2_items(max_samples: int, start_index: int = 0, end_index: Optional[int] = None) -> List[Dict]:
    """Load records from task2_items.jsonl, using line position as sample_index."""
    records = []
    with open(TASK2_FILE, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_idx < start_index:
                continue
            if end_index is not None and line_idx >= end_index:
                break
            rec = json.loads(line)
            rec["sample_index"] = line_idx
            records.append(rec)
            if len(records) >= max_samples:
                break
    return records


def load_cached_indexes(cache_file: Path) -> set:
    done: set = set()
    if not cache_file.exists():
        return done
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                idx = json.loads(line).get("sample_index")
                if idx is not None:
                    done.add(idx)
            except Exception:
                continue
    return done


# ─── Output parsing ────────────────────────────────────────────────────────────

def parse_behavior(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        m = re.search(r"```[a-zA-Z0-9]*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "expected_behavior" in obj:
            return obj["expected_behavior"].strip()
    except Exception:
        pass
    m = re.search(r'"expected_behavior"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        return m.group(1).replace('\\"', '"').strip()
    return text.strip()


# ─── vLLM backend ─────────────────────────────────────────────────────────────

def run_vllm(records, model, tensor_parallel_size, max_model_len, temperature, max_new_tokens) -> List[Dict]:
    from vllm import LLM, SamplingParams  # type: ignore

    print(f"Loading {model} via vLLM...", flush=True)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

    prompts: List[str] = []
    meta: List[Tuple[int, int]] = []  # (sample_index, attr_index)

    for rec in records:
        sidx = rec["sample_index"]
        prompt_text = rec.get("prompt_text", "")
        for iidx, item in enumerate(rec.get("profile_attributes", [])):
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(item.get("attribute", ""), prompt_text)}],
                tokenize=False, add_generation_prompt=True,
            )
            prompts.append(chat)
            meta.append((sidx, iidx))

    print(f"Generating {len(prompts)} prompts ({len(records)} samples)...", flush=True)
    generations = llm.generate(prompts, sampling)

    sample_items: Dict[int, Dict[int, Dict]] = {}
    sample_uid: Dict[int, str] = {}
    sample_pidx: Dict[int, int] = {}
    for rec in records:
        sidx = rec["sample_index"]
        sample_uid[sidx] = rec.get("user_id", "")
        sample_pidx[sidx] = rec.get("prompt_index", -1)
        sample_items[sidx] = {
            iidx: {"attribute": item.get("attribute", ""), "expected_behavior": ""}
            for iidx, item in enumerate(rec.get("profile_attributes", []))
        }

    debug_shown = 0
    for gen, (sidx, iidx) in zip(generations, meta):
        raw = gen.outputs[0].text if gen.outputs else ""
        if debug_shown < 4:
            print(f"  [DEBUG s={sidx} i={iidx}]: {raw[:200]!r}", flush=True)
            debug_shown += 1
        sample_items[sidx][iidx]["expected_behavior"] = parse_behavior(raw)

    return [
        {
            "model": model,
            "sample_index": sidx,
            "prompt_index": sample_pidx[sidx],
            "user_id": sample_uid[sidx],
            "items": [sample_items[sidx][i] for i in sorted(sample_items[sidx])],
        }
        for sidx in sorted(sample_uid)
    ]


# ─── OpenAI backend ───────────────────────────────────────────────────────────

def run_openai(records, model, temperature, max_new_tokens, retry_delay=2.0, max_retries=5) -> List[Dict]:
    import openai  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)

    def call(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_new_tokens,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"  OpenAI error ({e}), giving up.", file=sys.stderr)
                    return ""
        return ""

    return _run_api(records, model, call)


# ─── Anthropic backend ────────────────────────────────────────────────────────

def run_anthropic(records, model, temperature, max_new_tokens, retry_delay=2.0, max_retries=5) -> List[Dict]:
    import anthropic  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    def call(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(
                    model=model, max_tokens=max_new_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text if resp.content else ""
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"  Anthropic error ({e}), giving up.", file=sys.stderr)
                    return ""
        return ""

    return _run_api(records, model, call)


def _run_api(records, model, call_fn) -> List[Dict]:
    results = []
    total = sum(len(r.get("profile_attributes", [])) for r in records)
    done = 0
    for rec in records:
        sidx = rec["sample_index"]
        prompt_text = rec.get("prompt_text", "")
        items_out = []
        for item in rec.get("profile_attributes", []):
            attr = item.get("attribute", "")
            items_out.append({
                "attribute": attr,
                "expected_behavior": parse_behavior(call_fn(build_prompt(attr, prompt_text))),
            })
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] sample {sidx}", flush=True)
        results.append({
            "model": model,
            "sample_index": sidx,
            "prompt_index": rec.get("prompt_index", -1),
            "user_id": rec.get("user_id", ""),
            "items": items_out,
        })
    return results


# ─── Bedrock backend ─────────────────────────────────────────────────────────

def run_bedrock(records, model, region, temperature, max_new_tokens, retry_delay=2.0, max_retries=5) -> List[Dict]:
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore
    from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore

    cfg = Config(connect_timeout=30, read_timeout=300,
                 retries={"max_attempts": max_retries, "mode": "standard"})
    try:
        client = boto3.client("bedrock-runtime", region_name=region, config=cfg)
    except NoCredentialsError:
        print("ERROR: AWS credentials not found.", file=sys.stderr)
        sys.exit(1)

    def call(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                resp = client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": max_new_tokens, "temperature": temperature},
                )
                content = resp["output"]["message"]["content"]
                return "".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "Unknown")
                msg  = e.response.get("Error", {}).get("Message", str(e))
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    print(f"  Bedrock error ({code}: {msg}), giving up.", file=sys.stderr)
                    return ""
        return ""

    return _run_api(records, model, call)


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_").lower()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a single expected behavior per attribute (no explicit/implicit split)."
    )
    parser.add_argument("--backend", required=True, choices=["openai", "anthropic", "vllm", "bedrock"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not TASK2_FILE.exists():
        print(f"ERROR: {TASK2_FILE} not found. Run: python3 build_index.py", file=sys.stderr)
        sys.exit(1)

    BEHAVIORS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = BEHAVIORS_DIR / f"{model_tag(args.model)}.jsonl"

    range_desc = f"[{args.start_index}, {args.end_index if args.end_index is not None else '∞'})"
    print(f"Loading task2 items (max {args.max_samples}, range {range_desc})...", flush=True)
    all_records = load_task2_items(args.max_samples, start_index=args.start_index, end_index=args.end_index)
    print(f"  Loaded {len(all_records)} samples.", flush=True)

    already_done = set() if args.force else load_cached_indexes(cache_file)
    todo = [r for r in all_records if r["sample_index"] not in already_done]
    print(f"  {len(already_done)} cached, {len(todo)} to generate → {cache_file.name}", flush=True)

    if not todo:
        print("Nothing to do. Use --force to regenerate.", flush=True)
        return

    if args.force and cache_file.exists():
        cache_file.unlink()

    if args.backend == "vllm":
        results = run_vllm(todo, args.model, args.tensor_parallel_size, args.max_model_len,
                           args.temperature, args.max_new_tokens)
    elif args.backend == "openai":
        results = run_openai(todo, args.model, args.temperature, args.max_new_tokens)
    elif args.backend == "anthropic":
        results = run_anthropic(todo, args.model, args.temperature, args.max_new_tokens)
    elif args.backend == "bedrock":
        results = run_bedrock(todo, args.model, args.region, args.temperature, args.max_new_tokens)

    with open(cache_file, "a", encoding="utf-8") as out:
        for rec in results:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)} samples written → {cache_file}", flush=True)


if __name__ == "__main__":
    main()
