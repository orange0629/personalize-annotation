#!/usr/bin/env python3
"""
For each (user, prompt) pair in the Task-2 checklist, generate expected behavior
descriptions for EXPLICIT and IMPLICIT attribute incorporation.

Supported backends / models:
  openai    — e.g., gpt-4o
  anthropic — e.g., claude-3-5-sonnet-20241022
  vllm      — e.g., meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen3-72B

Each run writes to data/behaviors/<model_tag>.jsonl (one JSON line per sample).
Re-running is safe — already-cached sample_indexes are skipped unless --force is used.
To load all results later, simply read every *.jsonl file from data/behaviors/.

Output line format:
  {
    "model": "gpt-4o",
    "sample_index": N,
    "user_id": "...",
    "items": [
      {
        "attribute": "...",
        "explicit_behavior": "The response should ...",
        "implicit_behavior": "The response should ..."
      },
      ...
    ]
  }

Usage:
    python3 generate_expected_behaviors.py --backend openai --model gpt-4o

    python3 generate_expected_behaviors.py --backend anthropic \\
        --model claude-3-5-sonnet-20241022

    python3 generate_expected_behaviors.py --backend vllm \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --tensor-parallel-size 4
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

BASE_DIR        = Path(__file__).parent
CHECKLIST_FILE  = BASE_DIR / "data" / "extracted" / "checklist_sample.jsonl"
BEHAVIORS_DIR   = BASE_DIR / "data" / "behaviors"

# ─── Prompt templates ─────────────────────────────────────────────────────────

EXPLICIT_PROMPT = """You are helping design a personalized AI assistant.

**Task:** Given a user attribute and a prompt they sent, write the expected behavior of an AI response that *explicitly* incorporates this attribute.

**Explicit incorporation** means the response directly acknowledges or references the user's attribute. The personalization is visible — for example, "Since you are X...", "As someone who Y...", or "Given your background in Z...". The attribute is named or paraphrased in the response.

The expected behavior must describe what specifically changes due to this attribute — not just how to answer the prompt in general. Ask yourself: what would be different compared to a generic reply, because of this attribute?

Output a single JSON object and nothing else (expected_behavior should be one concise sentence under 30 words):
{{"expected_behavior": "The response should ..."}}

User Attribute: {attribute}
Prompt: {prompt}""".strip()

IMPLICIT_PROMPT = """You are helping design a personalized AI assistant.

**Task:** Given a user attribute and a prompt they sent, write the expected behavior of an AI response that *implicitly* incorporates this attribute.

**Implicit incorporation** means the response subtly adjusts its tone, vocabulary, examples, depth, or focus to align with the user's attribute WITHOUT ever directly naming or referencing it. The user experiences a better-fitted response but cannot tell why — the adjustment is invisible; only the effect is felt.

The expected behavior must describe what specifically changes due to this attribute — not just how to answer the prompt in general. Ask yourself: what would be different compared to a generic reply, because of this attribute?

Output a single JSON object and nothing else (expected_behavior should be one concise sentence under 30 words):
{{"expected_behavior": "The response should ..."}}

User Attribute: {attribute}
Prompt: {prompt}""".strip()


def build_prompts(attribute: str, prompt_text: str) -> Tuple[str, str]:
    """Return (explicit_prompt, implicit_prompt) for one (attribute, prompt) pair."""
    return (
        EXPLICIT_PROMPT.format(attribute=attribute, prompt=prompt_text[:800]),
        IMPLICIT_PROMPT.format(attribute=attribute, prompt=prompt_text[:800]),
    )


# ─── Data helpers ──────────────────────────────────────────────────────────────

def load_checklist(max_samples: int, start_index: int = 0, end_index: Optional[int] = None) -> List[Dict]:
    """Load checklist records with optional sample_index range [start_index, end_index)."""
    records = []
    with open(CHECKLIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sidx = rec.get("sample_index", 0)
            if sidx < start_index:
                continue
            if end_index is not None and sidx >= end_index:
                continue
            records.append(rec)
            if len(records) >= max_samples:
                break
    records.sort(key=lambda r: r.get("sample_index", 0))
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
    """
    Extract the 'expected_behavior' string from model output.
    The model is asked to return {"expected_behavior": "The response should ..."}.
    Falls back to the raw cleaned text if JSON parsing fails.
    """
    # Strip thinking tokens (reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    if text.startswith("```"):
        m = re.search(r"```[a-zA-Z0-9]*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # Try full JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "expected_behavior" in obj:
            return obj["expected_behavior"].strip()
    except Exception:
        pass
    # Try regex extraction
    m = re.search(r'"expected_behavior"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        return m.group(1).replace('\\"', '"').strip()
    return text.strip()


# ─── vLLM backend ─────────────────────────────────────────────────────────────

def run_vllm(
    records: List[Dict],
    model: str,
    tensor_parallel_size: int,
    max_model_len: int,
    temperature: float,
    max_new_tokens: int,
) -> List[Dict]:
    from vllm import LLM, SamplingParams  # type: ignore

    print(f"Loading {model} via vLLM...", flush=True)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

    # Build all prompts; meta[i] = (sample_index, item_index, impact_type)
    prompts: List[str] = []
    meta: List[Tuple[int, int, str]] = []

    for rec in records:
        sidx = rec["sample_index"]
        prompt_text = rec.get("prompt_text", "")
        for iidx, item in enumerate(rec.get("profile_attributes", [])):
            attr = item.get("attribute", "")
            exp_p, imp_p = build_prompts(attr, prompt_text)
            for impact_type, raw_p in [("explicit", exp_p), ("implicit", imp_p)]:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(chat)
                meta.append((sidx, iidx, impact_type))

    print(f"Generating {len(prompts)} prompts ({len(records)} samples)...", flush=True)
    generations = llm.generate(prompts, sampling)

    # Assemble
    sample_items: Dict[int, Dict[int, Dict]] = {}
    sample_uid: Dict[int, str] = {}
    for rec in records:
        sidx = rec["sample_index"]
        sample_uid[sidx] = rec.get("user_id", "")
        sample_items[sidx] = {
            iidx: {"attribute": item.get("attribute", ""), "explicit_behavior": "", "implicit_behavior": ""}
            for iidx, item in enumerate(rec.get("profile_attributes", []))
        }

    debug_shown = 0
    for gen, (sidx, iidx, impact_type) in zip(generations, meta):
        raw = gen.outputs[0].text if gen.outputs else ""
        if debug_shown < 4:
            print(f"  [DEBUG {impact_type} s={sidx} i={iidx}]: {raw[:200]!r}", flush=True)
            debug_shown += 1
        sample_items[sidx][iidx][f"{impact_type}_behavior"] = parse_behavior(raw)

    return [
        {
            "model": model,
            "sample_index": sidx,
            "user_id": sample_uid[sidx],
            "items": [sample_items[sidx][i] for i in sorted(sample_items[sidx])],
        }
        for sidx in sorted(sample_uid)
    ]


# ─── OpenAI backend ───────────────────────────────────────────────────────────

def run_openai(
    records: List[Dict],
    model: str,
    temperature: float,
    max_new_tokens: int,
    retry_delay: float = 2.0,
    max_retries: int = 5,
) -> List[Dict]:
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
                    wait = retry_delay * (2 ** attempt)
                    print(f"  OpenAI error ({e}), retry in {wait:.1f}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  OpenAI error ({e}), giving up.", file=sys.stderr)
                    return ""
        return ""

    return _run_api(records, model, call)


# ─── Anthropic backend ────────────────────────────────────────────────────────

def run_anthropic(
    records: List[Dict],
    model: str,
    temperature: float,
    max_new_tokens: int,
    retry_delay: float = 2.0,
    max_retries: int = 5,
) -> List[Dict]:
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
                    model=model,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text if resp.content else ""
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  Anthropic error ({e}), retry in {wait:.1f}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  Anthropic error ({e}), giving up.", file=sys.stderr)
                    return ""
        return ""

    return _run_api(records, model, call)


def _run_api(records: List[Dict], model: str, call_fn) -> List[Dict]:
    """Shared sequential loop for API-based backends."""
    results = []
    total = sum(len(r.get("profile_attributes", [])) for r in records)
    done = 0
    for rec in records:
        sidx = rec["sample_index"]
        prompt_text = rec.get("prompt_text", "")
        items_out = []
        for item in rec.get("profile_attributes", []):
            attr = item.get("attribute", "")
            exp_p, imp_p = build_prompts(attr, prompt_text)
            items_out.append({
                "attribute": attr,
                "explicit_behavior": parse_behavior(call_fn(exp_p)),
                "implicit_behavior": parse_behavior(call_fn(imp_p)),
            })
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] sample {sidx}", flush=True)
        results.append({
            "model": model,
            "sample_index": sidx,
            "user_id": rec.get("user_id", ""),
            "items": items_out,
        })
    return results


# ─── Bedrock backend ─────────────────────────────────────────────────────────

def run_bedrock(
    records: List[Dict],
    model: str,
    region: str,
    temperature: float,
    max_new_tokens: int,
    retry_delay: float = 2.0,
    max_retries: int = 5,
) -> List[Dict]:
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore
    from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore

    cfg = Config(
        connect_timeout=30,
        read_timeout=300,
        retries={"max_attempts": max_retries, "mode": "standard"},
    )
    try:
        client = boto3.client("bedrock-runtime", region_name=region, config=cfg)
    except NoCredentialsError:
        print("ERROR: AWS credentials not found. Configure AWS_PROFILE or "
              "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    def call(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                resp = client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "maxTokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
                content = resp["output"]["message"]["content"]
                return "".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "Unknown")
                msg  = e.response.get("Error", {}).get("Message", str(e))
                if attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  Bedrock error ({code}: {msg}), retry in {wait:.1f}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  Bedrock error ({code}: {msg}), giving up.", file=sys.stderr)
                    return ""
        return ""

    return _run_api(records, model, call)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def model_tag(model: str) -> str:
    """Convert a model name to a filesystem-safe tag (used as the output filename)."""
    return model.replace("/", "_").replace(":", "_").replace(" ", "_").lower()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate explicit/implicit expected behaviors for Task-2 annotation. "
            "Saves results to data/behaviors/<model_tag>.jsonl."
        )
    )
    parser.add_argument("--backend", required=True, choices=["openai", "anthropic", "vllm", "bedrock"])
    parser.add_argument(
        "--model", required=True,
        help=(
            "Model ID. E.g.: gpt-4o, claude-3-5-sonnet-20241022, "
            "meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen3-72B"
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="(vLLM only) tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="(vLLM only) max model context length")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0,
                        help="First sample_index to process (inclusive). Useful for debugging.")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Last sample_index to process (exclusive). E.g. --end-index 10 processes samples 0-9.")
    parser.add_argument("--region", default="us-east-1",
                        help="(bedrock only) AWS region, e.g. us-east-1, us-west-2.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and regenerate all samples.")
    args = parser.parse_args()

    if not CHECKLIST_FILE.exists():
        print(f"ERROR: {CHECKLIST_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    BEHAVIORS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = BEHAVIORS_DIR / f"{model_tag(args.model)}.jsonl"

    range_desc = f"[{args.start_index}, {args.end_index if args.end_index is not None else '∞'})"
    print(f"Loading checklist (max {args.max_samples} samples, index range {range_desc})...", flush=True)
    all_records = load_checklist(args.max_samples, start_index=args.start_index, end_index=args.end_index)
    print(f"  Loaded {len(all_records)} samples.", flush=True)

    already_done = set() if args.force else load_cached_indexes(cache_file)
    todo = [r for r in all_records if r.get("sample_index") not in already_done]
    print(f"  {len(already_done)} cached, {len(todo)} to generate → {cache_file.name}", flush=True)

    if not todo:
        print("Nothing to do. Use --force to regenerate.", flush=True)
        return

    if args.force and cache_file.exists():
        cache_file.unlink()

    if args.backend == "vllm":
        results = run_vllm(
            todo, args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.backend == "openai":
        results = run_openai(
            todo, args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.backend == "anthropic":
        results = run_anthropic(
            todo, args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.backend == "bedrock":
        results = run_bedrock(
            todo, args.model,
            region=args.region,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    with open(cache_file, "a", encoding="utf-8") as out:
        for rec in results:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)} samples written → {cache_file}", flush=True)


if __name__ == "__main__":
    main()
