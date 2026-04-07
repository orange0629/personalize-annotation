#!/usr/bin/env python3
"""
For each (user, prompt) pair in the Task-2 checklist, classify whether each
profile attribute should be taken into consideration when personalizing the
response to the prompt.

Answer is YES if the attribute would influence the response in any way —
whether explicitly (e.g., the attribute is mentioned) or implicitly (e.g.,
tone, vocabulary, examples, or depth are adjusted). Answer is NO only if
the attribute has absolutely zero effect on the response.

Supported backends / models:
  openai    — e.g., gpt-4o
  anthropic — e.g., claude-3-5-sonnet-20241022
  vllm      — e.g., meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen3-72B

Each run writes to data/relevance/<model_tag>.jsonl (one JSON line per sample).
Re-running is safe — already-cached sample_indexes are skipped unless --force.

Output line format:
  {
    "model": "gpt-4o",
    "sample_index": N,
    "user_id": "...",
    "items": [
      {
        "attribute": "...",
        "relevant": true,
        "raw_answer": "YES"
      },
      ...
    ]
  }

Usage:
    python3 classify_attribute_relevance.py --backend openai --model gpt-4o

    python3 classify_attribute_relevance.py --backend anthropic \\
        --model claude-3-5-sonnet-20241022

    python3 classify_attribute_relevance.py --backend vllm \\
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

BASE_DIR       = Path(__file__).parent
CHECKLIST_FILE = BASE_DIR / "data" / "extracted" / "checklist_sample.jsonl"
RELEVANCE_DIR  = BASE_DIR / "data" / "relevance"

# ─── Prompt template ──────────────────────────────────────────────────────────

RELEVANCE_PROMPT = """You are helping evaluate a personalized AI assistant.

**Task:** Given a prompt sent by a user and an attribute from the user's profile, decide whether the attribute should be taken into consideration when personalizing the AI's response.

**Criteria:**
- Answer YES if the attribute would change the response in ANY way compared to a generic reply — whether explicitly (e.g., directly mentioning the attribute) or implicitly (e.g., adjusting tone, vocabulary, examples, level of detail, or framing).
- Answer NO if the attribute has no effect on how a thoughtful AI would respond — the response would be identical with or without knowing this attribute.

**Output Format:**
You MUST end your response with a line containing only:
Answer: YES
or
Answer: NO

User Attribute: {attribute}
Prompt: {prompt}""".strip()


def build_prompt(attribute: str, prompt_text: str) -> str:
    return RELEVANCE_PROMPT.format(attribute=attribute, prompt=prompt_text[:800])


# ─── Data helpers ──────────────────────────────────────────────────────────────

def load_checklist(max_samples: int, start_index: int = 0, end_index: Optional[int] = None) -> List[Dict]:
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

def parse_relevance(text: str) -> Tuple[bool, str]:
    """
    Extract a YES/NO relevance decision from model output.

    Handles reasoning models that produce long chains of thought before the
    final answer. Priority order:
      1. "Answer: YES/NO" line — our explicitly requested format;
         last occurrence wins so any mid-reasoning examples are ignored
      2. \\boxed{YES} / \\boxed{NO} — common in math-style reasoning models
      3. JSON with a "relevant" / "answer" key
      4. Bold **YES** / **NO** (last occurrence)
      5. Last standalone YES/NO word in the cleaned text

    Returns (relevant: bool, raw_answer: str).
    """
    # Strip thinking tokens (reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences
    if text.startswith("```"):
        m = re.search(r"```[a-zA-Z0-9]*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    # 1. "Answer: YES/NO" line — last occurrence wins
    matches = re.findall(r"(?i)\bAnswer\s*:\s*(YES|NO)\b", text)
    if matches:
        ans = matches[-1].upper()
        return ans == "YES", ans

    # 2. \boxed{YES} / \boxed{NO} — last occurrence wins
    matches = re.findall(r"\\boxed\s*\{?\s*(YES|NO)\s*\}?", text, re.IGNORECASE)
    if matches:
        ans = matches[-1].upper()
        return ans == "YES", ans

    # 3. JSON with a "relevant" / "answer" / "decision" key
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("relevant", "answer", "decision"):
                val = str(obj.get(key, "")).strip().upper()
                if val in ("YES", "TRUE", "1"):
                    return True, "YES"
                if val in ("NO", "FALSE", "0"):
                    return False, "NO"
    except Exception:
        pass

    m = re.search(r'"(?:relevant|answer|decision)"\s*:\s*"?(YES|NO)"?', text, re.IGNORECASE)
    if m:
        ans = m.group(1).upper()
        return ans == "YES", ans

    # 4. Bold **YES** / **NO** — last occurrence
    matches = re.findall(r"\*\*(YES|NO)\*\*", text, re.IGNORECASE)
    if matches:
        ans = matches[-1].upper()
        return ans == "YES", ans

    # 5. Last standalone YES/NO word
    words = re.findall(r"\b(YES|NO)\b", text, re.IGNORECASE)
    if words:
        ans = words[-1].upper()
        return ans == "YES", ans

    # Fallback: unknown — default to False
    snippet = text.strip()[:80].replace("\n", " ")
    return False, f"UNKNOWN:{snippet}"


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

    prompts: List[str] = []
    meta: List[Tuple[int, int]] = []  # (sample_index, item_index)

    for rec in records:
        sidx = rec["sample_index"]
        prompt_text = rec.get("prompt_text", "")
        for iidx, item in enumerate(rec.get("profile_attributes", [])):
            attr = item.get("attribute", "")
            raw_p = build_prompt(attr, prompt_text)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(chat)
            meta.append((sidx, iidx))

    print(f"Generating {len(prompts)} prompts ({len(records)} samples)...", flush=True)
    generations = llm.generate(prompts, sampling)

    sample_items: Dict[int, Dict[int, Dict]] = {}
    sample_uid: Dict[int, str] = {}
    for rec in records:
        sidx = rec["sample_index"]
        sample_uid[sidx] = rec.get("user_id", "")
        sample_items[sidx] = {
            iidx: {"attribute": item.get("attribute", ""), "relevant": False, "raw_answer": "", "full_output": ""}
            for iidx, item in enumerate(rec.get("profile_attributes", []))
        }

    debug_shown = 0
    for gen, (sidx, iidx) in zip(generations, meta):
        raw = gen.outputs[0].text if gen.outputs else ""
        if debug_shown < 4:
            print(f"  [DEBUG s={sidx} i={iidx}]: {raw[:300]!r}", flush=True)
            debug_shown += 1
        relevant, raw_answer = parse_relevance(raw)
        sample_items[sidx][iidx]["relevant"] = relevant
        sample_items[sidx][iidx]["raw_answer"] = raw_answer
        sample_items[sidx][iidx]["full_output"] = raw

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
            raw = call_fn(build_prompt(attr, prompt_text))
            relevant, raw_answer = parse_relevance(raw)
            items_out.append({
                "attribute": attr,
                "relevant": relevant,
                "raw_answer": raw_answer,
                "full_output": raw,
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
    return model.replace("/", "_").replace(":", "_").replace(" ", "_").lower()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether each profile attribute is relevant for personalizing "
            "the response to the corresponding prompt. "
            "Saves results to data/relevance/<model_tag>.jsonl."
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
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Use 0 for deterministic classification.")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens to generate. Keep high to allow reasoning before the answer.")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0,
                        help="First sample_index to process (inclusive).")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Last sample_index to process (exclusive).")
    parser.add_argument("--region", default="us-east-1",
                        help="(bedrock only) AWS region, e.g. us-east-1, us-west-2.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and regenerate all samples.")
    args = parser.parse_args()

    if not CHECKLIST_FILE.exists():
        print(f"ERROR: {CHECKLIST_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RELEVANCE_DIR / f"{model_tag(args.model)}.jsonl"

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
