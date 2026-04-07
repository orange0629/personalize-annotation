#!/usr/bin/env python3
"""
Overall agreement analysis for Task 3 (attribute relevance classification).

Pools every (attribute, sample_index) judgment from:
  data/relevance/*.jsonl          — LLM model votes
  data/annotations/*_task3.jsonl — human annotator votes

For every pair of raters, computes:
  • % agreement on shared items
  • Cohen's kappa

Also computes Krippendorff's alpha across all raters and a YES-rate summary.

Usage:
    python3 analyze_relevance_agreement.py
    python3 analyze_relevance_agreement.py --skip-admin
    python3 analyze_relevance_agreement.py --output report.txt
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR      = Path(__file__).parent
RELEVANCE_DIR = BASE_DIR / "data" / "relevance"
ANNOT_DIR     = BASE_DIR / "data" / "annotations"

Key = Tuple[str, int]  # (attribute_text, sample_index)


# ─── Loaders ───────────────────────────────────────────────────────────────────

def load_model_votes(relevance_dir: Path = RELEVANCE_DIR) -> Dict[str, Dict[Key, bool]]:
    raters: Dict[str, Dict[Key, bool]] = {}
    if not relevance_dir.exists():
        return raters
    for fpath in sorted(relevance_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec  = json.loads(line)
                name = rec.get("model", fpath.stem)
                sidx = rec.get("sample_index")
                raters.setdefault(name, {})
                for item in rec.get("items", []):
                    attr = item.get("attribute", "")
                    rel  = item.get("relevant")
                    if attr and rel is not None:
                        raters[name][(attr, sidx)] = bool(rel)
    return raters


def load_human_votes(annot_dir: Path = ANNOT_DIR,
                     skip_admin: bool = False) -> Dict[str, Dict[Key, bool]]:
    raters: Dict[str, Dict[Key, bool]] = {}
    if not annot_dir.exists():
        return raters
    for fpath in sorted(annot_dir.glob("*_task3.jsonl")):
        name = fpath.name[: -len("_task3.jsonl")]
        if skip_admin and name.lower() == "admin":
            continue
        votes: Dict[Key, bool] = {}
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec  = json.loads(line)
                sidx = rec.get("index")
                for j in rec.get("relevance_judgments", []):
                    attr = j.get("attribute", "")
                    rel  = j.get("relevant")
                    if attr and rel is not None:
                        votes[(attr, sidx)] = bool(rel)
        if votes:
            raters[name] = votes
    return raters


# ─── Metrics ───────────────────────────────────────────────────────────────────

def pairwise(a: Dict[Key, bool], b: Dict[Key, bool]) -> dict:
    shared = [k for k in a if k in b]
    n = len(shared)
    if n == 0:
        return dict(n=0, agree=None, kappa=None, yes_a=None, yes_b=None)
    av = [a[k] for k in shared]
    bv = [b[k] for k in shared]
    p_o   = sum(x == y for x, y in zip(av, bv)) / n
    p_a   = sum(av) / n
    p_b   = sum(bv) / n
    p_e   = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) else float("nan")
    return dict(n=n, agree=p_o, kappa=kappa, yes_a=p_a, yes_b=p_b)


def krippendorff_alpha(all_votes: Dict[str, Dict[Key, bool]]) -> Optional[float]:
    """Binary nominal Krippendorff's alpha."""
    key_vals: Dict[Key, List[bool]] = defaultdict(list)
    for votes in all_votes.values():
        for k, v in votes.items():
            key_vals[k].append(v)

    disag = n_pairs = yes_total = no_total = 0
    for vals in key_vals.values():
        if len(vals) < 2:
            continue
        m = len(vals)
        for i in range(m):
            for j in range(i + 1, m):
                n_pairs += 1
                if vals[i] != vals[j]:
                    disag += 1
        yes_total += sum(vals)
        no_total  += m - sum(vals)

    if n_pairs == 0:
        return None
    d_o   = disag / n_pairs
    total = yes_total + no_total
    if total < 2:
        return None
    p_yes = yes_total / total
    d_e   = 2 * p_yes * (1 - p_yes)
    if d_e == 0:
        return 1.0 if d_o == 0 else float("nan")
    return 1.0 - d_o / d_e


def build_agreement_data(model_votes: Dict[str, Dict[Key, bool]],
                         human_votes: Dict[str, Dict[Key, bool]]) -> dict:
    """
    Compute all agreement statistics and return a structured dict
    suitable for both text reporting and HTML rendering.
    """
    all_raters = {
        **{f"[M] {k}": v for k, v in model_votes.items()},
        **{f"[H] {k}": v for k, v in human_votes.items()},
    }
    model_tags = [f"[M] {k}" for k in model_votes]
    human_tags = [f"[H] {k}" for k in human_votes]

    all_keys: set = set()
    for v in all_raters.values():
        all_keys |= v.keys()

    # Per-rater YES rates
    yes_rates = []
    for tag, votes in sorted(all_raters.items()):
        n = len(votes)
        ny = sum(votes.values())
        yes_rates.append({
            "rater": tag,
            "yes_rate": ny / n if n else None,
            "n": n,
            "is_model": tag.startswith("[M]"),
        })

    # Pairwise
    def pairs_stats(pairs):
        rows = []
        for a, b in pairs:
            s = pairwise(all_raters[a], all_raters[b])
            rows.append({"rater_a": a, "rater_b": b, **s})
        return rows

    mm_pairs = list(combinations(model_tags, 2))
    hh_pairs = list(combinations(human_tags, 2))
    mh_pairs = [(m, h) for m in model_tags for h in human_tags]

    def avg_stats(rows):
        valid = [r for r in rows if r["agree"] is not None]
        if not valid:
            return None, None
        ag = sum(r["agree"] for r in valid) / len(valid)
        ks = [r["kappa"] for r in valid if r["kappa"] is not None and not math.isnan(r["kappa"])]
        ak = sum(ks) / len(ks) if ks else None
        return ag, ak

    mm_rows = pairs_stats(mm_pairs)
    hh_rows = pairs_stats(hh_pairs)
    mh_rows = pairs_stats(mh_pairs)
    all_rows = pairs_stats(list(combinations(list(all_raters.keys()), 2)))

    summary = []
    for label, rows in [("Model × Model", mm_rows), ("Human × Human", hh_rows),
                         ("Model × Human", mh_rows), ("All pairs", all_rows)]:
        if not rows:
            continue
        ag, ak = avg_stats(rows)
        summary.append({"group": label, "avg_agree": ag, "avg_kappa": ak, "n_pairs": len(rows)})

    # Krippendorff's alpha
    alphas = {
        "models": krippendorff_alpha(model_votes),
        "humans": krippendorff_alpha(human_votes) if human_votes else None,
        "all":    krippendorff_alpha({**model_votes, **human_votes}),
    }

    # Per-attribute disagreement (most contested)
    key_votes_map: Dict[Key, Dict[str, bool]] = defaultdict(dict)
    for tag, votes in all_raters.items():
        for k, v in votes.items():
            key_votes_map[k][tag] = v

    contested = []
    for key, vd in key_votes_map.items():
        vals = list(vd.values())
        if len(vals) < 2:
            continue
        ny = sum(vals)
        nn = len(vals) - ny
        split = min(ny, nn) / len(vals)
        contested.append({
            "attribute": key[0],
            "sample_index": key[1],
            "split": split,
            "n_yes": ny,
            "n_no": nn,
            "n_total": len(vals),
            "yes_raters": [t for t, v in vd.items() if v],
            "no_raters":  [t for t, v in vd.items() if not v],
        })
    contested.sort(key=lambda x: x["split"], reverse=True)

    return {
        "n_raters": len(all_raters),
        "n_models": len(model_votes),
        "n_humans": len(human_votes),
        "n_total_keys": len(all_keys),
        "model_names": list(model_votes.keys()),
        "human_names": list(human_votes.keys()),
        "yes_rates": yes_rates,
        "pairwise": {
            "model_model": mm_rows,
            "human_human": hh_rows,
            "model_human": mh_rows,
        },
        "summary": summary,
        "alphas": alphas,
        "contested": contested[:30],
    }


# ─── Text report ───────────────────────────────────────────────────────────────

def _pct(v): return f"{v*100:.1f}%" if v is not None else "n/a"
def _k(v):
    if v is None: return "n/a"
    if math.isnan(v): return "NaN"
    return f"{v:+.3f}"
def _s(s, n=32): return s if len(s) <= n else "…" + s[-(n-1):]


def text_report(data: dict) -> str:
    out: List[str] = []
    W = 80

    def rule(c="─"): out.append(c * W)
    def h1(t): rule("═"); out.append(t); rule("═")
    def h2(t): out.append(""); out.append(f"── {t}"); rule()

    h1("TASK 3 — RELEVANCE AGREEMENT REPORT")
    out.append(f"  Models ({data['n_models']}):     {', '.join(data['model_names'])}")
    out.append(f"  Annotators ({data['n_humans']}): {', '.join(data['human_names']) or 'none'}")
    out.append(f"  Total (attr, sample) pairs seen by any rater: {data['n_total_keys']:,}")

    h2("YES RATE PER RATER")
    C = 36
    out.append(f"  {'Rater':<{C}}  {'YES rate':>9}  {'# labeled':>10}")
    out.append(f"  {'-'*C}  {'-'*9}  {'-'*10}")
    for r in data["yes_rates"]:
        out.append(f"  {_s(r['rater'],C):<{C}}  {_pct(r['yes_rate']):>9}  {r['n']:>10,}")

    h2("PAIRWISE AGREEMENT  ( % agree  |  Cohen's κ  |  N shared )")
    hdr = f"  {'Rater A':<{C}}  {'Rater B':<{C}}  {'Agree':>7}  {'κ':>7}  {'N':>7}"
    sep = f"  {'-'*C}  {'-'*C}  {'-'*7}  {'-'*7}  {'-'*7}"

    section_map = [
        ("Model × Model", data["pairwise"]["model_model"]),
        ("Human × Human", data["pairwise"]["human_human"]),
        ("Model × Human", data["pairwise"]["model_human"]),
    ]
    for title, rows in section_map:
        if not rows:
            continue
        out.append(f"\n  {title}")
        out.append(hdr); out.append(sep)
        for r in rows:
            out.append(
                f"  {_s(r['rater_a'],C):<{C}}  {_s(r['rater_b'],C):<{C}}"
                f"  {_pct(r['agree']):>7}  {_k(r['kappa']):>7}  {r['n']:>7,}"
            )

    h2("SUMMARY")
    out.append(f"  {'Group':<22}  {'Avg agree':>9}  {'Avg κ':>7}  {'Pairs':>6}")
    out.append(f"  {'-'*22}  {'-'*9}  {'-'*7}  {'-'*6}")
    for s in data["summary"]:
        out.append(
            f"  {s['group']:<22}  {_pct(s['avg_agree']):>9}"
            f"  {_k(s['avg_kappa']):>7}  {s['n_pairs']:>6}"
        )
    a = data["alphas"]
    out.append(f"\n  Krippendorff's α  (nominal, items with ≥2 raters)")
    out.append(f"    Models only:  {_k(a['models'])}")
    out.append(f"    Humans only:  {_k(a['humans'])}")
    out.append(f"    All raters:   {_k(a['all'])}")

    h2("MOST CONTESTED ATTRIBUTES  (top 20 by split)")
    shown = 0
    for c in data["contested"]:
        if shown >= 20 or c["split"] == 0:
            break
        out.append(
            f"\n  [{c['split']*100:.0f}% split | sample {c['sample_index']} | "
            f"YES {c['n_yes']}/{c['n_total']}]"
        )
        out.append(f"  {c['attribute'][:110]}")
        out.append(f"  YES: {', '.join(_s(r,22) for r in c['yes_raters'])}")
        out.append(f"  NO:  {', '.join(_s(r,22) for r in c['no_raters'])}")
        shown += 1
    if shown == 0:
        out.append("  (no disagreements found)")

    out.append("")
    return "\n".join(out)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task 3 relevance agreement report.")
    parser.add_argument("--skip-admin", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_votes = load_model_votes()
    human_votes = load_human_votes(skip_admin=args.skip_admin)

    if not model_votes and not human_votes:
        import sys
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    data = build_agreement_data(model_votes, human_votes)
    text = text_report(data)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
