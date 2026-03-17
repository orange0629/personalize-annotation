#!/usr/bin/env python3
"""
Inter-annotator agreement calculator for annotation_tool.

Task 1: per-attribute judgments  ("ok" / "unsure" / "bad")
Task 2: per-attribute ratings    (e.g. "necessary_explicit" / "necessary_implicit" /
                                   "optional" / "unnecessary" / "forbidden")

Metrics
-------
- Raw agreement (% exact match)
- Cohen's κ (pairwise, unweighted and linearly-weighted)
- Fleiss' κ  (multi-annotator, ≥ 3 annotators)

Usage
-----
  python calc_agreement.py [--dir data/annotations/annotations] [--task 1|2|both]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False
    print("[warn] numpy not found – Fleiss' κ will be skipped", file=sys.stderr)

try:
    from sklearn.metrics import cohen_kappa_score
    _HAS_SK = True
except ImportError:
    _HAS_SK = False
    print("[warn] scikit-learn not found – falling back to manual Cohen's κ", file=sys.stderr)


# ── Label orderings (for weighted κ) ─────────────────────────────────────────
TASK1_ORDER = ["ok", "unsure", "bad"]
TASK2_ORDER = [
    "necessary_explicit",
    "necessary_implicit",
    "optional",
    "unnecessary",
    "forbidden",
]
# New-scale aliases (map to canonical names for comparison)
TASK2_ALIASES: dict[str, str] = {
    "explicitly_apply":  "necessary_explicit",
    "implicitly_apply":  "necessary_implicit",
    "low_impact":        "optional",
    "irrelevant":        "unnecessary",
    "avoid":             "forbidden",
}

SKIP_RATING = {"none", "", None}


# ── Cohen's κ (manual fallback + weighted) ───────────────────────────────────

def _cohen_kappa_manual(a: list, b: list) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    label2i = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for ai, bi in zip(a, b):
        conf[label2i[ai]][label2i[bi]] += 1
    p_o = sum(conf[i][i] for i in range(k)) / n
    p_e = sum(
        (sum(conf[i][j] for j in range(k)) / n) * (sum(conf[j][i] for j in range(k)) / n)
        for i in range(k)
    )
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def _cohen_kappa_weighted(a: list, b: list, order: list[str]) -> float:
    """Linear-weighted Cohen's κ using the provided label ordering."""
    if not _HAS_NP:
        return float("nan")
    present = sorted(set(a) | set(b))
    # Assign positions based on order; unknowns go to end
    def pos(x: str) -> int:
        try:
            return order.index(x)
        except ValueError:
            return len(order)

    max_dist = max(abs(pos(x) - pos(y)) for x in present for y in present) or 1
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(present, key=pos)
    k = len(labels)
    label2i = {l: i for i, l in enumerate(labels)}

    conf = np.zeros((k, k), dtype=float)
    for ai, bi in zip(a, b):
        conf[label2i[ai]][label2i[bi]] += 1
    conf /= n

    weights = np.zeros((k, k))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            weights[i][j] = abs(pos(li) - pos(lj)) / max_dist

    row_sum = conf.sum(axis=1)
    col_sum = conf.sum(axis=0)
    expected = np.outer(row_sum, col_sum)

    p_o = 1.0 - (conf * weights).sum()
    p_e = 1.0 - (expected * weights).sum()
    return (p_o - p_e) / p_e if p_e != 0 else 1.0


def cohen_kappa(a: list, b: list, weighted: bool = False, order: list[str] | None = None) -> float:
    if weighted and order and _HAS_NP:
        return _cohen_kappa_weighted(a, b, order)
    if _HAS_SK:
        try:
            labels = sorted(set(a) | set(b))
            return cohen_kappa_score(a, b, labels=labels)
        except Exception:
            pass
    return _cohen_kappa_manual(a, b)


# ── Fleiss' κ ────────────────────────────────────────────────────────────────

def fleiss_kappa(ratings_matrix: "np.ndarray") -> float:
    """
    ratings_matrix : shape (N_items, N_labels)  — each cell = #raters who chose that label.
    """
    if not _HAS_NP:
        return float("nan")
    n_items, n_labels = ratings_matrix.shape
    n_raters = ratings_matrix[0].sum()
    if n_raters <= 1:
        return float("nan")

    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


# ── Data loading ─────────────────────────────────────────────────────────────

def load_annotations(ann_dir: Path, task: str) -> dict[str, list[dict]]:
    """Returns {annotator_name: [records…]} for the given task."""
    out: dict[str, list[dict]] = {}
    for path in sorted(ann_dir.glob(f"*_task{task}.jsonl")):
        annotator = path.stem.replace(f"_task{task}", "")
        records = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            out[annotator] = records
    return out


# ── Task-1 agreement ─────────────────────────────────────────────────────────

def task1_pairs(all_ann: dict[str, list[dict]]) -> dict[tuple, tuple[list, list]]:
    """
    Returns {(index, attr_pos): {annotator: label, …}} keyed by item.
    Then we extract pairwise lists for agreement computation.
    """
    # item_votes[(index, attr_pos)] = {annotator: label}
    item_votes: dict[tuple, dict[str, str]] = defaultdict(dict)
    for annotator, records in all_ann.items():
        for rec in records:
            for i, j in enumerate(rec.get("attr_judgments") or []):
                label = j.get("judgment")
                if label and label not in SKIP_RATING:
                    item_votes[(rec["index"], i)][annotator] = label
    return item_votes


def compute_task1(ann_dir: Path) -> None:
    print("=" * 60)
    print("TASK 1  — Attribute Judgment Agreement")
    print("=" * 60)

    all_ann = load_annotations(ann_dir, "1")
    if len(all_ann) < 2:
        print(f"  Only {len(all_ann)} annotator(s) for Task 1 — need ≥ 2 for agreement.\n")
        return

    annotators = sorted(all_ann.keys())
    print(f"  Annotators : {', '.join(annotators)}")

    item_votes = task1_pairs(all_ann)

    # Pairwise
    print(f"\n  {'Pair':<30}  {'N':>5}  {'Agree%':>7}  {'κ':>7}  {'κ_w':>7}")
    print("  " + "-" * 55)
    for a1, a2 in combinations(annotators, 2):
        seqs: list[tuple[str, str]] = [
            (votes[a1], votes[a2])
            for votes in item_votes.values()
            if a1 in votes and a2 in votes
        ]
        if not seqs:
            print(f"  {a1} vs {a2:<20}  {'0':>5}  {'—':>7}  {'—':>7}  {'—':>7}")
            continue
        la, lb = zip(*seqs)
        la, lb = list(la), list(lb)
        agree = sum(x == y for x, y in zip(la, lb)) / len(la) * 100
        k     = cohen_kappa(la, lb)
        k_w   = cohen_kappa(la, lb, weighted=True, order=TASK1_ORDER)
        pair  = f"{a1} vs {a2}"
        print(f"  {pair:<30}  {len(la):>5}  {agree:>6.1f}%  {k:>7.3f}  {k_w:>7.3f}")

    # Fleiss (≥3)
    if _HAS_NP and len(annotators) >= 3:
        labels = TASK1_ORDER
        lbl2i  = {l: i for i, l in enumerate(labels)}
        rows = []
        for votes in item_votes.values():
            if len(votes) >= 3:
                row = [0] * len(labels)
                for v in votes.values():
                    if v in lbl2i:
                        row[lbl2i[v]] += 1
                rows.append(row)
        if rows:
            mat = np.array(rows, dtype=float)
            fk  = fleiss_kappa(mat)
            print(f"\n  Fleiss' κ (N={len(rows)} items, {len(annotators)} raters): {fk:.3f}")

    # Per-label breakdown across all overlapping pairs
    print("\n  Per-label confusion (annotator A → B, aggregated pairwise):")
    _print_confusion(item_votes, annotators, TASK1_ORDER)
    print()


# ── Task-2 agreement ─────────────────────────────────────────────────────────

def normalise_rating(r: str | None) -> str | None:
    if r in SKIP_RATING:
        return None
    return TASK2_ALIASES.get(r, r)


def task2_pairs(all_ann: dict[str, list[dict]]) -> dict[tuple, dict[str, str]]:
    """
    Returns {(prompt_index, user_id, attr_text): {annotator: label}}
    """
    item_votes: dict[tuple, dict[str, str]] = defaultdict(dict)
    for annotator, records in all_ann.items():
        for rec in records:
            if rec.get("flagged"):
                continue
            pi = rec.get("prompt_index")
            uid = rec.get("user_id", "")
            for j in rec.get("relevance_judgments") or []:
                label = normalise_rating(j.get("rating"))
                if label is None:
                    continue
                attr = j.get("attribute", "").strip()
                key  = (pi, uid, attr)
                item_votes[key][annotator] = label
    return item_votes


def compute_task2(ann_dir: Path) -> None:
    print("=" * 60)
    print("TASK 2  — Attribute Relevance Rating Agreement")
    print("=" * 60)

    all_ann = load_annotations(ann_dir, "2")
    if len(all_ann) < 2:
        print(f"  Only {len(all_ann)} annotator(s) for Task 2 — need ≥ 2 for agreement.\n")
        return

    annotators = sorted(all_ann.keys())
    print(f"  Annotators : {', '.join(annotators)}")

    item_votes = task2_pairs(all_ann)
    total_items = len(item_votes)
    overlap = {k: v for k, v in item_votes.items() if len(v) >= 2}
    print(f"  Total items: {total_items}  |  Items with ≥2 annotations: {len(overlap)}")

    # ── Binary agreement: relevant vs not (necessary_* = relevant) ──────────
    def to_binary(label: str) -> str:
        return "relevant" if label.startswith("necessary") else "not_relevant"

    print(f"\n  {'Pair':<30}  {'N':>5}  {'Agree%':>7}  {'κ':>7}  {'κ_w':>7}  {'κ_bin':>7}")
    print("  " + "-" * 65)
    for a1, a2 in combinations(annotators, 2):
        seqs = [
            (votes[a1], votes[a2])
            for votes in item_votes.values()
            if a1 in votes and a2 in votes
        ]
        if not seqs:
            print(f"  {a1} vs {a2:<20}  no overlap")
            continue
        la, lb = list(zip(*seqs))
        la, lb = list(la), list(lb)
        agree   = sum(x == y for x, y in zip(la, lb)) / len(la) * 100
        k       = cohen_kappa(la, lb)
        k_w     = cohen_kappa(la, lb, weighted=True, order=TASK2_ORDER)
        ba, bb  = [to_binary(x) for x in la], [to_binary(x) for x in lb]
        k_bin   = cohen_kappa(ba, bb)
        pair    = f"{a1} vs {a2}"
        print(f"  {pair:<30}  {len(la):>5}  {agree:>6.1f}%  {k:>7.3f}  {k_w:>7.3f}  {k_bin:>7.3f}")

    # Fleiss (≥3)
    if _HAS_NP and len(annotators) >= 3:
        all_labels = sorted({l for votes in item_votes.values() for l in votes.values()})
        lbl2i = {l: i for i, l in enumerate(all_labels)}
        rows = []
        for votes in item_votes.values():
            if len(votes) >= 3:
                row = [0] * len(all_labels)
                for v in votes.values():
                    row[lbl2i[v]] += 1
                rows.append(row)
        if rows:
            mat = np.array(rows, dtype=float)
            fk  = fleiss_kappa(mat)
            print(f"\n  Fleiss' κ (N={len(rows)} items, {len(annotators)} raters): {fk:.3f}")

    # Per-label confusion
    print("\n  Per-label confusion (annotator A → B, aggregated pairwise):")
    _print_confusion(item_votes, annotators, TASK2_ORDER)

    # ── Per-prompt-index breakdown ───────────────────────────────────────────
    prompt_items: dict[Any, dict] = defaultdict(dict)
    for (pi, uid, attr), votes in item_votes.items():
        prompt_items[pi][(uid, attr)] = votes

    if len(prompt_items) > 1:
        print(f"\n  Per-prompt-index breakdown (pairwise pairs):")
        print(f"  {'prompt':>7}  {'items':>6}  {'overlap':>8}  {'agree%':>8}  {'κ':>7}")
        print("  " + "-" * 45)
        for pi in sorted(prompt_items.keys()):
            pv = prompt_items[pi]
            ol = {k: v for k, v in pv.items() if len(v) >= 2}
            if not ol:
                continue
            all_la, all_lb = [], []
            for a1, a2 in combinations(annotators, 2):
                for votes in ol.values():
                    if a1 in votes and a2 in votes:
                        all_la.append(votes[a1])
                        all_lb.append(votes[a2])
            if not all_la:
                continue
            ag = sum(x == y for x, y in zip(all_la, all_lb)) / len(all_la) * 100
            k  = cohen_kappa(all_la, all_lb)
            print(f"  {pi:>7}  {len(pv):>6}  {len(ol):>8}  {ag:>7.1f}%  {k:>7.3f}")
    print()


# ── Shared: confusion matrix pretty-print ────────────────────────────────────

def _print_confusion(
    item_votes: dict[tuple, dict[str, str]],
    annotators: list[str],
    order: list[str],
) -> None:
    # Aggregate all pairwise (a→b) counts
    all_labels: list[str] = []
    for votes in item_votes.values():
        all_labels.extend(votes.values())
    labels = [l for l in order if l in set(all_labels)]
    others = sorted(set(all_labels) - set(labels))
    labels = labels + others
    if not labels:
        return

    n = len(labels)
    conf: dict[tuple, int] = defaultdict(int)
    for votes in item_votes.values():
        for a1, a2 in combinations(annotators, 2):
            if a1 in votes and a2 in votes:
                conf[(votes[a1], votes[a2])] += 1
                conf[(votes[a2], votes[a1])] += 1

    col_w = max(len(l) for l in labels) + 2
    header = f"  {'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for row_l in labels:
        row = f"  {row_l:>{col_w}}"
        for col_l in labels:
            row += f"{conf[(row_l, col_l)]:>{col_w}}"
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-annotator agreement calculator.")
    parser.add_argument("--dir",  default="data/annotations",
                        help="Directory containing *_task1.jsonl and *_task2.jsonl files.")
    parser.add_argument("--task", default="both", choices=["1", "2", "both"],
                        help="Which task to evaluate (default: both).")
    args = parser.parse_args()

    ann_dir = Path(args.dir)
    if not ann_dir.is_dir():
        sys.exit(f"[error] Directory not found: {ann_dir}")

    if args.task in ("1", "both"):
        compute_task1(ann_dir)
    if args.task in ("2", "both"):
        compute_task2(ann_dir)


if __name__ == "__main__":
    main()
