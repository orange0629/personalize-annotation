#!/usr/bin/env python3
"""
Inter-annotator agreement calculator for annotation_tool.

Task 1: per-attribute judgments  ("ok" / "unsure" / "bad")
Task 2: per-attribute ratings    ("necessary_explicit" / "necessary_implicit" /
                                   "optional" / "unnecessary" / "forbidden")

Metrics
-------
Nominal:
  - Raw agreement (% exact match)
  - Cohen's κ            pairwise, unweighted
  - Fleiss' κ            multi-annotator nominal (≥3 raters)
Ordinal / Likert (treating scale as interval):
  - Cohen's κ_w²         pairwise, quadratic-weighted
  - Spearman's ρ         pairwise rank correlation
  - ICC(3,1)             pairwise, two-way mixed consistency (Likert standard)
  - ICC(1,1)             multi-rater, one-way random (handles unbalanced/missing data)
  - Krippendorff's α_ord multi-annotator ordinal (handles missing data)
Binary:
  - Cohen's κ_bin        pairwise, relevant vs not

Usage
-----
  python calc_agreement.py [--dir data/annotations] [--task 1|2|both] [--guide]
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
    print("[warn] numpy not found – ordinal metrics will be skipped", file=sys.stderr)

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[warn] scipy not found – Spearman's ρ will be skipped", file=sys.stderr)

try:
    from sklearn.metrics import cohen_kappa_score
    _HAS_SK = True
except ImportError:
    _HAS_SK = False
    print("[warn] scikit-learn not found – falling back to manual Cohen's κ", file=sys.stderr)


# ── Label orderings ───────────────────────────────────────────────────────────
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
    "explicitly_impact": "necessary_explicit",
    "implicitly_impact": "necessary_implicit",
    "low_impact":        "optional",
    "irrelevant":        "unnecessary",
    "avoid":             "forbidden",
}

SKIP_RATING = {"none", "", None}


# ── Cohen's κ (nominal) ───────────────────────────────────────────────────────

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


def cohen_kappa_nominal(a: list, b: list) -> float:
    if _HAS_SK:
        try:
            labels = sorted(set(a) | set(b))
            return cohen_kappa_score(a, b, labels=labels)
        except Exception:
            pass
    return _cohen_kappa_manual(a, b)


# ── Cohen's κ_w² (quadratic-weighted, ordinal) ───────────────────────────────

def cohen_kappa_quadratic(a: list, b: list, order: list[str]) -> float:
    """Quadratic-weighted Cohen's κ — standard for ordinal rating scales."""
    if not _HAS_NP:
        return float("nan")

    def pos(x: str) -> int:
        try:
            return order.index(x)
        except ValueError:
            return len(order)

    present = sorted(set(a) | set(b), key=pos)
    n = len(a)
    if n == 0:
        return float("nan")

    k = len(present)
    lbl2i = {l: i for i, l in enumerate(present)}
    positions = np.array([pos(l) for l in present], dtype=float)
    max_pos = positions.max() - positions.min() or 1.0

    # Quadratic weight matrix (0 on diagonal, 1 at max distance)
    weights = np.array(
        [[(positions[i] - positions[j]) ** 2 / max_pos ** 2 for j in range(k)] for i in range(k)]
    )

    conf = np.zeros((k, k), dtype=float)
    for ai, bi in zip(a, b):
        if ai in lbl2i and bi in lbl2i:
            conf[lbl2i[ai]][lbl2i[bi]] += 1
    conf /= n

    row_sum = conf.sum(axis=1, keepdims=True)
    col_sum = conf.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum

    p_o = 1.0 - (conf * weights).sum()
    p_e = 1.0 - (expected * weights).sum()
    return (p_o - p_e) / p_e if abs(p_e) > 1e-12 else 1.0


# ── Spearman's ρ ──────────────────────────────────────────────────────────────

def spearman_rho(a: list, b: list, order: list[str]) -> float:
    """Convert labels to ordinal ranks then compute Spearman's ρ."""
    if not _HAS_SCIPY or not _HAS_NP:
        return float("nan")

    def pos(x: str) -> float:
        try:
            return float(order.index(x))
        except ValueError:
            return float(len(order))

    ra = [pos(x) for x in a]
    rb = [pos(x) for x in b]
    if len(set(ra)) == 1 or len(set(rb)) == 1:
        return float("nan")  # zero variance → undefined
    rho, _ = spearmanr(ra, rb)
    return float(rho)


# ── Fleiss' κ (nominal, multi-rater) ─────────────────────────────────────────

def fleiss_kappa(ratings_matrix: "np.ndarray") -> float:
    """ratings_matrix: shape (N_items, N_labels) — count of raters per label."""
    if not _HAS_NP:
        return float("nan")
    n_items, _ = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())
    if n_raters <= 1:
        return float("nan")
    p_j   = ratings_matrix.sum(axis=0) / (n_items * n_raters)
    P_i   = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e   = (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


# ── Krippendorff's α (ordinal) ────────────────────────────────────────────────

def krippendorff_alpha_ordinal(item_votes: dict[tuple, dict[str, str]],
                               order: list[str]) -> float:
    """
    Krippendorff's α with ordinal distance metric.

    Ordinal distance²(c, k) = (Σ_{g=min(c,k)}^{max(c,k)} n_g  −  (n_c + n_k)/2)²
    where n_g is the global marginal count for label g.

    Handles different numbers of raters per item (missing data) correctly.
    """
    if not _HAS_NP:
        return float("nan")

    def rank(x: str) -> int:
        try:
            return order.index(x)
        except ValueError:
            return len(order)

    # Collect all ratings as a list-of-lists (one inner list per item)
    units: list[list[int]] = []
    for votes in item_votes.values():
        if len(votes) >= 2:
            units.append([rank(v) for v in votes.values()])

    if not units:
        return float("nan")

    # Global marginal frequency n_g for each rank
    all_ranks = [r for unit in units for r in unit]
    max_rank = max(all_ranks)
    n_g = np.zeros(max_rank + 1, dtype=float)
    for r in all_ranks:
        n_g[r] += 1

    def ordinal_dist_sq(c: int, k: int) -> float:
        lo, hi = min(c, k), max(c, k)
        return (n_g[lo:hi+1].sum() - (n_g[lo] + n_g[hi]) / 2.0) ** 2

    # Observed disagreement D_o
    D_o_num = 0.0
    D_o_den = 0.0
    for unit in units:
        m_u = len(unit)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(i + 1, m_u):
                D_o_num += ordinal_dist_sq(unit[i], unit[j])
        D_o_den += m_u * (m_u - 1) / 2

    if D_o_den == 0:
        return float("nan")
    D_o = D_o_num / D_o_den

    # Expected disagreement D_e — over all coincidences in the pairing table
    n_total = sum(len(u) for u in units)
    D_e = 0.0
    for c in range(max_rank + 1):
        for k in range(max_rank + 1):
            if c != k:
                D_e += n_g[c] * n_g[k] * ordinal_dist_sq(c, k)
    D_e /= n_total * (n_total - 1)

    return 1.0 - D_o / D_e if D_e > 0 else 1.0


# ── ICC ───────────────────────────────────────────────────────────────────────

def _to_numeric(labels: list[str], order: list[str]) -> "np.ndarray":
    """Map labels to integer positions (0-based) along the ordinal scale."""
    def pos(x: str) -> float:
        try:
            return float(order.index(x))
        except ValueError:
            return float(len(order))
    return np.array([pos(x) for x in labels], dtype=float)


def icc_pairwise(a: list[str], b: list[str], order: list[str]) -> tuple[float, float]:
    """
    ICC(3,1) — two-way mixed effects, consistency, single measures.
    Appropriate for fixed raters; measures how consistently they rank items.

    Also returns ICC(2,1) — absolute agreement variant.
    Both reduce to the same formula when rater means are equal.

    Returns (ICC_consistency, ICC_absolute).
    """
    if not _HAS_NP or len(a) < 2:
        return float("nan"), float("nan")

    ra = _to_numeric(a, order)
    rb = _to_numeric(b, order)
    n  = len(ra)
    k  = 2  # two raters

    grand_mean = (ra.mean() + rb.mean()) / 2
    item_means = (ra + rb) / 2

    # Sums of squares
    SS_b  = k * np.sum((item_means - grand_mean) ** 2)        # between items
    SS_wr = np.sum((ra - item_means) ** 2 + (rb - item_means) ** 2)  # within items (residual)
    rater_means = np.array([ra.mean(), rb.mean()])
    SS_r  = n * np.sum((rater_means - grand_mean) ** 2)       # between raters
    SS_e  = SS_wr - SS_r                                       # error

    MS_b  = SS_b  / (n - 1)
    MS_r  = SS_r  / (k - 1)
    MS_e  = SS_e  / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")

    # ICC(3,1) consistency  — rater bias not part of error
    icc3 = (MS_b - MS_e) / (MS_b + (k - 1) * MS_e) if MS_b + (k - 1) * MS_e > 0 else float("nan")

    # ICC(2,1) absolute agreement — rater bias IS part of error
    denom2 = MS_b + (k - 1) * MS_e + k * (MS_r - MS_e) / n
    icc2 = (MS_b - MS_e) / denom2 if denom2 > 0 else float("nan")

    return float(icc3), float(icc2)


def icc_oneway(item_votes: dict[tuple, dict[str, str]], order: list[str]) -> float:
    """
    ICC(1,1) — one-way random effects, single measures.
    Handles unbalanced designs (items with different numbers of raters).

    ICC(1,1) = (MSb - MSw) / (MSb + (k̄ - 1) * MSw)
    where k̄ is the harmonic mean number of raters per item.
    """
    if not _HAS_NP:
        return float("nan")

    units = [
        _to_numeric(list(votes.values()), order)
        for votes in item_votes.values()
        if len(votes) >= 2
    ]
    if not units:
        return float("nan")

    # Harmonic mean of group sizes
    sizes  = np.array([len(u) for u in units], dtype=float)
    k_harm = len(sizes) / np.sum(1.0 / sizes)

    grand_mean = np.concatenate(units).mean()
    n_items    = len(units)

    SS_b = sum(len(u) * (u.mean() - grand_mean) ** 2 for u in units)
    SS_w = sum(np.sum((u - u.mean()) ** 2) for u in units)

    df_b = n_items - 1
    df_w = sum(len(u) - 1 for u in units)

    if df_b == 0 or df_w == 0:
        return float("nan")

    MS_b = SS_b / df_b
    MS_w = SS_w / df_w

    denom = MS_b + (k_harm - 1) * MS_w
    return float((MS_b - MS_w) / denom) if denom > 0 else float("nan")


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

def task1_pairs(all_ann: dict[str, list[dict]]) -> dict[tuple, dict[str, str]]:
    item_votes: dict[tuple, dict[str, str]] = defaultdict(dict)
    for annotator, records in all_ann.items():
        for rec in records:
            for i, j in enumerate(rec.get("attr_judgments") or []):
                label = j.get("judgment")
                if label and label not in SKIP_RATING:
                    item_votes[(rec["index"], i)][annotator] = label
    return item_votes


def compute_task1(ann_dir: Path) -> None:
    print("=" * 70)
    print("TASK 1  — Attribute Judgment Agreement")
    print("=" * 70)

    all_ann = load_annotations(ann_dir, "1")
    if len(all_ann) < 2:
        print(f"  Only {len(all_ann)} annotator(s) for Task 1 — need ≥ 2 for agreement.\n")
        return

    annotators = sorted(all_ann.keys())
    print(f"  Annotators : {', '.join(annotators)}")
    print(f"  Scale order: {' < '.join(TASK1_ORDER)}")

    item_votes = task1_pairs(all_ann)

    hdr = f"  {'Pair':<30}  {'N':>5}  {'Agree%':>7}  {'κ':>7}  {'κ_w²':>7}  {'ρ':>7}  {'ICC(3,1)':>9}  {'ICC(2,1)':>9}"
    print(f"\n{hdr}")
    print("  " + "-" * 85)
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
        agree       = sum(x == y for x, y in zip(la, lb)) / len(la) * 100
        k           = cohen_kappa_nominal(la, lb)
        k_w2        = cohen_kappa_quadratic(la, lb, TASK1_ORDER)
        rho         = spearman_rho(la, lb, TASK1_ORDER)
        icc3, icc2  = icc_pairwise(la, lb, TASK1_ORDER)
        pair        = f"{a1} vs {a2}"
        print(f"  {pair:<30}  {len(la):>5}  {agree:>6.1f}%  {k:>7.3f}  {k_w2:>7.3f}  {rho:>7.3f}  {icc3:>9.3f}  {icc2:>9.3f}")

    # Multi-rater
    _print_multirater_task1(item_votes, annotators)

    print("\n  Per-label confusion (annotator A → B, aggregated pairwise):")
    _print_confusion(item_votes, annotators, TASK1_ORDER)
    print()


def _print_multirater_task1(item_votes: dict, annotators: list[str]) -> None:
    if not _HAS_NP:
        return
    overlap = {k: v for k, v in item_votes.items() if len(v) >= 2}
    icc1 = icc_oneway(overlap, TASK1_ORDER)
    alpha = krippendorff_alpha_ordinal(overlap, TASK1_ORDER)
    print(f"\n  Multi-rater (N={len(overlap)} overlapping items):")
    print(f"    ICC(1,1) one-way  : {icc1:.3f}")
    print(f"    Krippendorff α_ord: {alpha:.3f}")
    if len(annotators) >= 3:
        labels = TASK1_ORDER
        lbl2i  = {l: i for i, l in enumerate(labels)}
        rows = [
            [sum(1 for v in votes.values() if v == l) for l in labels]
            for votes in item_votes.values()
            if len(votes) >= 3
        ]
        if rows:
            fk = fleiss_kappa(np.array(rows, dtype=float))
            print(f"    Fleiss' κ (nominal): {fk:.3f}  (N={len(rows)} items with ≥3 annotations)")


# ── Task-2 agreement ─────────────────────────────────────────────────────────

def normalise_rating(r: str | None) -> str | None:
    if r in SKIP_RATING:
        return None
    return TASK2_ALIASES.get(r, r)


def task2_votes(all_ann: dict[str, list[dict]]) -> dict[tuple, dict[str, str]]:
    """Returns {(prompt_index, user_id, attr_text): {annotator: label}}"""
    item_votes: dict[tuple, dict[str, str]] = defaultdict(dict)
    for annotator, records in all_ann.items():
        for rec in records:
            if rec.get("flagged"):
                continue
            pi  = rec.get("prompt_index")
            uid = rec.get("user_id", "")
            for j in rec.get("relevance_judgments") or []:
                label = normalise_rating(j.get("rating"))
                if label is None:
                    continue
                attr = j.get("attribute", "").strip()
                item_votes[(pi, uid, attr)][annotator] = label
    return item_votes


def compute_task2(ann_dir: Path) -> None:
    print("=" * 70)
    print("TASK 2  — Attribute Relevance Rating Agreement")
    print("=" * 70)

    all_ann = load_annotations(ann_dir, "2")
    if len(all_ann) < 2:
        print(f"  Only {len(all_ann)} annotator(s) for Task 2 — need ≥ 2 for agreement.\n")
        return

    annotators = sorted(all_ann.keys())
    print(f"  Annotators : {', '.join(annotators)}")
    print(f"  Scale order: {' < '.join(TASK2_ORDER)}")

    item_votes = task2_votes(all_ann)
    overlap = {k: v for k, v in item_votes.items() if len(v) >= 2}
    print(f"  Total items: {len(item_votes)}  |  Overlapping (≥2 annotators): {len(overlap)}")

    def to_binary(label: str) -> str:
        return "relevant" if label.startswith("necessary") else "not_relevant"

    hdr = f"  {'Pair':<30}  {'N':>5}  {'Agree%':>7}  {'κ':>7}  {'κ_w²':>7}  {'ρ':>7}  {'ICC(3,1)':>9}  {'ICC(2,1)':>9}  {'κ_bin':>7}"
    print(f"\n{hdr}")
    print("  " + "-" * 95)
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
        agree       = sum(x == y for x, y in zip(la, lb)) / len(la) * 100
        k           = cohen_kappa_nominal(la, lb)
        k_w2        = cohen_kappa_quadratic(la, lb, TASK2_ORDER)
        rho         = spearman_rho(la, lb, TASK2_ORDER)
        icc3, icc2  = icc_pairwise(la, lb, TASK2_ORDER)
        ba, bb      = [to_binary(x) for x in la], [to_binary(x) for x in lb]
        k_bin       = cohen_kappa_nominal(ba, bb)
        pair        = f"{a1} vs {a2}"
        print(f"  {pair:<30}  {len(la):>5}  {agree:>6.1f}%  {k:>7.3f}  {k_w2:>7.3f}  {rho:>7.3f}  {icc3:>9.3f}  {icc2:>9.3f}  {k_bin:>7.3f}")

    # Multi-rater
    _print_multirater_task2(item_votes, annotators)

    # Per-label confusion
    print("\n  Per-label confusion (annotator A → B, aggregated pairwise):")
    _print_confusion(item_votes, annotators, TASK2_ORDER)

    # Per-prompt-index breakdown
    _print_per_prompt(item_votes, annotators)
    print()


def _print_multirater_task2(item_votes: dict, annotators: list[str]) -> None:
    if not _HAS_NP:
        return

    overlap = {k: v for k, v in item_votes.items() if len(v) >= 2}
    n_overlap = len(overlap)

    icc1  = icc_oneway(overlap, TASK2_ORDER)
    alpha = krippendorff_alpha_ordinal(overlap, TASK2_ORDER)

    print(f"\n  Multi-rater (N={n_overlap} overlapping items):")
    print(f"    ICC(1,1) one-way  : {icc1:.3f}  ← Likert/interval assumption")
    print(f"    Krippendorff α_ord: {alpha:.3f}  ← ordinal assumption")

    if len(annotators) >= 3:
        all_labels = sorted({l for votes in overlap.values() for l in votes.values()})
        lbl2i = {l: i for i, l in enumerate(all_labels)}
        rows_3 = [
            [sum(1 for v in votes.values() if v == l) for l in all_labels]
            for votes in item_votes.values()
            if len(votes) >= 3
        ]
        if rows_3:
            fk = fleiss_kappa(np.array(rows_3, dtype=float))
            print(f"    Fleiss' κ (nominal): {fk:.3f}  (N={len(rows_3)} items with ≥3 annotations)")


def _print_per_prompt(item_votes: dict, annotators: list[str]) -> None:
    prompt_items: dict[Any, dict] = defaultdict(dict)
    for (pi, uid, attr), votes in item_votes.items():
        prompt_items[pi][(uid, attr)] = votes

    if len(prompt_items) <= 1:
        return

    print(f"\n  Per-prompt breakdown:")
    print(f"  {'prompt':>7}  {'items':>6}  {'agree%':>7}  {'κ':>7}  {'κ_w²':>7}  {'ICC(1,1)':>9}  {'α_ord':>7}")
    print("  " + "-" * 65)
    for pi in sorted(prompt_items.keys()):
        pv = prompt_items[pi]
        sub_votes = {k: v for k, v in item_votes.items()
                     if k[0] == pi and len(v) >= 2}
        if not sub_votes:
            continue
        all_la, all_lb = [], []
        for a1, a2 in combinations(annotators, 2):
            for votes in sub_votes.values():
                if a1 in votes and a2 in votes:
                    all_la.append(votes[a1])
                    all_lb.append(votes[a2])
        if not all_la:
            continue
        ag    = sum(x == y for x, y in zip(all_la, all_lb)) / len(all_la) * 100
        k     = cohen_kappa_nominal(all_la, all_lb)
        k_w2  = cohen_kappa_quadratic(all_la, all_lb, TASK2_ORDER)
        icc1  = icc_oneway(sub_votes, TASK2_ORDER)
        alpha = krippendorff_alpha_ordinal(sub_votes, TASK2_ORDER)
        print(f"  {pi:>7}  {len(pv):>6}  {ag:>6.1f}%  {k:>7.3f}  {k_w2:>7.3f}  {icc1:>9.3f}  {alpha:>7.3f}")


# ── Shared: confusion matrix pretty-print ────────────────────────────────────

def _print_confusion(
    item_votes: dict[tuple, dict[str, str]],
    annotators: list[str],
    order: list[str],
) -> None:
    all_labels_used: set[str] = set()
    for votes in item_votes.values():
        all_labels_used.update(votes.values())
    labels = [l for l in order if l in all_labels_used]
    labels += sorted(all_labels_used - set(labels))
    if not labels:
        return

    conf: dict[tuple, int] = defaultdict(int)
    for votes in item_votes.values():
        for a1, a2 in combinations(annotators, 2):
            if a1 in votes and a2 in votes:
                conf[(votes[a1], votes[a2])] += 1
                conf[(votes[a2], votes[a1])] += 1

    # Short display names to keep columns narrow
    short = {l: l[:8] for l in labels}
    col_w = max(len(l) for l in labels) + 2
    s_col = max(len(s) for s in short.values()) + 2
    print("  " + f"{'':>{col_w}}" + "".join(f"{short[l]:>{s_col}}" for l in labels))
    for row_l in labels:
        row = f"  {row_l:>{col_w}}"
        for col_l in labels:
            row += f"{conf[(row_l, col_l)]:>{s_col}}"
        print(row)


# ── Interpretation guide ──────────────────────────────────────────────────────

def _print_guide() -> None:
    print("""
  Interpretation (κ / α / ICC):
    < 0.00   : Less than chance / poor
    0.00–0.20: Slight
    0.20–0.40: Fair
    0.40–0.60: Moderate
    0.60–0.80: Substantial / good
    0.80–1.00: Almost perfect / excellent

  Which metric to trust?
    κ          Nominal baseline. Ignores scale ordering — best for pure categories.
    κ_w²       Ordinal. Penalises large jumps more; good for 5-pt scale comparison.
    ρ           Rank correlation. Tells you if annotators agree on relative ordering.
    ICC(3,1)   Pairwise Likert reliability (consistency, rater bias excluded).
    ICC(2,1)   Pairwise Likert reliability (absolute agreement, rater bias included).
    ICC(1,1)   Multi-rater Likert reliability; handles unbalanced/missing raters.
    α_ord      Krippendorff ordinal: most principled for unbalanced annotation;
               equivalent to ICC when data are complete and balanced.
    κ_bin      Coarse check: do annotators agree on "relevant vs not"?
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-annotator agreement calculator.")
    parser.add_argument("--dir",  default="data/annotations",
                        help="Directory containing *_task1.jsonl and *_task2.jsonl files "
                             "(default: data/annotations).")
    parser.add_argument("--task", default="both", choices=["1", "2", "both"],
                        help="Which task to evaluate (default: both).")
    parser.add_argument("--guide", action="store_true",
                        help="Print κ/α interpretation guide.")
    args = parser.parse_args()

    ann_dir = Path(args.dir)
    if not ann_dir.is_dir():
        sys.exit(f"[error] Directory not found: {ann_dir}")

    if args.task in ("1", "both"):
        compute_task1(ann_dir)
    if args.task in ("2", "both"):
        compute_task2(ann_dir)
    if args.guide:
        _print_guide()


if __name__ == "__main__":
    main()
