#!/usr/bin/env python3
"""
Annotation Tool — Flask web application.

Two tasks:
  Task 1  /task1/<idx>  — Annotate user attributes (per user)
  Task 2  /task2/<idx>  — Annotate checklist relevance (per user-prompt pair)

Run:
    python3 app.py [--port 5050]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
INDEX_DIR   = BASE_DIR / "data" / "indexes"
ANNOT_DIR   = BASE_DIR / "data" / "annotations"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_BASE   = Path("/shared/0/projects/research-jam-summer-2024/tmp")
ATTRS_JSONL      = DATA_BASE / "attrs_per_conversation_final_merged_agglo_0.7_v2.jsonl"
CHECKLIST_JSONL  = DATA_BASE / "checklist_final_merged_agglo_0.7_v2_LIMA.jsonl"
CONVS_JSONL      = DATA_BASE / "wildchat_long_conversations_15_assistant_like_confidence_final.jsonl"

ATTRS_INDEX_DB      = INDEX_DIR / "attrs_index.db"
CHECKLIST_INDEX_DB  = INDEX_DIR / "checklist_index.db"
CONVS_INDEX_DB      = INDEX_DIR / "convs_index.db"

# ─── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "annotation-tool-secret-2024"


@app.context_processor
def inject_annotator():
    return {"current_annotator": annotator_name()}

# ─── File handles (open once, seek on demand) ─────────────────────────────────
_attrs_fh:    Optional[Any] = None
_checklist_fh: Optional[Any] = None
_convs_fh:    Optional[Any] = None


def get_attrs_fh():
    global _attrs_fh
    if _attrs_fh is None:
        _attrs_fh = open(ATTRS_JSONL, "rb")
    return _attrs_fh


def get_checklist_fh():
    global _checklist_fh
    if _checklist_fh is None:
        _checklist_fh = open(CHECKLIST_JSONL, "rb")
    return _checklist_fh


def get_convs_fh():
    global _convs_fh
    if _convs_fh is None:
        _convs_fh = open(CONVS_JSONL, "rb")
    return _convs_fh


# ─── Index helpers ────────────────────────────────────────────────────────────

def index_ready(db_path: Path) -> bool:
    return db_path.exists()


def attrs_count() -> int:
    if not index_ready(ATTRS_INDEX_DB):
        return 0
    with sqlite3.connect(str(ATTRS_INDEX_DB), timeout=10) as conn:
        return conn.execute("SELECT COUNT(*) FROM attrs_index").fetchone()[0]


def _checklist_has_sample(conn) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='checklist_sample'"
    ).fetchone() is not None


def checklist_count() -> int:
    if not index_ready(CHECKLIST_INDEX_DB):
        return 0
    with sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=10) as conn:
        if _checklist_has_sample(conn):
            return conn.execute("SELECT COUNT(*) FROM checklist_sample").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM checklist_index").fetchone()[0]


def fetch_attrs_record(line_index: int) -> Optional[Dict]:
    """Fetch one attrs record by its sequential line_index."""
    if not index_ready(ATTRS_INDEX_DB):
        return None
    with sqlite3.connect(str(ATTRS_INDEX_DB), timeout=10) as conn:
        row = conn.execute(
            "SELECT byte_offset FROM attrs_index WHERE line_index=?", (line_index,)
        ).fetchone()
    if not row:
        return None
    fh = get_attrs_fh()
    fh.seek(row[0])
    return json.loads(fh.readline())


def fetch_checklist_record(sample_index: int) -> Optional[Dict]:
    """Fetch one checklist record by sample_index (via checklist_sample if available)."""
    if not index_ready(CHECKLIST_INDEX_DB):
        return None
    with sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=10) as conn:
        if _checklist_has_sample(conn):
            row = conn.execute(
                """SELECT ci.byte_offset FROM checklist_sample cs
                   JOIN checklist_index ci ON cs.annotation_index = ci.annotation_index
                   WHERE cs.sample_index = ?""",
                (sample_index,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT byte_offset FROM checklist_index WHERE annotation_index=?",
                (sample_index,),
            ).fetchone()
    if not row:
        return None
    fh = get_checklist_fh()
    fh.seek(row[0])
    return json.loads(fh.readline())


def fetch_user_conversations(user_id: str) -> List[Dict]:
    """Fetch all conversation records for a user_id (hashed_ip)."""
    if not index_ready(CONVS_INDEX_DB):
        return []
    with sqlite3.connect(str(CONVS_INDEX_DB), timeout=10) as conn:
        rows = conn.execute(
            "SELECT byte_offset FROM convs_index WHERE hashed_ip=? ORDER BY id",
            (user_id,),
        ).fetchall()
    if not rows:
        return []
    fh = get_convs_fh()
    results = []
    for (offset,) in rows:
        fh.seek(offset)
        try:
            results.append(json.loads(fh.readline()))
        except Exception:
            continue
    return results


def get_checklist_meta_list() -> List[Dict]:
    """Return lightweight list of checklist records for the jump navigator."""
    if not index_ready(CHECKLIST_INDEX_DB):
        return []
    with sqlite3.connect(str(CHECKLIST_INDEX_DB), timeout=10) as conn:
        if _checklist_has_sample(conn):
            rows = conn.execute(
                """SELECT cs.sample_index, ci.user_id, ci.prompt_index, ci.prompt_snippet
                   FROM checklist_sample cs
                   JOIN checklist_index ci ON cs.annotation_index = ci.annotation_index
                   ORDER BY cs.sample_index"""
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT annotation_index, user_id, prompt_index, prompt_snippet FROM checklist_index ORDER BY annotation_index"
            ).fetchall()
    return [
        {"annotation_index": r[0], "user_id": r[1], "prompt_index": r[2], "prompt_snippet": r[3]}
        for r in rows
    ]


def get_attrs_meta_list() -> List[Dict]:
    """Return lightweight list of attrs records for the jump navigator."""
    if not index_ready(ATTRS_INDEX_DB):
        return []
    with sqlite3.connect(str(ATTRS_INDEX_DB), timeout=10) as conn:
        rows = conn.execute(
            "SELECT line_index, user_id FROM attrs_index ORDER BY line_index"
        ).fetchall()
    return [{"line_index": r[0], "user_id": r[1]} for r in rows]


# ─── Annotation helpers ───────────────────────────────────────────────────────

def annotator_name() -> Optional[str]:
    return session.get("annotator_name")


def annot_file(task: str) -> Path:
    name = annotator_name() or "unknown"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return ANNOT_DIR / f"{safe}_task{task}.jsonl"


def load_existing_annotations(task: str) -> Dict[str, Any]:
    """Load existing annotations into a dict keyed by record index string."""
    path = annot_file(task)
    result: Dict[str, Any] = {}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = str(rec.get("index", ""))
                if key:
                    result[key] = rec
            except Exception:
                continue
    return result


def save_annotation(task: str, record: Dict) -> None:
    """Upsert annotation: one record per index, rewrite file atomically."""
    record["annotator"] = annotator_name()
    record["timestamp"] = datetime.utcnow().isoformat()

    # Load all existing records, update the entry for this index, rewrite.
    all_records = load_existing_annotations(task)
    all_records[str(record["index"])] = record

    path = annot_file(task)
    tmp_path = path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in all_records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(path)  # atomic rename


# ─── Conversation HTML builder (adapted from visualize_score_assistant_like_usage.py) ──

def build_chat_html(conversations: Any, max_turns: int = 60, short_chars: int = 200, long_chars: int = 800) -> str:
    if not conversations:
        return '<div class="chat-empty">No messages.</div>'

    convs = conversations
    if isinstance(convs, list) and convs and isinstance(convs[0], dict):
        # Single flat conversation — wrap it
        convs = [convs]

    html_parts: List[str] = []
    turns_shown = 0

    for c_idx, conv_turns in enumerate(convs or []):
        if turns_shown >= max_turns:
            break
        if len(convs) > 1:
            html_parts.append(f'<div class="conv-label">Conversation {c_idx + 1}</div>')

        for turn in conv_turns:
            if turns_shown >= max_turns:
                break
            if not isinstance(turn, dict):
                continue
            role  = (turn.get("role") or "").upper()
            text  = (turn.get("content") or "").strip()
            if not text:
                continue

            role_class = "msg-user" if role == "USER" else (
                "msg-assistant" if role in {"ASSISTANT", "SYSTEM", "TOOL"} else "msg-other"
            )
            short_text = text[:short_chars] + ("..." if len(text) > short_chars else "")
            long_text  = text[:long_chars]  + ("..." if len(text) > long_chars  else "")

            safe_role  = html.escape(role)
            safe_short = html.escape(short_text)
            safe_long  = html.escape(long_text)

            html_parts.append(
                f'<details class="msg-details">'
                f'<summary><div class="msg {role_class}">'
                f'<div class="msg-role">{safe_role}</div>'
                f'<div class="msg-text">{safe_short}</div>'
                f'</div></summary>'
                f'<div class="msg-long"><div class="msg {role_class} msg-expanded">'
                f'<div class="msg-role">{safe_role} (full)</div>'
                f'<div class="msg-text">{safe_long}</div>'
                f'</div></div></details>'
            )
            turns_shown += 1

    if turns_shown == 0:
        return '<div class="chat-empty">No messages.</div>'
    if turns_shown >= max_turns:
        html_parts.append('<div class="chat-more-hint">Preview truncated.</div>')
    return "".join(html_parts)


# ─── Routes: auth ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if name:
            session["annotator_name"] = name
            return redirect(url_for("task_select"))
    return render_template("login.html", current_name=annotator_name())


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/select")
def task_select():
    if not annotator_name():
        return redirect(url_for("index"))
    t1_count = attrs_count()
    t2_count = checklist_count()
    t1_ready = index_ready(ATTRS_INDEX_DB) and index_ready(CONVS_INDEX_DB)
    t2_ready = index_ready(CHECKLIST_INDEX_DB)
    return render_template(
        "select.html",
        annotator=annotator_name(),
        t1_count=t1_count,
        t2_count=t2_count,
        t1_ready=t1_ready,
        t2_ready=t2_ready,
    )


# ─── Routes: Task 1 — User Attribute Annotation ───────────────────────────────

@app.route("/task1")
def task1_redirect():
    if not annotator_name():
        return redirect(url_for("index"))
    return redirect(url_for("task1_view", idx=0))


@app.route("/task1/<int:idx>")
def task1_view(idx: int):
    if not annotator_name():
        return redirect(url_for("index"))

    total = attrs_count()
    if total == 0:
        return render_template("error.html", msg="Attrs index not built yet. Run build_index.py first.")

    idx = max(0, min(idx, total - 1))
    record = fetch_attrs_record(idx)
    if not record:
        return render_template("error.html", msg=f"Could not fetch attrs record {idx}.")

    user_id = record.get("user_id", "")
    merged_attrs = record.get("merged_attributes", [])
    # Strip embeddings for display
    for a in merged_attrs:
        a.pop("embedding", None)

    # Fetch conversations
    conv_records = fetch_user_conversations(user_id)
    # Build chat HTML per conversation record
    conv_htmls = []
    for cr in conv_records:
        conv = cr.get("conversation")
        ch = build_chat_html(conv) if conv else '<div class="chat-empty">No conversation data.</div>'
        conv_htmls.append({
            "conversation_hash": cr.get("conversation_hash", ""),
            "model": cr.get("model", ""),
            "timestamp": cr.get("timestamp", ""),
            "html": ch,
        })

    # Load existing annotation for this index
    existing = load_existing_annotations("1").get(str(idx))

    return render_template(
        "task1.html",
        idx=idx,
        total=total,
        user_id=user_id,
        merged_attrs=merged_attrs,
        conv_htmls=conv_htmls,
        annotator=annotator_name(),
        existing=existing,
    )


@app.route("/task1/<int:idx>/save", methods=["POST"])
def task1_save(idx: int):
    if not annotator_name():
        return jsonify({"ok": False, "error": "Not logged in"}), 401

    data = request.get_json(force=True)
    # data = {
    #   "attr_judgments": [{"attribute": ..., "ok": bool, "note": str}, ...],
    #   "missing_attrs": str,
    #   "overall_note": str
    # }
    record = {
        "task": "1",
        "index": idx,
        "attr_judgments": data.get("attr_judgments", []),
        "missing_attrs": data.get("missing_attrs", ""),
        "overall_note": data.get("overall_note", ""),
    }
    save_annotation("1", record)
    return jsonify({"ok": True})


@app.route("/task1/list")
def task1_list():
    """Return JSON list of attrs records for jump navigator."""
    if not annotator_name():
        return jsonify([])
    meta = get_attrs_meta_list()
    existing = load_existing_annotations("1")
    for m in meta:
        m["annotated"] = str(m["line_index"]) in existing
    return jsonify(meta)


# ─── Routes: Task 2 — Checklist Relevance Annotation ─────────────────────────

@app.route("/task2")
def task2_redirect():
    if not annotator_name():
        return redirect(url_for("index"))
    return redirect(url_for("task2_view", idx=0))


@app.route("/task2/<int:idx>")
def task2_view(idx: int):
    if not annotator_name():
        return redirect(url_for("index"))

    total = checklist_count()
    if total == 0:
        return render_template("error.html", msg="Checklist index not built yet. Run build_index.py first.")

    idx = max(0, min(idx, total - 1))
    record = fetch_checklist_record(idx)
    if not record:
        return render_template("error.html", msg=f"Could not fetch checklist record {idx}.")

    user_id      = record.get("user_id", "")
    prompt_text  = record.get("prompt_text", "")
    profile_attrs = record.get("profile_attributes", [])
    items        = record.get("items", [])  # model-selected relevant attributes
    prompt_index = record.get("prompt_index", -1)

    # Strip embeddings
    for a in profile_attrs:
        a.pop("embedding", None)

    # Build lookup: attribute text -> item dict (for expected_behavior / relevance)
    items_by_attr: Dict[str, Dict] = {it.get("attribute", ""): it for it in items}

    # Sort: model-relevant attributes first, then the rest (stable within each group)
    profile_attrs = sorted(
        profile_attrs,
        key=lambda a: 0 if a.get("attribute", "") in items_by_attr else 1,
    )

    # Load existing annotation
    existing = load_existing_annotations("2").get(str(idx))

    return render_template(
        "task2.html",
        idx=idx,
        total=total,
        user_id=user_id,
        prompt_text=prompt_text,
        prompt_index=prompt_index,
        profile_attrs=profile_attrs,
        items=items,
        items_by_attr=items_by_attr,
        annotator=annotator_name(),
        existing=existing,
    )


@app.route("/task2/<int:idx>/save", methods=["POST"])
def task2_save(idx: int):
    if not annotator_name():
        return jsonify({"ok": False, "error": "Not logged in"}), 401

    data = request.get_json(force=True)
    # data = {
    #   "relevance_judgments": [{"attribute": ..., "was_relevant": bool, "annotator_says_relevant": bool}, ...],
    #   "note": str
    # }
    record = {
        "task": "2",
        "index": idx,
        "relevance_judgments": data.get("relevance_judgments", []),
        "note": data.get("note", ""),
    }
    save_annotation("2", record)
    return jsonify({"ok": True})


@app.route("/task2/list")
def task2_list():
    """Return JSON list of checklist records for jump navigator."""
    if not annotator_name():
        return jsonify([])
    meta = get_checklist_meta_list()
    existing = load_existing_annotations("2")
    for m in meta:
        m["annotated"] = str(m["annotation_index"]) in existing
    return jsonify(meta)


# ─── Error template route ─────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify({
        "attrs_index_ready":     index_ready(ATTRS_INDEX_DB),
        "checklist_index_ready": index_ready(CHECKLIST_INDEX_DB),
        "convs_index_ready":     index_ready(CONVS_INDEX_DB),
        "attrs_count":           attrs_count(),
        "checklist_count":       checklist_count(),
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
