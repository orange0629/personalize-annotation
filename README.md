# Annotation Tool

Two annotation tasks for the LLM personalization research project.

## Tasks

**Task 1 — User Attribute Annotation**
For each user: view conversation history + inferred attributes.
Judge each attribute (makes sense / doesn't / unsure), note missing attributes.

**Task 2 — Checklist Relevance Annotation**
For each user–prompt pair: given the full user profile and the model's "relevant"
attribute subset, verify which items are truly relevant to the prompt.
Items are pre-checked (relevant); uncheck to mark as not relevant.

## Setup

### 1. Build indexes (one-time, runs in background)

```bash
cd annotation_tool/

# Build all three indexes (attrs, convs, checklist).
# The checklist index takes longest (70 GB file) — let it run.
python3 build_index.py

# Faster: skip the large checklist index to start Task 1 immediately
python3 build_index.py --skip-checklist
# Then later:
python3 build_index.py --skip-attrs --skip-convs
```

To rebuild the sample with a different N: python3 build_index.py --skip-attrs --skip-convs --skip-checklist --force --n-per-user 10

Indexes are stored in `data/indexes/` and only need to be built once.

### 2. Start the server

```bash
python3 app.py --port 5050
```

Open http://localhost:5050 in your browser.

### 3. Annotate

1. Enter your name → per-annotator annotation files are created.
2. Choose Task 1 or Task 2.
3. Use ← → arrow keys or the Prev/Next buttons to navigate.
4. Press **J** to jump to any instance.
5. Ctrl+S (or the Save button) to save the current annotation.

Annotations are saved in real-time to:
- `data/annotations/<name>_task1.jsonl`
- `data/annotations/<name>_task2.jsonl`

## Annotation format

### Task 1 output (`*_task1.jsonl`)
```json
{
  "task": "1",
  "index": 42,
  "annotator": "Alice",
  "timestamp": "2024-01-01T12:00:00",
  "attr_judgments": [
    {"judgment": "ok", "note": ""},
    {"judgment": "bad", "note": "This doesn't follow from the conversations"},
    {"judgment": "unsure", "note": "Could be right"}
  ],
  "missing_attrs": "The user seems to prefer Python over Java\n...",
  "overall_note": "Profile looks mostly accurate"
}
```

### Task 2 output (`*_task2.jsonl`)
```json
{
  "task": "2",
  "index": 7,
  "annotator": "Alice",
  "timestamp": "2024-01-01T12:00:00",
  "relevance_judgments": [
    {
      "attribute": "The user values safety.",
      "expected_behavior": "...",
      "relevance_score": 0.95,
      "was_relevant": true,
      "annotator_says_relevant": true,
      "changed": false
    },
    {
      "attribute": "The user prefers Python.",
      "was_relevant": true,
      "annotator_says_relevant": false,
      "changed": true
    }
  ],
  "note": "The Python preference is not relevant to this outdoor safety question."
}
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| ← / → | Previous / Next instance |
| J | Open jump-to modal |
| Escape | Close modal |
| Ctrl+S | Save annotation |
