"""
local_vector_store.py

Very simple local "vector DB" based on JSONL + cosine similarity over
bag-of-words vectors. Lives in the same folder as execution.

Each stored record contains:
- user_query
- final_decision_text (full final answer from the council)
- metadata (any dict you want to keep, including summaries, rankings, etc.)

On retrieval, vectors are computed on the fly from text, and a cosine
similarity search is performed over all stored records.
"""

from __future__ import annotations

import os
import json
import math
import datetime
from typing import List, Dict, Any


DB_FILENAME = "council_vector_db.jsonl"


def _get_db_path() -> str:
    """
    Return the path of the local JSONL file used as vector DB.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DB_FILENAME)


def _tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercase, split on whitespace, strip punctuation.
    """
    if not text:
        return []
    text = text.lower()
    # Replace some punctuation with spaces
    for ch in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'", "\n", "\t"]:
        text = text.replace(ch, " ")
    tokens = [t for t in text.split(" ") if t]
    return tokens


def _text_to_vector(text: str) -> Dict[str, float]:
    """
    Convert text to a sparse frequency vector: {token: count}.
    """
    vec: Dict[str, float] = {}
    for token in _tokenize(text):
        vec[token] = vec.get(token, 0.0) + 1.0
    return vec


def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse frequency vectors.
    """
    if not v1 or not v2:
        return 0.0

    # Intersection of keys
    intersection = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in intersection)

    norm1 = math.sqrt(sum(v * v for v in v1.values()))
    norm2 = math.sqrt(sum(v * v for v in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


def store_council_run(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    stage3_result: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Append a single council run to the local JSONL "vector DB".

    The record contains:
        - id
        - timestamp
        - user_query
        - final_decision_text
        - metadata (augmented with a small final_decision_preview)
    """
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    final_decision_text = stage3_result.get("response", "") if stage3_result else ""
    final_preview = (final_decision_text[:300] + "...") if len(final_decision_text) > 300 else final_decision_text

    # We avoid storing the whole stage1/stage2 content to keep file smaller,
    # but you can add them into the record if you want full history.
    record = {
        "id": metadata.get("run_id"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user_query": user_query,
        "final_decision_text": final_decision_text,
        "final_decision_preview": final_preview,
        "metadata": metadata,
    }

    # If run_id not set, generate one
    if record["id"] is None:
        import uuid as _uuid
        record["id"] = str(_uuid.uuid4())
        metadata["run_id"] = record["id"]

    # Append as JSON line
    with open(db_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def retrieve_similar_context(
    user_query: str,
    top_k: int = 3,
    min_similarity: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Retrieve up to top_k previous runs with highest cosine similarity
    to the given user_query.

    Returns a list of records with an extra "similarity" field, sorted
    by similarity descending. If no DB file or no matches above min_similarity,
    returns [].
    """
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return []

    q_vec = _text_to_vector(user_query)
    if not q_vec:
        return []

    results: List[Dict[str, Any]] = []
    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            combined_text = (record.get("user_query", "") or "") + " \n " + (
                record.get("final_decision_text", "") or ""
            )
            r_vec = _text_to_vector(combined_text)
            sim = _cosine_similarity(q_vec, r_vec)
            if sim >= min_similarity:
                record["similarity"] = sim
                results.append(record)

    # Sort by similarity descending
    results.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)

    if top_k is not None and top_k > 0:
        results = results[:top_k]

    return results
