"""
council_different_views.py

3-stage LLM Council orchestration with:
- A double reasoning run using 2 biased "views" / "ideas"
  that evaluate the initial responses from different perspectives.
- A random "what if" intervention that sometimes injects doubt
  into the reasoning of each view.
- Automatic idea generation per query, driven by a deliberation mode
  (e.g., balanced, risk_averse, exploratory).
- A local vector DB (JSONL in the same folder) that stores past runs and is
  consulted for retrieval-augmented context based on cosine similarity.
- Observability and inspection metadata for debugging and analysis.

Pipeline:

Stage 0 (optional retrieval):
    - If local vector DB has entries, retrieve similar past runs by cosine
      similarity on (question + final decision), and append a short summary
      block to the current user query as helpful background.

Stage 0.5 (idea generation):
    - Based on the user query and a "mode" (balanced / risk_averse / exploratory),
      automatically propose IDEA 1 and IDEA 2 descriptions, unless the caller
      provides them manually.

Stage 1:
    - Query COUNCIL_MODELS for the (possibly augmented) user question.
    - Each response gets a GUID (response_id).

Reasoning Graphs (between Stage 1 and Stage 2):
    - Randomly split Stage 1 responses into 2 groups.
    - Graph #1 is biased toward IDEA 1.
    - Graph #2 is biased toward IDEA 2.
    - Each graph has 2 nodes:
        1. Summarization node (neutral summaries + issues).
        2. Evaluation node, biased by the stance (idea) with:
            - ok / not_ok
            - stance_alignment: supporting | neutral | contradicting
           Sometimes a random external "what if" is injected to create doubt.

    - If ALL responses are deemed not_ok (across graphs),
      a "theoretical_context" string is created and appended
      as retroactive context to the original question for Stage 2 & 3.

Stage 2:
    - Each council model ranks anonymized responses.

Stage 3:
    - Chairman model synthesizes a final answer, seeing the
      possibly "enhanced" question (original + theoretical_context
      + any retrieved past context).

Post-run:
    - The full run is stored in the local vector DB for future retrieval.
    - Metadata contains observability info and an optional debug bundle.

Public entry points:
    - run_full_council(
          user_query: str,
          mode: str = "balanced",
          idea1_description: str | None = None,
          idea2_description: str | None = None,
          debug: bool = False,
      )
    - generate_conversation_title(user_query: str)
"""

from typing import List, Dict, Any, Tuple, Optional
import uuid
import random
import json
import asyncio

from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
from .local_vector_store import store_council_run, retrieve_similar_context


# Probability that a "what if" intervention is triggered in a reasoning graph
WHAT_IF_PROBABILITY = 0.35

# Model used to generate competing ideas per query (can be the same as CHAIRMAN)
IDEA_GENERATION_MODEL = CHAIRMAN_MODEL  # NEW


# ---------------------------------------------------------------------------
# Idea generation per query (automatic "IDEA 1" and "IDEA 2")
# ---------------------------------------------------------------------------

async def generate_ideas_for_query(
    user_query: str,
    mode: str = "balanced",
) -> Dict[str, Any]:
    """
    Generate two competing high-level approaches (IDEA 1 and IDEA 2)
    for the given question, guided by a deliberation mode.

    Modes (suggested semantics):
        - "balanced": IDEA 1 more cautious, IDEA 2 more ambitious.
        - "risk_averse": both ideas should be cautious, but IDEA 1 very conservative,
                         IDEA 2 slightly more flexible.
        - "exploratory": IDEA 1 somewhat grounded, IDEA 2 more radical/creative.

    Returns:
        {
          "idea1_description": str,
          "idea2_description": str,
          "mode_used": mode,
          "raw": str  # raw model output (for observability)
        }
    """
    mode = (mode or "balanced").strip().lower()
    if mode not in {"balanced", "risk_averse", "exploratory"}:
        mode = "balanced"

    mode_instructions = {
        "balanced": (
            "Generate one cautious, stability-focused idea (IDEA 1) "
            "and one more ambitious or innovative idea (IDEA 2)."
        ),
        "risk_averse": (
            "Both ideas should be cautious and focused on robustness and minimizing risk. "
            "IDEA 1 should be very conservative; IDEA 2 can be slightly more flexible "
            "but still clearly risk-averse."
        ),
        "exploratory": (
            "Generate one reasonably grounded idea (IDEA 1) and one highly exploratory, "
            "creative or unconventional idea (IDEA 2) that pushes boundaries."
        ),
    }

    prompt = f"""You are an AI that proposes two competing high-level ideas for approaching a question.

Question:
{user_query}

Deliberation mode: {mode}

Mode instructions:
{mode_instructions[mode]}

Your task:
1. Propose two distinct high-level approaches labeled IDEA 1 and IDEA 2.
2. Each idea should be 1-3 sentences, high-level but concrete enough to guide a strategy.
3. The two ideas must genuinely differ in assumptions or priorities, not just wording.

Return ONLY valid JSON with this exact structure, and no extra text:

{{
  "idea1_description": "...",
  "idea2_description": "..."
}}"""

    messages = [{"role": "user", "content": prompt}]
    resp = await query_model(IDEA_GENERATION_MODEL, messages)
    raw_text = resp.get("content", "") if resp else ""

    idea1 = "Cautious baseline approach derived from the question."
    idea2 = "More ambitious alternative approach derived from the question."

    if raw_text:
        try:
            parsed = json.loads(raw_text)
            idea1 = parsed.get("idea1_description", idea1) or idea1
            idea2 = parsed.get("idea2_description", idea2) or idea2
        except Exception:
            # Fallback: try to split text in two paragraphs or sentences as a hack
            parts = [p.strip() for p in raw_text.split("\n") if p.strip()]
            if len(parts) >= 2:
                idea1, idea2 = parts[0], parts[1]

    return {
        "idea1_description": idea1,
        "idea2_description": idea2,
        "mode_used": mode,
        "raw": raw_text,
    }


# ---------------------------------------------------------------------------
# Stage 1: collect individual responses
# ---------------------------------------------------------------------------

async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question (possibly retrieval-augmented)

    Returns:
        List of dicts with keys:
            - id: GUID for this response
            - model: model name
            - response: text content from the model
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "id": str(uuid.uuid4()),  # GUID to track this response
                "model": model,
                "response": response.get("content", ""),
            })

    return stage1_results


# ---------------------------------------------------------------------------
# Stage 2: peer rankings
# ---------------------------------------------------------------------------

async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The (possibly enhanced) user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of:
            - rankings list (one entry per council model)
            - label_to_model mapping (e.g. "Response A" -> "gpt-4.1")
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get("content", "")
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed,
            })

    return stage2_results, label_to_model


# ---------------------------------------------------------------------------
# Stage 3: chairman synthesis
# ---------------------------------------------------------------------------

async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The (possibly enhanced) user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question (possibly with internal theoretical context and retrieved past decisions appended):
{user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis.",
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get("content", ""),
    }


# ---------------------------------------------------------------------------
# Ranking parsing & aggregation
# ---------------------------------------------------------------------------

def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order (e.g. ["Response C", "Response A", "Response B"])
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            numbered_matches = re.findall(r"\d+\.\s*Response [A-Z]", ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r"Response [A-Z]", m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r"Response [A-Z]", ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    import re as _re
    matches = _re.findall(r"Response [A-Z]", ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with:
            - model: model name
            - average_rank: float
            - rankings_count: number of rankings included
        Sorted best to worst (lower average_rank is better).
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking["ranking"]

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions),
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x["average_rank"])

    return aggregate


# ---------------------------------------------------------------------------
# Title generation (optional helper)
# ---------------------------------------------------------------------------

async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use a fast, cheap model for title generation
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get("content", "New Conversation").strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')  # remove surrounding quotes if any

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


# ---------------------------------------------------------------------------
# "What if" generator
# ---------------------------------------------------------------------------

async def _maybe_generate_what_if(
    user_query: str,
    stance_label: str,
    stance_description: str,
) -> str:
    """
    With probability WHAT_IF_PROBABILITY, generate a short 'what if' scenario
    from an external model that challenges the current stance.

    Returns:
        - Empty string if no what-if is generated.
        - Otherwise, a short text starting with "What if".
    """
    if random.random() >= WHAT_IF_PROBABILITY:
        return ""

    prompt = f"""You are an external contrarian AI injecting doubt into an ongoing internal debate.

Question:
{user_query}

Current stance ({stance_label}):
{stance_description}

Based on this stance, propose ONE short 'what if' scenario that could challenge,
complicate, or cast doubt on this stance.

Requirements:
- Start with the exact words "What if".
- Maximum 2 sentences.
- Do not explain or justify it, just state the scenario.

What-if scenario:"""

    messages = [{"role": "user", "content": prompt}]
    resp = await query_model(CHAIRMAN_MODEL, messages)

    if resp is None:
        return ""

    text = resp.get("content", "").strip()
    # Light cleanup: ensure it starts with "What if" if there is any content
    if text and not text.lower().startswith("what if"):
        text = "What if " + text.lstrip(".,;:!? ").lstrip()

    return text


# ---------------------------------------------------------------------------
# Reasoning graphs with different views (IDEA 1 vs IDEA 2)
# ---------------------------------------------------------------------------

async def _run_single_reasoning_graph(
    graph_id: int,
    user_query: str,
    group: List[Dict[str, Any]],
    stance_label: str,
    stance_description: str,
) -> Dict[str, Any]:
    """
    Run a single 2-node reasoning graph over a subset of Stage 1 responses,
    biased toward a specific idea / stance, with a possible 'what if' intervention.

    Node 1: Summarization (as neutral as possible).
    Node 2: Evaluation from the perspective of the stance,
            sometimes influenced by an external 'what if' idea.

    stance_label: short id, e.g. "IDEA 1"
    stance_description: natural language description of the stance.
    """
    if not group:
        return {
            "graph_id": graph_id,
            "stance_label": stance_label,
            "stance_description": stance_description,
            "what_if": "",
            "group_ids": [],
            "raw_summary": "",
            "raw_evaluation": "",
            "summaries": [],
            "evaluations": [],
        }

    # Compact JSON payload for the group
    group_payload = [
        {
            "response_id": r["id"],
            "model": r["model"],
            "answer": r["response"],
        }
        for r in group
    ]

    # Possibly generate a 'what if' perturbation for this stance
    what_if_text = await _maybe_generate_what_if(
        user_query=user_query,
        stance_label=stance_label,
        stance_description=stance_description,
    )

    # ---------- Node 1: Summarization (neutral) ----------
    summary_prompt = f"""You are reasoning graph #{graph_id}, node 1 (summarization).

Your stance: {stance_label} = {stance_description}
For this node, be as neutral and descriptive as possible.

Question:
{user_query}

You are given candidate answers from different models as JSON:

{json.dumps(group_payload)}

For each item in the JSON array, create:
- a very short neutral summary of the answer (max 3 sentences)
- a short description of potential issues or weaknesses (if any)

Return ONLY valid JSON with this exact structure, and no extra text:

{{
  "summaries": [
    {{
      "response_id": "...",
      "summary": "...",
      "potential_issues": "..."
    }},
    ...
  ]
}}"""

    summary_messages = [{"role": "user", "content": summary_prompt}]
    summary_resp = await query_model(CHAIRMAN_MODEL, summary_messages)
    summary_text = summary_resp.get("content", "") if summary_resp else ""

    summaries: List[Dict[str, Any]] = []
    summary_for_eval: Any = summary_text  # either dict or raw text

    try:
        summary_json = json.loads(summary_text)
        summaries = summary_json.get("summaries", [])
        summary_for_eval = summary_json
    except Exception:
        summaries = []

    # Prepare extra what-if text for evaluation prompt, if any
    extra_what_if_segment = ""
    if what_if_text:
        extra_what_if_segment = f"""

An external model has injected the following 'what if' challenge that might
introduce doubt about {stance_label}. Consider it when you judge the answers,
but do not assume it is true; treat it as a hypothesis that raises potential
failure modes or edge cases:

{what_if_text}
"""

    # ---------- Node 2: Evaluation (biased toward stance + what-if) ----------
    eval_prompt = f"""You are reasoning graph #{graph_id}, node 2 (evaluation).

Your stance: {stance_label} = {stance_description}

Your job is to evaluate each answer FROM THIS STANCE.

Question:
{user_query}

You are given:
1. The original answers (as JSON):
{json.dumps(group_payload)}

2. The analysis summaries for each answer (as JSON or text):
{json.dumps(summary_for_eval) if isinstance(summary_for_eval, dict) else summary_for_eval}
{extra_what_if_segment}
For every "response_id":
1. Decide whether, from the perspective of {stance_label}, the answer is:
   - "supporting" the idea
   - "neutral" toward the idea
   - "contradicting" the idea

2. Decide whether the answer is acceptable ("ok") or not acceptable ("not_ok")
   *from the same stance*. Be conservative: mark "ok" only if the answer is
   substantially correct and reasonably complete, given the stance and the
   potential doubts raised by the what-if (if present).

Return ONLY valid JSON with this exact structure, and no extra text:

{{
  "evaluations": [
    {{
      "response_id": "...",
      "ok": true,
      "stance_alignment": "supporting" | "neutral" | "contradicting",
      "reason": "..."
    }},
    ...
  ]
}}"""

    eval_messages = [{"role": "user", "content": eval_prompt}]
    eval_resp = await query_model(CHAIRMAN_MODEL, eval_messages)
    eval_text = eval_resp.get("content", "") if eval_resp else ""

    evaluations: List[Dict[str, Any]] = []
    try:
        eval_json = json.loads(eval_text)
        evaluations = eval_json.get("evaluations", [])
    except Exception:
        evaluations = []

    return {
        "graph_id": graph_id,
        "stance_label": stance_label,
        "stance_description": stance_description,
        "what_if": what_if_text,
        "group_ids": [r["id"] for r in group],
        "raw_summary": summary_text,
        "raw_evaluation": eval_text,
        "summaries": summaries,
        "evaluations": evaluations,
    }


async def run_reasoning_graphs(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    idea1_description: str,
    idea2_description: str,
) -> Dict[str, Any]:
    """
    Run a double reasoning pass with 2 graphs in parallel.

    Graph 1 is biased toward IDEA 1.
    Graph 2 is biased toward IDEA 2.
    Each graph may receive a random 'what if' from an external model that
    injects doubt about its stance.

    Returns:
        {
          "graphs": [...],
          "evaluations_by_id": {response_id: [eval, ...]},
          "all_not_ok": bool,
          "theoretical_context": str
        }
    """
    if len(stage1_results) < 2:
        return {
            "graphs": [],
            "evaluations_by_id": {},
            "all_not_ok": False,
            "theoretical_context": "",
        }

    # Randomly shuffle and split into 2 groups
    shuffled = stage1_results[:]
    random.shuffle(shuffled)
    mid = len(shuffled) // 2
    group1 = shuffled[:mid]
    group2 = shuffled[mid:]

    # Run both graphs in parallel
    graph_tasks = [
        _run_single_reasoning_graph(
            graph_id=1,
            user_query=user_query,
            group=group1,
            stance_label="IDEA 1",
            stance_description=idea1_description,
        ),
        _run_single_reasoning_graph(
            graph_id=2,
            user_query=user_query,
            group=group2,
            stance_label="IDEA 2",
            stance_description=idea2_description,
        ),
    ]
    graphs = await asyncio.gather(*graph_tasks)

    # Aggregate evaluations by response_id
    evaluations_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for g in graphs:
        for ev in g.get("evaluations", []):
            rid = ev.get("response_id")
            if not rid:
                continue
            # Attach stance info to each evaluation
            ev_with_stance = {
                **ev,
                "stance_label": g.get("stance_label"),
                "stance_description": g.get("stance_description"),
                "what_if": g.get("what_if", ""),
            }
            evaluations_by_id.setdefault(rid, []).append(ev_with_stance)

    # Decide if ALL Stage 1 responses have been marked "not_ok"
    all_not_ok = False
    if evaluations_by_id:
        all_not_ok = True
        for r in stage1_results:
            rid = r["id"]
            evs = evaluations_by_id.get(rid, [])
            if not evs:
                all_not_ok = False
                break
            # If any evaluation says ok=True, then not all are bad
            if any(ev.get("ok", False) for ev in evs):
                all_not_ok = False
                break

    # Build theoretical context text, including stance alignment
    theoretical_context = ""
    if all_not_ok:
        lines: List[str] = []
        for r in stage1_results:
            rid = r["id"]
            model = r["model"]
            evs = evaluations_by_id.get(rid, [])
            if not evs:
                continue
            # For brevity, include all evaluations (can be filtered by stance if needed)
            for ev in evs:
                stance = ev.get("stance_label")
                alignment = ev.get("stance_alignment")
                reason = ev.get("reason", "")
                what_if_text = ev.get("what_if", "")
                base_line = (
                    f"- Model {model} (id={rid}) under {stance} "
                    f"viewed as {alignment} and not_ok: {reason}"
                )
                if what_if_text:
                    base_line += f" (influenced by what-if: {what_if_text})"
                lines.append(base_line)

        if lines:
            theoretical_context = (
                "Internal reasoning graphs with two opposing ideas and occasional "
                "external 'what if' interventions judged all initial answers as "
                "not fully satisfactory. From IDEA 1 and IDEA 2 perspectives, "
                "they observed:\n"
                + "\n".join(lines)
            )

    return {
        "graphs": graphs,
        "evaluations_by_id": evaluations_by_id,
        "all_not_ok": all_not_ok,
        "theoretical_context": theoretical_context,
    }


# ---------------------------------------------------------------------------
# Orchestrator: run_full_council with retrieval + programmable ideas + observability
# ---------------------------------------------------------------------------

async def run_full_council(
    user_query: str,
    mode: str = "balanced",
    idea1_description: Optional[str] = None,
    idea2_description: Optional[str] = None,
    debug: bool = False,
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process with:

    - Stage 0: consult local vector DB and inject retrieved context if available.
    - Stage 0.5: generate IDEA 1 and IDEA 2 automatically based on the question
                 and the deliberation mode, unless manual descriptions are given.
    - Stage 1: collect individual responses.
    - Reasoning graphs: 2 biased views with optional what-if interventions.
    - Stage 2: peer rankings.
    - Stage 3: chairman synthesis.
    - Post-run: store this run into the local vector DB.

    Args:
        user_query: The user's question
        mode: deliberation mode for idea generation ("balanced", "risk_averse", "exploratory")
        idea1_description: manual override for IDEA 1 (if provided, skips auto-generation for that idea)
        idea2_description: manual override for IDEA 2
        debug: if True, metadata includes heavier debug info (raw retrieved entries, enhanced query, etc.)

    Returns:
        Tuple of:
            - stage1_results
            - stage2_results
            - stage3_result
            - metadata (label_to_model, aggregate_rankings, reasoning_graphs, ideas, observability, ...)
    """
    original_user_query = user_query

    # --- Stage 0: Retrieval from local vector DB (if any) ---
    retrieved = retrieve_similar_context(original_user_query, top_k=3, min_similarity=0.15)

    retrieved_block = ""
    if retrieved:
        lines = []
        for idx, rec in enumerate(retrieved, start=1):
            sim = rec.get("similarity", 0.0)
            q = (rec.get("user_query", "") or "").strip()
            preview = (rec.get("final_decision_preview", "") or "").strip()
            lines.append(
                f"{idx}. Similarity={sim:.2f}\n"
                f"   Past question: {q}\n"
                f"   Past final decision (truncated): {preview}"
            )
        retrieved_block = (
            "The following are past council decisions for similar questions. "
            "You may reuse their insights if helpful, but do not copy them blindly:\n"
            + "\n".join(lines)
        )
        user_query = original_user_query + "\n\n" + retrieved_block

    # --- Stage 0.5: Idea generation / configuration ---
    auto_ideas = False
    idea_gen_raw = ""

    if not idea1_description or not idea2_description:
        auto_ideas = True
        ideas = await generate_ideas_for_query(original_user_query, mode=mode)
        if not idea1_description:
            idea1_description = ideas["idea1_description"]
        if not idea2_description:
            idea2_description = ideas["idea2_description"]
        idea_gen_raw = ideas.get("raw", "")
        mode_used = ideas.get("mode_used", mode)
    else:
        mode_used = (mode or "balanced").strip().lower()

    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query)

    # If no models responded successfully, return error
    if not stage1_results:
        error_result = {
            "model": "error",
            "response": "All models failed to respond. Please try again.",
        }
        # Still optionally store something, but probably not useful; here we skip.
        return [], [], error_result, {}

    # Run 2 biased reasoning graphs (IDEA 1 vs IDEA 2, with possible what-if)
    reasoning_meta = await run_reasoning_graphs(
        user_query,
        stage1_results,
        idea1_description=idea1_description,
        idea2_description=idea2_description,
    )

    # Retroaction: if all responses are not_ok, inject theoretical data
    # as additional context into the question seen by Stage 2 & 3.
    enhanced_query = user_query
    if reasoning_meta.get("all_not_ok") and reasoning_meta.get("theoretical_context"):
        enhanced_query = (
            f"{user_query}\n\n"
            "Internal meta-analysis from two opposing ideas "
            "(IDEA 1 vs IDEA 2), with occasional external 'what if' "
            "interventions. Treat this as theoretical context, "
            "not as ground truth:\n"
            f"{reasoning_meta['theoretical_context']}"
        )

    # Stage 2: Collect rankings (models see enhanced query if applicable)
    stage2_results, label_to_model = await stage2_collect_rankings(
        enhanced_query,
        stage1_results,
    )

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer (also gets enhanced query)
    stage3_result = await stage3_synthesize_final(
        enhanced_query,
        stage1_results,
        stage2_results,
    )

    # Prepare metadata, including reasoning graphs and idea descriptions
    run_id = str(uuid.uuid4())

    # Observability: summarize what-if events and retrieval matches
    what_if_events = []
    for g in reasoning_meta.get("graphs", []):
        w = g.get("what_if", "")
        if w:
            what_if_events.append({
                "graph_id": g.get("graph_id"),
                "stance_label": g.get("stance_label"),
                "what_if": w,
            })

    retrieval_matches_summary = []
    if retrieved:
        for rec in retrieved:
            retrieval_matches_summary.append({
                "id": rec.get("id"),
                "similarity": rec.get("similarity", 0.0),
                "question_preview": (rec.get("user_query", "") or "")[:200],
            })

    observability = {
        "mode": mode_used,
        "auto_ideas": auto_ideas,
        "retrieval_used": bool(retrieved),
        "retrieval_matches": retrieval_matches_summary,
        "what_if_events": what_if_events,
        "all_not_ok": reasoning_meta.get("all_not_ok", False),
    }

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "reasoning_graphs": reasoning_meta,
        "ideas": {
            "idea1_description": idea1_description,
            "idea2_description": idea2_description,
            "mode": mode_used,
            "auto_generated": auto_ideas,
            "idea_generation_raw": idea_gen_raw,
        },
        "retrieval_used": bool(retrieved),
        "observability": observability,
    }

    # Debug bundle (heavier artifacts for inspection)
    if debug:
        metadata["debug"] = {
            "original_user_query": original_user_query,
            "retrieved_raw": retrieved,
            "retrieved_block": retrieved_block,
            "final_enhanced_query_for_stage2_and_3": enhanced_query,
        }

    # --- Post-run: store everything in local vector DB ---
    store_council_run(
        user_query=original_user_query,
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        stage3_result=stage3_result,
        metadata=metadata,
    )

    return stage1_results, stage2_results, stage3_result, metadata
