"""
spiking_brain_ideas.py

"Brain-like" orchestration layer on top of council_different_views.

Concepts:
- 6 brain zones (analysis, world_modeling, planning, creative, social, meta),
  each with its own role, neuron count, thresholds, decay, and doubt sensitivity.
- A simple spiking neural simulation that turns a query into an activation map.
- Zone relevance is computed via a combination of:
    - a keyword-based heuristic, and
    - an LLM classifier (JSON output), weighted by config.
- Primary and secondary zones chosen from spiking activations.
- Zone-guided ideas:
    - IDEA 1 = primary zone perspective
    - IDEA 2 = secondary zone perspective
- A brain-level "doubt interrupt" that sometimes challenges the first council
  run and triggers a second run with meta-doubt context appended.
- Spiking parameters are configurable via spiking_config.

Public entry:
    async def run_brain_council_spiking_brain(
        user_query: str,
        debug: bool = False,
        spiking_config: Optional[Dict[str, Any]] = None,
    )

This function:
    - runs spiking "brain" dynamics,
    - builds zone-based ideas and a deliberation mode,
    - calls council_different_views.run_full_council(),
    - optionally triggers a brain-level doubt interrupt and re-runs,
    - returns (stage1_results, stage2_results, stage3_result, metadata).
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, Optional
import math
import random
import uuid
import json

from .council_different_views import run_full_council
from .openrouter import query_model
from .config import CHAIRMAN_MODEL


# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

ZONES: Dict[str, Dict[str, Any]] = {
    "analysis": {
        "id": "analysis",
        "name": "Analytical / Logical",
        "neuron_count": 32,
        "base_threshold": 1.0,
        "decay": 0.8,
        "noise_scale": 0.25,
        "doubt_sensitivity": 0.25,
        "default_mode": "risk_averse",
        "role_description": (
            "Focuses on precise, step-by-step reasoning, mathematical clarity, "
            "and explicit assumptions. Prefers safe, logically justified conclusions."
        ),
        "keywords": [
            "math", "prove", "proof", "derive", "calculate", "optimize",
            "algorithm", "complexity", "theorem", "logic"
        ],
    },
    "world_modeling": {
        "id": "world_modeling",
        "name": "World Modeling",
        "neuron_count": 32,
        "base_threshold": 1.0,
        "decay": 0.8,
        "noise_scale": 0.2,
        "doubt_sensitivity": 0.2,
        "default_mode": "balanced",
        "role_description": (
            "Builds causal models of how systems behave over time. Considers "
            "uncertainty, external constraints, and long-term dynamics."
        ),
        "keywords": [
            "predict", "forecast", "impact", "system", "market", "environment",
            "economy", "trend", "behavior", "simulate"
        ],
    },
    "planning": {
        "id": "planning",
        "name": "Planning / Execution",
        "neuron_count": 24,
        "base_threshold": 1.1,
        "decay": 0.85,
        "noise_scale": 0.2,
        "doubt_sensitivity": 0.3,
        "default_mode": "balanced",
        "role_description": (
            "Organizes concrete steps, prioritizes tasks, and trades off effort, "
            "time, and resources. Thinks in roadmaps, milestones, and constraints."
        ),
        "keywords": [
            "plan", "roadmap", "steps", "milestone", "timeline", "strategy",
            "execute", "implementation", "schedule"
        ],
    },
    "creative": {
        "id": "creative",
        "name": "Creative / Imaginative",
        "neuron_count": 24,
        "base_threshold": 0.9,
        "decay": 0.75,
        "noise_scale": 0.35,
        "doubt_sensitivity": 0.15,
        "default_mode": "exploratory",
        "role_description": (
            "Generates novel ideas, analogies, and unconventional perspectives. "
            "Comfortable with ambiguity and speculative thinking."
        ),
        "keywords": [
            "idea", "brainstorm", "create", "story", "design", "concept",
            "innovative", "novel", "invention", "vision"
        ],
    },
    "social": {
        "id": "social",
        "name": "Social / Emotional",
        "neuron_count": 20,
        "base_threshold": 1.1,
        "decay": 0.8,
        "noise_scale": 0.25,
        "doubt_sensitivity": 0.25,
        "default_mode": "balanced",
        "role_description": (
            "Reasons about people, teams, users, motivations, and communication. "
            "Considers empathy, persuasion, and user experience."
        ),
        "keywords": [
            "people", "team", "user", "customer", "stakeholder", "communication",
            "feedback", "emotion", "motivation", "relationship"
        ],
    },
    "meta": {
        "id": "meta",
        "name": "Meta / Self-Reflection",
        "neuron_count": 16,
        "base_threshold": 1.2,
        "decay": 0.85,
        "noise_scale": 0.3,
        "doubt_sensitivity": 0.4,
        "default_mode": "risk_averse",
        "role_description": (
            "Reflects on the reasoning process itself, looks for hidden assumptions, "
            "failure modes, and ways the whole approach could be wrong."
        ),
        "keywords": [
            "reasoning", "think", "improve", "evaluate", "reflection",
            "bias", "assumption", "metacognitive"
        ],
    },
}


# ---------------------------------------------------------------------------
# Spiking configuration defaults (can be overridden per run)
# ---------------------------------------------------------------------------

BRAIN_SPIKING_DEFAULTS: Dict[str, Any] = {
    # How many discrete timesteps of spiking dynamics to run
    "steps": 3,
    # Multiply each zone's noise_scale by this factor
    "global_noise_scale": 1.0,
    # Multiply effective decay by this factor (clamped to [0, 1])
    "global_decay_scale": 1.0,
    # Whether to use the LLM classifier for zone relevance
    "use_llm_relevance": True,
    # Weight of LLM relevance vs heuristic relevance (0 = only heuristic, 1 = only LLM)
    "llm_relevance_weight": 0.6,
}


# ---------------------------------------------------------------------------
# Simple spiking neuron simulation
# ---------------------------------------------------------------------------

def _init_brain_state() -> Dict[str, Any]:
    """
    Initialize neuron states for each zone.

    BrainState structure:
        {
          "zones": {
             zone_id: {
                 "neurons": [
                     {"potential": float, "threshold": float, "decay": float}, ...
                 ],
                 "last_spikes": [0/1, ...]
             },
             ...
          }
        }
    """
    brain_state: Dict[str, Any] = {"zones": {}}

    for zid, zinfo in ZONES.items():
        neurons = []
        n = zinfo["neuron_count"]
        base_th = zinfo["base_threshold"]
        decay = zinfo["decay"]
        for _ in range(n):
            th = base_th + random.gauss(0.0, 0.05)  # small jitter on threshold
            neurons.append({
                "potential": 0.0,
                "threshold": th,
                "decay": decay,
            })
        brain_state["zones"][zid] = {
            "neurons": neurons,
            "last_spikes": [0] * n,
        }

    return brain_state


def _compute_zone_relevance_keywords(user_query: str) -> Dict[str, float]:
    """
    Heuristic zone relevance based on keyword hits.
    Returns a dict zone_id -> relevance_score in [0, 1] (normalized).
    """
    text = (user_query or "").lower()
    relevance: Dict[str, float] = {}

    for zid, zinfo in ZONES.items():
        score = 0.0
        for kw in zinfo.get("keywords", []):
            if kw in text:
                score += 1.0
        # Small baseline so unused zones aren't totally dead
        if score == 0.0:
            score = 0.2
        relevance[zid] = score

    max_score = max(relevance.values()) if relevance else 1.0
    if max_score <= 0:
        return {zid: 0.5 for zid in ZONES.keys()}

    for zid in relevance:
        relevance[zid] = relevance[zid] / max_score

    return relevance


async def _classify_zone_relevance_llm(user_query: str) -> Dict[str, float]:
    """
    Use an LLM to assign relevance scores (0-1) to each zone, based on the question.

    Returns:
        dict zone_id -> relevance_score in [0, 1]. Missing or invalid zones are ignored.
        If the call fails or parsing fails, returns {}.
    """
    zones_description = []
    for zid, zinfo in ZONES.items():
        zones_description.append(
            {
                "id": zid,
                "name": zinfo["name"],
                "role": zinfo["role_description"],
            }
        )

    prompt = f"""You are a classifier that assigns relevance scores to different brain zones.

Question:
{user_query}

Brain zones:
{json.dumps(zones_description, ensure_ascii=False, indent=2)}

Your task:
1. For each zone, assign a relevance score between 0.0 and 1.0 (inclusive), where:
   - 0.0 means "essentially irrelevant",
   - 1.0 means "absolutely central for reasoning about this question".
2. Favor multiple zones being moderately active when in doubt, instead of only one.

Return ONLY valid JSON mapping zone ids to scores, for example:

{{
  "analysis": 0.8,
  "world_modeling": 0.7,
  "planning": 0.4,
  "creative": 0.3,
  "social": 0.2,
  "meta": 0.5
}}"""

    messages = [{"role": "user", "content": prompt}]
    resp = await query_model(CHAIRMAN_MODEL, messages)
    raw_text = resp.get("content", "") if resp else ""
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {}

    llm_scores: Dict[str, float] = {}
    for zid in ZONES.keys():
        v = parsed.get(zid)
        try:
            if v is None:
                continue
            v = float(v)
            # clamp to [0, 1]
            v = max(0.0, min(1.0, v))
            llm_scores[zid] = v
        except Exception:
            continue

    return llm_scores


async def _compute_zone_relevance(
    user_query: str,
    spiking_config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute zone relevance as a combination of heuristic and LLM-based scores,
    according to spiking_config options.

    spiking_config keys:
        - use_llm_relevance: bool
        - llm_relevance_weight: float in [0, 1]
    """
    heuristic = _compute_zone_relevance_keywords(user_query)

    use_llm = bool(spiking_config.get("use_llm_relevance", True))
    llm_weight = float(spiking_config.get("llm_relevance_weight", 0.6))
    llm_weight = max(0.0, min(1.0, llm_weight))

    if not use_llm:
        return heuristic

    llm_scores = await _classify_zone_relevance_llm(user_query)
    if not llm_scores:
        return heuristic

    combined: Dict[str, float] = {}
    for zid in ZONES.keys():
        h = heuristic.get(zid, 0.5)
        l = llm_scores.get(zid, h)
        combined[zid] = (1.0 - llm_weight) * h + llm_weight * l

    # Normalize to [0, 1] to keep things comparable
    max_val = max(combined.values()) if combined else 1.0
    if max_val > 0.0:
        for zid in combined:
            combined[zid] = combined[zid] / max_val

    return combined


def _run_spiking_dynamics(
    brain_state: Dict[str, Any],
    zone_relevance: Dict[str, float],
    steps: int,
    global_noise_scale: float,
    global_decay_scale: float,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """
    Run a few timesteps of spiking dynamics.

    Args:
        brain_state: initial brain state
        zone_relevance: zone_id -> relevance score in [0, 1]
        steps: number of simulation steps
        global_noise_scale: factor applied to each zone's noise_scale
        global_decay_scale: factor applied to effective decay (clamped to [0, 1])

    Returns:
        updated_brain_state,
        activation_scores: zone_id -> [0, 1],
        spiking_log: {
            zone_id: {
                "total_spikes": int,
                "expected_from_relevance": float,
            },
            ...
        }
    """
    zone_spike_counts = {zid: 0 for zid in ZONES.keys()}
    global_decay_scale = max(0.0, min(1.0, global_decay_scale))

    for _ in range(max(1, steps)):
        for zid, zone_state in brain_state["zones"].items():
            zinfo = ZONES[zid]
            neurons = zone_state["neurons"]
            base_noise = zinfo["noise_scale"]
            noise_scale = base_noise * global_noise_scale
            relevance = zone_relevance.get(zid, 0.0)

            new_spikes = []
            for neuron in neurons:
                potential = neuron["potential"]
                # Input current: relevance + random noise
                input_current = relevance + random.gauss(0.0, noise_scale)
                eff_decay = neuron["decay"] * global_decay_scale
                eff_decay = max(0.0, min(1.0, eff_decay))

                potential = eff_decay * potential + input_current

                if potential >= neuron["threshold"]:
                    spike = 1
                    potential = 0.0  # reset after spike
                else:
                    spike = 0

                neuron["potential"] = potential
                new_spikes.append(spike)
                zone_spike_counts[zid] += spike

            zone_state["last_spikes"] = new_spikes

    activation_scores: Dict[str, float] = {}
    spiking_log: Dict[str, Any] = {}

    for zid, zinfo in ZONES.items():
        n = zinfo["neuron_count"]
        total_spikes = zone_spike_counts[zid]
        # Normalize by (neurons * steps)
        denom = float(max(n * max(1, steps), 1))
        activation = total_spikes / denom
        activation_scores[zid] = activation
        spiking_log[zid] = {
            "total_spikes": total_spikes,
            "expected_from_relevance": zone_relevance.get(zid, 0.0),
        }

    # Re-normalize activations to [0, 1] for easier interpretation
    max_act = max(activation_scores.values()) if activation_scores else 1.0
    if max_act > 0:
        for zid in activation_scores:
            activation_scores[zid] = activation_scores[zid] / max_act

    return brain_state, activation_scores, spiking_log


# ---------------------------------------------------------------------------
# Zone & mode selection
# ---------------------------------------------------------------------------

def _select_primary_secondary(
    activation_scores: Dict[str, float]
) -> Tuple[str, Optional[str]]:
    """
    Pick primary and secondary zones based on activation scores.
    Returns (primary_zone_id, secondary_zone_id or None).
    """
    if not activation_scores:
        # fallback: analysis + world_modeling
        return "analysis", "world_modeling"

    sorted_zones = sorted(
        activation_scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    primary = sorted_zones[0][0]
    secondary = sorted_zones[1][0] if len(sorted_zones) > 1 else None
    return primary, secondary


def _zone_to_mode(primary_zone_id: str) -> str:
    """
    Translate primary zone into a deliberation mode for the council.
    """
    z = ZONES.get(primary_zone_id)
    if z is None:
        return "balanced"

    mode = z.get("default_mode", "balanced")
    mode = (mode or "balanced").strip().lower()
    if mode not in {"balanced", "risk_averse", "exploratory"}:
        mode = "balanced"
    return mode


# ---------------------------------------------------------------------------
# Zone-based idea generation (LLM)
# ---------------------------------------------------------------------------

async def _build_zone_based_ideas(
    user_query: str,
    primary_zone_id: str,
    secondary_zone_id: Optional[str],
) -> Tuple[str, str, str]:
    """
    Use an LLM to produce IDEA 1 and IDEA 2 based on zone roles.

    IDEA 1: primary zone perspective
    IDEA 2: secondary zone perspective (or a variant of primary if None)

    Returns:
        idea1_description, idea2_description, raw_text_from_llm
    """
    primary = ZONES[primary_zone_id]
    if secondary_zone_id is None:
        secondary_zone_id = primary_zone_id
    secondary = ZONES[secondary_zone_id]

    prompt = f"""You are orchestrating two different high-level approaches to a question,
each from a different "brain zone" with a distinct reasoning style.

Question:
{user_query}

Primary zone:
- ID: {primary['id']}
- Name: {primary['name']}
- Role: {primary['role_description']}

Secondary zone:
- ID: {secondary['id']}
- Name: {secondary['name']}
- Role: {secondary['role_description']}

Your task:
1. Propose IDEA 1: a high-level approach that reflects the PRIMARY zone's style.
2. Propose IDEA 2: a high-level approach that reflects the SECONDARY zone's style.
3. Each idea should be 1-3 sentences, high-level but concrete enough to shape strategy.
4. The two ideas must genuinely differ in assumptions or priorities, not just wording.

Return ONLY valid JSON with this exact structure, and no extra text:

{{
  "idea1_description": "...",
  "idea2_description": "..."
}}"""

    messages = [{"role": "user", "content": prompt}]
    resp = await query_model(CHAIRMAN_MODEL, messages)
    raw_text = resp.get("content", "") if resp else ""

    fallback1 = (
        f"A high-level approach guided by the {primary['name']} zone: "
        f"{primary['role_description']}"
    )
    fallback2 = (
        f"A high-level approach guided by the {secondary['name']} zone: "
        f"{secondary['role_description']}"
    )

    idea1 = fallback1
    idea2 = fallback2

    if raw_text:
        try:
            parsed = json.loads(raw_text)
            idea1 = parsed.get("idea1_description", idea1) or idea1
            idea2 = parsed.get("idea2_description", idea2) or idea2
        except Exception:
            parts = [p.strip() for p in raw_text.split("\n") if p.strip()]
            if len(parts) >= 2:
                idea1, idea2 = parts[0], parts[1]

    return idea1, idea2, raw_text


# ---------------------------------------------------------------------------
# Brain-level doubt interrupt (LLM)
# ---------------------------------------------------------------------------

async def _maybe_trigger_brain_doubt(
    user_query: str,
    activation_scores: Dict[str, float],
    spiking_log: Dict[str, Any],
    primary_zone_id: str,
    secondary_zone_id: Optional[str],
    council_metadata: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Decide (probabilistically) whether to trigger a brain-level doubt interrupt.

    Uses:
      - zone doubting tendencies (doubt_sensitivity),
      - spiking "surprises" (zones that spiked more than relevance suggests),
      - council metadata (e.g. all_not_ok) to boost probability.

    If triggered, uses an LLM to generate a meta-doubt text that
    will be appended to the question in a second council run.

    Returns:
        meta_doubt_text ("" if no doubt triggered),
        doubt_info (for observability)
    """
    secondary_zone_id = secondary_zone_id or primary_zone_id

    primary_zone = ZONES[primary_zone_id]
    secondary_zone = ZONES[secondary_zone_id]

    # Base probability from zones
    base_prob = 0.5 * (
        primary_zone["doubt_sensitivity"] + secondary_zone["doubt_sensitivity"]
    )

    # Check for "surprise" spikes: high spikes relative to relevance
    surprise_boost = 0.0
    for zid, slog in spiking_log.items():
        spikes = slog["total_spikes"]
        rel = slog["expected_from_relevance"]
        if rel <= 0.1 and spikes > 0:
            surprise_boost += 0.05
        if rel > 0.0 and spikes > 3 * rel * 10:
            surprise_boost += 0.05

    # Council signals: all_not_ok, strong disagreements, etc.
    reasoning_meta = council_metadata.get("reasoning_graphs", {})
    all_not_ok = bool(reasoning_meta.get("all_not_ok", False))

    council_boost = 0.0
    if all_not_ok:
        council_boost += 0.3

    aggregate = council_metadata.get("aggregate_rankings", [])
    if aggregate:
        ranks = [item.get("average_rank", 0.0) for item in aggregate]
        if len(ranks) >= 2:
            mean_r = sum(ranks) / len(ranks)
            var = sum((r - mean_r) ** 2 for r in ranks) / len(ranks)
            std = math.sqrt(var)
            if std > 0.7:
                council_boost += 0.15

    probability = min(0.9, max(0.0, base_prob + surprise_boost + council_boost))
    roll = random.random()

    doubt_info = {
        "base_prob": base_prob,
        "surprise_boost": surprise_boost,
        "council_boost": council_boost,
        "final_probability": probability,
        "random_roll": roll,
        "triggered": False,
    }

    if roll >= probability:
        # No doubt interrupt
        return "", doubt_info

    doubt_info["triggered"] = True

    meta_prompt = f"""You are the META / self-reflection zone of a brain-like system.

The system has already produced a first round of reasoning and a provisional answer
to the following question:

Question:
{user_query}

Primary zone: {primary_zone['id']} ({primary_zone['name']})
Secondary zone: {secondary_zone['id']} ({secondary_zone['name']})

From your meta perspective, assume the current overall reasoning might be flawed.

Your task:
1. Identify 1-3 deep failure modes, blind spots, or alternative framings that could
   make the current line of reasoning seriously incomplete or misleading.
2. Focus on high-level structural concerns (assumptions, missing dimensions, wrong
   objective), not tiny details.
3. Be concise but concrete.

Return a short text starting with: "META DOUBT:" then your critique in 1-3 sentences.
"""

    messages = [{"role": "user", "content": meta_prompt}]
    resp = await query_model(CHAIRMAN_MODEL, messages)
    meta_text = resp.get("content", "").strip() if resp else ""

    if meta_text and not meta_text.upper().startswith("META DOUBT:"):
        meta_text = "META DOUBT: " + meta_text

    doubt_info["meta_doubt_excerpt"] = meta_text[:300] if meta_text else ""
    return meta_text, doubt_info


# ---------------------------------------------------------------------------
# Main entry point: run_brain_council_spiking_brain
# ---------------------------------------------------------------------------

async def run_brain_council_spiking_brain(
    user_query: str,
    debug: bool = False,
    spiking_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    High-level orchestration that wraps council_different_views.run_full_council
    in a brain-like layer with zones, spiking activations, and brain-level doubt.

    Spiking configuration:
        spiking_config is a dict merged over BRAIN_SPIKING_DEFAULTS, with keys:
            - steps: int
            - global_noise_scale: float
            - global_decay_scale: float
            - use_llm_relevance: bool
            - llm_relevance_weight: float in [0, 1]

    Returns:
        (stage1_results, stage2_results, stage3_result, metadata)
    """
    original_user_query = user_query
    brain_run_id = str(uuid.uuid4())

    # Merge spiking config with defaults
    cfg = dict(BRAIN_SPIKING_DEFAULTS)
    if spiking_config:
        cfg.update(spiking_config)

    # --- 1. Initialize and compute relevance ---
    brain_state = _init_brain_state()
    zone_relevance = await _compute_zone_relevance(original_user_query, cfg)

    # --- 2. Spiking dynamics ---
    brain_state, activation_scores, spiking_log = _run_spiking_dynamics(
        brain_state,
        zone_relevance,
        steps=int(cfg.get("steps", 3)),
        global_noise_scale=float(cfg.get("global_noise_scale", 1.0)),
        global_decay_scale=float(cfg.get("global_decay_scale", 1.0)),
    )

    # --- 3. Primary & secondary zones, mode ---
    primary_zone_id, secondary_zone_id = _select_primary_secondary(activation_scores)
    mode = _zone_to_mode(primary_zone_id)

    # --- 4. Zone-based ideas via LLM ---
    idea1, idea2, ideas_raw = await _build_zone_based_ideas(
        original_user_query,
        primary_zone_id,
        secondary_zone_id,
    )

    # --- 5. First council run (with zone-based ideas) ---
    stage1_1, stage2_1, stage3_1, meta1 = await run_full_council(
        original_user_query,
        mode=mode,
        idea1_description=idea1,
        idea2_description=idea2,
        debug=debug,
    )

    # --- 6. Brain-level doubt interrupt ---
    meta_doubt_text, doubt_info = await _maybe_trigger_brain_doubt(
        original_user_query,
        activation_scores,
        spiking_log,
        primary_zone_id,
        secondary_zone_id,
        council_metadata=meta1,
    )

    # If doubt didn't trigger, we are done after first run
    if not meta_doubt_text:
        final_stage1 = stage1_1
        final_stage2 = stage2_1
        final_stage3 = stage3_1
        final_meta = meta1
        second_run_used = False
        meta2 = None
    else:
        # Second run: append meta-doubt text to the question as extra context
        doubted_query = (
            f"{original_user_query}\n\n"
            "The META zone raised the following concern about the previous reasoning. "
            "Use this as additional context to refine and possibly correct the answer, "
            "without blindly trusting it:\n"
            f"{meta_doubt_text}"
        )

        stage1_2, stage2_2, stage3_2, meta2 = await run_full_council(
            doubted_query,
            mode=mode,
            idea1_description=idea1,
            idea2_description=idea2,
            debug=debug,
        )

        final_stage1 = stage1_2
        final_stage2 = stage2_2
        final_stage3 = stage3_2
        final_meta = meta2
        second_run_used = True

    # --- 7. Enrich metadata with brain observability ---
    brain_observability = {
        "brain_run_id": brain_run_id,
        "zones": list(ZONES.keys()),
        "zone_relevance": zone_relevance,
        "activation_scores": activation_scores,
        "spiking_log": spiking_log,
        "primary_zone": primary_zone_id,
        "secondary_zone": secondary_zone_id,
        "mode": mode,
        "zone_ideas": {
            "idea1_description": idea1,
            "idea2_description": idea2,
            "raw_idea_text": ideas_raw,
        },
        "doubt": {
            "used": second_run_used,
            "brain_doubt_info": doubt_info,
            "meta_doubt_text_truncated": doubt_info.get("meta_doubt_excerpt", ""),
        },
        "spiking_config_used": cfg,
    }

    combined_metadata = dict(final_meta) if final_meta is not None else {}
    combined_metadata["brain_observability"] = brain_observability
    combined_metadata["brain_initial_run_metadata"] = meta1
    if meta2 is not None:
        combined_metadata["brain_second_run_metadata"] = meta2

    if debug:
        combined_metadata.setdefault("brain_debug", {})
        combined_metadata["brain_debug"].update({
            "original_user_query": original_user_query,
            "meta_doubt_text_full": meta_doubt_text,
        })

    return final_stage1, final_stage2, final_stage3, combined_metadata




# USAGE
# import asyncio
# from brain_ideas import run_brain_council_spiking_brain  

# async def main():
#     stage1, stage2, final, meta = await run_brain_council_spiking_brain(
#         "Design a scalable LLM API gateway",
#         debug=True,
#     )

#     print("FINAL ANSWER:\n")
#     print(final.get("response", final))

#     # If you want to peek at brain metadata:
#     print("\nPRIMARY ZONE:", meta["brain_observability"]["primary_zone"])
#     print("SECONDARY ZONE:", meta["brain_observability"]["secondary_zone"])
#     print("ACTIVATION SCORES:", meta["brain_observability"]["activation_scores"])

# if __name__ == "__main__":
#     asyncio.run(main())
