# Spiking Brain Orchestrator (`spiking_brain_ideas.py`)

This document explains the **spiking brain-style orchestrator** built on top of the **LLM Council** pipeline by Andrej Karpathy.

The brain layer wraps the council with:

- **Cognitive zones** (analysis, planning, creative, meta, etc.)
- **Spiking dynamics** (neurons integrating input + noise over time)
- **Zone-based idea generation** (IDEA 1 / IDEA 2 from different perspectives)
- A **brain-level doubt interrupt** that can trigger a second council run

It is implemented in `brain_ideas.py` and primarily exposed via:

```python
async def run_brain_council_spiking_brain(
    user_query: str,
    debug: bool = False,
    spiking_config: Optional[Dict[str, Any]] = None,
) -> (stage1_results, stage2_results, stage3_result, metadata)
```

---

## 1. High-Level Overview

The brain orchestrator does the following:

1. **Understands the query**: estimates which cognitive zones are relevant.
2. **Runs a spiking simulation**: neurons in each zone integrate input + noise across a few timesteps.
3. **Chooses zones**:
   - **Primary zone** – main perspective
   - **Secondary zone** – complementary perspective
4. **Builds two ideas**:
   - IDEA 1 = primary zone perspective  
   - IDEA 2 = secondary zone perspective  
5. **Calls the council**:
   - Uses `run_full_council` with these ideas and a mode influenced by the primary zone.
6. **Meta-doubt**:
   - Checks disagreement, surprises in spiking, “all_not_ok” signals, and zone doubt sensitivity.
   - May trigger a second council run with a META critique appended to the query.

The result is a system that **doesn’t just chain prompts**, but **chooses how to think** about a problem and when to doubt itself.

---

## 2. Cognitive Zones

The brain is divided into **six zones**, each with its own “thinking style” and spiking parameters:

- `analysis` – Analytical / logical reasoning
- `world_modeling` – Causal / systemic thinking
- `planning` – Goal-directed planning / execution
- `creative` – Divergent / imaginative thinking
- `social` – Social / emotional reasoning
- `meta` – Meta-cognition, self-reflection, critique

Each zone has a configuration similar to:

```python
{
    "id": "analysis",
    "name": "Analytical Reasoning",
    "neuron_count": ...,
    "base_threshold": ...,
    "decay": ...,
    "noise_scale": ...,
    "doubt_sensitivity": ...,
    "default_mode": "balanced" | "risk_averse" | "exploratory",
    "role_description": "...how this zone thinks...",
    "keywords": ["analyze", "evaluate", "tradeoff", ...],
}
```

---

## 3. Spiking Dynamics

### 3.1 Brain State Initialization

```python
_init_brain_state()
```

```python
brain_state = {
    "zones": {
        zone_id: {
            "neurons": [
                {"potential": 0.0, "threshold": ..., "decay": ...},
                ...
            ],
            "last_spikes": [...],
        },
        ...
    }
}
```

### 3.2 Spiking Simulation

```python
_run_spiking_dynamics(
    brain_state,
    zone_relevance,
    steps,
    global_noise_scale,
    global_decay_scale,
)
```

Each timestep:

```python
input_current = zone_relevance[zid] + random.gauss(0, noise_scale * global_noise_scale)
potential = decay * potential + input_current
if potential >= threshold: spike
```

Outputs:

```python
{
  "activation_scores": {zone_id: float},
  "spiking_log": {
    zone_id: {
      "total_spikes": int,
      "expected_from_relevance": float
    }
  }
}
```

---

## 4. Idea Generation

```python
_build_zone_based_ideas(...)
# Returns descriptions for IDEA 1 (primary zone) and IDEA 2 (secondary zone)
```

---

## 5. Brain-Level Doubt Mechanism

```python
_maybe_trigger_brain_doubt(...)
```

Triggers if:

- All council responses are `not_ok`
- High disagreement
- Unexpected spiking activity
- High `doubt_sensitivity` zone active

If triggered:

1. Generate `"META DOUBT: ..."` text  
2. Append to query  
3. Run **second council pass**

---

## 6. Core Entry: 'run_brain_council_spiking_brain'

### Pipeline

1. Compute zone relevance
2. Run spiking dynamics
3. Select zones
4. Generate ideas
5. Run council
6. Optional meta-doubt rerun
7. Return full metadata with brain observability

---

## 7. Configurable Parameters

```python
BRAIN_SPIKING_DEFAULTS = {
    "steps": 3,
    "global_noise_scale": 1.0,
    "global_decay_scale": 1.0,
    "use_llm_relevance": True,
    "llm_relevance_weight": 0.6,
}
```

Override like:

```python
spiking_config = {
    "steps": 5,
    "global_noise_scale": 1.5,
    "global_decay_scale": 0.9,
}
await run_brain_council_spiking_brain("...", spiking_config=spiking_config)
```

---

## 8. Example Usage

```python
async def main():
    stage1, stage2, final, meta = await run_brain_council(
        "Design a scalable LLM API gateway",
        debug=True,
    )

    print(final.get("response"))
    print(meta["brain_observability"]["primary_zone"])
```

---

## 9. Metadata Examples

```json
{
  "primary_zone": "analysis",
  "secondary_zone": "planning",
  "activation_scores": {...},
  "doubt": {"used": false},
  "zone_ideas": {
    "idea1_description": "...",
    "idea2_description": "..."
  }
}
```

---

## 10. Extension Ideas

- Persist brain state over multiple calls (long-term memory)
- Use embeddings instead of bag-of-words for retrieval
- Experiment with new zone types
- Add reinforcement feedback loops

---

## Summary

The system combines:

| Component | Purpose |
|----------|----------|
| Council | Multi-model competitive reasoning |
| Brain | Cognitive orchestration & doubt logic |
| Retrieval | Context from past decisions |

Use the brain layer when you want **self-reflective, multi-perspective reasoning** with the ability to **challenge its own conclusions**.
