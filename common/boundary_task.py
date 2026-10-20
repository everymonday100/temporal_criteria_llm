"""boundary_task.py — Scalar tracking task for boundary experiment.

Two designs:
  R (reactive): agent writes NL, framework parses a_t, applies to latent threshold
  O (observation): agent emits [SET: a_t], framework applies directly to y_t

Task dynamics:
  Design R:
    threshold_t ∈ [0, 100], init ~ U(20, 80)
    y_t = clip(threshold_t + σ_t · ξ_t, 0, 100)
    threshold_{t+1} = threshold_t + a_t
    external: σ_t jumps by +Δσ at step s

  Design O:
    y_t ∈ [0, 100], init ~ U(20, 80)
    y_{t+1} = clip(y_t + a_t·c + δ_t + η_t, 0, 100)
    external: δ_t = ±δ_mag at step s

Parameters:
  c = 0.5, σ_0 = 0.2, σ_ext = 0.6, Δσ = 0.4
  δ_mag = 2.0, η_t ~ N(0, σ_0²)
  event step s ~ Uniform({5, 10, 15, 20, 25}) per stream
  event type: σ-shift (R) or δ-injection (O), random per stream
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Task parameters
C = 0.5
SIGMA_0 = 0.2
SIGMA_EXT = 0.6
DELTA_SIGMA = 0.4
DELTA_MAG = 2.0
CLIP_MIN, CLIP_MAX = 0.0, 100.0
THRESHOLD_INIT_RANGE = (20.0, 80.0)
EVENT_STEPS = [5, 10, 15, 20, 25]
ACTION_SPACE = [-2, -1, 0, +1, +2]


@dataclass
class StreamStep:
    t: int
    y_t: float
    action: Optional[int]
    action_text: str
    delta_y: float
    is_external: bool
    external_type: Optional[str]
    threshold_t: Optional[float]


@dataclass
class Stream:
    stream_id: int
    design: str
    steps: List[StreamStep]
    threshold_init: float
    event_step: int
    event_type: str
    parse_success_rate: float


def parse_action_r(text: str) -> Optional[int]:
    """Extract action from NL response (Design R).

    Priority order:
      1. Structured [SET: n] — if model accidentally uses O format
      2. Explicit action verbs: 'by +1', 'add -1', 'decrease 2', etc.
      3. Bare signed integers: '+1', '-2'
      4. Any integer in range

    This is deliberately lenient: Phase 1 measures whether the model
    produces parseable output, not whether parsing is perfect.
    Constrained decoding is a fallback if this rate is too low.
    """
    for pattern in [
        r'\[SET:\s*([-+]?\d+)\]',
        r'(?:by|add|subtract|increase|decrease|adjust(?:\s+by)?|go)\s*([-+]?\d+)',
        r'([-+]\d+)',
        r'\b(-?\d+)\b',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
            except ValueError:
                continue
            if val in ACTION_SPACE:
                return val
    return None


def parse_action_o(text: str) -> Optional[int]:
    """Extract action from structured command (Design O).

    Expected format: [SET: +1] or [SET: -2].
    """
    match = re.search(r'\[SET:\s*([-+]?\d+)\]', text)
    if match:
        val = int(match.group(1))
        if val in ACTION_SPACE:
            return val
    return None


def generate_stream(
    stream_id: int,
    design: str,
    M: int,
    rng: np.random.Generator,
    agent_fn=None,
) -> Stream:
    """Generate one stream of M steps.

    If agent_fn is None, uses random actions (for Phase 2 without LoRA
    or for training stream generation). For Phase 1, agent_fn is the
    frozen or LoRA model.
    """
    threshold_init = rng.uniform(*THRESHOLD_INIT_RANGE)
    threshold_t = threshold_init

    if design == "O":
        y_t = threshold_init
    else:
        y_t = clip(threshold_t + rng.normal(0, SIGMA_0),
                   CLIP_MIN, CLIP_MAX)

    event_step = int(rng.choice(EVENT_STEPS))
    event_type = "σ-shift" if design == "R" else "δ-injection"

    steps = []
    parse_successes = 0

    for t in range(M):
        is_external = (t == event_step)

        if agent_fn is None:
            a_t = int(rng.choice(ACTION_SPACE))
            if design == "O":
                action_text = f"[SET: {a_t:+d}]"
            else:
                action_text = f"I'll adjust by {a_t:+d}"
        else:
            prompt = build_prompt(design, y_t, steps, step_t=t)
            action_text = agent_fn(y_t, prompt)
            if design == "O":
                a_t = parse_action_o(action_text)
            else:
                a_t = parse_action_r(action_text)

        if a_t is not None:
            parse_successes += 1

        # Apply action + external event
        if design == "R":
            if a_t is not None:
                threshold_t = threshold_t + a_t
            sigma_t = SIGMA_EXT if is_external else SIGMA_0
            y_next = clip(threshold_t + rng.normal(0, sigma_t),
                          CLIP_MIN, CLIP_MAX)
            delta_y = y_next - y_t
            y_t = y_next
        else:  # Design O
            delta_y_base = (a_t * C) if a_t is not None else 0.0
            delta_ext = (rng.choice([-1, +1]) * DELTA_MAG
                         if is_external else 0.0)
            eta_t = rng.normal(0, SIGMA_0)
            y_next = clip(y_t + delta_y_base + delta_ext + eta_t,
                          CLIP_MIN, CLIP_MAX)
            delta_y = y_next - y_t
            y_t = y_next

        steps.append(StreamStep(
            t=t,
            y_t=y_t,
            action=a_t,
            action_text=action_text,
            delta_y=delta_y,
            is_external=is_external,
            external_type=event_type if is_external else None,
            threshold_t=threshold_t if design == "R" else None,
        ))

    return Stream(
        stream_id=stream_id,
        design=design,
        steps=steps,
        threshold_init=threshold_init,
        event_step=event_step,
        event_type=event_type,
        parse_success_rate=parse_successes / M,
    )


FEW_SHOT_R = """Example 1:
Current reading: 50.0. Choose adjustment (-2 to +2).
Agent: +1
Framework: adjustment applied: +1. New reading: 50.5.

Example 2:
Current reading: 50.5. Choose adjustment (-2 to +2).
Agent: -1
Framework: adjustment applied: -1. New reading: 49.8.

Example 3:
Current reading: 49.8. Choose adjustment (-2 to +2).
Agent: 0
Framework: adjustment applied: 0. New reading: 50.1.

"""

FEW_SHOT_O = """Example 1:
Current reading: 50.0. Emit [SET: n] with n in {-2,-1,0,+1,+2}.
Agent: [SET: +1]
Framework: Applied: [SET: +1]. New reading: 50.5.

Example 2:
Current reading: 50.5. Emit [SET: n] with n in {-2,-1,0,+1,+2}.
Agent: [SET: -1]
Framework: Applied: [SET: -1]. New reading: 49.8.

Example 3:
Current reading: 49.8. Emit [SET: n] with n in {-2,-1,0,+1,+2}.
Agent: [SET: 0]
Framework: Applied: [SET: 0]. New reading: 50.1.

"""


def build_prompt(design: str, y_t: float,
                  history: List[StreamStep],
                  step_t: int = 0) -> str:
    """Build prompt with few-shot examples AND actual stream history."""
    if design == "R":
        head = FEW_SHOT_R
        prompt = head + "--- Current stream ---\n"
        for step in history:
            prev_y = step.y_t - step.delta_y
            act = step.action if step.action is not None else 0
            prompt += (f"Current reading: {prev_y:.1f}. "
                       f"Choose adjustment (-2 to +2).\n"
                       f"Agent: {act:+d}\n"
                       f"Framework: adjustment applied: {act:+d}. "
                       f"New reading: {step.y_t:.1f}.\n\n")
        prompt += (f"Current reading: {y_t:.1f}. "
                   f"Choose adjustment (-2 to +2).\nAgent:")
    else:  # Design O
        head = FEW_SHOT_O
        prompt = head + "--- Current stream ---\n"
        for step in history:
            prev_y = step.y_t - step.delta_y
            act = step.action if step.action is not None else 0
            prompt += (f"Current reading: {prev_y:.1f}. "
                       f"Emit [SET: n] with n in {{-2,-1,0,+1,+2}}.\n"
                       f"Agent: [SET: {act:+d}]\n"
                       f"Framework: Applied: [SET: {act:+d}]. "
                       f"New reading: {step.y_t:.1f}.\n\n")
        prompt += (f"Current reading: {y_t:.1f}. "
                   f"Emit [SET: n] with n in {{-2,-1,0,+1,+2}}.\nAgent:")
    return prompt


def clip(x: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, x))


def check_saturation(stream: Stream) -> float:
    """Fraction of steps where Design R threshold is near clip edges."""
    if stream.design != "R":
        return 0.0
    saturated = 0
    for step in stream.steps:
        if step.threshold_t is None:
            continue
        if step.threshold_t < 5 or step.threshold_t > 95:
            saturated += 1
    return saturated / max(1, len(stream.steps))