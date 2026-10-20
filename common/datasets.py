"""datasets.py — fixed prompt sets and task generators for Stage 1."""
import json
import random
from pathlib import Path
from common.generation import _normalize_number

import numpy as np


DATA_DIR = Path(r"E:\OllamaModels\temporal_llm\data")
SM_PROMPTS_PATH = DATA_DIR / "stage1" / "sm_prompts.json"
GSM8K_LOCAL = DATA_DIR / "gsm8k_test.jsonl"


# ----------------------------------------------------------------------
# Self-model prompts (same 25 as in Paper 2 / llm_probe_arc.py)
# ----------------------------------------------------------------------
SM_PROMPTS = [
    "Consider the role of rivers in shaping early human settlements.",
    "Explain why the sky is blue, at the level of Rayleigh scattering.",
    "What makes a proof in mathematics convincing? Discuss.",
    "Describe how neurons communicate across a synapse.",
    "Analyze the causes of the fall of the Roman Empire.",
    "Discuss the nature of time in modern physics.",
    "Explain how a transformer model attends to context.",
    "What is the difference between weather and climate?",
    "Describe how a compiler translates source code into machine code.",
    "Discuss the ethical implications of autonomous weapons.",
    "Explain how the immune system recognizes pathogens.",
    "What is the role of symmetry in physics?",
    "Describe how photosynthesis converts light into chemical energy.",
    "Discuss the relationship between language and thought.",
    "Explain how public-key cryptography works.",
    "Analyze the structure of a Shakespearean sonnet.",
    "Describe how tectonic plates move and interact.",
    "What is the significance of Godel's incompleteness theorems?",
    "Explain how antibiotics work and why resistance evolves.",
    "Discuss the role of randomness in evolution.",
    "Describe how the internet routes a packet.",
    "Analyze the economic causes of the 2008 financial crisis.",
    "Explain how a neural network learns from data.",
    "What is the difference between a virus and a bacterium?",
    "Discuss the concept of emergence in complex systems.",
]


def load_sm_prompts(n: int = 25):
    return SM_PROMPTS[:n]


# ----------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------
def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsm8k(n: int = 400):
    """Load GSM8K test set. Try local JSONL first, then HF openai/gsm8k."""
    if GSM8K_LOCAL.exists():
        raw = _read_jsonl(GSM8K_LOCAL)
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        raw = [{"question": ex["question"], "answer": ex["answer"]}
               for ex in ds]

    problems = []
    for i, ex in enumerate(raw[:n]):
        raw_answer = ex["answer"].strip()
        ref = raw_answer.split("####")[-1].strip().replace(",", "")
        problems.append({
            "i": i,
            "question": ex["input"].strip(),
            "code": ex["code"].strip(),
            "reference": _normalize_number(ex["target"]),
        })
    return problems


def gsm8k_prompt(q: str) -> str:
    return (
        f"Solve the following math problem step by step. "
        f"End your answer with a line of the form '#### <number>'.\n\n"
        f"Problem: {q}\n\nSolution:"
    )


# ----------------------------------------------------------------------
# Boundary dataset (BD): synthetic episodes with self/external events
# ----------------------------------------------------------------------
BD_TEMPLATES = [
    "The following describes a simple physical system. Answer briefly.",
    "Read the following description and provide a short response.",
    "Consider the scenario below and answer concisely.",
    "Given the following setup, respond with a brief analysis.",
    "Read carefully and answer the question that follows.",
]

BD_SCENARIOS = [
    "A pendulum swings with damping. Its period depends on length and gravity.",
    "A ball rolls down an incline. Friction coefficient is {mu}.",
    "A fluid flows through a pipe. Flow rate depends on pressure and radius.",
    "A spring is compressed by {x} cm. Its stiffness is {k} N/m.",
    "Two resistors in parallel carry current according to Ohm's law.",
    "A capacitor discharges through a resistor with time constant RC.",
    "A wave propagates through a medium with speed {v} m/s.",
    "A lever balances masses on either side of a fulcrum.",
    "An object falls in a gravitational field with drag.",
    "A gas expands adiabatically in a piston chamber.",
]


def load_bd_tasks(n: int = 300, seed: int = 42):
    """BD dataset — synthetic episodes.

    Each episode: a physics scenario prompt, model generates a response,
    observation y = Phi(response). Later we construct self-caused and
    external events from y.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        tmpl = BD_TEMPLATES[i % len(BD_TEMPLATES)]
        scen = BD_SCENARIOS[i % len(BD_SCENARIOS)]
        scen = scen.replace("{mu}", f"{rng.uniform(0.1, 0.5):.2f}")
        scen = scen.replace("{x}", f"{rng.uniform(1, 10):.1f}")
        scen = scen.replace("{k}", f"{rng.uniform(50, 500):.0f}")
        scen = scen.replace("{v}", f"{rng.uniform(1, 100):.0f}")
        prompt = f"{tmpl}\n\n{scen}\n\nAnalysis:"
        tasks.append({"i": i, "prompt": prompt, "scenario": scen})
    return tasks


def bd_prompt(task) -> str:
    return task["prompt"]


# ----------------------------------------------------------------------
# Continuity task (CT): trajectories of 3 episodes on one topic
# ----------------------------------------------------------------------
CT_TOPIC_KEYWORDS = [
    "percentage", "fraction", "total", "average", "rate",
    "distance", "time", "cost", "profit", "interest",
    "sum", "difference", "product", "ratio", "days",
]


def _topic_of(question: str):
    q = question.lower()
    for kw in CT_TOPIC_KEYWORDS:
        if kw in q:
            return kw
    return "_other"


def load_ct_tasks(n_traj: int = 50, n_episodes: int = 3, seed: int = 42):
    """CT dataset — 50 trajectories × 3 GSM8K problems of the same topic."""
    problems = load_gsm8k(n=400)
    by_topic = {}
    for p in problems:
        t = _topic_of(p["question"])
        by_topic.setdefault(t, []).append(p)

    rng = random.Random(seed)
    topics = [t for t, lst in by_topic.items() if len(lst) >= n_episodes]
    rng.shuffle(topics)

    traj = []
    for k in range(min(n_traj, len(topics))):
        t = topics[k]
        picks = rng.sample(by_topic[t], n_episodes)
        traj.append({
            "traj": k,
            "topic": t,
            "episodes": picks,
        })
    return traj
    

# Добавить в конец common/datasets.py

def load_gsm_hard(n=None):
    """Load GSM-HARD: GSM8K with larger, less common numbers.

    Structure: input (question), code (Python solution), target (answer).
    Qwen2.5-3B accuracy on GSM-HARD is ~30-50% (vs 66% on GSM8K),
    giving CE 1.5-3.0 and a strong training signal.
    """
    from datasets import load_dataset
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    problems = []
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        problems.append({
            "i": i,
            "question": ex["input"].strip(),
            "code": ex["code"].strip(),
            "reference": _normalize_number(ex["target"]),
        })
    return problems


def gsm_hard_prompt(q):
    return (
        f"Solve the following math problem step by step. "
        f"End your answer with a line of the form '#### <number>'.\n\n"
        f"Problem: {q}\n\nSolution:"
    )