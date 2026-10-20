"""boundary_gen_streams.py — Generate and cache streams for boundary experiment.

Phase 1: 20 streams per design, run frozen Qwen, check:
  - parse success rate >= 0.90
  - saturation rate <= 0.10 (Design R)
Phase 2: full train/test streams, no agent (random actions).

Usage:
  python stage1/boundary_gen_streams.py --phase 1 --n_streams 20
  python stage1/boundary_gen_streams.py --phase 2 --n_train 400 --n_test 600
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.model import load_model
from common.boundary_task import (
    generate_stream, check_saturation,
)

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\boundary")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 2], required=True)
    p.add_argument("--n_streams", type=int, default=20)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--M", type=int, default=30)
    p.add_argument("--model", default="qwen", choices=["qwen", "mamba2"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_agent_fn(model, tok, stream_id):
    """Closure producing a deterministic agent_fn for a given stream.

    Seed depends on (stream_id, step_counter) so the agent varies
    across steps and across streams, and the whole run is reproducible.
    """
    counter = [0]

    def agent_fn(y_t, prompt):
        seed = (stream_id * 7919 + counter[0] * 31 + 42) % (2**31 - 1)
        counter[0] += 1
        return _agent_fn_frozen(model, tok, prompt, seed=seed)

    return agent_fn


def _agent_fn_frozen(model, tok, prompt, seed=42):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tok.eos_token_id,
        )
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True)
    # Truncate hallucinated Framework line
    response = response.split('\n')[0].strip()
    return response


def phase1_smoke(args):
    print(f"[boundary] Phase 1: {args.n_streams} streams per design")

    rng = np.random.default_rng(args.seed)
    tok, model = load_model(args.model)
    model.eval()

    results = {"R": [], "O": []}
    t0 = time.time()

    for design in ["R", "O"]:
        print(f"\n[boundary] Design {design}:")
        parse_rates = []
        sat_rates = []
        sample_outputs = []

        for i in range(args.n_streams):
            agent_fn = make_agent_fn(model, tok, stream_id=i)
            stream = generate_stream(
                stream_id=i, design=design, M=args.M,
                rng=rng, agent_fn=agent_fn,
            )
            parse_rates.append(stream.parse_success_rate)
            sat_rates.append(check_saturation(stream))

            # Save 2 sample outputs per design for inspection
            if i < 2:
                for step in stream.steps[:2]:
                    sample_outputs.append({
                        "design": design,
                        "stream": i,
                        "step": step.t,
                        "y_t": round(step.y_t, 2),
                        "raw": step.action_text[:80],
                        "parsed": step.action,
                    })

            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{args.n_streams}]  "
                      f"parse_rate={stream.parse_success_rate:.2f}  "
                      f"({elapsed:.0f}s)")

        mean_parse = float(np.mean(parse_rates))
        mean_sat = float(np.mean(sat_rates))

        print(f"\n[boundary] Design {design}:")
        print(f"  parse_rate:  mean={mean_parse:.3f}  "
              f"min={min(parse_rates):.2f}  max={max(parse_rates):.2f}")
        print(f"  sat_rate:    mean={mean_sat:.3f}  "
              f"max={max(sat_rates):.2f}")

        ok_parse = mean_parse >= 0.90
        ok_sat = (design != "R") or (mean_sat <= 0.10)
        print(f"  parse {'PASS' if ok_parse else 'FAIL'}  "
              f"sat {'PASS' if ok_sat else 'FAIL'}")

        results[design] = {
            "parse_rates": parse_rates,
            "saturation_rates": sat_rates,
            "mean_parse": mean_parse,
            "mean_sat": mean_sat,
            "sample_outputs": sample_outputs,
            "pass": bool(ok_parse and ok_sat),
        }

    # Save
    out_path = OUT_DIR / "phase1_smoke.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")

    # Sample inspection
    print("\n[samples from Design R]:")
    for s in results["R"]["sample_outputs"]:
        print(f"  stream={s['stream']} step={s['step']}  "
              f"y={s['y_t']}  raw={s['raw']!r}  parsed={s['parsed']}")
    print("\n[samples from Design O]:")
    for s in results["O"]["sample_outputs"]:
        print(f"  stream={s['stream']} step={s['step']}  "
              f"y={s['y_t']}  raw={s['raw']!r}  parsed={s['parsed']}")

    all_pass = (results["R"]["pass"] and results["O"]["pass"])
    print(f"\n{'PASSED' if all_pass else 'FAILED'}: Phase 1")
    return all_pass


def phase2_full(args):
    print(f"[boundary] Phase 2: {args.n_train} train + {args.n_test} test")

    rng = np.random.default_rng(args.seed)

    print(f"\n[boundary] Generating {args.n_train} train streams...")
    train_streams = []
    for i in range(args.n_train):
        for j, design in enumerate(["R", "O"]):
            stream = generate_stream(
                stream_id=i * 2 + j, design=design, M=args.M,
                rng=rng, agent_fn=None,
            )
            train_streams.append(stream)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.n_train}]")

    print(f"\n[boundary] Generating {args.n_test} test streams...")
    test_streams = []
    for i in range(args.n_test):
        for j, design in enumerate(["R", "O"]):
            stream = generate_stream(
                stream_id=i * 2 + j + 10000, design=design, M=args.M,
                rng=rng, agent_fn=None,
            )
            test_streams.append(stream)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{args.n_test}]")

    with open(OUT_DIR / "train_streams.pkl", "wb") as f:
        pickle.dump(train_streams, f)
    with open(OUT_DIR / "test_streams.pkl", "wb") as f:
        pickle.dump(test_streams, f)

    print(f"\n[DONE] {len(train_streams)} train + "
          f"{len(test_streams)} test streams")


def main():
    args = parse_args()
    if args.phase == 1:
        phase1_smoke(args)
    else:
        phase2_full(args)


if __name__ == "__main__":
    main()