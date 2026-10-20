# Signal without Action: Structural Boundaries of Functional Temporal Properties in Frozen Language Models

This repository contains the official code for the paper **"Signal without Action"**. We apply an operational battery of four functional temporal properties (self-model, agency, boundary, continuity) to frozen sequence models (Qwen2.5-3B, Mamba2-1.3B, and xLAM-2-3B).

## Key Findings
*   **Self-model:** Fails in the token coordinate, but is recovered via arc-length reparameterization (R² = 0.755 on Qwen). Localized to conditional heteroskedasticity.
*   **Agency:** Standard cross-entropy provides a scalar, not directional, gradient. A controller trained with CE collapses to a constant direction (cos ≈ 0.99). 
*   **Weak-form Agency:** Applying a contrastive hinge loss on a tool-calling base (xLAM) breaks the bottleneck (cross-class cos < 0), achieving 42.5% accuracy versus 0% for the frozen baseline, bounded by LoRA fine-tuning ceiling.
*   **Boundary:** Interface thesis holds. Observation-channel design achieves AUC = 1.000 vs Reactive AUC = 0.718.
*   **Continuity:** Signal presence ≠ decision value. Cost-Aware expected-F1 outperforms Argmax (0.524 vs 0.469) by regularizing overconfident hallucinations.

## Repository Structure
*   `common/`: Core model wrappers, dataset loaders, and evaluation utilities.
    *   `controller_model.py`: Qwen/xLAM wrapper with latent controller injection at layer 18.
    *   `boundary_model.py` / `boundary_task.py`: Forward model and stream generation for the Boundary track.
*   `stage1/`: Experiment scripts.
    *   `train_controller.py` / `eval_controller.py`: Original CE controller on GSM-HARD.
    *   `train_xlam_controller_v2.py` / `eval_xlam_matrix.py`: Contrastive controller on xLAM tool-calling.
    *   `audit_loss_anisotropy.py`: Loss-gradient audit (C2 prerequisite).
    *   `continuity_gen_stream.py` / `continuity_probe_m3.py`: Continuity track pipeline.
    *   `boundary_extract_hidden.py` / `boundary_train_fm.py`: Boundary track pipeline.

## Setup & Reproducibility
**Environment:** Python 3.12, PyTorch 2.9.1, Transformers 5.7, `peft`, `mamba-ssm`. Runs on a single RTX 4060 (8 GB).

### Agency (Qwen2.5-3B vs xLAM-2-3B)
```bash
# Original CE controller (Constant Direction)
python stage1/train_controller.py --n_train 600 --epochs 2
python stage1/eval_controller.py --n_test 300 --n_train_skip 700

# Contrastive xLAM controller (Weak-form Agency)
python stage1/train_xlam_controller_v2.py --lambda_contrast 1.0
python stage1/eval_xlam_matrix.py
```
### Boundary (Forward-model)
```bash
python stage1/boundary_extract_hidden.py --streams train_streams.pkl --variant F
python stage1/boundary_train_fm.py --design {R,O} --variant F
```
### Continuity (Decision-rule dependence)
```bash
python stage1/continuity_gen_stream.py
python stage1/continuity_extract_context.py
python stage1/continuity_probe_m3.py
```
### Citation
If you use this code in your research, please cite:

```bibtex
@article{kuzmin2026signal,
  title={Signal without Action: Structural Boundaries of Functional Temporal Properties in Frozen Language Models},
  author={Kuzmin, Yakov},
  journal={Preprint},
  year={2026}
}
```
