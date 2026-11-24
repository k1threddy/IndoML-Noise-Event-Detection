# IndoML 2026: Noise Event Detection

## Project summary

This repository contains our implementation and experiments for the IndoML 2026 Noise Event Detection task. We developed a seven-class sound-event detection system using pretrained ATST audio representations together with a task-specific CNN–RNN detection network.

Our experiments focused on training with the **gold subset containing verified temporal event annotations**. The work explored supervised loss formulations and temporal event detection, rather than attempting to reproduce every component of the reference paper.

## 1. Model overview

The system combines two audio feature paths:

- **ATST representation path:** log-mel input at 16 kHz, using 64 mel bins and a 160-sample hop (10 ms).
- **SED feature path:** log-mel features at 16 kHz, using 128 mel bins and a 256-sample hop (16 ms).

The features are processed by the CNN–RNN detection architecture to produce frame-level predictions for seven sound-event classes:

1. Animal
2. Appliance / machine
3. Baby / child
4. Human non-speech
5. Phone signal / alarm
6. Singing / music
7. Vehicle / traffic

The training pipeline uses 10-second windows with a 5-second stride. The training script aligns strong targets to a 16 ms frame duration.

## 2. Training data and supervision

The experiments use the Vaani Noise Event Dataset. The repository includes the gold-window metadata (`gold_windows.jsonl`) and validation ground truth (`gt_validation.jsonl`); the audio files themselves are not included.

Our training focus was the gold-labelled subset, where event timestamps had been verified. This provided temporal supervision for frame-level detection and allowed us to study the effect of different loss formulations using trusted event boundaries.

The current training configuration uses a frozen pretrained ATST backbone. The task-specific detection layers are trained on the available labelled data.

## 3. Training objective

We experimented with a combined objective containing:

- **Strong-label BCE (weight 1.0):** frame-level supervised binary cross-entropy.
- **Class-specific Tversky (weight 1.0):** computes a Tversky loss independently for each event class.
- **Class-agnostic Tversky (weight 3.0):** collapses class predictions into an any-event probability and optimizes temporal event detection.
- **Weak-label loss (weight 0.5):** uses clip-level weak-label supervision.

The Tversky parameters in the checked-in configuration are \(\alpha=0.6\) and \(\beta=1.5\). The training configuration uses a batch size of 16, a maximum of 20 epochs, and separate learning rates for the CNN, RNN, and other trainable parameters.

### Frozen and fine-tuned ATST

The training script currently included in this repository is the **frozen-ATST configuration**: all parameters in the ATST branch, including transformer blocks 10 and 11, have `requires_grad=False`. The CNN–RNN detection layers remain trainable.

We also explored unfreezing the final two ATST blocks in a late experiment. That experiment was performed close to the submission deadline and was not sufficiently investigated to treat it as a separately validated result. The checked-in training script therefore represents the frozen configuration.

## 4. Difference from the reference paper

The reference paper describes a semi-supervised Mean Teacher framework. In that approach, a teacher network is maintained as an exponential moving average (EMA) of the student, generates pseudo-labels for unlabelled samples, and supplies a consistency-learning signal.

We did **not** implement the Mean Teacher component in this project. Our experiments were scoped around the gold-labelled subset with verified event timestamps. Since the training focus was on data with trusted temporal annotations, we prioritized supervised temporal learning and comparisons between BCE/Tversky-based objectives, rather than adding a pseudo-label and consistency-training pipeline for unlabelled data.

Accordingly, this repository does not contain:
- an EMA teacher network or EMA update;
- teacher-generated pseudo-labels for unlabelled samples; or
- a Mean Teacher consistency loss.

This is a deliberate difference in experimental scope, not a claim that Mean Teacher is unnecessary when gold annotations are available. Semi-supervised learning could still be explored by incorporating additional unlabelled data.

## 5. Evaluation

We report two distinct event-detection metrics:

- **Event-based F1:** predicted events are matched to reference events using onset and offset tolerances. The evaluation code uses a tolerance of the larger of 20% of the reference-event duration or 50 ms, followed by closest-first matching.
- **Segment-based Dice:** events are rasterized onto a 10 ms time grid and Dice is computed per clip, then averaged.

The segment-Dice evaluation uses a 10 ms grid even though the model's training/output frame duration is 16 ms. Event-based F1 and segment-based Dice measure different aspects of performance and should be reported separately.

### Preliminary result

An intermediate evaluation using the epoch-4 checkpoint produced:

| Metric | Result |
|---|---:|
| Event-based F1 | 0.345 |
| Segment-based Dice (10 ms grid) | 0.677 |

These are preliminary, checkpoint-specific results, not a final benchmark claim. Other experiments produced different values; comparisons should always specify the checkpoint and evaluation protocol.

## 6. Code and reproduction notes

| File | Purpose |
|---|---|
| `train_gold_tversky.py` | Training with BCE and Tversky-based objectives |
| `infer_validation_16ms.py` | Validation inference |
| `evaluate_validation.py` | Event F1 and segment-Dice evaluation |
| `dataset.py`, `encoder.py` | Dataset loading and temporal label encoding |
| `make_train_gt.py`, `make_gt_validation.py` | Ground-truth metadata preparation |
| `nnet/` | CNN, RNN, ATST and BEATs model components |

Dependencies are specified in `pyproject.toml` and `uv.lock`. The project uses Python 3.11+ and can be set up with:

```bash
uv sync
```

The pretrained ATST checkpoint and raw dataset are not included. Configure their paths in the scripts for the local environment before running training or inference. The current scripts contain lab-specific dataset/output paths and a CUDA device selection that may need to be changed on another machine.

**Metadata path note:** `gold_windows.jsonl` is committed in the repository root, while `make_train_gt.py` currently expects it under `metadata/gold_windows.jsonl`. Update the generator's input path or arrange the file accordingly before using that script.

## Scope

The repository documents the architecture, supervised training experiments, inference and evaluation work carried out for the task. It should be read as an implementation and experimental report focused on gold temporal annotations, not as a complete reproduction of the reference paper's semi-supervised Mean Teacher pipeline.
