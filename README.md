# IndoML 2026: Noise Event Detection

Experimental implementation and reproduction work for the IndoML 2026 Noise Event Detection task.

This repository contains the training, inference, preprocessing, and evaluation code for a seven-class sound-event detection (SED) system built around an ATST audio representation branch and a CNN–RNN detection network.

> **Reproduction note:** This repository documents the implementation that was actually run. It is not a complete reproduction of every component in the reference paper. In particular, the Mean Teacher (EMA teacher + consistency-loss) procedure described in the paper was **not implemented** in these experiments.

## Contents

- [Overview](#overview)
- [Model and feature pipeline](#model-and-feature-pipeline)
- [Event classes](#event-classes)
- [Dataset and metadata](#dataset-and-metadata)
- [Environment setup](#environment-setup)
- [Pretrained checkpoint](#pretrained-checkpoint)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Experimental configuration](#experimental-configuration)
- [Results](#results)
- [Implementation notes and limitations](#implementation-notes-and-limitations)
- [Repository structure](#repository-structure)

## Overview

The system combines pretrained audio representations with task-specific spectral features for frame-level sound-event detection.

The current published training configuration uses a **frozen ATST backbone**. The trainable detection network is optimized using a combination of:

- Strong-label binary cross-entropy (BCE)
- Class-specific Tversky loss
- Class-agnostic Tversky loss
- Weak-label loss

The training script uses 10-second audio windows with a 5-second stride, at a 16 kHz sample rate.

## Model and feature pipeline

At a high level, the implementation follows this flow:

1. Load and prepare a 10-second audio window.
2. Extract pretrained ATST features.
3. Compute task-specific log-mel spectral features for the SED branch.
4. Process the features through the CNN–RNN detection architecture.
5. Produce frame-level class predictions for the seven event classes.
6. Train against strong and weak annotations using the configured losses.

### Feature time resolutions

The implementation uses different hops for the two feature paths:

| Feature path | Configuration | Frame hop |
|---|---|---:|
| ATST input mel | 16 kHz, 64 mel bins, 160-sample hop | 10 ms |
| SED mel | 16 kHz, 128 mel bins, 256-sample hop | 16 ms |

The training script defines its strong-target frame duration as 256 / 16000 seconds (16 ms). Do not confuse this model-output frame duration with the 10 ms grid used by the event-level segment-Dice evaluation script.

The model implementation and feature alignment are in `nnet/`, `dataset.py`, and `train_gold_tversky.py`.

## Event classes

The training code uses the following seven classes:

1. `animal`
2. `appliance_machine`
3. `baby_child`
4. `human_non_speech`
5. `phone_signal_alarm`
6. `singing_music`
7. `vehicle_traffic`

## Dataset and metadata

The experiments use the Vaani Noise Event Dataset supplied for the task.

The repository includes JSONL metadata files:

- `gold_windows.jsonl`: window-level gold training metadata consumed by the training script.
- `gt_validation.jsonl`: validation ground-truth events consumed by the evaluation script.

The audio dataset itself is not included. Configure the dataset paths for the environment where the code is run.

The training and validation scripts currently contain a lab-specific validation path:

```text
/mnt/Work/Datasets/indoml/Vaani-Noise-Event-Dataset/validation
```

Update `VALIDATION_DIR` in the relevant scripts to point to the local validation dataset location.

### Metadata-generation scripts

- `make_train_gt.py`: creates clip-level training ground truth from window metadata.
- `make_gt_validation.py`: creates clip-level validation ground truth from the validation metadata JSON.

**Path note:** The checked-in `gold_windows.jsonl` is in the repository root, while `make_train_gt.py` currently looks for `metadata/gold_windows.jsonl`. If you use this generator, update its `INPUT_FILE` path to the root-level file (or place the input file in the expected `metadata/` directory). The training script itself reads the root-level `gold_windows.jsonl`.

## Environment setup

The project requires Python 3.11 or later. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

Using uv:

```bash
uv sync
source .venv/bin/activate
```

The training and inference scripts use PyTorch and torchaudio. A compatible PyTorch/torchaudio installation and CUDA environment are needed to use a GPU. The scripts fall back to CPU when CUDA is unavailable, although training will be substantially slower.

The scripts currently set `CUDA_VISIBLE_DEVICES = "1"`. Change this to the appropriate GPU index for your machine, or remove the setting if you want to use the default device visibility.

## Pretrained checkpoint

The ATST pretrained checkpoint is not committed to this repository.

The scripts expect it at:

```text
checkpoints/atst_as2M.ckpt
```

Obtain the checkpoint from its authorized source and place it at that location, or update `ATST_CHECKPOINT` in the training and inference scripts. Check the original checkpoint's terms before redistributing it.

Training checkpoints and generated inference predictions are runtime artifacts and are intentionally excluded from Git.

## Training

The main training entry point is:

```text
train_gold_tversky.py
```

From the repository root, after configuring the dataset and checkpoint paths:

```bash
uv run python train_gold_tversky.py
```

The checked-in configuration includes:

| Setting | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Window duration | 10 s |
| Window stride | 5 s |
| Batch size | 16 |
| Maximum epochs | 20 |
| CNN learning rate | 5e-5 |
| RNN learning rate | 3e-4 |
| Other trainable parameters learning rate | 5e-5 |
| Strong BCE weight | 1.0 |
| Class-specific Tversky weight | 1.0 |
| Class-agnostic Tversky weight | 3.0 |
| Weak-label loss weight | 0.5 |
| Tversky alpha | 0.6 |
| Tversky beta | 1.5 |
| Validation threshold | 0.50 |

### Frozen ATST configuration

In the checked-in training script, all parameters in `model.atst_frame` have `requires_grad = False`. The final two ATST transformer blocks (indices 10 and 11) are also explicitly left frozen.

Consequently, the ATST backbone is frozen in this published configuration. The CNN/RNN and other trainable detection layers are optimized.

A late fine-tuning experiment was also explored by unfreezing ATST blocks 10 and 11. That experiment is distinct from the frozen configuration documented by the current training script.

## Inference

The validation inference entry point is:

```text
infer_validation_16ms.py
```

Run it from the repository root after setting the checkpoint and validation dataset paths:

```bash
uv run python infer_validation_16ms.py
```

The script currently expects the selected model checkpoint at:

```text
work/kevanth/checkpoints_tversky/best_classagnostic_dice.pt
```

It writes predictions to:

```text
work/kevanth/inference/pred_validation.jsonl
```

Update these paths if your directory layout differs. The `work/` directory and model checkpoints are generated locally and are not tracked by Git.

## Evaluation

Run the evaluation script after generating the prediction JSONL:

```bash
uv run python evaluate_validation.py
```

The script compares predictions against `gt_validation.jsonl` and reports two different metrics:

### Event-based F1

Predicted events are matched to reference events using onset and offset tolerances. For each reference event, the tolerance is:

```text
max(20% of reference-event duration, 50 ms)
```

The script uses closest-first greedy matching and aggregates TP, FP, and FN across clips to calculate precision, recall, and event F1.

### Segment-based Dice

Events are rasterized onto a **10 ms** time grid. Dice is calculated per clip and averaged across the reference clips.

Event-based F1 and segment-based Dice measure different properties and are not expected to be numerically identical.

The prediction path in `evaluate_validation.py` is currently set to `work/kevanth/inference/pred_validation.jsonl`. Update `PRED_FILE` if inference writes to a different location.

## Experimental configuration

The current training objective combines the following terms:

- **Strong BCE:** supervised frame-level binary cross-entropy on strongly labelled data.
- **Class-specific Tversky:** computes a Tversky loss independently for each event class and averages across classes.
- **Class-agnostic Tversky:** collapses class predictions into an any-event probability and computes a temporal Tversky loss against the any-event target.
- **Weak-label loss:** uses clip-level weak labels.

The checked-in configuration uses weights of 1.0, 1.0, 3.0, and 0.5 for these terms, respectively.

### Mean Teacher status

The reference methodology also describes a Mean Teacher approach with an exponential-moving-average (EMA) teacher, teacher-generated pseudo-labels for unlabelled data, and a student–teacher consistency loss.

**That Mean Teacher component was not implemented in the experiments represented by this repository.** There is no EMA teacher update or Mean Teacher consistency-loss term in the checked-in training implementation. The use of a pretrained ATST backbone does not, by itself, constitute a student–teacher framework.

## Results

Results depend on the checkpoint, inference threshold, post-processing, and evaluation protocol. Record the checkpoint and configuration alongside any reported score.

One intermediate evaluation shared during development, using an epoch-4 checkpoint, reported:

| Metric | Intermediate result |
|---|---:|
| Event-based F1 | 0.33 |
| Segment-based Dice (10 ms grid) | 0.68 |

These are **preliminary, checkpoint-specific values**, not a final benchmark claim. Earlier experiments produced different results, and the values should not be compared with frame-wise validation metrics unless the evaluation protocol is the same.

## Implementation notes and limitations

- The raw dataset and pretrained ATST checkpoint are not included.
- Dataset and checkpoint paths must be configured for the target environment.
- The checked-in training script is the frozen-ATST configuration.
- The late ATST fine-tuning experiment is not presented as the frozen baseline.
- Mean Teacher training was not implemented.
- The metadata generator `make_train_gt.py` currently expects the training JSONL under `metadata/`, whereas the committed training JSONL is at the repository root; adjust the path before running that generator.
- The scripts contain lab-specific paths and a CUDA device selection that may need changing on another machine.

## Repository structure

```text
.
├── nnet/
│   ├── atst/
│   └── beats/
├── dataset.py
├── encoder.py
├── train_gold_tversky.py
├── infer_validation_16ms.py
├── evaluate_validation.py
├── make_train_gt.py
├── make_gt_validation.py
├── gold_windows.jsonl
├── gt_validation.jsonl
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Acknowledgements

This work uses the task dataset and pretrained ATST components. Please consult the original dataset and model sources for their respective terms of use and citation requirements.
