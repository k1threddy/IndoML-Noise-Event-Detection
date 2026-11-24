import os

# Must be before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import DataLoader

from torchaudio.transforms import (
    MelSpectrogram,
    AmplitudeToDB,
)

from nnet.CRNN_e2e import CRNN
from dataset import VaaniStrongSet


# ============================================================
# Configuration
# ============================================================

GOLD_METADATA = (
    "gold_windows.jsonl"
)

ATST_CHECKPOINT = (
    "checkpoints/atst_as2M.ckpt"
)

CHECKPOINT_DIR = Path(
    "work/kevanth/checkpoints_tversky"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Raw validation data
# ------------------------------------------------------------

VALIDATION_DIR = Path(
    "/mnt/Work/Datasets/indoml/Vaani-Noise-Event-Dataset/validation"
)

VALIDATION_METADATA = (
    VALIDATION_DIR / "validationMetadata.json"
)

GT_FILE = (
    "gt_validation.jsonl"
)


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0

# Maximum number of epochs
EPOCHS = 20

CNN_LR = 5e-5
RNN_LR = 3e-4
OTHER_LR = 5e-5
ATST_LR = 5e-6

SAMPLE_RATE = 16000
AUDIO_LEN = 10.0


# ============================================================
# NEW LOSS
# ============================================================

STRONG_BCE_WEIGHT = 1.0

CLASS_TVERSKY_WEIGHT = 1.0

AGNOSTIC_TVERSKY_WEIGHT = 3.0

WEAK_WEIGHT = 0.5

TVERSKY_ALPHA = 0.6
TVERSKY_BETA = 1.5

DICE_SMOOTH = 1.0


# ============================================================
# Train Dice early-stop guardrail
# ============================================================

TRAIN_DICE_CUTOFF = 0.90

STOP_ON_TRAIN_DICE = True


# ============================================================
# Validation
# ============================================================

VAL_THRESHOLD = 0.50

# 256 samples / 16000 Hz = 16 ms
FRAME_DURATION = 256 / 16000.0

WINDOW_SIZE = 10.0
WINDOW_STRIDE = 5.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Classes
# ============================================================

CLASSES = [
    "animal",
    "appliance_machine",
    "baby_child",
    "human_non_speech",
    "phone_signal_alarm",
    "singing_music",
    "vehicle_traffic",
]


# ============================================================
# Collate
# ============================================================

def collate_fn(batch):

    batch = [
        item
        for item in batch
        if item is not None
    ]

    if len(batch) == 0:
        return None

    audio = torch.stack(
        [item[0] for item in batch]
    )

    atst_feats = torch.stack(
        [item[1] for item in batch]
    )

    labels = torch.stack(
        [item[2] for item in batch]
    )

    padded_indxs = torch.stack(
        [item[3] for item in batch]
    )

    metadata = [
        item[4]
        for item in batch
    ]

    return (
        audio,
        atst_feats,
        labels,
        padded_indxs,
        metadata,
    )


# ============================================================
# ATST normalization
# ============================================================

class ATSTNorm(nn.Module):

    def __init__(self):

        super().__init__()

        self.amp_to_db = AmplitudeToDB(
            stype="power",
            top_db=80,
        )

        self.min_value = -79.6482
        self.max_value = 50.6842

    def forward(self, spec):

        spec = self.amp_to_db(spec)

        spec = spec.clamp(
            min=-50,
            max=80,
        )

        spec = (
            (spec - self.min_value)
            /
            (
                self.max_value
                - self.min_value
            )
        )

        return spec * 2.0 - 1.0


# ============================================================
# SED Mel spectrogram
# ============================================================

sed_mel = MelSpectrogram(
    sample_rate=16000,
    n_fft=2048,
    win_length=2048,
    hop_length=256,
    f_min=0,
    f_max=8000,
    n_mels=128,
    window_fn=torch.hamming_window,
    wkwargs={"periodic": False},
    power=1,
).to(DEVICE)

sed_amp_to_db = AmplitudeToDB(
    stype="amplitude",
    top_db=80,
)

sed_amp_to_db.amin = 1e-5


def prepare_sed_features(audio):

    mel = sed_mel(audio)

    # [B, 1, 128, T]
    mel = mel.squeeze(1)

    mel = sed_amp_to_db(mel)

    mel = mel.clamp(
        min=-50,
        max=80,
    )

    min_val = mel.amin(
        dim=(1, 2),
        keepdim=True,
    )

    max_val = mel.amax(
        dim=(1, 2),
        keepdim=True,
    )

    mel = (
        (mel - min_val)
        /
        (
            max_val
            - min_val
            + 1e-8
        )
    )

    mel = mel * 2.0 - 1.0

    return mel


# ============================================================
# Frame mask
# ============================================================

def make_frame_mask(
    metadata,
    n_frames,
    device,
):

    masks = []

    for item in metadata:

        real_duration = float(
            item["real_duration"]
        )

        valid_frames = int(
            np.ceil(
                real_duration
                / FRAME_DURATION
            )
        )

        valid_frames = min(
            valid_frames,
            n_frames,
        )

        mask = torch.zeros(
            n_frames,
            dtype=torch.bool,
        )

        mask[:valid_frames] = True

        masks.append(mask)

    return torch.stack(
        masks
    ).to(device)


# ============================================================
# Align temporal dimensions
# ============================================================

def align_strong_targets(
    strong_pred,
    labels,
    frame_mask,
):

    n_frames = min(
        strong_pred.shape[-1],
        labels.shape[-1],
        frame_mask.shape[-1],
    )

    strong_pred = strong_pred[
        ..., :n_frames
    ]

    labels = labels[
        ..., :n_frames
    ]

    frame_mask = frame_mask[
        ..., :n_frames
    ]

    return (
        strong_pred,
        labels,
        frame_mask,
    )


# ============================================================
# CLASS-SPECIFIC Tversky loss
# ============================================================

def masked_tversky_loss(
    predictions,
    targets,
    frame_mask,
    alpha=0.6,
    beta=1.4,
    smooth=1.0,
):
    """
    Class-specific Tversky.

    predictions:
        [B, C, T]

    targets:
        [B, C, T]

    Tversky is calculated independently
    for every class and then averaged.
    """

    mask = (
        frame_mask
        .unsqueeze(1)
        .float()
    )

    predictions = (
        predictions * mask
    )

    targets = (
        targets * mask
    )

    predictions = predictions.flatten(
        start_dim=2
    )

    targets = targets.flatten(
        start_dim=2
    )

    TP = (
        predictions
        * targets
    ).sum(dim=2)

    FP = (
        predictions
        * (1.0 - targets)
    ).sum(dim=2)

    FN = (
        (1.0 - predictions)
        * targets
    ).sum(dim=2)

    tversky = (
        TP + smooth
    ) / (
        TP
        + alpha * FP
        + beta * FN
        + smooth
    )

    return 1.0 - tversky.mean()


# ============================================================
# CLASS-AGNOSTIC probability
# ============================================================

def class_agnostic_probability(
    class_probabilities
):
    """
    Convert [B, C, T] class probabilities into
    probability that ANY class is active.

    P(any active) = 1 - product(1 - P(class))

    This is preferable to amax() for the TRAINING LOSS
    because gradients can flow through all class channels.
    """

    return (
        1.0
        - torch.prod(
            1.0 - class_probabilities,
            dim=1,
        )
    )


# ============================================================
# CLASS-AGNOSTIC Tversky loss
# ============================================================

def masked_agnostic_tversky_loss(
    predictions,
    targets,
    frame_mask,
    alpha=0.6,
    beta=1.4,
    smooth=1.0,
):
    """
    Class-agnostic Tversky.

    predictions:
        [B, C, T]

    targets:
        [B, C, T]

    First collapses the class dimension into:

        probability(any event)

    Then calculates one temporal Tversky score.
    """

    # --------------------------------------------------------
    # Convert class probabilities to event probability
    # --------------------------------------------------------

    predictions = (
        class_agnostic_probability(
            predictions
        )
    )

    # --------------------------------------------------------
    # Collapse target classes
    # --------------------------------------------------------

    targets = targets.amax(
        dim=1
    )

    # --------------------------------------------------------
    # Apply temporal mask
    # --------------------------------------------------------

    mask = frame_mask.float()

    predictions = (
        predictions * mask
    )

    targets = (
        targets * mask
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    predictions = predictions.flatten(
        start_dim=1
    )

    targets = targets.flatten(
        start_dim=1
    )

    # --------------------------------------------------------
    # Tversky components
    # --------------------------------------------------------

    TP = (
        predictions
        * targets
    ).sum(dim=1)

    FP = (
        predictions
        * (1.0 - targets)
    ).sum(dim=1)

    FN = (
        (1.0 - predictions)
        * targets
    ).sum(dim=1)

    # --------------------------------------------------------
    # Tversky
    # --------------------------------------------------------

    tversky = (
        TP + smooth
    ) / (
        TP
        + alpha * FP
        + beta * FN
        + smooth
    )

    return 1.0 - tversky.mean()


# ============================================================
# Model
# ============================================================

def build_model():

    model = CRNN(

        nclass=7,
        n_in_channel=1,

        activation="cg",
        dropout=0.5,

        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,

        embedding_size=768,

        model_init=None,
        atst_init=ATST_CHECKPOINT,
        atst_dropout=0.0,

        kernel_size=[
            3, 3, 3, 3,
            3, 3, 3, 3,
        ],

        padding=[
            1, 1, 1, 1,
            1, 1, 1, 1,
        ],

        stride=[
            1, 1, 1, 1,
            1, 1, 1, 1,
        ],

        nb_filters=[
            16,
            32,
            64,
            128,
            128,
            128,
            128,
        ],

        pooling=[
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ],
    )

    # --------------------------------------------------------
    # Freeze ATST
    # --------------------------------------------------------

    for param in model.atst_frame.parameters():

        param.requires_grad = False

    # --------------------------------------------------------
    # Fine-tune final two ATST blocks
    # --------------------------------------------------------

    for param in (
        model.atst_frame
        .atst
        .blocks[10]
        .parameters()
    ):

        param.requires_grad = False

    for param in (
        model.atst_frame
        .atst
        .blocks[11]
        .parameters()
    ):

        param.requires_grad = False

    return model


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model):

    cnn_params = []
    rnn_params = []
    other_params = []
    atst_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name.startswith("cnn."):

            cnn_params.append(param)

        elif name.startswith("rnn."):

            rnn_params.append(param)

        elif (
            name.startswith(
                "atst_frame.atst.blocks.10"
            )
            or
            name.startswith(
                "atst_frame.atst.blocks.11"
            )
        ):

            atst_params.append(param)

        else:

            other_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {
                "params": cnn_params,
                "lr": CNN_LR,
            },
            {
                "params": rnn_params,
                "lr": RNN_LR,
            },
            {
                "params": other_params,
                "lr": OTHER_LR,
            },
            {
                "params": atst_params,
                "lr": ATST_LR,
            },
        ]
    )

    return optimizer


# ============================================================
# Validation windowing
# ============================================================

def make_windows(
    duration,
    window_size=WINDOW_SIZE,
    stride=WINDOW_STRIDE,
):

    if duration <= window_size:

        return [
            (0.0, duration)
        ]

    windows = []

    start = 0.0

    while start < duration:

        end = min(
            start + window_size,
            duration,
        )

        windows.append(
            (start, end)
        )

        if end >= duration:
            break

        start += stride

    return windows


# ============================================================
# Audio
# ============================================================

def load_audio(
    wav_path,
):

    waveform, sample_rate = (
        torchaudio.load(
            str(wav_path)
        )
    )

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != SAMPLE_RATE:

        waveform = (
            torchaudio.functional.resample(
                waveform,
                sample_rate,
                SAMPLE_RATE,
            )
        )

        sample_rate = SAMPLE_RATE

    return waveform, sample_rate


def extract_and_pad_window(
    waveform,
    window_start,
    window_end,
):

    start_sample = int(
        round(
            window_start
            * SAMPLE_RATE
        )
    )

    end_sample = int(
        round(
            window_end
            * SAMPLE_RATE
        )
    )

    window = waveform[
        :,
        start_sample:end_sample,
    ]

    target_samples = int(
        round(
            WINDOW_SIZE
            * SAMPLE_RATE
        )
    )

    current_samples = (
        window.shape[-1]
    )

    real_samples = current_samples

    if current_samples < target_samples:

        padding = (
            target_samples
            - current_samples
        )

        window = F.pad(
            window,
            (0, padding),
            mode="constant",
            value=0.0,
        )

    elif current_samples > target_samples:

        window = window[
            :,
            :target_samples,
        ]

        real_samples = target_samples

    return (
        window,
        real_samples,
    )


# ============================================================
# Load validation GT
# ============================================================

def load_validation_gt():

    gt_data = {}

    with open(
        GT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            gt_data[
                record["clip_id"]
            ] = [
                (
                    float(event["onset"]),
                    float(event["offset"]),
                )
                for event in record["events"]
            ]

    return gt_data


# ============================================================
# Validation audio index
# ============================================================

def build_audio_index():

    audio_index = {}

    for folder_name in [
        "naturalNoisyAudio",
        "syntheticNoiseAudio",
    ]:

        folder = (
            VALIDATION_DIR
            / folder_name
        )

        for wav_path in folder.rglob(
            "*.wav"
        ):

            audio_index[
                wav_path.name
            ] = wav_path

    print(
        f"Indexed "
        f"{len(audio_index)} validation WAV files"
    )

    return audio_index


# ============================================================
# Create class-agnostic validation target
# ============================================================

def events_to_frames(
    events,
    num_frames,
):

    target = np.zeros(
        num_frames,
        dtype=np.float32,
    )

    for onset, offset in events:

        start = int(
            np.floor(
                onset
                / FRAME_DURATION
            )
        )

        end = int(
            np.ceil(
                offset
                / FRAME_DURATION
            )
        )

        start = max(
            0,
            start,
        )

        end = min(
            num_frames,
            end,
        )

        if end > start:

            target[
                start:end
            ] = 1.0

    return target


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    gt_data,
    validation_metadata,
    audio_index,
):

    model.eval()

    total_val_loss = 0.0
    total_val_batches = 0

    total_bce = 0.0
    total_dice = 0.0

    frame_tp = 0
    frame_fp = 0
    frame_fn = 0

    for idx, item in enumerate(
        validation_metadata
    ):

        filename = item[
            "segmentFileName"
        ]

        if filename not in audio_index:

            raise FileNotFoundError(
                f"Could not find validation "
                f"audio: {filename}"
            )

        clip_id = Path(
            filename
        ).stem

        if clip_id not in gt_data:

            continue

        wav_path = (
            audio_index[
                filename
            ]
        )

        waveform, _ = load_audio(
            wav_path
        )

        duration = (
            waveform.shape[-1]
            / SAMPLE_RATE
        )

        gt_events = gt_data[
            clip_id
        ]

        windows = make_windows(
            duration
        )

        for (
            window_start,
            window_end,
        ) in windows:

            (
                window,
                real_samples,
            ) = extract_and_pad_window(
                waveform,
                window_start,
                window_end,
            )

            audio = (
                window
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # SED features
            # ------------------------------------------------

            sed_features = (
                prepare_sed_features(
                    audio
                )
            )

            # ------------------------------------------------
            # ATST
            # ------------------------------------------------

            atst_mel = (
                ATST_VALIDATION_MEL(
                    audio
                )
            )

            atst_features = (
                ATST_VALIDATION_NORM(
                    atst_mel
                )
            )

            atst_features = (
                atst_features
                .squeeze(1)
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            strong_pred, weak_pred = model(
                sed_features,
                atst_features,
            )

            # ------------------------------------------------
            # Class-agnostic prediction
            # ------------------------------------------------

            pred_any = (
                strong_pred.amax(
                    dim=1
                )
            )

            n_frames = (
                pred_any.shape[-1]
            )

            # ------------------------------------------------
            # Local GT
            # ------------------------------------------------

            local_events = []

            for onset, offset in gt_events:

                local_onset = (
                    onset
                    - window_start
                )

                local_offset = (
                    offset
                    - window_start
                )

                if (
                    local_offset <= 0
                    or local_onset
                    >= WINDOW_SIZE
                ):
                    continue

                local_onset = max(
                    0.0,
                    local_onset,
                )

                local_offset = min(
                    WINDOW_SIZE,
                    local_offset,
                )

                local_events.append(
                    (
                        local_onset,
                        local_offset,
                    )
                )

            target_np = events_to_frames(
                local_events,
                n_frames,
            )

            target = torch.from_numpy(
                target_np
            ).to(DEVICE)

            target = target.unsqueeze(0)

            # ------------------------------------------------
            # Real-audio mask
            # ------------------------------------------------

            valid_frames = int(
                np.ceil(
                    (
                        real_samples
                        / SAMPLE_RATE
                    )
                    / FRAME_DURATION
                )
            )

            valid_frames = min(
                valid_frames,
                n_frames,
            )

            frame_mask = torch.zeros(
                (1, n_frames),
                dtype=torch.bool,
                device=DEVICE,
            )

            frame_mask[
                :,
                :valid_frames
            ] = True

            # ------------------------------------------------
            # Align
            # ------------------------------------------------

            min_frames = min(
                pred_any.shape[-1],
                target.shape[-1],
                frame_mask.shape[-1],
            )

            pred_any = pred_any[
                ..., :min_frames
            ]

            target = target[
                ..., :min_frames
            ]

            frame_mask = frame_mask[
                ..., :min_frames
            ]

            # ------------------------------------------------
            # BCE
            # ------------------------------------------------

            bce_all = F.binary_cross_entropy(
                pred_any,
                target,
                reduction="none",
            )

            bce = (
                (
                    bce_all
                    * frame_mask.float()
                ).sum()
                /
                (
                    frame_mask.sum()
                    + 1e-8
                )
            )

            # ------------------------------------------------
            # Soft Dice
            # ------------------------------------------------

            mask = frame_mask.float()

            pred_masked = (
                pred_any * mask
            )

            target_masked = (
                target * mask
            )

            intersection = (
                pred_masked
                * target_masked
            ).sum()

            denominator = (
                pred_masked.sum()
                + target_masked.sum()
            )

            dice = (
                2.0 * intersection
                + DICE_SMOOTH
            ) / (
                denominator
                + DICE_SMOOTH
            )

            dice_loss = (
                1.0 - dice
            )

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            val_loss = (
                STRONG_BCE_WEIGHT
                * bce
                + dice_loss
            )

            total_val_loss += (
                val_loss.item()
            )

            total_bce += (
                bce.item()
            )

            total_dice += (
                dice_loss.item()
            )

            total_val_batches += 1

            # ------------------------------------------------
            # Hard frame metrics
            # ------------------------------------------------

            pred_binary = (
                pred_any
                >= VAL_THRESHOLD
            )

            target_binary = (
                target
                > 0.5
            )

            mask_bool = frame_mask

            tp = (
                pred_binary
                & target_binary
                & mask_bool
            ).sum().item()

            fp = (
                pred_binary
                & ~target_binary
                & mask_bool
            ).sum().item()

            fn = (
                ~pred_binary
                & target_binary
                & mask_bool
            ).sum().item()

            frame_tp += tp
            frame_fp += fp
            frame_fn += fn

        if (
            idx == 0
            or (idx + 1) % 100 == 0
        ):

            print(
                f"Validation "
                f"[{idx + 1}/"
                f"{len(validation_metadata)}]"
            )

    # ========================================================
    # Aggregate validation metrics
    # ========================================================

    val_loss = (
        total_val_loss
        /
        max(
            total_val_batches,
            1,
        )
    )

    val_bce = (
        total_bce
        /
        max(
            total_val_batches,
            1,
        )
    )

    val_dice_loss = (
        total_dice
        /
        max(
            total_val_batches,
            1,
        )
    )

    precision = (
        frame_tp
        /
        (
            frame_tp
            + frame_fp
        )
        if (
            frame_tp
            + frame_fp
        )
        else 0.0
    )

    recall = (
        frame_tp
        /
        (
            frame_tp
            + frame_fn
        )
        if (
            frame_tp
            + frame_fn
        )
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    denominator = (
        2 * frame_tp
        + frame_fp
        + frame_fn
    )

    frame_dice = (
        2.0 * frame_tp
        / denominator
        if denominator
        else 1.0
    )

    return {
        "loss": val_loss,
        "bce": val_bce,
        "dice_loss": val_dice_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": frame_dice,
        "tp": frame_tp,
        "fp": frame_fp,
        "fn": frame_fn,
    }


# ============================================================
# ATST validation features
# ============================================================

ATST_VALIDATION_MEL = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=160,
    f_min=60,
    f_max=7800,
    n_mels=64,
    power=2.0,
).to(DEVICE)

ATST_VALIDATION_NORM = ATSTNorm().to(
    DEVICE
)


# ============================================================
# TRAIN DICE
# ============================================================

@torch.no_grad()
def evaluate_train_dice(
    model,
    train_loader,
    atst_norm,
):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_examples = 0

    for batch_idx, batch in enumerate(
        train_loader
    ):

        if batch is None:
            continue

        (
            audio,
            atst_feats,
            labels,
            padded_indxs,
            metadata,
        ) = batch

        audio = audio.to(
            DEVICE,
            non_blocking=True,
        )

        atst_feats = atst_feats.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        sed_feats = (
            prepare_sed_features(
                audio
            )
        )

        atst_feats = atst_norm(
            atst_feats
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        strong_pred, weak_pred = model(
            sed_feats,
            atst_feats,
        )

        # ----------------------------------------------------
        # Frame mask
        # ----------------------------------------------------

        frame_mask = (
            make_frame_mask(
                metadata,
                strong_pred.shape[-1],
                DEVICE,
            )
        )

        # ----------------------------------------------------
        # Align
        # ----------------------------------------------------

        (
            strong_pred,
            labels,
            frame_mask,
        ) = align_strong_targets(
            strong_pred,
            labels,
            frame_mask,
        )

        # ----------------------------------------------------
        # Class-agnostic prediction
        #
        # We use the SAME definition as validation:
        # amax over classes.
        # ----------------------------------------------------

        pred_any = (
            strong_pred.amax(
                dim=1
            )
        )

        target_any = (
            labels.amax(
                dim=1
            )
        )

        # ----------------------------------------------------
        # Hard predictions
        # ----------------------------------------------------

        pred_binary = (
            pred_any
            >= VAL_THRESHOLD
        )

        target_binary = (
            target_any
            > 0.5
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        tp = (
            pred_binary
            & target_binary
            & frame_mask
        ).sum().item()

        fp = (
            pred_binary
            & ~target_binary
            & frame_mask
        ).sum().item()

        fn = (
            ~pred_binary
            & target_binary
            & frame_mask
        ).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_examples += (
            audio.shape[0]
        )

        if (
            batch_idx == 0
            or (batch_idx + 1) % 100 == 0
        ):

            print(
                f"Train evaluation "
                f"[{batch_idx + 1}/"
                f"{len(train_loader)}]"
            )

    # ========================================================
    # Aggregate
    # ========================================================

    precision = (
        total_tp
        /
        (
            total_tp
            + total_fp
        )
        if (
            total_tp
            + total_fp
        )
        else 0.0
    )

    recall = (
        total_tp
        /
        (
            total_tp
            + total_fn
        )
        if (
            total_tp
            + total_fn
        )
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    denominator = (
        2 * total_tp
        + total_fp
        + total_fn
    )

    dice = (
        2.0 * total_tp
        / denominator
        if denominator
        else 1.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print(
        "FRESH CLASS-AGNOSTIC TVERSKY TRAINING"
    )
    print("=" * 70)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Frame duration:",
        f"{FRAME_DURATION * 1000:.1f} ms",
    )

    print()
    print("Training objective:")
    print(
        "BCE + "
        "1.0*ClassTversky + "
        "3.0*AgnosticTversky + "
        "0.5*Weak"
    )

    print(
        "Tversky alpha:",
        TVERSKY_ALPHA,
    )

    print(
        "Tversky beta:",
        TVERSKY_BETA,
    )

    print()
    print(
        "Train Dice cutoff:",
        TRAIN_DICE_CUTOFF,
    )

    # ========================================================
    # Training dataset
    # ========================================================

    print()
    print(
        "Loading training dataset..."
    )

    train_dataset = VaaniStrongSet(
        GOLD_METADATA,
        audio_len=AUDIO_LEN,
        sample_rate=SAMPLE_RATE,
    )

    print(
        "Training examples:",
        len(train_dataset),
    )

    # --------------------------------------------------------
    # Training loader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Separate evaluation loader
    #
    # No shuffle.
    # --------------------------------------------------------

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ========================================================
    # Validation
    # ========================================================

    print()
    print(
        "Loading validation metadata..."
    )

    with open(
        VALIDATION_METADATA,
        "r",
    ) as f:

        validation_metadata = json.load(
            f
        )

    gt_data = load_validation_gt()

    audio_index = (
        build_audio_index()
    )

    print(
        "Validation clips:",
        len(validation_metadata),
    )

    print(
        "Validation GT clips:",
        len(gt_data),
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Building FRESH model..."
    )

    model = build_model().to(
        DEVICE
    )

    # ========================================================
    # IMPORTANT:
    # NO CHECKPOINT RESUME
    # ========================================================

    print()
    print(
        "Starting from fresh model initialization."
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        model
    )

    # ========================================================
    # ATST normalization
    # ========================================================

    atst_norm = ATSTNorm().to(
        DEVICE
    )

    # ========================================================
    # Best validation metrics
    # ========================================================

    best_val_f1 = -1.0
    best_val_dice = -1.0

    best_train_dice = -1.0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        running_strong_bce = 0.0
        running_class_tversky = 0.0
        running_agnostic_tversky = 0.0
        running_weak = 0.0

        # ====================================================
        # Training batches
        # ====================================================

        for batch_idx, batch in enumerate(
            train_loader
        ):

            if batch is None:
                continue

            (
                audio,
                atst_feats,
                labels,
                padded_indxs,
                metadata,
            ) = batch

            audio = audio.to(
                DEVICE,
                non_blocking=True,
            )

            atst_feats = atst_feats.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            # ------------------------------------------------
            # SED features
            # ------------------------------------------------

            sed_feats = (
                prepare_sed_features(
                    audio
                )
            )

            # ------------------------------------------------
            # ATST
            # ------------------------------------------------

            atst_feats = atst_norm(
                atst_feats
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            strong_pred, weak_pred = model(
                sed_feats,
                atst_feats,
            )

            # ------------------------------------------------
            # Frame mask
            # ------------------------------------------------

            frame_mask = (
                make_frame_mask(
                    metadata,
                    strong_pred.shape[-1],
                    DEVICE,
                )
            )

            # ------------------------------------------------
            # Align
            # ------------------------------------------------

            (
                strong_pred,
                labels,
                frame_mask,
            ) = align_strong_targets(
                strong_pred,
                labels,
                frame_mask,
            )

            # =================================================
            # STRONG BCE
            # =================================================

            strong_bce_all = (
                F.binary_cross_entropy(
                    strong_pred,
                    labels,
                    reduction="none",
                )
            )

            strong_bce = (
                (
                    strong_bce_all
                    * frame_mask
                    .unsqueeze(1)
                    .float()
                ).sum()
                /
                (
                    frame_mask.sum()
                    * labels.shape[1]
                    + 1e-8
                )
            )

            # =================================================
            # CLASS-SPECIFIC TVERSKY
            # =================================================

            class_tversky_loss = (
                masked_tversky_loss(
                    strong_pred,
                    labels,
                    frame_mask,
                    alpha=TVERSKY_ALPHA,
                    beta=TVERSKY_BETA,
                    smooth=DICE_SMOOTH,
                )
            )

            # =================================================
            # CLASS-AGNOSTIC TVERSKY
            # =================================================

            agnostic_tversky_loss = (
                masked_agnostic_tversky_loss(
                    strong_pred,
                    labels,
                    frame_mask,
                    alpha=TVERSKY_ALPHA,
                    beta=TVERSKY_BETA,
                    smooth=DICE_SMOOTH,
                )
            )

            # =================================================
            # WEAK LOSS
            # =================================================

            weak_labels = labels.amax(
                dim=-1
            )

            weak_loss = (
                F.binary_cross_entropy(
                    weak_pred,
                    weak_labels,
                )
            )

            # =================================================
            # TOTAL LOSS
            # =================================================

            loss = (
                STRONG_BCE_WEIGHT
                * strong_bce

                +

                CLASS_TVERSKY_WEIGHT
                * class_tversky_loss

                +

                AGNOSTIC_TVERSKY_WEIGHT
                * agnostic_tversky_loss

                +

                WEAK_WEIGHT
                * weak_loss
            )

            # =================================================
            # BACKPROP
            # =================================================

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            # =================================================
            # Logging
            # =================================================

            running_loss += (
                loss.item()
            )

            running_strong_bce += (
                strong_bce.item()
            )

            running_class_tversky += (
                class_tversky_loss.item()
            )

            running_agnostic_tversky += (
                agnostic_tversky_loss.item()
            )

            running_weak += (
                weak_loss.item()
            )

            if batch_idx % 100 == 0:

                print(
                    f"Epoch "
                    f"[{epoch}/{EPOCHS}] "
                    f"Batch "
                    f"[{batch_idx}/"
                    f"{len(train_loader)}] "
                    f"Loss: "
                    f"{loss.item():.5f} "
                    f"BCE: "
                    f"{strong_bce.item():.5f} "
                    f"ClassTV: "
                    f"{class_tversky_loss.item():.5f} "
                    f"AgnosticTV: "
                    f"{agnostic_tversky_loss.item():.5f} "
                    f"Weak: "
                    f"{weak_loss.item():.5f}"
                )

        # ====================================================
        # Epoch training loss
        # ====================================================

        num_batches = max(
            len(train_loader),
            1,
        )

        epoch_loss = (
            running_loss
            / num_batches
        )

        epoch_bce = (
            running_strong_bce
            / num_batches
        )

        epoch_class_tversky = (
            running_class_tversky
            / num_batches
        )

        epoch_agnostic_tversky = (
            running_agnostic_tversky
            / num_batches
        )

        epoch_weak = (
            running_weak
            / num_batches
        )

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch} TRAINING LOSS"
        )
        print("=" * 70)

        print(
            f"Total loss             : "
            f"{epoch_loss:.6f}"
        )

        print(
            f"Strong BCE             : "
            f"{epoch_bce:.6f}"
        )

        print(
            f"Class Tversky loss     : "
            f"{epoch_class_tversky:.6f}"
        )

        print(
            f"Agnostic Tversky loss  : "
            f"{epoch_agnostic_tversky:.6f}"
        )

        print(
            f"Weak loss              : "
            f"{epoch_weak:.6f}"
        )

        # ====================================================
        # TRAINING SET DICE
        # ====================================================

        print()
        print(
            "Running training-set evaluation..."
        )

        train_results = (
            evaluate_train_dice(
                model,
                train_eval_loader,
                atst_norm,
            )
        )

        train_dice = (
            train_results["dice"]
        )

        train_precision = (
            train_results["precision"]
        )

        train_recall = (
            train_results["recall"]
        )

        train_f1 = (
            train_results["f1"]
        )

        best_train_dice = max(
            best_train_dice,
            train_dice,
        )

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch} TRAIN METRICS"
        )
        print("=" * 70)

        print(
            f"Train precision : "
            f"{train_precision:.6f}"
        )

        print(
            f"Train recall    : "
            f"{train_recall:.6f}"
        )

        print(
            f"Train F1        : "
            f"{train_f1:.6f}"
        )

        print(
            f"Train Dice      : "
            f"{train_dice:.6f}"
        )

        print(
            f"Train TP        : "
            f"{train_results['tp']}"
        )

        print(
            f"Train FP        : "
            f"{train_results['fp']}"
        )

        print(
            f"Train FN        : "
            f"{train_results['fn']}"
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        print()
        print(
            "Running raw validation..."
        )

        val_results = validate(
            model,
            gt_data,
            validation_metadata,
            audio_index,
        )

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch} VALIDATION"
        )
        print("=" * 70)

        print(
            f"Val loss       : "
            f"{val_results['loss']:.6f}"
        )

        print(
            f"Val BCE        : "
            f"{val_results['bce']:.6f}"
        )

        print(
            f"Val Dice loss  : "
            f"{val_results['dice_loss']:.6f}"
        )

        print(
            f"Val precision  : "
            f"{val_results['precision']:.6f}"
        )

        print(
            f"Val recall     : "
            f"{val_results['recall']:.6f}"
        )

        print(
            f"Val F1         : "
            f"{val_results['f1']:.6f}"
        )

        print(
            f"Val Dice       : "
            f"{val_results['dice']:.6f}"
        )

        print(
            f"TP             : "
            f"{val_results['tp']}"
        )

        print(
            f"FP             : "
            f"{val_results['fp']}"
        )

        print(
            f"FN             : "
            f"{val_results['fn']}"
        )

        # ====================================================
        # Save regular checkpoint
        # ====================================================

        checkpoint_path = (
            CHECKPOINT_DIR
            /
            f"classagnostic_epoch_"
            f"{epoch}.pt"
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss":
                    epoch_loss,

                "train_precision":
                    train_precision,

                "train_recall":
                    train_recall,

                "train_f1":
                    train_f1,

                "train_dice":
                    train_dice,

                "val_loss":
                    val_results["loss"],

                "val_f1":
                    val_results["f1"],

                "val_dice":
                    val_results["dice"],

                "class_tversky_weight":
                    CLASS_TVERSKY_WEIGHT,

                "agnostic_tversky_weight":
                    AGNOSTIC_TVERSKY_WEIGHT,

                "tversky_alpha":
                    TVERSKY_ALPHA,

                "tversky_beta":
                    TVERSKY_BETA,
            },
            checkpoint_path,
        )

        print()
        print(
            "Saved:",
            checkpoint_path,
        )

        # ====================================================
        # BEST VALIDATION F1
        # ====================================================

        if (
            val_results["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_results["f1"]
            )

            best_path = (
                CHECKPOINT_DIR
                /
                "best_classagnostic_f1.pt"
            )

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        epoch_loss,

                    "train_precision":
                        train_precision,

                    "train_recall":
                        train_recall,

                    "train_f1":
                        train_f1,

                    "train_dice":
                        train_dice,

                    "val_loss":
                        val_results["loss"],

                    "val_f1":
                        val_results["f1"],

                    "val_dice":
                        val_results["dice"],

                    "class_tversky_weight":
                        CLASS_TVERSKY_WEIGHT,

                    "agnostic_tversky_weight":
                        AGNOSTIC_TVERSKY_WEIGHT,

                    "tversky_alpha":
                        TVERSKY_ALPHA,

                    "tversky_beta":
                        TVERSKY_BETA,
                },
                best_path,
            )

            print(
                "NEW BEST VAL F1:",
                f"{best_val_f1:.6f}",
            )

        # ====================================================
        # BEST VALIDATION DICE
        # ====================================================

        if (
            val_results["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_results["dice"]
            )

            best_path = (
                CHECKPOINT_DIR
                /
                "best_classagnostic_dice.pt"
            )

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        epoch_loss,

                    "train_precision":
                        train_precision,

                    "train_recall":
                        train_recall,

                    "train_f1":
                        train_f1,

                    "train_dice":
                        train_dice,

                    "val_loss":
                        val_results["loss"],

                    "val_f1":
                        val_results["f1"],

                    "val_dice":
                        val_results["dice"],

                    "class_tversky_weight":
                        CLASS_TVERSKY_WEIGHT,

                    "agnostic_tversky_weight":
                        AGNOSTIC_TVERSKY_WEIGHT,

                    "tversky_alpha":
                        TVERSKY_ALPHA,

                    "tversky_beta":
                        TVERSKY_BETA,
                },
                best_path,
            )

            print(
                "NEW BEST VAL DICE:",
                f"{best_val_dice:.6f}",
            )

        # ====================================================
        # Epoch summary
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            f"Epoch {epoch} SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Train Dice : "
            f"{train_dice:.6f}"
        )

        print(
            f"Val Dice   : "
            f"{val_results['dice']:.6f}"
        )

        print(
            f"Val F1     : "
            f"{val_results['f1']:.6f}"
        )

        print(
            f"Val Recall : "
            f"{val_results['recall']:.6f}"
        )

        print(
            f"Best Val Dice : "
            f"{best_val_dice:.6f}"
        )

        print(
            f"Best Val F1   : "
            f"{best_val_f1:.6f}"
        )

        # ====================================================
        # HARD TRAIN DICE CUTOFF
        # ====================================================

        if (
            STOP_ON_TRAIN_DICE
            and train_dice
            >= TRAIN_DICE_CUTOFF
        ):

            print()
            print("=" * 70)
            print(
                "TRAIN DICE CUTOFF REACHED"
            )
            print("=" * 70)

            print(
                f"Train Dice = "
                f"{train_dice:.6f}"
            )

            print(
                f"Cutoff     = "
                f"{TRAIN_DICE_CUTOFF:.6f}"
            )

            print()
            print(
                "Stopping training to prevent "
                "further memorization."
            )

            break

        print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()