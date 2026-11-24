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
from scipy.ndimage import median_filter

from torchaudio.transforms import (
    MelSpectrogram,
    AmplitudeToDB,
)

from nnet.CRNN_e2e import CRNN


# ============================================================
# Configuration
# ============================================================


CHECKPOINT = Path(
    "work/kevanth/checkpoints_tversky/best_classagnostic_dice.pt"
)

OUTPUT_FILE = Path(
    "work/kevanth/inference/pred_validation.jsonl"
)

VALIDATION_DIR = Path(
    "/mnt/Work/Datasets/indoml/Vaani-Noise-Event-Dataset/validation"
)

VALIDATION_METADATA = (
    VALIDATION_DIR / "validationMetadata.json"
)

SAMPLE_RATE = 16000

WINDOW_SIZE = 10.0
WINDOW_STRIDE = 5.0
MEDIAN_FILTER_SIZE = 5

# ------------------------------------------------------------
# Detection threshold
# ------------------------------------------------------------

VAL_THRESHOLD = 0.85

# ------------------------------------------------------------
# Frame duration
#
# hop = 256 samples
# 256 / 16000 = 16 ms
# ------------------------------------------------------------

FRAME_DURATION = 256 / 16000.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# ATST checkpoint
# ============================================================

ATST_CHECKPOINT = "checkpoints/atst_as2M.ckpt"


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
# ATST validation feature extraction
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

ATST_VALIDATION_NORM = ATSTNorm().to(DEVICE)


# ============================================================
# Build model
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
            3, 3, 3,
        ],

        padding=[
            1, 1, 1, 1,
            1, 1, 1,
        ],

        stride=[
            1, 1, 1, 1,
            1, 1, 1,
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

        # No temporal pooling.
        # First dimension = time.
        # Second dimension = frequency.
        #
        # Therefore [1, 2]:
        #   time     -> unchanged
        #   frequency -> /2
        #
        # Seven such layers reduce:
        # 128 frequency bins -> 1
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

    return model


# ============================================================
# Load audio
# ============================================================

def load_audio(wav_path):

    waveform, sample_rate = torchaudio.load(
        str(wav_path)
    )

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != SAMPLE_RATE:

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            SAMPLE_RATE,
        )

        sample_rate = SAMPLE_RATE

    return waveform, sample_rate


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
# Extract + pad window
# ============================================================

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
# Build validation audio index
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
# Frame probabilities -> events
#
# IMPORTANT:
#
# This function ONLY groups CONTIGUOUS positive frames.
#
# There is NO gap filling.
#
# Example:
#
# frames:
#
# 111111000001111111
# ^^^^^^     ^^^^^^^
#
# becomes:
#
# event 1
# event 2
#
# It does NOT become one event.
# ============================================================

def probabilities_to_events(
    probabilities,
    threshold,
    frame_duration,
    valid_duration,
):

    binary = (
        probabilities >= threshold
    )

    events = []

    start_frame = None

    for i, active in enumerate(binary):

        # ----------------------------------------------------
        # Start a new event
        # ----------------------------------------------------

        if active and start_frame is None:

            start_frame = i

        # ----------------------------------------------------
        # End the current event
        #
        # The first negative frame terminates the event.
        #
        # No gap filling is performed.
        # ----------------------------------------------------

        elif (
            not active
            and start_frame is not None
        ):

            onset = (
                start_frame
                * frame_duration
            )

            offset = (
                i
                * frame_duration
            )

            events.append(
                (
                    onset,
                    offset,
                )
            )

            start_frame = None

    # --------------------------------------------------------
    # Event continuing until final frame
    # --------------------------------------------------------

    if start_frame is not None:

        onset = (
            start_frame
            * frame_duration
        )

        offset = (
            len(binary)
            * frame_duration
        )

        events.append(
            (
                onset,
                offset,
            )
        )

    # --------------------------------------------------------
    # Clip events to actual audio duration
    # --------------------------------------------------------

    clipped_events = []

    for onset, offset in events:

        onset = max(
            0.0,
            min(
                onset,
                valid_duration,
            ),
        )

        offset = max(
            0.0,
            min(
                offset,
                valid_duration,
            ),
        )

        if offset > onset:

            clipped_events.append(
                (
                    onset,
                    offset,
                )
            )

    return clipped_events


# ============================================================
# Merge overlapping / near-adjacent events
#
# IMPORTANT:
#
# This is NOT frame-gap filling.
#
# This is only used after local events have been converted
# to absolute time, to consolidate duplicate/overlapping
# predictions originating from the overlapping 10-second
# inference windows.
#
# gap_tolerance = 16 ms.
# ============================================================

def merge_events(
    events,
    gap_tolerance=FRAME_DURATION,
):

    if not events:
        return []

    events = sorted(
        events,
        key=lambda x: x[0],
    )

    merged = []

    current_onset, current_offset = (
        events[0]
    )

    for onset, offset in events[1:]:

        if onset <= (
            current_offset
            + gap_tolerance
        ):

            current_offset = max(
                current_offset,
                offset,
            )

        else:

            merged.append(
                (
                    current_onset,
                    current_offset,
                )
            )

            current_onset = onset
            current_offset = offset

    merged.append(
        (
            current_onset,
            current_offset,
        )
    )

    return merged


# ============================================================
# Run inference on one audio file
# ============================================================

@torch.no_grad()
def infer_audio(
    model,
    wav_path,
):

    waveform, _ = load_audio(
        wav_path
    )

    duration = (
        waveform.shape[-1]
        / SAMPLE_RATE
    )

    windows = make_windows(
        duration
    )

    all_events = []

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

        # ----------------------------------------------------
        # SED features
        # ----------------------------------------------------

        sed_features = (
            prepare_sed_features(
                audio
            )
        )

        # ----------------------------------------------------
        # ATST features
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        strong_pred, weak_pred = model(
            sed_features,
            atst_features,
        )

        # ----------------------------------------------------
        # Class-agnostic prediction
        #
        # [1, 7, T]
        #       |
        #     amax
        #       v
        # [1, T]
        # ----------------------------------------------------

        pred_any = (
            strong_pred
            .amax(dim=1)
            .squeeze(0)
        )

        probabilities = (
            pred_any
            .detach()
            .cpu()
            .numpy()
        )
        probabilities = median_filter(
            probabilities,
            size=MEDIAN_FILTER_SIZE,
        )

        n_frames = len(
            probabilities
        )

        # ----------------------------------------------------
        # Only keep frames corresponding to real audio
        # ----------------------------------------------------

        real_duration = (
            real_samples
            / SAMPLE_RATE
        )

        valid_frames = min(
            int(
                np.ceil(
                    real_duration
                    / FRAME_DURATION
                )
            ),
            n_frames,
        )

        probabilities = (
            probabilities[
                :valid_frames
            ]
        )

        # ----------------------------------------------------
        # Convert window-local frames into events
        #
        # IMPORTANT:
        # Separate positive runs remain separate.
        # No gap filling.
        # ----------------------------------------------------

        local_events = (
            probabilities_to_events(
                probabilities,
                VAL_THRESHOLD,
                FRAME_DURATION,
                real_duration,
            )
        )

        # ----------------------------------------------------
        # Convert local event times into global time
        # ----------------------------------------------------

        for (
            local_onset,
            local_offset,
        ) in local_events:

            absolute_onset = (
                window_start
                + local_onset
            )

            absolute_offset = (
                window_start
                + local_offset
            )

            absolute_onset = max(
                0.0,
                min(
                    absolute_onset,
                    duration,
                ),
            )

            absolute_offset = max(
                0.0,
                min(
                    absolute_offset,
                    duration,
                ),
            )

            if (
                absolute_offset
                > absolute_onset
            ):

                all_events.append(
                    (
                        absolute_onset,
                        absolute_offset,
                    )
                )

    # --------------------------------------------------------
    # Merge duplicate/overlapping events created by the
    # overlapping 10-second inference windows.
    #
    # This is retained intentionally.
    # --------------------------------------------------------

    all_events = merge_events(
        all_events
    )

    # --------------------------------------------------------
    # Round for JSON readability
    # --------------------------------------------------------

    events = []

    for onset, offset in all_events:

        events.append(
            {
                "onset": round(
                    float(onset),
                    3,
                ),
                "offset": round(
                    float(offset),
                    3,
                ),
            }
        )

    return events


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Threshold:",
        VAL_THRESHOLD,
    )

    print(
        "Frame duration:",
        f"{FRAME_DURATION * 1000:.1f} ms",
    )

    print(
        "Window size:",
        f"{WINDOW_SIZE:.1f} s",
    )

    print(
        "Window stride:",
        f"{WINDOW_STRIDE:.1f} s",
    )

    print(
        "Frame-gap filling:",
        "DISABLED",
    )

    print(
        "Cross-window merge:",
        "ENABLED",
        f"(gap tolerance = "
        f"{FRAME_DURATION * 1000:.1f} ms)",
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validation metadata
    # --------------------------------------------------------

    print()
    print(
        "Loading validation metadata..."
    )

    with open(
        VALIDATION_METADATA,
        "r",
        encoding="utf-8",
    ) as f:

        validation_metadata = json.load(
            f
        )

    print(
        "Validation clips:",
        len(validation_metadata),
    )

    # --------------------------------------------------------
    # Audio index
    # --------------------------------------------------------

    audio_index = (
        build_audio_index()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Building model..."
    )

    model = (
        build_model()
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print(
        "Loading checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Checkpoint validation F1:",
        checkpoint.get(
            "val_f1",
            "unknown",
        ),
    )

    print(
        "Checkpoint validation Dice:",
        checkpoint.get(
            "val_dice",
            "unknown",
        ),
    )

    model.eval()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "Starting validation inference..."
    )

    results = []

    for idx, item in enumerate(
        validation_metadata
    ):

        filename = item[
            "segmentFileName"
        ]

        if filename not in audio_index:

            print(
                "WARNING: missing audio:",
                filename,
            )

            continue

        wav_path = (
            audio_index[
                filename
            ]
        )

        clip_id = Path(
            filename
        ).stem

        events = infer_audio(
            model,
            wav_path,
        )

        record = {
            "clip_id": clip_id,
            "events": events,
        }

        results.append(
            record
        )

        print(
            f"[{idx + 1}/"
            f"{len(validation_metadata)}] "
            f"{clip_id}: "
            f"{len(events)} events"
        )

    # --------------------------------------------------------
    # Write ONE JSONL file
    # --------------------------------------------------------

    print()
    print(
        "Writing:",
        OUTPUT_FILE,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for record in results:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 70)
    print(
        "INFERENCE COMPLETE"
    )
    print("=" * 70)

    print(
        "Clips written:",
        len(results),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    total_events = sum(
        len(
            record["events"]
        )
        for record in results
    )

    print(
        "Total predicted events:",
        total_events,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()