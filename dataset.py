import io
import pandas as pd
import torch
import torchaudio
import json
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from encoder import ManyHotEncoder


# Vaani dataset

DATASET_DIR = Path("/mnt/Work/Datasets/indoml/Vaani-Noise-Event-Dataset")
DATA_DIR = DATASET_DIR / "data"

TARGET_SAMPLE_RATE = 16000
WINDOW_SIZE = 10.0
STRIDE = 5.0


# Windowing

def make_windows(duration, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Create overlapping windows for train/validation.

    Windows start every `stride` seconds and contain at most
    `window_size` seconds of real audio.

    Examples:
        4.57 s  -> [(0.0, 4.57)]
        10.0 s  -> [(0.0, 10.0)]
        14.0 s  -> [(0.0, 10.0), (5.0, 14.0)]
        15.0 s  -> [(0.0, 10.0), (5.0, 15.0)]
    """
    if duration <= window_size:
        return [(0.0, duration)]

    windows = []
    start = 0.0

    while start < duration:
        end = min(start + window_size, duration)
        windows.append((start, end))

        if end >= duration:
            break

        start += stride

    return windows


# Ground-truth event processing

def get_window_events(events, window_start, window_end):
    """
    Intersect Vaani events with a window and convert their
    timestamps from global coordinates to window-relative
    coordinates.
    """
    window_events = []

    for event in events:
        event_start = float(event["start"])
        event_end = float(event["end"])

        # Intersection between event and window
        overlap_start = max(event_start, window_start)
        overlap_end = min(event_end, window_end)

        # No overlap
        if overlap_start >= overlap_end:
            continue

        window_events.append(
            {
                "event_label": event["category"],
                "onset": overlap_start - window_start,
                "offset": overlap_end - window_start,
            }
        )

    return window_events


# Audio loading


def load_audio(audio_data):
    """
    Decode the audio stored in a Vaani parquet row.

    Returns:
        waveform: Tensor [channels, samples]
        sample_rate: int
    """
    waveform, sample_rate = torchaudio.load(
        io.BytesIO(audio_data["bytes"])
    )

    return waveform, sample_rate


# Audio window extraction + padding

def extract_and_pad_window(
    waveform,
    window_start,
    window_end,
    sample_rate,
    target_duration=WINDOW_SIZE,
):
    """
    Extract a real-audio window and zero-pad it to exactly
    `target_duration` seconds.
    """
    start_sample = int(round(window_start * sample_rate))
    end_sample = int(round(window_end * sample_rate))

    window = waveform[:, start_sample:end_sample]

    target_samples = int(round(target_duration * sample_rate))

    current_samples = window.shape[-1]

    if current_samples < target_samples:
        padding = target_samples - current_samples

        window = torch.nn.functional.pad(
            window,
            (0, padding),
            mode="constant",
            value=0.0,
        )

    elif current_samples > target_samples:
        window = window[:, :target_samples]

    return window


# Prepare one Vaani clip

def prepare_clip(row, source_file):
    """
    Convert one Vaani clip into window-level metadata.

    No audio is written to disk. Each window stores the
    information required to extract the audio later.
    """
    duration = float(row["duration"])
    events = row["NoiseSubCategoryTimeStamp"]

    windows = make_windows(duration)

    samples = []

    for window_start, window_end in windows:

        window_events = get_window_events(
            events,
            window_start,
            window_end,
        )

        samples.append(
            {
                "source_file": str(source_file),
                "audio_path": row["audio"]["path"],
                "window_start": window_start,
                "window_end": window_end,
                "real_duration": window_end - window_start,
                "events": window_events,
            }
        )

    return samples


# Prepare metadata from one parquet file

def prepare_parquet_file(parquet_file):
    """
    Prepare all verified-timestamp clips from one parquet file.
    """
    df = pd.read_parquet(parquet_file)

    verified_df = df[
        df["annotationQuality"] == "verified_timestamps"
    ]

    samples = []

    for _, row in verified_df.iterrows():
        samples.extend(
            prepare_clip(
                row,
                parquet_file,
            )
        )

    return samples

# Vaani classes

CLASSES = [
    "animal",
    "appliance_machine",
    "baby_child",
    "human_non_speech",
    "phone_signal_alarm",
    "singing_music",
    "vehicle_traffic",
]

# ATST feature extraction


class ATSTTransform:
    """
    Feature extraction used by ATST-SED.

    Input:
        waveform: [1, samples]

    Output:
        mel spectrogram: [1, n_mels, time]
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=160,
        f_min=60,
        f_max=7800,
    ):
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    def __call__(self, waveform):
        return self.mel(waveform)


# Vaani Strong Dataset


class VaaniStrongSet(Dataset):
    """
    Dataset for Gold/Silver Vaani samples.

    Each metadata entry corresponds to one 10-second window.

    Returns:
        audio:
            [1, 160000]

        atst_feat:
            [1, 64, T]

        strong:
            [7, 156]

        padded_indx:
            boolean mask indicating padded region

        metadata:
            original metadata entry
    """

    def __init__(
        self,
        metadata_file,
        audio_len=10.0,
        sample_rate=16000,
    ):
        self.metadata_file = Path(metadata_file)
        self.audio_len = audio_len
        self.sample_rate = sample_rate

        # Load metadata


        with open(self.metadata_file, "r") as f:
            self.metadata = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        # Encoder


        self.encoder = ManyHotEncoder(
            labels=CLASSES,
            audio_len=audio_len,
            frame_len=2048,
            frame_hop=256,
            net_pooling=1,
            fs=sample_rate,
        )

        # ATST feature extractor

        self.atst_transform = ATSTTransform(
            sample_rate=sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=160,
            f_min=60,
            f_max=7800,
        )
        

        # Cache one parquet shard at a time.
        #
        # We don't want to keep every audio shard in RAM.

        self._cached_parquet = None
        self._cached_dataframe = None

        self.target_samples = int(
            round(audio_len * sample_rate)
        )

    def __len__(self):
        return len(self.metadata)

    # Parquet loading

    def _load_parquet_row(self, parquet_file, row_index):

        parquet_file = str(parquet_file)

        # Reuse currently loaded shard
        if parquet_file != self._cached_parquet:

            self._cached_dataframe = pd.read_parquet(
                parquet_file
            )

            self._cached_parquet = parquet_file

        row = self._cached_dataframe.iloc[row_index]

        return row

    # Audio loading

    def _load_audio(self, audio_data):

        waveform, sample_rate = torchaudio.load(
            io.BytesIO(audio_data["bytes"])
        )

        # Vaani should already be 16 kHz, but don't silently
        # assume that forever.
        if sample_rate != self.sample_rate:

            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate,
            )

            sample_rate = self.sample_rate

        # Convert stereo/multichannel to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    # Extract requested window

    def _extract_window(
        self,
        waveform,
        window_start,
        window_end,
    ):
        start_sample = int(
            round(window_start * self.sample_rate)
        )

        end_sample = int(
            round(window_end * self.sample_rate)
        )

        window = waveform[
            :,
            start_sample:end_sample,
        ]

        real_samples = window.shape[-1]

        # Pad to exactly 10 seconds
        if real_samples < self.target_samples:

            padding = (
                self.target_samples
                - real_samples
            )

            window = F.pad(
                window,
                (0, padding),
                mode="constant",
                value=0.0,
            )

        # Safety truncation
        elif real_samples > self.target_samples:

            window = window[
                :, :self.target_samples
            ]

            real_samples = self.target_samples

        return window, real_samples

    # Main


    def __getitem__(self, index):

        item = self.metadata[index]


        # Load original audio from parquet

        row = self._load_parquet_row(
            item["parquet_file"],
            item["row_index"],
        )

        try:
            waveform = self._load_audio(
                row["audio"]
            )
        except Exception as e:
            print("SKIPPING UNDECODABLE AUDIO")
            print("parquet_file:", item["parquet_file"])
            print("row_index:", item["row_index"])
            print("window_start:", item["window_start"])
            print("window_end:", item["window_end"])
            print("error:", repr(e))
            print("=" * 80) 
            return None

        # Extract 10-secondprint("\n" + "=" * 80)


        window, real_samples = self._extract_window(
            waveform,
            item["window_start"],
            item["window_end"],
        )

        # ATST features


        atst_feat = self.atst_transform(window).squeeze(0)


        # Strong labels
        #
        # Metadata already contains LOCAL window timestamps.
        #
        # Example:
        # {
        #   "event_label": "animal",
        #   "onset": 1.2,
        #   "offset": 3.4
        # }

        events = item.get("events", [])

        if len(events) == 0:

            strong = np.zeros(
                (
                    self.encoder.n_frames,
                    len(CLASSES),
                ),
                dtype=np.float32,
            )

        else:

            events_df = pd.DataFrame(events)

            strong = self.encoder.encode_strong_df(
                events_df
            ).astype(np.float32)

        strong = torch.from_numpy(strong)

        # ATST-SED expects [classes, frames]
        strong = strong.transpose(0, 1)

        # Padding mask
        #
        # True = padded region
        # False = real audio

        padded_indx = torch.zeros(
            self.target_samples,
            dtype=torch.bool,
        )

        if real_samples < self.target_samples:

            padded_indx[real_samples:] = True

        # Convert audio to float32-

        window = window.float()

        return (
            window,
            atst_feat,
            strong,
            padded_indx,
            item,
        )