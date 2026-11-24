import numpy as np
import pandas as pd
from dcase_util.data import DecisionEncoder


class ManyHotEncoder:
    """
    Encode weak and strong sound-event labels into many-hot arrays.

    Strong labels are represented frame-by-frame:
        (n_frames, n_classes)

    Weak labels are represented as:
        (n_classes,)
    """

    def __init__(
        self,
        labels,
        audio_len,
        frame_len,
        frame_hop,
        net_pooling=1,
        fs=16000,
    ):
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.fs = fs
        self.net_pooling = net_pooling

        n_samples = self.audio_len * self.fs

        self.n_frames = int(
            int(n_samples / self.frame_hop) / self.net_pooling
        )

    def encode_weak(self, labels):
        """Encode clip-level labels into a many-hot vector."""

        if isinstance(labels, str):
            if labels == "empty":
                return np.zeros(len(self.labels)) - 1

            labels = labels.split(",")

        if isinstance(labels, pd.DataFrame):
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]

        y = np.zeros(len(self.labels))

        for label in labels:
            if not pd.isna(label):
                i = self.labels.index(label)
                y[i] = 1

        return y

    def _time_to_frame(self, time):
        """Convert time in seconds to network frame index."""

        samples = time * self.fs
        frame = samples / self.frame_hop

        return np.clip(
            frame / self.net_pooling,
            a_min=0,
            a_max=self.n_frames,
        )

    def _frame_to_time(self, frame):
        """Convert network frame index back to seconds."""

        frame = (
            frame
            * self.net_pooling
            / (self.fs / self.frame_hop)
        )

        return np.clip(
            frame,
            a_min=0,
            a_max=self.audio_len,
        )

    def encode_strong_df(self, label_df):
        """
        Encode strong event annotations into a frame-level
        many-hot target.

        Returns:
            numpy array of shape (n_frames, n_classes)
        """

        assert any(
            x is not None
            for x in [self.audio_len, self.frame_len, self.frame_hop]
        )

        n_frames = self.n_frames

        if isinstance(label_df, str):
            if label_df == "empty":
                return np.zeros(
                    (n_frames, len(self.labels))
                ) - 1

        y = np.zeros(
            (n_frames, len(self.labels))
        )

        if isinstance(label_df, pd.DataFrame):

            if {
                "onset",
                "offset",
                "event_label",
            }.issubset(label_df.columns):

                for _, row in label_df.iterrows():

                    if not pd.isna(row["event_label"]):

                        i = self.labels.index(
                            row["event_label"]
                        )

                        onset = int(
                            self._time_to_frame(
                                row["onset"]
                            )
                        )

                        offset = int(
                            np.ceil(
                                self._time_to_frame(
                                    row["offset"]
                                )
                            )
                        )

                        y[onset:offset, i] = 1

        elif isinstance(
            label_df,
            (pd.Series, list, np.ndarray),
        ):

            if isinstance(label_df, pd.Series):

                if {
                    "onset",
                    "offset",
                    "event_label",
                }.issubset(label_df.index):

                    if not pd.isna(
                        label_df["event_label"]
                    ):

                        i = self.labels.index(
                            label_df["event_label"]
                        )

                        onset = int(
                            self._time_to_frame(
                                label_df["onset"]
                            )
                        )

                        offset = int(
                            np.ceil(
                                self._time_to_frame(
                                    label_df["offset"]
                                )
                            )
                        )

                        y[onset:offset, i] = 1

                    return y

            for event_label in label_df:

                # Weak label supplied to strong encoder
                if isinstance(event_label, str):

                    if event_label != "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # [label, onset, offset]
                elif len(event_label) == 3:

                    if event_label[0] != "":

                        i = self.labels.index(
                            event_label[0]
                        )

                        onset = int(
                            self._time_to_frame(
                                event_label[1]
                            )
                        )

                        offset = int(
                            np.ceil(
                                self._time_to_frame(
                                    event_label[2]
                                )
                            )
                        )

                        y[onset:offset, i] = 1

                else:
                    raise NotImplementedError(
                        "cannot encode strong, "
                        f"type mismatch: {type(event_label)}"
                    )

        else:
            raise NotImplementedError(
                "encode_strong_df expects a pandas "
                "DataFrame, Series, list, or numpy array."
            )

        return y

    def decode_weak(self, labels):
        """Decode a many-hot weak-label vector."""

        result_labels = []

        for i, value in enumerate(labels):

            if value == 1:
                result_labels.append(
                    self.labels[i]
                )

        return result_labels

    def decode_strong(self, labels):
        """Decode frame-level predictions into events."""

        result_labels = []

        for i, label_column in enumerate(labels.T):

            change_indices = (
                DecisionEncoder()
                .find_contiguous_regions(
                    label_column
                )
            )

            for row in change_indices:

                result_labels.append(
                    [
                        self.labels[i],
                        self._frame_to_time(row[0]),
                        self._frame_to_time(row[1]),
                    ]
                )

        return result_labels

    def state_dict(self):
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "net_pooling": self.net_pooling,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):

        return cls(
            labels=state_dict["labels"],
            audio_len=state_dict["audio_len"],
            frame_len=state_dict["frame_len"],
            frame_hop=state_dict["frame_hop"],
            net_pooling=state_dict["net_pooling"],
            fs=state_dict["fs"],
        )
