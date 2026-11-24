import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent

INPUT_FILE = (
    HERE / "metadata/gold_windows.jsonl"
)

OUTPUT_FILE = (
    HERE / "gt_gold_train.jsonl"
)


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            data.append(
                json.loads(line)
            )

    return data


# ============================================================
# decision_unify
#
# Same logic as inference.
#
# Window-local events
#        ↓
# Global timestamps
#        ↓
# Union overlapping events
# ============================================================

def decision_unify(
    window_events,
    window_starts,
):

    all_events = []

    # --------------------------------------------------------
    # Local -> global coordinates
    # --------------------------------------------------------

    for events, window_start in zip(
        window_events,
        window_starts,
    ):

        for event in events:

            global_onset = (
                window_start
                + float(event["onset"])
            )

            global_offset = (
                window_start
                + float(event["offset"])
            )

            all_events.append(
                {
                    "onset": global_onset,
                    "offset": global_offset,
                }
            )

    # --------------------------------------------------------
    # No events
    # --------------------------------------------------------

    if not all_events:
        return []

    # --------------------------------------------------------
    # Sort by onset
    # --------------------------------------------------------

    all_events.sort(
        key=lambda event: (
            event["onset"],
            event["offset"],
        )
    )

    # --------------------------------------------------------
    # Union overlapping events
    # --------------------------------------------------------

    unified_events = [
        all_events[0].copy()
    ]

    for event in all_events[1:]:

        current = unified_events[-1]

        if (
            event["onset"]
            <= current["offset"]
        ):

            current["offset"] = max(
                current["offset"],
                event["offset"],
            )

        else:

            unified_events.append(
                event.copy()
            )

    return unified_events


# ============================================================
# Group windows by original clip
# ============================================================

def group_by_clip(metadata):

    clips = {}

    for item in metadata:

        # Original Vaani clip identity.
        key = (
            item["parquet_file"],
            item["row_index"],
        )

        if key not in clips:

            clips[key] = {
                "audio_path": item[
                    "audio_path"
                ],
                "windows": [],
            }

        clips[key]["windows"].append(
            item
        )

    return clips


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("GENERATING GOLD TRAINING GT")
    print("=" * 80)

    print(
        f"Input : {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Load window-level metadata
    # --------------------------------------------------------

    metadata = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nGold windows: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Group windows into original clips
    # --------------------------------------------------------

    clips = group_by_clip(
        metadata
    )

    print(
        f"Original Gold clips: {len(clips)}"
    )

    # --------------------------------------------------------
    # Generate clip-level GT
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for clip_idx, (
            key,
            clip,
        ) in enumerate(
            clips.items(),
            start=1,
        ):

            # Sort windows by their global start time.
            windows = sorted(
                clip["windows"],
                key=lambda item: (
                    item["window_start"]
                ),
            )

            window_events = [
                item.get(
                    "events",
                    []
                )
                for item in windows
            ]

            window_starts = [
                float(
                    item["window_start"]
                )
                for item in windows
            ]

            # Convert local GT events to global
            # clip-level events and merge overlaps.
            events = decision_unify(
                window_events,
                window_starts,
            )

            # Use exactly the same clip ID convention
            # as infer_train.py.
            clip_id = Path(
                str(
                    clip["audio_path"]
                )
            ).stem

            record = {
                "clip_id": clip_id,
                "events": events,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (
                clip_idx == 1
                or clip_idx % 1000 == 0
                or clip_idx == len(clips)
            ):

                print(
                    f"[{clip_idx}/"
                    f"{len(clips)}] "
                    f"{clip_id}: "
                    f"{len(events)} GT events"
                )

    print()
    print("=" * 80)
    print("GT GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Gold windows : {len(metadata)}"
    )

    print(
        f"Gold clips   : {len(clips)}"
    )

    print(
        f"Saved GT     : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()