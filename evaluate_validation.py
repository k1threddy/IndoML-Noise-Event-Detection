import json
from pathlib import Path


HERE = Path(__file__).resolve().parent

GT_FILE = HERE / "gt_validation.jsonl"
#GT_FILE = HERE / "gt_gold_train.jsonl"
PRED_FILE = HERE /"work"/"kevanth"/"inference"/ "pred_validation.jsonl"


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            data[record["clip_id"]] = [
                (
                    float(event["onset"]),
                    float(event["offset"]),
                )
                for event in record["events"]
            ]

    return data


# ============================================================
# Event-based F1
# ============================================================

def match_events(
    ref_events,
    pred_events,
    tolerance_frac=0.20,
):
    matched_ref = set()
    matched_pred = set()
    candidates = []

    for ri, (r_on, r_off) in enumerate(ref_events):

        # Exact Codabench tolerance:
        # max(20% of reference duration, 50 ms)
        tol = max(
            tolerance_frac * (r_off - r_on),
            0.05,
        )

        for pi, (p_on, p_off) in enumerate(pred_events):

            if (
                abs(p_on - r_on) <= tol
                and
                abs(p_off - r_off) <= tol
            ):
                candidates.append(
                    (
                        abs(p_on - r_on)
                        + abs(p_off - r_off),
                        ri,
                        pi,
                    )
                )

    # Closest-first greedy matching
    for _, ri, pi in sorted(candidates):

        if (
            ri not in matched_ref
            and pi not in matched_pred
        ):
            matched_ref.add(ri)
            matched_pred.add(pi)

    tp = len(matched_ref)
    fp = len(pred_events) - tp
    fn = len(ref_events) - tp

    return tp, fp, fn


def event_based_f1(ref_data, pred_data):

    TP = 0
    FP = 0
    FN = 0

    # Evaluate every reference clip.
    for clip_id, ref_events in ref_data.items():

        pred_events = pred_data.get(
            clip_id,
            [],
        )

        tp, fp, fn = match_events(
            ref_events,
            pred_events,
        )

        TP += tp
        FP += fp
        FN += fn

    # Extra prediction clips are false positives.
    for clip_id, pred_events in pred_data.items():

        if clip_id not in ref_data:
            FP += len(pred_events)

    precision = (
        TP / (TP + FP)
        if TP + FP
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return f1, precision, recall, TP, FP, FN


# ============================================================
# Segment Dice
# ============================================================

def events_to_frames(
    events,
    max_time,
    frame_len=0.01,
):
    """
    Rasterize events onto a 10 ms grid.
    """

    n_frames = int(
        max_time / frame_len
    ) + 1

    mask = [0] * n_frames

    for on, off in events:

        start = int(
            on / frame_len
        )

        end = min(
            int(off / frame_len) + 1,
            n_frames,
        )

        for i in range(start, end):
            mask[i] = 1

    return mask


def segment_dice(ref_data, pred_data):

    scores = []

    for clip_id, ref_events in ref_data.items():

        pred_events = pred_data.get(
            clip_id,
            [],
        )

        all_events = (
            ref_events + pred_events
        )

        # Both reference and prediction are empty.
        if not all_events:
            scores.append(1.0)
            continue

        max_time = (
            max(
                off
                for _, off in all_events
            )
            + 0.5
        )

        ref_mask = events_to_frames(
            ref_events,
            max_time,
        )

        pred_mask = events_to_frames(
            pred_events,
            max_time,
        )

        intersection = sum(
            r & p
            for r, p in zip(
                ref_mask,
                pred_mask,
            )
        )

        total = (
            sum(ref_mask)
            + sum(pred_mask)
        )

        dice = (
            1.0
            if total == 0
            else 2.0 * intersection / total
        )

        scores.append(dice)

    return (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )


# ============================================================
# Main
# ============================================================

def main():

    print("Loading GT...")
    gt_data = load_jsonl(GT_FILE)

    print("Loading predictions...")
    pred_data = load_jsonl(PRED_FILE)

    print()
    print(f"GT clips          : {len(gt_data)}")
    print(f"Prediction clips  : {len(pred_data)}")

    # Sanity check
    missing_predictions = (
        set(gt_data) - set(pred_data)
    )

    extra_predictions = (
        set(pred_data) - set(gt_data)
    )

    print(
        f"Missing predictions: "
        f"{len(missing_predictions)}"
    )

    print(
        f"Extra predictions  : "
        f"{len(extra_predictions)}"
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    (
        f1,
        precision,
        recall,
        tp,
        fp,
        fn,
    ) = event_based_f1(
        gt_data,
        pred_data,
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = segment_dice(
        gt_data,
        pred_data,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    print(f"Event F1      : {f1:.6f}")
    print(f"Segment Dice  : {dice:.6f}")

    print()
    print(f"Precision     : {precision:.6f}")
    print(f"Recall        : {recall:.6f}")

    print()
    print(f"TP            : {tp}")
    print(f"FP            : {fp}")
    print(f"FN            : {fn}")

    print("=" * 50)


if __name__ == "__main__":
    main()