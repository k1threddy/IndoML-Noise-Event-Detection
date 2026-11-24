import json
from pathlib import Path


VALIDATION_DIR = Path(
    "/mnt/Work/Datasets/indoml/Vaani-Noise-Event-Dataset/validation"
)

METADATA_FILE = (
    VALIDATION_DIR / "validationMetadata.json"
)

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "gt_validation.jsonl"
)


def main():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    print(f"Validation clips: {len(metadata)}")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for item in metadata:

            clip_id = Path(
                item["segmentFileName"]
            ).stem

            events = []

            timestamps = item.get(
                "NoiseSubCategoryTimeStamp"
            )

            if timestamps is not None:

                for event in timestamps:

                    if not event:
                        continue

                    events.append(
                        {
                            "onset": float(
                                event["start"]
                            ),
                            "offset": float(
                                event["end"]
                            ),
                        }
                    )

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

    print(f"Saved GT to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()