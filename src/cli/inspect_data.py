from __future__ import annotations

import argparse
import json

from src.data.manifest import build_subject_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Algonauts subject layout")
    parser.add_argument("--dataset-root", default="/data/algonauts_2023_challenge_data")
    parser.add_argument("--subject", type=int, required=True)
    args = parser.parse_args()

    manifest = build_subject_manifest(args.dataset_root, args.subject)
    summary = {
        "subject_id": manifest.subject_id,
        "training_images_dir": str(manifest.training_images_dir),
        "training_fmri_lh": str(manifest.training_fmri_lh),
        "training_fmri_rh": str(manifest.training_fmri_rh),
        "released_test_images_dir": str(manifest.released_test_images_dir),
        "challenge_test_images_dir": str(manifest.challenge_test_images_dir),
        "roi_dir": str(manifest.roi_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
