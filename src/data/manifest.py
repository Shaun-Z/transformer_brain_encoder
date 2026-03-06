from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SubjectManifest:
    dataset_root: Path
    subject: int
    subject_id: str
    train_subject_dir: Path
    test_subject_dir: Path
    training_images_dir: Path
    training_fmri_dir: Path
    training_fmri_lh: Path
    training_fmri_rh: Path
    released_test_images_dir: Path
    challenge_test_images_dir: Path
    roi_dir: Path


def _require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def build_subject_manifest(dataset_root: str | Path, subject: int) -> SubjectManifest:
    root = Path(dataset_root).expanduser().resolve()
    subject_id = f"subj{int(subject):02d}"

    train_subject_dir = _require_path(root / "train_data" / subject_id, "training subject directory")
    test_subject_dir = root / "test_data" / subject_id
    training_images_dir = _require_path(
        train_subject_dir / "training_split" / "training_images",
        "training images directory",
    )
    training_fmri_dir = _require_path(
        train_subject_dir / "training_split" / "training_fmri",
        "training fMRI directory",
    )
    training_fmri_lh = _require_path(training_fmri_dir / "lh_training_fmri.npy", "left hemisphere training data")
    training_fmri_rh = _require_path(training_fmri_dir / "rh_training_fmri.npy", "right hemisphere training data")
    released_test_images_dir = _require_path(
        train_subject_dir / "test_split" / "test_images",
        "released test images directory",
    )
    roi_dir = _require_path(train_subject_dir / "roi_masks", "ROI directory")
    challenge_test_images_dir = test_subject_dir / "test_split" / "test_images"

    return SubjectManifest(
        dataset_root=root,
        subject=int(subject),
        subject_id=subject_id,
        train_subject_dir=train_subject_dir,
        test_subject_dir=test_subject_dir,
        training_images_dir=training_images_dir,
        training_fmri_dir=training_fmri_dir,
        training_fmri_lh=training_fmri_lh,
        training_fmri_rh=training_fmri_rh,
        released_test_images_dir=released_test_images_dir,
        challenge_test_images_dir=challenge_test_images_dir,
        roi_dir=roi_dir,
    )
