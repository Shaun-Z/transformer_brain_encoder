from src.data.manifest import build_subject_manifest
from src.data.rois import load_roi_bundle


def test_load_roi_bundle_returns_masks():
    manifest = build_subject_manifest("/data/algonauts_2023_challenge_data", subject=1)
    bundle = load_roi_bundle(manifest, "rois_all")
    assert bundle.lh_challenge_rois.ndim == 2
    assert bundle.rh_challenge_rois.ndim == 2
    assert bundle.num_queries > 0
