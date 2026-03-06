from src.data.manifest import build_subject_manifest


def test_build_subject_manifest_uses_real_layout():
    manifest = build_subject_manifest("/data/algonauts_2023_challenge_data", subject=1)
    assert manifest.subject_id == "subj01"
    assert manifest.training_fmri_lh.exists()
    assert manifest.training_images_dir.exists()
    assert manifest.roi_dir.exists()
