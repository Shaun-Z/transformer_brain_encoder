from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.manifest import SubjectManifest


ROI_MAPPING_FILES = (
    "mapping_prf-visualrois.npy",
    "mapping_floc-bodies.npy",
    "mapping_floc-faces.npy",
    "mapping_floc-places.npy",
    "mapping_floc-words.npy",
    "mapping_streams.npy",
)

LH_ROI_FILES = (
    "lh.prf-visualrois_challenge_space.npy",
    "lh.floc-bodies_challenge_space.npy",
    "lh.floc-faces_challenge_space.npy",
    "lh.floc-places_challenge_space.npy",
    "lh.floc-words_challenge_space.npy",
    "lh.streams_challenge_space.npy",
)

RH_ROI_FILES = (
    "rh.prf-visualrois_challenge_space.npy",
    "rh.floc-bodies_challenge_space.npy",
    "rh.floc-faces_challenge_space.npy",
    "rh.floc-places_challenge_space.npy",
    "rh.floc-words_challenge_space.npy",
    "rh.streams_challenge_space.npy",
)


@dataclass(slots=True)
class RoiBundle:
    readout_name: str
    lh_challenge_rois: torch.Tensor
    rh_challenge_rois: torch.Tensor
    lh_roi_names: list[str]
    rh_roi_names: list[str]
    num_queries: int


def _load_mapping(path: Path) -> dict[int, str]:
    return np.load(path, allow_pickle=True).item()


def _readout_indices(readout_name: str) -> tuple[list[int], int]:
    if readout_name == "visuals":
        return [0], 16
    if readout_name == "bodies":
        return [1], 16
    if readout_name == "faces":
        return [2], 16
    if readout_name == "places":
        return [3], 16
    if readout_name == "words":
        return [4], 16
    if readout_name in {"streams", "streams_inc"}:
        return [5], 16
    if readout_name == "hemis":
        return [5], 2
    if readout_name in {"voxels", "rois_all"}:
        return [0, 1, 2, 3, 4], -1
    raise ValueError(f"Unsupported readout '{readout_name}'")


def load_roi_bundle(manifest: SubjectManifest, readout_name: str) -> RoiBundle:
    roi_name_maps = [_load_mapping(manifest.roi_dir / filename) for filename in ROI_MAPPING_FILES]
    lh_rois = [np.load(manifest.roi_dir / filename) for filename in LH_ROI_FILES]
    rh_rois = [np.load(manifest.roi_dir / filename) for filename in RH_ROI_FILES]

    selected_indices, num_queries = _readout_indices(readout_name)
    lh_masks: list[torch.Tensor] = []
    rh_masks: list[torch.Tensor] = []
    lh_roi_names: list[str] = []
    rh_roi_names: list[str] = []

    for roi_index in selected_indices:
        lh_tensor = torch.as_tensor(lh_rois[roi_index])
        rh_tensor = torch.as_tensor(rh_rois[roi_index])
        roi_mapping = roi_name_maps[roi_index]
        for label in range(1, len(roi_mapping)):
            lh_masks.append(torch.where(lh_tensor == label, 1, 0))
            rh_masks.append(torch.where(rh_tensor == label, 1, 0))
            lh_roi_names.append(roi_mapping[label])
            rh_roi_names.append(roi_mapping[label])

    lh_stack = torch.vstack(lh_masks)
    rh_stack = torch.vstack(rh_masks)

    lh_unknown = torch.where(lh_stack.sum(0) == 0, 1, 0)
    rh_unknown = torch.where(rh_stack.sum(0) == 0, 1, 0)
    lh_stack = torch.cat((lh_stack, lh_unknown[None, :]), dim=0)
    rh_stack = torch.cat((rh_stack, rh_unknown[None, :]), dim=0)

    if readout_name == "voxels":
        num_queries = lh_stack.shape[1] + rh_stack.shape[1]
    elif readout_name == "rois_all":
        num_queries = lh_stack.shape[0] + rh_stack.shape[0]

    return RoiBundle(
        readout_name=readout_name,
        lh_challenge_rois=lh_stack,
        rh_challenge_rois=rh_stack,
        lh_roi_names=lh_roi_names,
        rh_roi_names=rh_roi_names,
        num_queries=num_queries,
    )
