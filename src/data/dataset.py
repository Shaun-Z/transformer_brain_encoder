from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms

from src.data.manifest import SubjectManifest


@dataclass(slots=True)
class Sample:
    image: torch.Tensor
    lh_fmri: torch.Tensor
    rh_fmri: torch.Tensor


def default_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class AlgonautsDataset(Dataset[Sample]):
    def __init__(
        self,
        manifest: SubjectManifest,
        indices: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest = manifest
        self.indices = np.asarray(indices)
        self.transform = transform or default_image_transform()

        self.image_paths = sorted(Path(manifest.training_images_dir).glob("*.png"))
        self.lh_fmri = np.load(manifest.training_fmri_lh)
        self.rh_fmri = np.load(manifest.training_fmri_rh)

        if len(self.image_paths) != self.lh_fmri.shape[0] or len(self.image_paths) != self.rh_fmri.shape[0]:
            raise ValueError(
                "Training images and fMRI arrays have mismatched lengths: "
                f"{len(self.image_paths)} images, {self.lh_fmri.shape[0]} LH, {self.rh_fmri.shape[0]} RH."
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> Sample:
        dataset_index = int(self.indices[item])
        image = Image.open(self.image_paths[dataset_index]).convert("RGB")
        image_tensor = self.transform(image)
        image_tensor = pad_to_patch_multiple(image_tensor, patch_size=14)

        return Sample(
            image=image_tensor,
            lh_fmri=torch.from_numpy(self.lh_fmri[dataset_index]).float(),
            rh_fmri=torch.from_numpy(self.rh_fmri[dataset_index]).float(),
        )


def collate_samples(batch: list[Sample]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([sample.image for sample in batch], dim=0),
        "lh_fmri": torch.stack([sample.lh_fmri for sample in batch], dim=0),
        "rh_fmri": torch.stack([sample.rh_fmri for sample in batch], dim=0),
    }


def pad_to_patch_multiple(image_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = image_tensor.shape
    padded_height = ((height + patch_size - 1) // patch_size) * patch_size
    padded_width = ((width + patch_size - 1) // patch_size) * patch_size
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom == 0 and pad_right == 0:
        return image_tensor
    return TF.pad(image_tensor, [0, 0, pad_right, pad_bottom])
